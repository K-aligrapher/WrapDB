#include <io.h>

#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>

#include "exec/engine.h"
#include "gpu/device.h"
#include "gpu/gpu_backend.h"
#include "storage/csv.h"

using namespace warpdb;

namespace {

struct Settings {
  bool csv = false;       // CSV output (machine readable)
  bool protocol = false;  // emit @@END / @@ERR markers after every statement
  bool timer = true;
};

std::string csv_escape(const std::string& s) {
  if (s.find_first_of(",\"\n\r") == std::string::npos) return s;
  std::string o = "\"";
  for (char c : s) {
    if (c == '"') o += '"';
    o += c;
  }
  return o + "\"";
}

void print_result(const QueryResult& r, const Settings& st) {
  if (r.explain) {
    for (auto& l : r.trace) std::printf("  %s\n", l.c_str());
    return;
  }
  const size_t nc = r.cols.size();
  if (st.csv) {
    for (size_t j = 0; j < nc; ++j) std::printf("%s%s", j ? "," : "", csv_escape(r.names[j]).c_str());
    std::printf("\n");
    for (size_t i = 0; i < r.nrows; ++i) {
      for (size_t j = 0; j < nc; ++j)
        std::printf("%s%s", j ? "," : "", csv_escape(format_cell(*r.cols[j], i)).c_str());
      std::printf("\n");
    }
    return;
  }
  const size_t shown = std::min<size_t>(r.nrows, 40);
  std::vector<size_t> w(nc);
  std::vector<std::vector<std::string>> cells(shown, std::vector<std::string>(nc));
  for (size_t j = 0; j < nc; ++j) {
    w[j] = r.names[j].size();
    for (size_t i = 0; i < shown; ++i) {
      cells[i][j] = format_cell(*r.cols[j], i);
      w[j] = std::max(w[j], cells[i][j].size());
    }
  }
  auto line = [&]() {
    for (size_t j = 0; j < nc; ++j) std::printf("+%s", std::string(w[j] + 2, '-').c_str());
    std::printf("+\n");
  };
  line();
  for (size_t j = 0; j < nc; ++j) std::printf("| %-*s ", static_cast<int>(w[j]), r.names[j].c_str());
  std::printf("|\n");
  line();
  for (size_t i = 0; i < shown; ++i) {
    for (size_t j = 0; j < nc; ++j) {
      bool num = r.cols[j]->kind == Value::I || r.cols[j]->kind == Value::F;
      if (num && !r.cols[j]->is_date) std::printf("| %*s ", static_cast<int>(w[j]), cells[i][j].c_str());
      else std::printf("| %-*s ", static_cast<int>(w[j]), cells[i][j].c_str());
    }
    std::printf("|\n");
  }
  line();
  if (r.nrows > shown) std::printf("... %zu rows total (%zu shown)\n", r.nrows, shown);
  else std::printf("%zu row%s\n", r.nrows, r.nrows == 1 ? "" : "s");
}

std::vector<std::string> split_ws(const std::string& s) {
  std::istringstream is(s);
  std::vector<std::string> v;
  for (std::string t; is >> t;) v.push_back(t);
  return v;
}

const char* HELP =
    "Commands:\n"
    "  .load <table> <file> [delim] [noheader]   load a CSV file (delim: a character, or 'tab' / 'pipe')\n"
    "  .tables                                   list tables\n"
    "  .schema <table>                           show columns\n"
    "  .mode csv|table                           output format\n"
    "  .timer on|off                             show timings\n"
    "  .gpu                                      show GPU info\n"
    "  .device cpu|gpu                           choose where queries execute\n"
    "  .gpustats                                 GPU memory / column-cache statistics\n"
    "  .dropcache                                evict resident GPU columns (next GPU query runs cold)\n"
    "  .quit                                     exit\n"
    "SQL: SELECT ... FROM ... [JOIN ... ON ...] [WHERE] [GROUP BY] [HAVING] [ORDER BY] [LIMIT]; prefix with EXPLAIN to see the execution trace.\n";

std::shared_ptr<gpu::GpuBackend> g_gpu;  // created at startup if a CUDA device exists

// returns false to quit
bool run_command(Engine& eng, Settings& st, const std::string& line) {
  auto a = split_ws(line);
  const std::string& c = a[0];
  if (c == ".quit" || c == ".exit") return false;
  if (c == ".device") {
    if (a.size() < 2 || (a[1] != "cpu" && a[1] != "gpu")) throw Error("usage: .device cpu|gpu");
    if (a[1] == "gpu" && !g_gpu) throw Error("no CUDA device available");
    eng.device = a[1];
    std::printf("device = %s\n", eng.device.c_str());
    return true;
  }
  if (c == ".gpustats" || c == ".dropcache") {
    if (!g_gpu) throw Error("no CUDA device available");
    if (c == ".dropcache") {
      g_gpu->drop_cache();
      std::printf("GPU column cache cleared\n");
      return true;
    }
    auto m = g_gpu->mem_stats();
    std::printf("column cache: %.1f / %.1f MB   allocator: %.1f MB in use, %.1f MB peak, %.1f MB reserved   uploaded total: %.1f MB\n",
                m.cache_bytes / 1048576.0, m.cache_budget / 1048576.0, m.in_use / 1048576.0, m.peak / 1048576.0,
                m.reserved / 1048576.0, m.uploaded_total / 1048576.0);
    return true;
  }
  if (c == ".help") { std::printf("%s", HELP); return true; }
  if (c == ".mode" && a.size() > 1) { st.csv = a[1] == "csv"; return true; }
  if (c == ".timer" && a.size() > 1) { st.timer = a[1] == "on"; return true; }
  if (c == ".tables") {
    for (auto& kv : eng.catalog.all())
      std::printf("%-20s %10zu rows  %8.1f MB\n", kv.second->name.c_str(), kv.second->nrows, kv.second->bytes() / 1048576.0);
    return true;
  }
  if (c == ".schema" && a.size() > 1) {
    auto t = eng.catalog.get(a[1]);
    if (!t) throw Error("table not found: " + a[1]);
    for (size_t i = 0; i < t->names.size(); ++i) std::printf("  %-24s %s\n", t->names[i].c_str(), type_name(t->cols[i]->type));
    return true;
  }
  if (c == ".gpu") {
    gpu::DeviceInfo d;
    if (!gpu::query_device(d)) { std::printf("no CUDA device\n"); return true; }
    std::printf("%s (sm_%d%d), %.2f GB VRAM, %d SMs, %.0f GB/s\n", d.name.c_str(), d.sm_major, d.sm_minor,
                d.total_mem / 1073741824.0, d.sm_count, d.peak_bandwidth_gbs());
    return true;
  }
  if (c == ".load") {
    if (a.size() < 3) throw Error("usage: .load <table> <file> [delim] [noheader]");
    CsvOptions o;
    for (size_t i = 3; i < a.size(); ++i) {
      if (a[i] == "noheader") o.header = false;
      else if (a[i] == "tab") o.delimiter = '\t';
      else if (a[i] == "pipe") o.delimiter = '|';
      else if (a[i].size() == 1) o.delimiter = a[i][0];
      else throw Error("bad option: " + a[i]);
    }
    Timer t;
    auto tab = load_csv(a[2], a[1], o);
    eng.catalog.add(tab);
    std::printf("loaded %s: %zu rows, %zu columns, %.1f MB in %.0f ms\n", tab->name.c_str(), tab->nrows,
                tab->names.size(), tab->bytes() / 1048576.0, t.ms());
    return true;
  }
  throw Error("unknown command: " + c + " (try .help)");
}

}  // namespace

int main(int argc, char** argv) {
  Engine eng;
  Settings st;
  std::string once;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--csv") st.csv = true;
    else if (a == "--protocol") { st.protocol = true; st.timer = false; }
    else if (a == "-c" && i + 1 < argc) once = argv[++i];
    else if (a == "--help" || a == "-h") { std::printf("warpdb [--csv] [--protocol] [-c \"sql\"]\n%s", HELP); return 0; }
  }
  {
    gpu::DeviceInfo d;
    if (gpu::query_device(d)) {
      try {
        g_gpu = gpu::make_gpu_backend();
        eng.set_gpu_backend(g_gpu);
      } catch (const std::exception&) {
        g_gpu.reset();
      }
    }
  }
  const bool tty = _isatty(_fileno(stdin)) != 0;
  if (tty && once.empty())
    std::printf("WarpDB v0.1 (%s) - type .help for help\n", g_gpu ? "CPU + GPU engines; .device gpu to use the GPU" : "CPU engine");

  auto run_sql = [&](const std::string& sql) {
    try {
      QueryResult r = eng.execute(sql);
      print_result(r, st);
      if (st.timer && !r.explain) std::printf("(%.2f ms)\n", r.ms);
      if (st.protocol) std::printf("@@END %.3f\n", r.ms);
    } catch (const std::exception& e) {
      if (st.protocol) std::printf("@@ERR %s\n", e.what());
      else std::printf("error: %s\n", e.what());
    }
    std::fflush(stdout);
  };

  if (!once.empty()) {
    run_sql(once);
    return 0;
  }

  std::string buf, line;
  if (tty) std::printf("warpdb> ");
  while (std::getline(std::cin, line)) {
    while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
    bool is_cmd = buf.empty() && !line.empty() && line[0] == '.';
    if (is_cmd) {
      try {
        if (!run_command(eng, st, line)) break;
        if (st.protocol) std::printf("@@END 0\n");
      } catch (const std::exception& e) {
        if (st.protocol) std::printf("@@ERR %s\n", e.what());
        else std::printf("error: %s\n", e.what());
      }
      std::fflush(stdout);
    } else if (!line.empty()) {
      buf += line + "\n";
      if (line.back() == ';') {
        run_sql(buf);
        buf.clear();
      }
    }
    if (tty) std::printf(buf.empty() ? "warpdb> " : "   ...> ");
  }
  if (!buf.empty()) run_sql(buf);
  return 0;
}
