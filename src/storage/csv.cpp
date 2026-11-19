#include "storage/csv.h"

#include <charconv>
#include <cstring>
#include <fstream>
#include <unordered_map>

namespace warpdb {
namespace {

// Splits one line into at most `max_fields` views. Returns the number of fields.
size_t split_line(const char* p, const char* e, char delim, std::string_view* out, size_t max_fields) {
  size_t n = 0;
  for (;;) {
    if (n >= max_fields) throw Error("csv: too many fields in line");
    if (p < e && *p == '"') {
      const char* q = static_cast<const char*>(std::memchr(p + 1, '"', e - (p + 1)));
      if (!q) throw Error("csv: unterminated quoted field");
      if (q + 1 < e && q[1] == '"') throw Error("csv: escaped quotes inside fields are not supported");
      out[n++] = std::string_view(p + 1, q - (p + 1));
      p = q + 1;
      if (p < e && *p != delim) throw Error("csv: garbage after closing quote");
    } else {
      const char* f = p < e ? static_cast<const char*>(std::memchr(p, delim, e - p)) : nullptr;
      if (!f) f = e;
      out[n++] = std::string_view(p, f - p);
      p = f;
    }
    if (p < e && *p == delim) {
      ++p;
      if (p == e) {  // trailing delimiter -> one more (empty) field
        if (n >= max_fields) throw Error("csv: too many fields in line");
        out[n++] = std::string_view();
        break;
      }
      continue;
    }
    break;
  }
  return n;
}

enum : uint8_t { CAN_I32 = 1, CAN_I64 = 2, CAN_F64 = 4, CAN_DATE = 8, CAN_ALL = 15 };

inline bool parse_i64(std::string_view s, int64_t& v) {
  if (s.empty()) return false;
  auto r = std::from_chars(s.data(), s.data() + s.size(), v);
  return r.ec == std::errc() && r.ptr == s.data() + s.size();
}
inline bool parse_f64(std::string_view s, double& v) {
  if (s.empty()) return false;
  char c = s[0];
  if (!((c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.')) return false;
  if (c == '+') s.remove_prefix(1);
  auto r = std::from_chars(s.data(), s.data() + s.size(), v);
  return r.ec == std::errc() && r.ptr == s.data() + s.size();
}

uint8_t classify(std::string_view s, uint8_t mask) {
  if (mask & (CAN_I32 | CAN_I64)) {
    int64_t v;
    if (parse_i64(s, v)) {
      if (v < INT32_MIN || v > INT32_MAX) mask &= static_cast<uint8_t>(~CAN_I32);
    } else {
      mask &= static_cast<uint8_t>(~(CAN_I32 | CAN_I64));
    }
  }
  if (mask & CAN_F64) {
    double d;
    if (!parse_f64(s, d)) mask &= static_cast<uint8_t>(~CAN_F64);
  }
  if (mask & CAN_DATE) {
    int32_t d;
    if (!parse_date(s, d)) mask &= static_cast<uint8_t>(~CAN_DATE);
  }
  return mask;
}

}  // namespace

std::shared_ptr<Table> load_csv(const std::string& path, const std::string& table_name,
                                const CsvOptions& opt) {
  // ---- read whole file ----
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw Error("cannot open file: " + path);
  size_t size = static_cast<size_t>(f.tellg());
  f.seekg(0);
  std::string buf(size, '\0');
  f.read(buf.data(), static_cast<std::streamsize>(size));
  f.close();

  const char* base = buf.data();
  size_t off = 0;
  if (size >= 3 && (uint8_t)base[0] == 0xEF && (uint8_t)base[1] == 0xBB && (uint8_t)base[2] == 0xBF) off = 3;

  // ---- line starts ----
  std::vector<uint64_t> starts;
  starts.reserve(size / 64 + 16);
  {
    size_t p = off;
    while (p < size) {
      starts.push_back(p);
      const char* nl = static_cast<const char*>(std::memchr(base + p, '\n', size - p));
      if (!nl) break;
      p = static_cast<size_t>(nl - base) + 1;
    }
  }
  auto line_range = [&](size_t i, const char*& b, const char*& e) {
    b = base + starts[i];
    e = (i + 1 < starts.size()) ? base + starts[i + 1] - 1 : base + size;
    if (e > b && e[-1] == '\r') --e;
    if (e > b && *(e - 1) == '\n') --e;
  };
  // drop trailing empty lines
  while (!starts.empty()) {
    const char *b, *e;
    line_range(starts.size() - 1, b, e);
    if (b == e) starts.pop_back();
    else break;
  }
  if (starts.empty()) throw Error("csv: empty file");

  // ---- header / column count ----
  std::vector<std::string> names;
  size_t first_data = 0;
  std::vector<std::string_view> fields(4096);
  {
    const char *b, *e;
    line_range(0, b, e);
    size_t nf = split_line(b, e, opt.delimiter, fields.data(), fields.size());
    if (opt.header) {
      for (size_t i = 0; i < nf; ++i) names.push_back(to_lower(std::string(fields[i])));
      if (!names.empty() && names.back().empty()) names.pop_back();  // trailing delimiter
      first_data = 1;
    } else if (!opt.column_names.empty()) {
      names = opt.column_names;
    } else {
      size_t cnt = nf;
      if (cnt > 0 && fields[cnt - 1].empty()) --cnt;
      for (size_t i = 0; i < cnt; ++i) names.push_back("c" + std::to_string(i));
    }
  }
  const size_t ncols = names.size();
  if (ncols == 0) throw Error("csv: no columns");
  const size_t nrows = starts.size() - first_data;

  auto split_row = [&](size_t row, std::string_view* out) {
    const char *b, *e;
    line_range(first_data + row, b, e);
    size_t nf = split_line(b, e, opt.delimiter, out, ncols + 1);
    if (nf == ncols + 1 && out[ncols].empty()) nf = ncols;  // trailing delimiter
    if (nf != ncols)
      throw Error("csv: line " + std::to_string(first_data + row + 1) + " has " + std::to_string(nf) +
                  " fields, expected " + std::to_string(ncols));
  };

  // ---- pass 1: type inference ----
  size_t nch = std::max<size_t>(1, num_chunks(nrows, 1 << 14));
  std::vector<std::vector<uint8_t>> masks(nch, std::vector<uint8_t>(ncols, CAN_ALL));
  parallel_for(nrows, 1 << 14, [&](size_t c, size_t b, size_t e) {
    std::vector<std::string_view> fl(ncols + 1);
    auto& m = masks[c];
    for (size_t r = b; r < e; ++r) {
      split_row(r, fl.data());
      for (size_t j = 0; j < ncols; ++j)
        if (m[j]) m[j] = classify(fl[j], m[j]);
    }
  });
  std::vector<Type> types(ncols);
  for (size_t j = 0; j < ncols; ++j) {
    uint8_t m = CAN_ALL;
    for (auto& v : masks) m &= v[j];
    if (nrows == 0) m = 0;
    types[j] = (m & CAN_I32)    ? Type::INT32
               : (m & CAN_I64)  ? Type::INT64
               : (m & CAN_F64)  ? Type::FLOAT64
               : (m & CAN_DATE) ? Type::DATE
                                : Type::VARCHAR;
  }

  // ---- pass 2: parse ----
  auto tab = std::make_shared<Table>();
  tab->name = table_name;
  tab->names = names;
  tab->nrows = nrows;
  std::vector<std::vector<std::string_view>> svs(ncols);  // temp views for VARCHAR columns
  for (size_t j = 0; j < ncols; ++j) {
    auto col = std::make_shared<Column>();
    col->type = types[j];
    switch (types[j]) {
      case Type::INT32:
      case Type::DATE: col->i32.resize(nrows); break;
      case Type::INT64: col->i64.resize(nrows); break;
      case Type::FLOAT64: col->f64.resize(nrows); break;
      case Type::VARCHAR: svs[j].resize(nrows); break;
    }
    tab->cols.push_back(col);
  }
  parallel_for(nrows, 1 << 14, [&](size_t, size_t b, size_t e) {
    std::vector<std::string_view> fl(ncols + 1);
    for (size_t r = b; r < e; ++r) {
      split_row(r, fl.data());
      for (size_t j = 0; j < ncols; ++j) {
        Column& c = *tab->cols[j];
        switch (c.type) {
          case Type::INT32: {
            int64_t v;
            if (!parse_i64(fl[j], v)) throw Error("csv: bad int");
            c.i32[r] = static_cast<int32_t>(v);
            break;
          }
          case Type::INT64:
            if (!parse_i64(fl[j], c.i64[r])) throw Error("csv: bad int");
            break;
          case Type::FLOAT64: {
            double d;
            if (!parse_f64(fl[j], d)) throw Error("csv: bad double");
            c.f64[r] = d;
            break;
          }
          case Type::DATE:
            if (!parse_date(fl[j], c.i32[r])) throw Error("csv: bad date");
            break;
          case Type::VARCHAR: svs[j][r] = fl[j]; break;
        }
      }
    }
  });

  // ---- dictionary-encode VARCHAR columns (sorted dictionary), parallel across columns ----
  std::vector<size_t> vcols;
  for (size_t j = 0; j < ncols; ++j)
    if (types[j] == Type::VARCHAR) vcols.push_back(j);
  parallel_for(vcols.size(), 1, [&](size_t, size_t b, size_t e) {
    for (size_t k = b; k < e; ++k) {
      size_t j = vcols[k];
      Column& c = *tab->cols[j];
      auto& v = svs[j];
      std::unordered_map<std::string_view, int32_t> ids;
      std::vector<std::string_view> uniq;
      std::vector<int32_t> prov(nrows);
      for (size_t r = 0; r < nrows; ++r) {
        auto it = ids.find(v[r]);
        if (it == ids.end()) {
          it = ids.emplace(v[r], static_cast<int32_t>(uniq.size())).first;
          uniq.push_back(v[r]);
        }
        prov[r] = it->second;
      }
      std::vector<int32_t> order(uniq.size());
      for (size_t u = 0; u < order.size(); ++u) order[u] = static_cast<int32_t>(u);
      std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b2) { return uniq[a] < uniq[b2]; });
      std::vector<int32_t> rank(uniq.size());
      c.dict.resize(uniq.size());
      for (size_t r = 0; r < order.size(); ++r) {
        rank[order[r]] = static_cast<int32_t>(r);
        c.dict[r] = std::string(uniq[order[r]]);
      }
      c.i32.resize(nrows);
      for (size_t r = 0; r < nrows; ++r) c.i32[r] = rank[prov[r]];
      std::vector<std::string_view>().swap(v);
    }
  });
  return tab;
}

}  // namespace warpdb
