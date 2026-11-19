// Cost-based CPU/GPU placement: estimates each stage's cardinalities, prices every stage on both devices with
// the calibrated cost model, adds the PCIe cost of columns that are not resident in VRAM, and picks the cheaper
// device. The reasoning is rendered as an EXPLAIN table so every decision can be audited.
#include <cmath>
#include <cstdio>
#include <set>

#include "exec/engine.h"

namespace warpdb {
namespace {

void collect_cols(const Expr& e, std::set<std::pair<int, int>>& out) {
  if (e.kind == Expr::Col) out.insert({e.tbl, e.col});
  for (auto& a : e.args) collect_cols(*a, out);
}

std::string group3(double v) {  // 1234567 -> "1,234,567"
  long long n = static_cast<long long>(std::llround(v));
  std::string s = std::to_string(n < 0 ? -n : n), out;
  for (size_t i = 0; i < s.size(); ++i) {
    if (i && (s.size() - i) % 3 == 0) out += ',';
    out += s[i];
  }
  return n < 0 ? "-" + out : out;
}

std::string fmt(const char* f, double a) {
  char b[64];
  std::snprintf(b, sizeof b, f, a);
  return b;
}

}  // namespace

PlacementEstimate Engine::estimate_placement(PlannedQuery& pq) {
  PlacementEstimate est;
  const int T = static_cast<int>(pq.tabs.size());
  const Select& s = pq.sel;

  auto add_stage = [&](StageStat st) {
    StageEstimate e;
    st.gpu = false;
    e.cpu_ms = cost.predict(st);
    st.gpu = true;
    e.gpu_ms = cost.predict(st);
    e.stat = st;
    est.stages.push_back(e);
  };

  // ---- scans: selectivity measured on a sample of each table ----
  std::vector<size_t> rel_n(T);
  for (int k = 0; k < T; ++k) {
    const double n = static_cast<double>(pq.tabs[k]->nrows);
    double sel = sample_selectivity(pq, k);
    double out = n == 0 ? 0 : std::max(1.0, std::round(n * sel));
    rel_n[k] = static_cast<size_t>(out);
    StageStat st;
    st.op = StageOp::Scan;
    st.label = pq.tabs[k]->name;
    st.rows_in = n;
    st.rows_out = out;
    st.k1 = distinct_columns(pq.push[k]);
    st.k2 = expr_nodes(pq.push[k]);
    add_stage(st);
  }

  // ---- joins: replay the greedy order the executors will use, with estimated sizes ----
  double cur_n = 0;
  {
    std::vector<bool> joined(T, false);
    int start = pick_start(pq, rel_n);
    joined[start] = true;
    cur_n = static_cast<double>(rel_n[start]);
    int ntab = 1;
    for (int remaining = T - 1; remaining > 0; --remaining) {
      int best = pick_next(pq, joined, static_cast<size_t>(cur_n), rel_n);
      if (best < 0) break;  // cross join: execution will report the error
      int nk = 0;
      double out = join_estimate(pq, joined, static_cast<size_t>(cur_n), best, rel_n, &nk);
      out = std::min(std::max(1.0, out), 4e9);
      StageStat st;
      st.op = StageOp::Join;
      st.label = pq.tabs[best]->name;
      st.rows_in = cur_n;
      st.rows_aux = static_cast<double>(rel_n[best]);
      st.rows_out = out;
      st.k1 = nk;
      st.k2 = ++ntab;
      add_stage(st);
      cur_n = out;
      joined[best] = true;
    }
  }
  if (!pq.residual.empty()) cur_n = std::max(1.0, cur_n * 0.3);

  // ---- aggregation ----
  double n_after = cur_n;
  if (pq.agg_mode) {
    double groups = 1;
    if (!s.group_by.empty()) {
      groups = 1;
      for (auto& g : s.group_by) groups *= key_ndv(pq, g, static_cast<size_t>(cur_n));
      groups = std::min(groups, cur_n);
    }
    StageStat st;
    st.op = StageOp::Aggregate;
    st.rows_in = cur_n;
    st.rows_out = groups;
    st.k1 = static_cast<int>(s.group_by.size());
    st.k2 = static_cast<int>(pq.aggs.size());
    st.k3 = agg_expr_nodes(pq);
    add_stage(st);
    n_after = s.having ? std::max(1.0, groups * 0.5) : groups;
  }

  // ---- sort / limit / projection ----
  if (!s.order_by.empty()) {
    StageStat st;
    st.op = StageOp::Sort;
    st.rows_in = n_after;
    st.rows_out = s.limit >= 0 ? std::min(n_after, static_cast<double>(s.limit)) : n_after;
    st.k1 = static_cast<int>(s.order_by.size());
    add_stage(st);
  }
  if (s.limit >= 0) n_after = std::min(n_after, static_cast<double>(s.limit));
  {
    StageStat st;
    st.op = StageOp::Project;
    st.rows_in = n_after;
    st.rows_out = n_after;
    st.k1 = static_cast<int>(s.items.size());
    add_stage(st);
  }

  // ---- residency: every referenced base column must be in VRAM for the GPU plan ----
  std::set<std::pair<int, int>> cols;
  for (auto& v : pq.push) for (auto& e : v) collect_cols(*e, cols);
  for (auto& e : pq.edges) { collect_cols(*e.ea, cols); collect_cols(*e.eb, cols); }
  for (auto& e : pq.residual) collect_cols(*e, cols);
  for (auto& e : s.group_by) collect_cols(*e, cols);
  for (auto& e : pq.aggs) collect_cols(*e, cols);
  for (auto& it : s.items) collect_cols(*it.e, cols);
  if (s.having) collect_cols(*s.having, cols);
  for (auto& o : s.order_by) collect_cols(*o.e, cols);

  double total_bytes = 0, cold_bytes = 0;
  est.cols_total = static_cast<int>(cols.size());
  for (auto& [t, c] : cols) {
    const Column& col = *pq.tabs[t]->cols[c];
    double bytes = static_cast<double>(col.size()) * ((col.type == Type::INT64 || col.type == Type::FLOAT64) ? 8 : 4);
    total_bytes += bytes;
    if (!gpu_ || !gpu_->column_resident(col)) {
      cold_bytes += bytes;
      ++est.cols_cold;
    }
  }
  est.upload_mb = cold_bytes / 1e6;
  if (est.cols_cold > 0) {
    StageStat up;
    up.op = StageOp::Upload;
    up.gpu = true;
    up.rows_in = est.upload_mb;
    est.upload_ms = cost.predict(up);
  }

  est.gpu_feasible = gpu_ != nullptr;
  if (!gpu_) est.gpu_reason = "no GPU backend";
  else if (total_bytes > static_cast<double>(gpu_->cache_budget_bytes())) {
    est.gpu_feasible = false;
    est.gpu_reason = "working set " + fmt("%.0f", total_bytes / 1e6) + " MB exceeds the VRAM column budget of " +
                     fmt("%.0f", static_cast<double>(gpu_->cache_budget_bytes()) / 1e6) + " MB";
  }

  for (auto& st : est.stages) {
    est.cpu_ms += st.cpu_ms;
    est.gpu_ms += st.gpu_ms;
  }
  est.gpu_ms += est.upload_ms;
  est.choice = (est.gpu_feasible && est.gpu_ms < est.cpu_ms) ? "gpu" : "cpu";

  // ---- explanation ----
  auto& L = est.lines;
  char b[200];
  L.push_back(std::string("placement estimate (cost model ") + (cost.calibrated ? "calibrated" : "UNCALIBRATED priors") + ")");
  std::snprintf(b, sizeof b, "  %-22s %-28s %10s %10s", "stage", "rows in -> out", "CPU ms", "GPU ms");
  L.push_back(b);
  for (auto& e : est.stages) {
    std::string name = stage_op_name(e.stat.op);
    if (!e.stat.label.empty()) name += " " + e.stat.label;
    std::string rows = group3(e.stat.rows_in) + (e.stat.op == StageOp::Join ? " x " + group3(e.stat.rows_aux) : "") +
                       " -> " + group3(e.stat.rows_out);
    std::snprintf(b, sizeof b, "  %-22.22s %-28.28s %10.3f %10.3f", name.c_str(), rows.c_str(), e.cpu_ms, e.gpu_ms);
    L.push_back(b);
  }
  std::snprintf(b, sizeof b, "  %-22s %-28s %10s %10.3f", "upload (PCIe)",
                (std::to_string(est.cols_cold) + "/" + std::to_string(est.cols_total) + " cold cols, " +
                 fmt("%.1f", est.upload_mb) + " MB")
                    .c_str(),
                "-", est.upload_ms);
  L.push_back(b);
  std::snprintf(b, sizeof b, "  %-22s %-28s %10.3f %10.3f", "TOTAL", "", est.cpu_ms, est.gpu_ms);
  L.push_back(b);
  std::string why;
  if (!est.gpu_feasible) why = " (GPU not feasible: " + est.gpu_reason + ")";
  else if (est.choice == "gpu") why = " (" + fmt("%.1f", est.cpu_ms / std::max(est.gpu_ms, 1e-9)) + "x faster than CPU)";
  else why = " (" + fmt("%.1f", est.gpu_ms / std::max(est.cpu_ms, 1e-9)) + "x faster than GPU" +
             (est.cols_cold ? ", including PCIe upload" : "") + ")";
  L.push_back("  decision: " + std::string(est.choice == "gpu" ? "GPU" : "CPU") + why);
  return est;
}

QueryResult Engine::explain_only(PlannedQuery& pq) {
  ensure_calibrated();
  PlacementEstimate est = estimate_placement(pq);
  QueryResult r;
  r.explain = true;
  r.plan_lines = est.lines;
  r.device_used = device == "auto" ? est.choice : device;
  return r;
}

}  // namespace warpdb
