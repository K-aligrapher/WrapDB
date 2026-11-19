#include "optimizer/cost_model.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>

namespace warpdb {

const char* stage_op_name(StageOp op) {
  switch (op) {
    case StageOp::Scan: return "scan";
    case StageOp::Join: return "join";
    case StageOp::Aggregate: return "aggregate";
    case StageOp::Sort: return "sort";
    case StageOp::Project: return "project";
    case StageOp::Upload: return "upload";
  }
  return "?";
}

double stage_work(const StageStat& s) {
  auto mx = [](double v, double lo) { return v < lo ? lo : v; };
  switch (s.op) {
    case StageOp::Scan:  // every touched column is gathered, and every operator node is a pass over the rows
      return s.rows_in * (0.2 + s.k1 + 0.7 * s.k2);
    case StageOp::Join:  // key gathers for both inputs + compose of the output row-id lists
      return (s.rows_in + s.rows_aux) * mx(s.k1, 1) + s.rows_out * mx(s.k2, 1);
    case StageOp::Aggregate:  // key + argument evaluation and one table update per row
      return s.rows_in * (1.0 + s.k1 + s.k2 + 0.7 * s.k3);
    case StageOp::Sort: {
      double n = mx(s.rows_in, 2);
      return n * std::log2(n) * mx(s.k1, 1);
    }
    case StageOp::Project:  // expression evaluation + materialization (+ device->host copy on the GPU)
      return s.rows_out * mx(s.k1, 1);
    case StageOp::Upload: return s.rows_in;  // megabytes moved over PCIe
  }
  return 0;
}

// ---- LinearModel ------------------------------------------------------------------------------------------

void LinearModel::reset(double a, double b) {
  a_ = a;
  b_ = b;
  n_ = 0;
  s0 = sx = sxx = sy = sxy = 0;
}

void LinearModel::restore(double a, double b, size_t n) {
  a_ = a;
  b_ = b;
  n_ = n;
}

double LinearModel::predict(double work) const { return std::max(0.0, a_ + b_ * work); }

void LinearModel::observe(double x, double y) {
  constexpr double kForget = 0.995;                 // effective window ~200 samples
  const double w = 1.0 / std::max(y * y, 0.0025);   // relative-error weighting (floor at 0.05 ms)
  s0 = s0 * kForget + w;
  sx = sx * kForget + w * x;
  sxx = sxx * kForget + w * x * x;
  sy = sy * kForget + w * y;
  sxy = sxy * kForget + w * x * y;
  ++n_;
  solve();
}

void LinearModel::solve() {
  if (n_ < 2) {
    if (sxx > 0) b_ = std::max(0.0, sxy / sxx);  // single point: through the origin
    return;
  }
  double det = s0 * sxx - sx * sx;
  double a = a_, b = b_;
  if (det > 1e-9 * std::abs(s0 * sxx)) {
    b = (s0 * sxy - sx * sy) / det;
    a = (sy - b * sx) / s0;
  } else if (sxx > 0) {  // all samples have (nearly) the same work: only the rate is identifiable
    a = 0;
    b = sxy / sxx;
  }
  if (b < 0 || a < 0) {  // physically meaningless: refit through the origin
    a = 0;
    b = sxx > 0 ? std::max(0.0, sxy / sxx) : 0.0;
  }
  a_ = a;
  b_ = b;
}

// ---- CostModel ------------------------------------------------------------------------------------------

CostModel::CostModel() {
  // Neutral priors; calibration overwrites them. (a in ms, b in ms per unit of work)
  for (int op = 0; op < kNumStageOps; ++op) {
    models_[op][0].reset(0.02, 2e-6);  // CPU
    models_[op][1].reset(0.08, 2e-7);  // GPU
  }
}

void CostModel::observe(const StageStat& s) {
  LinearModel& m = model(s.op, s.gpu);
  double x = stage_work(s);
  double y = s.ms;
  if (m.samples() >= 8) {  // clip outliers (first-use kernel loads, OS hiccups) instead of letting them skew the fit
    double p = m.predict(x);
    y = std::min(y, 8.0 * p + 5.0);
  }
  m.observe(x, y);
}

void CostModel::seed_upload_from_hw() {
  if (hw.h2d_gbs > 0) model(StageOp::Upload, true).reset(0.05, 1.0 / hw.h2d_gbs);  // 1 GB/s == 1 MB/ms
}

// ---- persistence (flat JSON: {"key": number|"string", ...}) --------------------------------------------------

std::string CostModel::to_json() const {
  std::ostringstream o;
  o.precision(10);
  o << "{\n";
  o << "  \"version\": " << kProfileVersion << ",\n";
  o << "  \"fingerprint\": \"" << hw.fingerprint << "\",\n";
  o << "  \"has_gpu\": " << (hw.has_gpu ? 1 : 0) << ",\n";
  o << "  \"h2d_gbs\": " << hw.h2d_gbs << ",\n";
  o << "  \"d2h_gbs\": " << hw.d2h_gbs << ",\n";
  o << "  \"launch_us\": " << hw.launch_us << ",\n";
  o << "  \"gpu_membw_gbs\": " << hw.gpu_membw_gbs << ",\n";
  for (int op = 0; op < kNumStageOps; ++op)
    for (int g = 0; g < 2; ++g) {
      const LinearModel& m = models_[op][g];
      std::string k = std::string("m.") + stage_op_name(static_cast<StageOp>(op)) + (g ? ".gpu" : ".cpu");
      o << "  \"" << k << ".a\": " << m.a() << ",\n";
      o << "  \"" << k << ".b\": " << m.b() << ",\n";
      o << "  \"" << k << ".n\": " << m.samples() << ",\n";
      o << "  \"" << k << ".s\": [" << m.s0 << ", " << m.sx << ", " << m.sxx << ", " << m.sy << ", " << m.sxy << "],\n";
    }
  o << "  \"calibrated\": " << (calibrated ? 1 : 0) << "\n}\n";
  return o.str();
}

namespace {
bool find_number(const std::string& t, const std::string& key, double& out) {
  size_t p = t.find("\"" + key + "\"");
  if (p == std::string::npos) return false;
  p = t.find(':', p);
  if (p == std::string::npos) return false;
  out = std::strtod(t.c_str() + p + 1, nullptr);
  return true;
}
bool find_numbers(const std::string& t, const std::string& key, double* out, int n) {
  size_t p = t.find("\"" + key + "\"");
  if (p == std::string::npos) return false;
  p = t.find('[', p);
  if (p == std::string::npos) return false;
  const char* c = t.c_str() + p + 1;
  for (int i = 0; i < n; ++i) {
    char* e;
    out[i] = std::strtod(c, &e);
    if (e == c) return false;
    c = e;
    while (*c == ',' || *c == ' ') ++c;
  }
  return true;
}
}  // namespace

bool CostModel::from_json(const std::string& text, const std::string& expected_fingerprint) {
  double ver = 0;
  if (!find_number(text, "version", ver) || static_cast<int>(ver) != kProfileVersion) return false;  // stale format
  size_t p = text.find("\"fingerprint\"");
  if (p == std::string::npos) return false;
  size_t q1 = text.find('"', text.find(':', p) + 1);
  size_t q2 = text.find('"', q1 + 1);
  if (q1 == std::string::npos || q2 == std::string::npos) return false;
  std::string fp = text.substr(q1 + 1, q2 - q1 - 1);
  if (fp != expected_fingerprint) return false;  // profile of a different machine / GPU
  CostModel tmp;
  tmp.hw.fingerprint = fp;
  double v;
  if (find_number(text, "has_gpu", v)) tmp.hw.has_gpu = v != 0;
  find_number(text, "h2d_gbs", tmp.hw.h2d_gbs);
  find_number(text, "d2h_gbs", tmp.hw.d2h_gbs);
  find_number(text, "launch_us", tmp.hw.launch_us);
  find_number(text, "gpu_membw_gbs", tmp.hw.gpu_membw_gbs);
  for (int op = 0; op < kNumStageOps; ++op)
    for (int g = 0; g < 2; ++g) {
      LinearModel& m = tmp.models_[op][g];
      std::string k = std::string("m.") + stage_op_name(static_cast<StageOp>(op)) + (g ? ".gpu" : ".cpu");
      double a, b, n;
      if (!find_number(text, k + ".a", a) || !find_number(text, k + ".b", b) || !find_number(text, k + ".n", n))
        return false;
      m.restore(a, b, static_cast<size_t>(n));
      double s[5];
      if (find_numbers(text, k + ".s", s, 5)) {
        m.s0 = s[0]; m.sx = s[1]; m.sxx = s[2]; m.sy = s[3]; m.sxy = s[4];
      }
    }
  double cal = 0;
  find_number(text, "calibrated", cal);
  tmp.calibrated = cal != 0;
  *this = tmp;
  return true;
}

}  // namespace warpdb
