#pragma once
// Cost model for CPU/GPU placement.
//
// Every executor reports its work as a list of StageStats. The same stage vocabulary is used to
//   (1) PREDICT the cost of a not-yet-executed query on each device (estimated cardinalities), and
//   (2) LEARN from executed queries (observed cardinalities and times).
// The model for one (stage kind, device) pair is   time_ms = a + b * work,   where `work` is a
// stage-specific measure of data touched (rows x columns, ...). a captures fixed costs (kernel launches,
// thread start-up), b the per-unit throughput. Parameters are fitted by weighted least squares with
// RELATIVE-error weighting (so tiny queries, where placement mistakes are most visible, are fitted as
// accurately as huge ones) and exponential forgetting (the model tracks drift: thermals, clocks, load).
#include <map>
#include <string>
#include <vector>

namespace warpdb {

// Bump whenever stage_work() or the stage vocabulary changes: a stored model is only valid for the work
// definition it was fitted with (older profiles are discarded and recalibrated automatically).
constexpr int kProfileVersion = 2;

enum class StageOp { Scan = 0, Join, Aggregate, Sort, Project, Upload };
constexpr int kNumStageOps = 6;
const char* stage_op_name(StageOp op);

struct StageStat {
  StageOp op = StageOp::Scan;
  bool gpu = false;
  std::string label;
  double rows_in = 0;   // Scan/Agg/Sort: input rows.  Join: left (current) rows.  Upload: megabytes.
  double rows_aux = 0;  // Join: right (new table) rows
  double rows_out = 0;
  int k1 = 0;           // Scan: columns touched | Join: #keys | Agg: #group keys | Sort: #keys | Project: #items
  int k2 = 0;           // Scan: operator nodes evaluated | Join: #tables in output | Agg: #aggregates
  int k3 = 0;           // Agg: operator nodes in group keys and aggregate arguments
  double ms = 0;
};

// The regressor of the stage's linear model (what "amount of work" the stage represents).
double stage_work(const StageStat& s);

class LinearModel {
 public:
  void reset(double a, double b);  // prior; forgets all samples
  void observe(double work, double ms);
  double predict(double work) const;
  double a() const { return a_; }
  double b() const { return b_; }
  size_t samples() const { return n_; }

  // raw state for persistence
  double s0 = 0, sx = 0, sxx = 0, sy = 0, sxy = 0;
  void restore(double a, double b, size_t n);

 private:
  void solve();
  double a_ = 0.05, b_ = 1e-5;
  size_t n_ = 0;
};

// Measured properties of this machine (filled by the GPU backend's micro-benchmarks).
struct HwProfile {
  std::string fingerprint;  // GPU name + CPU thread count; a stored profile is only reused if it matches
  bool has_gpu = false;
  double h2d_gbs = 0;       // pageable host -> device
  double d2h_gbs = 0;
  double launch_us = 0;     // empty kernel launch + synchronize
  double gpu_membw_gbs = 0; // device-to-device copy
};

class CostModel {
 public:
  CostModel();
  LinearModel& model(StageOp op, bool gpu) { return models_[static_cast<int>(op)][gpu ? 1 : 0]; }
  const LinearModel& model(StageOp op, bool gpu) const { return models_[static_cast<int>(op)][gpu ? 1 : 0]; }
  double predict(const StageStat& s) const { return model(s.op, s.gpu).predict(stage_work(s)); }
  // Online learning from an executed stage (outliers such as first-launch spikes are clipped).
  void observe(const StageStat& s);
  void seed_upload_from_hw();

  HwProfile hw;
  bool calibrated = false;

  std::string to_json() const;
  bool from_json(const std::string& text, const std::string& expected_fingerprint);

 private:
  LinearModel models_[kNumStageOps][2];
};

}  // namespace warpdb
