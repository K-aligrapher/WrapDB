#pragma once
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "exec/plan.h"
#include "exec/value.h"
#include "optimizer/cost_model.h"
#include "storage/table.h"

namespace warpdb {

struct QueryResult {
  std::vector<std::string> names;
  std::vector<VP> cols;  // all materialized, each with nrows elements
  size_t nrows = 0;
  std::vector<std::string> trace;       // per-stage execution trace
  std::vector<std::string> plan_lines;  // placement estimate (EXPLAIN / auto mode)
  std::vector<StageStat> stages;        // structured per-stage measurements (feed the cost model)
  std::string device_used;              // "cpu" | "gpu"
  double ms = 0;
  bool explain = false;                 // print plan_lines/trace instead of rows
};

// Predicted cost of one stage on both devices.
struct StageEstimate {
  StageStat stat;  // cardinalities are ESTIMATES; ms is unused
  double cpu_ms = 0, gpu_ms = 0;
};

struct PlacementEstimate {
  std::vector<StageEstimate> stages;
  int cols_total = 0, cols_cold = 0;  // referenced base columns / those not yet resident in VRAM
  double upload_mb = 0, upload_ms = 0;
  double cpu_ms = 0, gpu_ms = 0;      // gpu_ms includes upload_ms
  bool gpu_feasible = false;
  std::string gpu_reason;             // why the GPU is not feasible, if so
  std::string choice = "cpu";
  std::vector<std::string> lines;     // human-readable breakdown
};

// One automatic placement decision, kept so the optimizer can be audited and evaluated.
struct DecisionRecord {
  std::string sql;
  double est_cpu_ms = 0, est_gpu_ms = 0;
  std::string chosen;
  double actual_ms = 0;  // wall time of the executed query (excludes the optimizer itself)
  double decide_ms = 0;  // time spent estimating
};

class Engine {
 public:
  Catalog catalog;

  // Parse + bind + classify, without executing.
  PlannedQuery plan(const std::string& sql);
  // plan() then run on `device`: "cpu", "gpu", or "auto" (cost-based placement).
  QueryResult execute(const std::string& sql);
  QueryResult execute_cpu(PlannedQuery& pq);

  void set_gpu_backend(std::shared_ptr<Backend> b) { gpu_ = std::move(b); }
  bool has_gpu() const { return gpu_ != nullptr; }
  Backend* gpu_backend() const { return gpu_.get(); }
  std::string device = "cpu";  // "cpu" | "gpu" | "auto"

  // ---- statistics & shared planning helpers ----
  double column_ndv(const std::shared_ptr<Table>& t, int col);
  double key_ndv(const PlannedQuery& pq, const ExprP& e, size_t n);
  int pick_start(const PlannedQuery& pq, const std::vector<size_t>& rel_n) const;
  int pick_next(const PlannedQuery& pq, const std::vector<bool>& joined, size_t cur_n,
                const std::vector<size_t>& rel_n);
  // Estimated rows of (current result JOIN table k); sets *keys to the number of equality keys used.
  double join_estimate(const PlannedQuery& pq, const std::vector<bool>& joined, size_t cur_n, int k,
                       const std::vector<size_t>& rel_n, int* keys);
  // Fraction of table k's rows passing its pushed-down predicates, measured on a strided sample.
  double sample_selectivity(const PlannedQuery& pq, int k);

  // ---- placement optimizer ----
  CostModel cost;
  bool learn = true;  // refine the cost model from every executed query
  std::vector<DecisionRecord> decisions;
  std::function<void(const std::string&)> on_notice;  // progress messages (e.g. "calibrating...")
  std::string profile_path;                            // where the calibrated model is persisted ("" = nowhere)

  PlacementEstimate estimate_placement(PlannedQuery& pq);
  void ensure_calibrated();
  void calibrate();                  // measure hardware + fit the cost model with synthetic workloads
  void warm_up_gpu();                // pay one-off GPU start-up costs outside user queries
  bool load_profile();
  void save_profile() const;
  std::string hw_fingerprint() const;

 private:
  QueryResult explain_only(PlannedQuery& pq);
  std::unordered_map<uint64_t, double> ndv_cache_;
  std::unordered_map<std::string, double> sel_cache_;  // sampled selectivities, keyed by table + predicate text
  std::shared_ptr<Backend> gpu_;
};

}  // namespace warpdb
