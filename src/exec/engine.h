#pragma once
#include <string>
#include <unordered_map>
#include <vector>

#include "exec/plan.h"
#include "exec/value.h"
#include "storage/table.h"

namespace warpdb {

struct QueryResult {
  std::vector<std::string> names;
  std::vector<VP> cols;  // all materialized, each with nrows elements
  size_t nrows = 0;
  std::vector<std::string> trace;  // per-stage execution trace (shown by EXPLAIN)
  double ms = 0;
  bool explain = false;
};

class Engine {
 public:
  Catalog catalog;

  // Parse + bind + classify, without executing.
  PlannedQuery plan(const std::string& sql);
  // plan() then run on the selected device ("cpu" always works; "gpu" needs a registered backend).
  QueryResult execute(const std::string& sql);
  QueryResult execute_cpu(PlannedQuery& pq);

  void set_gpu_backend(std::shared_ptr<Backend> b) { gpu_ = std::move(b); }
  bool has_gpu() const { return gpu_ != nullptr; }
  std::string device = "cpu";  // "cpu" | "gpu"

  // Approximate distinct-value count of a column (lazily computed linear-counting sketch).
  double column_ndv(const std::shared_ptr<Table>& t, int col);
  // NDV of a join-key expression inside a relation of n rows (columns only; other expressions: n).
  double key_ndv(const PlannedQuery& pq, const ExprP& e, size_t n);

  // Greedy cost-based join ordering shared by all backends. rel_n[k] = rows of table k after its scan filters.
  int pick_start(const PlannedQuery& pq, const std::vector<size_t>& rel_n) const;
  // Next table to join into the current result (cur_n rows), by smallest estimated output; -1 if none connect.
  int pick_next(const PlannedQuery& pq, const std::vector<bool>& joined, size_t cur_n,
                const std::vector<size_t>& rel_n);

 private:
  std::unordered_map<uint64_t, double> ndv_cache_;
  std::shared_ptr<Backend> gpu_;
};

}  // namespace warpdb
