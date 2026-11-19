#pragma once
#include <string>
#include <unordered_map>
#include <vector>

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
  QueryResult execute(const std::string& sql);

 private:
  // Approximate distinct-value counts per (table, column), computed lazily (linear counting).
  double column_ndv(const std::shared_ptr<Table>& t, int col, int tbl_key);
  std::unordered_map<uint64_t, double> ndv_cache_;
};

}  // namespace warpdb
