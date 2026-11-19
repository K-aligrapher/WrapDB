#pragma once
#include <memory>
#include <string>
#include <vector>

#include "optimizer/cost_model.h"
#include "sql/ast.h"
#include "storage/table.h"

namespace warpdb {

using TableVec = std::vector<std::shared_ptr<Table>>;

// Equality join condition between two different FROM tables: ea references only table ta, eb only tb.
struct Edge {
  int ta, tb;
  ExprP ea, eb;
};

// A bound, classified query: everything an execution backend (CPU or GPU) needs.
struct PlannedQuery {
  Select sel;  // select items / group_by / having / order_by / limit, all bound to (table, column)
  TableVec tabs;                          // FROM tables, by FROM index
  std::vector<ExprP> aggs;                // distinct aggregate calls appearing anywhere
  bool agg_mode = false;
  std::vector<std::vector<ExprP>> push;   // single-table predicates, per table (applied at the scan)
  std::vector<Edge> edges;                // equi-join conditions
  std::vector<ExprP> residual;            // everything else, applied after the joins
};

struct QueryResult;
class Engine;

// An execution backend other than the built-in CPU executor (e.g. the CUDA backend).
class Backend {
 public:
  virtual ~Backend() = default;
  virtual const char* name() const = 0;
  virtual QueryResult execute(Engine& eng, PlannedQuery& pq) = 0;

  // ---- hooks used by the placement optimizer (defaults describe "no device memory") ----
  virtual bool column_resident(const Column&) const { return false; }  // is the column already in VRAM?
  virtual size_t cache_budget_bytes() const { return 0; }              // VRAM available for resident columns
  virtual void forget_columns(const std::vector<uint64_t>&) {}         // evict specific columns (by Column::uid)
  virtual void drop_cache() {}
  virtual std::string fingerprint() const { return ""; }                // identifies the device model
  virtual bool measure_hardware(HwProfile&) { return false; }          // PCIe / launch micro-benchmarks
};

// Number of distinct (table, column) pairs referenced by the expressions.
int distinct_columns(const std::vector<ExprP>& exprs);
// Number of operator nodes (arithmetic, comparisons, CASE, ...) the expressions evaluate per row.
int expr_nodes(const std::vector<ExprP>& exprs);
// Operator nodes in the GROUP BY keys and aggregate arguments of a query.
int agg_expr_nodes(const PlannedQuery& pq);

}  // namespace warpdb
