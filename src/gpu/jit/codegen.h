#pragma once
// SQL expression trees -> CUDA C++ source for the JIT. Pure host code (no CUDA headers).
//
// Two pipeline shapes are generated:
//   gen_filter_plan : scan -> filter(all conjuncts, short-circuit) -> compaction to a row-id list
//   gen_agg_plan    : [scan ->] filter -> group keys -> hash-table slot -> every aggregate's update,
//                     in ONE kernel that reads the base columns directly (no intermediate vectors)
//
// The generated code must agree bit-for-bit with the interpreted (unfused) kernels, so promotion rules,
// constant folding (including the decimal-literal rounding) and string predicates (dictionary masks built on
// the host and shared with the interpreter through cached_dict_mask) are all reproduced here.
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "exec/plan.h"
#include "exec/value.h"

namespace warpdb::gpu {

// Thrown when a query uses something the JIT does not handle; the caller falls back to the unfused kernels.
struct JitUnsupported : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct ValType {
  Value::Kind kind = Value::I;
  bool is_date = false;
  std::shared_ptr<const Column> dict;  // VARCHAR: the column whose dictionary the codes refer to
};

struct MaskSpec {  // per-dictionary-code predicate result, uploaded as a uint8 kernel argument
  std::shared_ptr<const Column> dictcol;
  std::string key;
  std::function<void(std::vector<uint8_t>&)> fill;
};

enum class JAgg { Sum, Count, Avg, Min, Max };
struct JAggDesc {
  JAgg func = JAgg::Count;
  ExprP arg;  // null for count(*)
};

struct JitPlan {
  std::string source;
  std::string entry = "k_main";
  // Kernel argument layout (in this order after the leading `n`):
  std::vector<std::pair<int, int>> cols;  // (table, column): device pointers to base columns
  std::vector<int> rid_tables;            // row-id arrays, one per relation table (empty = identity input)
  std::vector<MaskSpec> masks;            // dictionary masks
  // Aggregate pipelines additionally describe their outputs:
  int K = 0, A = 0;
  std::vector<ValType> key_types;
  std::vector<JAggDesc> aggs;
  std::vector<ValType> agg_out_types;
  std::vector<bool> acc_is_f;  // accumulator element type is double (else int64)
};

JitPlan gen_filter_plan(const PlannedQuery& pq, int tbl, const std::vector<ExprP>& preds);

// rel_tbls: tables of the input relation (their row-id arrays are kernel arguments unless `identity`, in which
// case rel_tbls must hold exactly one table and row i of the pipeline is row i of that table).
JitPlan gen_agg_plan(const PlannedQuery& pq, const std::vector<int>& rel_tbls, bool identity,
                     const std::vector<ExprP>& preds, const std::vector<ExprP>& keys,
                     const std::vector<JAggDesc>& aggs);

}  // namespace warpdb::gpu
