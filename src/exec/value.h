#pragma once
#include <memory>
#include <string>
#include <vector>

#include "sql/ast.h"
#include "storage/table.h"

namespace warpdb {

// A (possibly scalar) evaluated vector. Integers, dates and dictionary codes are widened to int64 so
// expression kernels only deal with three payload types.
struct Value {
  enum Kind : uint8_t { I, F, B, S, L };  // int64 | double | bool | string codes (+dict) | string literal
  Kind kind = I;
  bool is_date = false;  // I only: logical DATE
  bool scalar = false;   // true => one element broadcast to n rows
  size_t n = 0;          // logical row count
  std::vector<int64_t> i;                 // I and S
  std::vector<double> f;                  // F
  std::vector<uint8_t> b;                 // B
  std::shared_ptr<const Column> dict;     // S: owning column (dictionary lives there)
  std::string lit;                        // L

  size_t stride() const { return scalar ? 0 : 1; }
};
using VP = std::shared_ptr<const Value>;

// Evaluation context: resolves column references / pre-computed expressions.
struct Scope {
  size_t n = 0;
  virtual ~Scope() = default;
  // Returns a value if this scope can answer `e` directly (columns, aggregates, group keys), else null.
  virtual VP intercept(const Expr& e) = 0;
  // New scope containing rows idx[0..k) of this scope, in that order.
  virtual std::unique_ptr<Scope> gather(const std::vector<uint32_t>& idx) const = 0;
};

VP eval(const Expr& e, Scope& sc);
VP gather_column(const std::shared_ptr<Column>& c, const std::vector<uint32_t>& rowids);
VP gather_value(const VP& v, const std::vector<uint32_t>& idx);
VP materialize(const VP& v, size_t n);  // broadcast scalars to n rows
std::vector<uint32_t> select_true(const Value& b);
std::string format_cell(const Value& v, size_t row);
bool like_match(const std::string& s, const std::string& pattern);  // SQL LIKE with % and _

}  // namespace warpdb
