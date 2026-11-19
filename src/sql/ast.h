#pragma once
#include <memory>
#include <string>
#include <vector>

#include "common/util.h"

namespace warpdb {

struct Expr;
using ExprP = std::shared_ptr<Expr>;

struct Expr {
  enum Kind { Col, Int, Float, Str, Date, Interval, Bin, Not, Neg, Between, In, Like, Case, Func, Agg, Star };
  Kind kind = Int;
  std::string name;   // Col: column; Str: text; Func/Agg: lowercase function; Interval: unit; Bin: operator
  std::string table;  // Col: optional qualifier
  int64_t ival = 0;   // Int value; Date days; Interval count
  double fval = 0;    // Float value
  bool negated = false;   // NOT BETWEEN / NOT IN / NOT LIKE
  bool has_else = false;  // Case
  // Children. Bin: [l, r]; Not/Neg: [x]; Between: [x, lo, hi]; In: [x, v1, v2...]; Like: [x, pattern];
  // Case: [when1, then1, ..., (else)]; Func/Agg: arguments (Agg count(*): [Star])
  std::vector<ExprP> args;
  // Filled by the binder.
  int tbl = -1, col = -1;

  // Canonical text; bound columns print as "t<tbl>.<col>" so equal expressions compare equal.
  const std::string& str() const;

 private:
  mutable std::string cache_;
};

ExprP make_expr(Expr::Kind k);

struct TableRef {
  std::string name, alias;
};
struct SelectItem {
  ExprP e;
  std::string alias;  // may be empty
  std::string text;   // original SQL text of the expression
};
struct OrderItem {
  ExprP e;
  bool desc = false;
};

struct Select {
  bool explain = false;
  bool distinct = false;
  bool star = false;
  std::vector<SelectItem> items;
  std::vector<TableRef> from;
  std::vector<ExprP> on_conds;  // from JOIN ... ON
  ExprP where;
  std::vector<ExprP> group_by;
  ExprP having;
  std::vector<OrderItem> order_by;
  int64_t limit = -1;
};

Select parse_sql(const std::string& sql);

}  // namespace warpdb
