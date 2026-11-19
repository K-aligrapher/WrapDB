#include <cmath>
#include <numeric>
#include <set>

#include "exec/value.h"

namespace warpdb {
namespace {

constexpr size_t GRAIN = 1 << 15;

enum class Op { Add, Sub, Mul, Div, Mod, Eq, Ne, Lt, Le, Gt, Ge, And, Or };

Op op_of(const std::string& s) {
  if (s == "+") return Op::Add;
  if (s == "-") return Op::Sub;
  if (s == "*") return Op::Mul;
  if (s == "/") return Op::Div;
  if (s == "%") return Op::Mod;
  if (s == "=") return Op::Eq;
  if (s == "<>") return Op::Ne;
  if (s == "<") return Op::Lt;
  if (s == "<=") return Op::Le;
  if (s == ">") return Op::Gt;
  if (s == ">=") return Op::Ge;
  if (s == "and") return Op::And;
  if (s == "or") return Op::Or;
  throw Error("unknown operator: " + s);
}

Op flip(Op o) {
  switch (o) {
    case Op::Lt: return Op::Gt;
    case Op::Le: return Op::Ge;
    case Op::Gt: return Op::Lt;
    case Op::Ge: return Op::Le;
    default: return o;
  }
}

template <class R, class A, class B, class F>
void map2(R* o, const A* a, size_t sa, const B* b, size_t sb, size_t m, F f) {
  parallel_for(m, GRAIN, [&](size_t, size_t s, size_t e) {
    for (size_t i = s; i < e; ++i) o[i] = f(a[i * sa], b[i * sb]);
  });
}

std::shared_ptr<Value> new_value(Value::Kind k, size_t n, bool scalar) {
  auto v = std::make_shared<Value>();
  v->kind = k;
  v->n = n;
  v->scalar = scalar;
  size_t m = scalar ? 1 : n;
  switch (k) {
    case Value::I:
    case Value::S: v->i.resize(m); break;
    case Value::F: v->f.resize(m); break;
    case Value::B: v->b.resize(m); break;
    default: break;
  }
  return v;
}

bool is_num(const Value& v) { return v.kind == Value::I || v.kind == Value::F; }

VP arith(Op op, const Value& a, const Value& b, size_t n) {
  if (!is_num(a) || !is_num(b)) throw Error("arithmetic requires numeric operands");
  bool sc = a.scalar && b.scalar;
  size_t m = sc ? 1 : n, sa = a.stride(), sb = b.stride();
  bool ints = a.kind == Value::I && b.kind == Value::I && op != Op::Div;
  if (ints) {
    auto v = new_value(Value::I, n, sc);
    int64_t* o = v->i.data();
    const int64_t *pa = a.i.data(), *pb = b.i.data();
    switch (op) {
      case Op::Add: map2(o, pa, sa, pb, sb, m, [](int64_t x, int64_t y) { return x + y; }); break;
      case Op::Sub: map2(o, pa, sa, pb, sb, m, [](int64_t x, int64_t y) { return x - y; }); break;
      case Op::Mul: map2(o, pa, sa, pb, sb, m, [](int64_t x, int64_t y) { return x * y; }); break;
      case Op::Mod: map2(o, pa, sa, pb, sb, m, [](int64_t x, int64_t y) { return y ? x % y : int64_t(0); }); break;
      default: throw Error("bad integer operator");
    }
    if (op == Op::Add || op == Op::Sub) v->is_date = a.is_date != b.is_date && !(op == Op::Sub && b.is_date);
    return v;
  }
  auto v = new_value(Value::F, n, sc);
  double* o = v->f.data();
  auto go = [&](auto* pa, auto* pb) {
    switch (op) {
      case Op::Add: map2(o, pa, sa, pb, sb, m, [](auto x, auto y) { return double(x) + double(y); }); break;
      case Op::Sub: map2(o, pa, sa, pb, sb, m, [](auto x, auto y) { return double(x) - double(y); }); break;
      case Op::Mul: map2(o, pa, sa, pb, sb, m, [](auto x, auto y) { return double(x) * double(y); }); break;
      case Op::Div: map2(o, pa, sa, pb, sb, m, [](auto x, auto y) { return double(x) / double(y); }); break;
      case Op::Mod: map2(o, pa, sa, pb, sb, m, [](auto x, auto y) { return std::fmod(double(x), double(y)); }); break;
      default: throw Error("bad arithmetic operator");
    }
  };
  if (a.kind == Value::I && b.kind == Value::I) go(a.i.data(), b.i.data());
  else if (a.kind == Value::I) go(a.i.data(), b.f.data());
  else if (b.kind == Value::I) go(a.f.data(), b.i.data());
  else go(a.f.data(), b.f.data());
  // Constant folding of decimal-looking literals (0.06 - 0.01 must equal 0.05, as in SQL DECIMAL
  // arithmetic): round scalar results to 10 decimals. Temporary until a real DECIMAL type exists.
  if (sc && std::isfinite(v->f[0])) v->f[0] = std::round(v->f[0] * 1e10) / 1e10;
  return v;
}

template <class T, class U>
void cmp_run(Op op, uint8_t* o, const T* a, size_t sa, const U* b, size_t sb, size_t m) {
  switch (op) {
    case Op::Eq: map2(o, a, sa, b, sb, m, [](T x, U y) -> uint8_t { return x == y; }); break;
    case Op::Ne: map2(o, a, sa, b, sb, m, [](T x, U y) -> uint8_t { return x != y; }); break;
    case Op::Lt: map2(o, a, sa, b, sb, m, [](T x, U y) -> uint8_t { return x < y; }); break;
    case Op::Le: map2(o, a, sa, b, sb, m, [](T x, U y) -> uint8_t { return x <= y; }); break;
    case Op::Gt: map2(o, a, sa, b, sb, m, [](T x, U y) -> uint8_t { return x > y; }); break;
    case Op::Ge: map2(o, a, sa, b, sb, m, [](T x, U y) -> uint8_t { return x >= y; }); break;
    default: throw Error("bad comparison operator");
  }
}

VP mask_to_bool(const Value& s, const std::vector<uint8_t>& mask, size_t n) {
  auto v = new_value(Value::B, n, false);
  uint8_t* o = v->b.data();
  const int64_t* c = s.i.data();
  const uint8_t* mk = mask.data();
  parallel_for(n, GRAIN, [&](size_t, size_t b, size_t e) {
    for (size_t i = b; i < e; ++i) o[i] = mk[c[i]];
  });
  return v;
}

VP cmp_str_lit(Op op, const Value& s, const std::string& lit, size_t n) {
  const auto& d = s.dict->dict;
  std::vector<uint8_t> mask(d.size());
  for (size_t k = 0; k < d.size(); ++k) {
    int c = d[k].compare(lit);
    bool r = false;
    switch (op) {
      case Op::Eq: r = c == 0; break;
      case Op::Ne: r = c != 0; break;
      case Op::Lt: r = c < 0; break;
      case Op::Le: r = c <= 0; break;
      case Op::Gt: r = c > 0; break;
      case Op::Ge: r = c >= 0; break;
      default: throw Error("bad string comparison");
    }
    mask[k] = r;
  }
  return mask_to_bool(s, mask, n);
}

VP compare(Op op, const Value& a, const Value& b, size_t n) {
  if (a.kind == Value::S && b.kind == Value::L) return cmp_str_lit(op, a, b.lit, n);
  if (a.kind == Value::L && b.kind == Value::S) return cmp_str_lit(flip(op), b, a.lit, n);
  if (a.kind == Value::S && b.kind == Value::S && a.dict.get() != b.dict.get())
    throw Error("comparing VARCHAR columns from different dictionaries is not supported yet");
  bool sc = a.scalar && b.scalar;
  size_t m = sc ? 1 : n, sa = a.stride(), sb = b.stride();
  auto v = new_value(Value::B, n, sc);
  uint8_t* o = v->b.data();
  auto intlike = [](const Value& x) { return x.kind == Value::I || x.kind == Value::S; };
  if (intlike(a) && intlike(b)) cmp_run<int64_t, int64_t>(op, o, a.i.data(), sa, b.i.data(), sb, m);
  else if (intlike(a) && b.kind == Value::F) cmp_run<int64_t, double>(op, o, a.i.data(), sa, b.f.data(), sb, m);
  else if (a.kind == Value::F && intlike(b)) cmp_run<double, int64_t>(op, o, a.f.data(), sa, b.i.data(), sb, m);
  else if (a.kind == Value::F && b.kind == Value::F) cmp_run<double, double>(op, o, a.f.data(), sa, b.f.data(), sb, m);
  else throw Error("cannot compare these operand types");
  return v;
}

VP logic(Op op, const Value& a, const Value& b, size_t n) {
  if (a.kind != Value::B || b.kind != Value::B) throw Error("AND/OR require boolean operands");
  bool sc = a.scalar && b.scalar;
  auto v = new_value(Value::B, n, sc);
  size_t m = sc ? 1 : n;
  if (op == Op::And) map2(v->b.data(), a.b.data(), a.stride(), b.b.data(), b.stride(), m, [](uint8_t x, uint8_t y) -> uint8_t { return x & y; });
  else map2(v->b.data(), a.b.data(), a.stride(), b.b.data(), b.stride(), m, [](uint8_t x, uint8_t y) -> uint8_t { return x | y; });
  return v;
}

VP negate(const Value& a, size_t n) {
  if (a.kind != Value::B) throw Error("NOT requires a boolean operand");
  auto v = new_value(Value::B, n, a.scalar);
  size_t m = a.scalar ? 1 : n;
  const uint8_t* p = a.b.data();
  uint8_t* o = v->b.data();
  parallel_for(m, GRAIN, [&](size_t, size_t s, size_t e) {
    for (size_t i = s; i < e; ++i) o[i] = !p[i];
  });
  return v;
}

bool like_match(const std::string& s, const std::string& p) {
  size_t si = 0, pi = 0, star = std::string::npos, mark = 0;
  while (si < s.size()) {
    if (pi < p.size() && (p[pi] == '_' || p[pi] == s[si])) { ++si; ++pi; }
    else if (pi < p.size() && p[pi] == '%') { star = pi++; mark = si; }
    else if (star != std::string::npos) { pi = star + 1; si = ++mark; }
    else return false;
  }
  while (pi < p.size() && p[pi] == '%') ++pi;
  return pi == p.size();
}

VP bool_not(VP v, size_t n) { return negate(*v, n); }

}  // namespace

// ---------------------------------------------------------------------------------------------
VP gather_column(const std::shared_ptr<Column>& c, const std::vector<uint32_t>& rid) {
  size_t n = rid.size();
  const uint32_t* r = rid.data();
  switch (c->type) {
    case Type::INT32:
    case Type::DATE:
    case Type::VARCHAR: {
      auto v = new_value(c->type == Type::VARCHAR ? Value::S : Value::I, n, false);
      v->is_date = c->type == Type::DATE;
      if (c->type == Type::VARCHAR) v->dict = c;
      int64_t* o = v->i.data();
      const int32_t* src = c->i32.data();
      parallel_for(n, GRAIN, [&](size_t, size_t b, size_t e) {
        for (size_t i = b; i < e; ++i) o[i] = src[r[i]];
      });
      return v;
    }
    case Type::INT64: {
      auto v = new_value(Value::I, n, false);
      int64_t* o = v->i.data();
      const int64_t* src = c->i64.data();
      parallel_for(n, GRAIN, [&](size_t, size_t b, size_t e) {
        for (size_t i = b; i < e; ++i) o[i] = src[r[i]];
      });
      return v;
    }
    case Type::FLOAT64: {
      auto v = new_value(Value::F, n, false);
      double* o = v->f.data();
      const double* src = c->f64.data();
      parallel_for(n, GRAIN, [&](size_t, size_t b, size_t e) {
        for (size_t i = b; i < e; ++i) o[i] = src[r[i]];
      });
      return v;
    }
  }
  throw Error("bad column type");
}

VP gather_value(const VP& v, const std::vector<uint32_t>& idx) {
  size_t n = idx.size();
  if (v->scalar) {
    auto c = std::make_shared<Value>(*v);
    c->n = n;
    return c;
  }
  auto o = new_value(v->kind, n, false);
  o->is_date = v->is_date;
  o->dict = v->dict;
  o->lit = v->lit;
  const uint32_t* r = idx.data();
  auto run = [&](auto* dst, const auto* src) {
    parallel_for(n, GRAIN, [&](size_t, size_t b, size_t e) {
      for (size_t i = b; i < e; ++i) dst[i] = src[r[i]];
    });
  };
  switch (v->kind) {
    case Value::I:
    case Value::S: run(o->i.data(), v->i.data()); break;
    case Value::F: run(o->f.data(), v->f.data()); break;
    case Value::B: run(o->b.data(), v->b.data()); break;
    default: break;
  }
  return o;
}

VP materialize(const VP& v, size_t n) {
  if (!v->scalar) return v;
  auto o = std::make_shared<Value>(*v);
  o->scalar = false;
  o->n = n;
  switch (v->kind) {
    case Value::I:
    case Value::S: o->i.assign(n, v->i[0]); break;
    case Value::F: o->f.assign(n, v->f[0]); break;
    case Value::B: o->b.assign(n, v->b[0]); break;
    default: break;
  }
  return o;
}

std::vector<uint32_t> select_true(const Value& b) {
  if (b.kind != Value::B) throw Error("predicate must be boolean");
  size_t n = b.n;
  std::vector<uint32_t> out;
  if (b.scalar) {
    if (b.b[0]) {
      out.resize(n);
      std::iota(out.begin(), out.end(), 0u);
    }
    return out;
  }
  size_t nch = num_chunks(n, 1 << 16);
  std::vector<size_t> cnt(nch + 1, 0);
  const uint8_t* p = b.b.data();
  parallel_for(n, 1 << 16, [&](size_t c, size_t s, size_t e) {
    size_t k = 0;
    for (size_t i = s; i < e; ++i) k += p[i] != 0;
    cnt[c] = k;
  });
  size_t total = 0;
  for (size_t c = 0; c < nch; ++c) {
    size_t k = cnt[c];
    cnt[c] = total;
    total += k;
  }
  out.resize(total);
  uint32_t* o = out.data();
  parallel_for(n, 1 << 16, [&](size_t c, size_t s, size_t e) {
    size_t w = cnt[c];
    for (size_t i = s; i < e; ++i)
      if (p[i]) o[w++] = static_cast<uint32_t>(i);
  });
  return out;
}

std::string format_cell(const Value& v, size_t row) {
  size_t k = v.scalar ? 0 : row;
  char buf[64];
  switch (v.kind) {
    case Value::I:
      if (v.is_date) return format_date(static_cast<int32_t>(v.i[k]));
      return std::to_string(v.i[k]);
    case Value::F: {
      double d = v.f[k];
      std::snprintf(buf, sizeof buf, "%.10g", d);
      return buf;
    }
    case Value::B: return v.b[k] ? "true" : "false";
    case Value::S: return v.dict->dict[static_cast<size_t>(v.i[k])];
    case Value::L: return v.lit;
  }
  return "?";
}

VP eval(const Expr& e, Scope& sc) {
  if (VP v = sc.intercept(e)) return v;
  const size_t n = sc.n;
  switch (e.kind) {
    case Expr::Int: {
      auto v = new_value(Value::I, n, true);
      v->i[0] = e.ival;
      return v;
    }
    case Expr::Float: {
      auto v = new_value(Value::F, n, true);
      v->f[0] = e.fval;
      return v;
    }
    case Expr::Date: {
      auto v = new_value(Value::I, n, true);
      v->i[0] = e.ival;
      v->is_date = true;
      return v;
    }
    case Expr::Str: {
      auto v = new_value(Value::L, n, true);
      v->lit = e.name;
      return v;
    }
    case Expr::Bin: {
      Op op = op_of(e.name);
      VP a = eval(*e.args[0], sc);
      VP b = eval(*e.args[1], sc);
      switch (op) {
        case Op::Add: case Op::Sub: case Op::Mul: case Op::Div: case Op::Mod: return arith(op, *a, *b, n);
        case Op::And: case Op::Or: return logic(op, *a, *b, n);
        default: return compare(op, *a, *b, n);
      }
    }
    case Expr::Not: return bool_not(eval(*e.args[0], sc), n);
    case Expr::Neg: {
      VP a = eval(*e.args[0], sc);
      if (!is_num(*a)) throw Error("unary minus requires a numeric operand");
      bool scal = a->scalar;
      size_t m = scal ? 1 : n;
      if (a->kind == Value::I) {
        auto v = new_value(Value::I, n, scal);
        for (size_t i = 0; i < m; ++i) v->i[i] = -a->i[i];
        return v;
      }
      auto v = new_value(Value::F, n, scal);
      for (size_t i = 0; i < m; ++i) v->f[i] = -a->f[i];
      return v;
    }
    case Expr::Between: {
      VP x = eval(*e.args[0], sc), lo = eval(*e.args[1], sc), hi = eval(*e.args[2], sc);
      VP r = logic(Op::And, *compare(Op::Ge, *x, *lo, n), *compare(Op::Le, *x, *hi, n), n);
      return e.negated ? bool_not(r, n) : r;
    }
    case Expr::In: {
      VP x = eval(*e.args[0], sc);
      VP r;
      if (x->kind == Value::S) {
        std::set<std::string> set;
        for (size_t k = 1; k < e.args.size(); ++k) {
          VP it = eval(*e.args[k], sc);
          if (it->kind != Value::L) throw Error("IN list for a VARCHAR column must contain string literals");
          set.insert(it->lit);
        }
        const auto& d = x->dict->dict;
        std::vector<uint8_t> mask(d.size());
        for (size_t k = 0; k < d.size(); ++k) mask[k] = set.count(d[k]) != 0;
        r = mask_to_bool(*x, mask, n);
      } else {
        for (size_t k = 1; k < e.args.size(); ++k) {
          VP c = compare(Op::Eq, *x, *eval(*e.args[k], sc), n);
          r = r ? VP(logic(Op::Or, *r, *c, n)) : c;
        }
        if (!r) throw Error("empty IN list");
      }
      return e.negated ? bool_not(r, n) : r;
    }
    case Expr::Like: {
      VP x = eval(*e.args[0], sc), p = eval(*e.args[1], sc);
      if (x->kind != Value::S || p->kind != Value::L) throw Error("LIKE requires a VARCHAR column and a string pattern");
      const auto& d = x->dict->dict;
      std::vector<uint8_t> mask(d.size());
      for (size_t k = 0; k < d.size(); ++k) mask[k] = like_match(d[k], p->lit);
      VP r = mask_to_bool(*x, mask, n);
      return e.negated ? bool_not(r, n) : r;
    }
    case Expr::Case: {
      size_t nw = (e.args.size() - (e.has_else ? 1 : 0)) / 2;
      std::vector<VP> conds, thens;
      bool anyf = false;
      for (size_t k = 0; k < nw; ++k) {
        conds.push_back(eval(*e.args[2 * k], sc));
        thens.push_back(eval(*e.args[2 * k + 1], sc));
        if (conds.back()->kind != Value::B) throw Error("CASE WHEN condition must be boolean");
        if (!is_num(*thens.back())) throw Error("CASE branches must be numeric");
        anyf |= thens.back()->kind == Value::F;
      }
      VP els;
      if (e.has_else) {
        els = eval(*e.args.back(), sc);
        if (!is_num(*els)) throw Error("CASE branches must be numeric");
        anyf |= els->kind == Value::F;
      }
      auto v = new_value(anyf ? Value::F : Value::I, n, false);
      auto get = [&](const Value& x, size_t i) -> double { return x.kind == Value::F ? x.f[i * x.stride()] : double(x.i[i * x.stride()]); };
      auto geti = [&](const Value& x, size_t i) -> int64_t { return x.i[i * x.stride()]; };
      parallel_for(n, GRAIN, [&](size_t, size_t b, size_t en) {
        for (size_t i = b; i < en; ++i) {
          if (anyf) {
            double r = els ? get(*els, i) : 0.0;
            for (size_t k = nw; k-- > 0;)
              if (conds[k]->b[i * conds[k]->stride()]) r = get(*thens[k], i);
            v->f[i] = r;
          } else {
            int64_t r = els ? geti(*els, i) : 0;
            for (size_t k = nw; k-- > 0;)
              if (conds[k]->b[i * conds[k]->stride()]) r = geti(*thens[k], i);
            v->i[i] = r;
          }
        }
      });
      return v;
    }
    case Expr::Func: {
      VP a = eval(*e.args[0], sc);
      if (e.name == "abs") {
        if (!is_num(*a)) throw Error("abs requires a numeric operand");
        auto v = new_value(a->kind, n, a->scalar);
        size_t m = a->scalar ? 1 : n;
        for (size_t i = 0; i < m; ++i) {
          if (a->kind == Value::I) v->i[i] = a->i[i] < 0 ? -a->i[i] : a->i[i];
          else v->f[i] = std::fabs(a->f[i]);
        }
        return v;
      }
      if (a->kind != Value::I || !a->is_date) throw Error(e.name + "() requires a DATE argument");
      auto v = new_value(Value::I, n, a->scalar);
      size_t m = a->scalar ? 1 : n;
      int which = e.name == "year" ? 0 : e.name == "month" ? 1 : 2;
      parallel_for(m, GRAIN, [&](size_t, size_t b, size_t en) {
        for (size_t i = b; i < en; ++i) {
          int y;
          unsigned mo, d;
          civil_from_days(static_cast<int32_t>(a->i[i]), y, mo, d);
          v->i[i] = which == 0 ? y : which == 1 ? static_cast<int64_t>(mo) : static_cast<int64_t>(d);
        }
      });
      return v;
    }
    case Expr::Col: throw Error("internal: unresolved column " + e.name);
    case Expr::Agg: throw Error("aggregate function " + e.name + "() is not allowed here");
    case Expr::Interval: throw Error("INTERVAL is only supported when added to / subtracted from a DATE literal");
    case Expr::Star: throw Error("* is not allowed here");
  }
  throw Error("unsupported expression");
}

}  // namespace warpdb
