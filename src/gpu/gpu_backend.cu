// CUDA execution backend. Mirrors the CPU executor stage by stage (scan+filter, joins, aggregate,
// sort/limit, projection) but every operator is a GPU kernel and all intermediates stay in VRAM.
// Intermediates are late-materialized row-id lists, like on the CPU.
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <set>
#include <unordered_map>

#include "exec/value.h"
#include "gpu/gpu_backend.h"
#include "gpu/jit/codegen.h"
#include "gpu/jit/jit.h"
#include "gpu/op_kernels.cuh"

namespace warpdb::gpu {
namespace {

using Kind = Value::Kind;

// ------------------------------------------------------------------------------------------------
// Device values
// ------------------------------------------------------------------------------------------------
struct DValue {
  Kind kind = Value::I;
  bool is_date = false;
  bool scalar = false;  // one broadcast element: payload lives on the host (si / sf / sb / lit)
  size_t n = 0;
  GpuVec<int64_t> i;  // I and S (dictionary codes)
  GpuVec<double> f;
  GpuVec<uint8_t> b;
  int64_t si = 0;
  double sf = 0;
  uint8_t sb = 0;
  std::shared_ptr<const Column> dict;
  std::string lit;
};
using DVP = std::shared_ptr<const DValue>;

std::shared_ptr<DValue> new_dvec(Kind k, size_t n) {
  auto v = std::make_shared<DValue>();
  v->kind = k;
  v->n = n;
  if (k == Value::I || k == Value::S) v->i.allocate(n);
  else if (k == Value::F) v->f.allocate(n);
  else if (k == Value::B) v->b.allocate(n);
  return v;
}
std::shared_ptr<DValue> new_dscalar(Kind k, size_t n) {
  auto v = std::make_shared<DValue>();
  v->kind = k;
  v->n = n;
  v->scalar = true;
  return v;
}

struct Sel {
  GpuVec<uint32_t> idx;
  size_t k = 0;
};

struct DScope {
  size_t n = 0;
  virtual ~DScope() = default;
  virtual DVP intercept(const Expr& e) = 0;
  virtual std::unique_ptr<DScope> gather(const GpuVec<uint32_t>& idx, size_t k) = 0;
};

// operand visitors -----------------------------------------------------------------------------------
template <class F>
void visit_num(const DValue& v, F&& f) {
  if (v.kind == Value::I || v.kind == Value::S) {
    if (v.scalar) f(ops::Sca<int64_t>{v.si});
    else f(ops::Vec<int64_t>{v.i.data()});
  } else if (v.kind == Value::F) {
    if (v.scalar) f(ops::Sca<double>{v.sf});
    else f(ops::Vec<double>{v.f.data()});
  } else {
    throw Error("numeric operand expected");
  }
}
template <class F>
void visit_bool(const DValue& v, F&& f) {
  if (v.kind != Value::B) throw Error("boolean operand expected");
  if (v.scalar) f(ops::Sca<uint8_t>{v.sb});
  else f(ops::Vec<uint8_t>{v.b.data()});
}

bool is_numeric(const DValue& v) { return v.kind == Value::I || v.kind == Value::F; }
bool intlike(const DValue& v) { return v.kind == Value::I || v.kind == Value::S; }
double as_double(const DValue& v) { return v.kind == Value::F ? v.sf : static_cast<double>(v.si); }

// --- arithmetic / comparison / logic -----------------------------------------------------------------
template <class Op>
DVP bin_arith(Op op, const DValue& a, const DValue& b, size_t n, bool int_result) {
  if (!is_numeric(a) || !is_numeric(b)) throw Error("arithmetic requires numeric operands");
  const bool sc = a.scalar && b.scalar;
  if (sc) {
    auto o = new_dscalar(int_result ? Value::I : Value::F, n);
    if (int_result) {
      o->si = static_cast<int64_t>(op(a.si, b.si));
    } else {
      double r = static_cast<double>(op(as_double(a), as_double(b)));
      if (std::isfinite(r)) r = std::round(r * 1e10) / 1e10;  // decimal-literal emulation, same as CPU
      o->sf = r;
    }
    return o;
  }
  auto o = new_dvec(int_result ? Value::I : Value::F, n);
  visit_num(a, [&](auto A) {
    visit_num(b, [&](auto B) {
      if (int_result) ops::map2<int64_t>(A, B, op, o->i.data(), n);
      else ops::map2<double>(A, B, op, o->f.data(), n);
    });
  });
  return o;
}

template <class Op>
DVP bin_cmp(Op op, const DValue& a, const DValue& b, size_t n) {
  if (!(is_numeric(a) || a.kind == Value::S) || !(is_numeric(b) || b.kind == Value::S))
    throw Error("cannot compare these operand types");
  const bool sc = a.scalar && b.scalar;
  if (sc) {
    auto o = new_dscalar(Value::B, n);
    if (a.kind == Value::F || b.kind == Value::F) o->sb = op(as_double(a), as_double(b));
    else o->sb = op(a.si, b.si);
    return o;
  }
  auto o = new_dvec(Value::B, n);
  visit_num(a, [&](auto A) { visit_num(b, [&](auto B) { ops::map2<uint8_t>(A, B, op, o->b.data(), n); }); });
  return o;
}

DVP bin_logic(bool is_and, const DValue& a, const DValue& b, size_t n) {
  if (a.kind != Value::B || b.kind != Value::B) throw Error("AND/OR require boolean operands");
  if (a.scalar && b.scalar) {
    auto o = new_dscalar(Value::B, n);
    o->sb = is_and ? (a.sb & b.sb) : (a.sb | b.sb);
    return o;
  }
  auto o = new_dvec(Value::B, n);
  visit_bool(a, [&](auto A) {
    visit_bool(b, [&](auto B) {
      if (is_and) ops::map2<uint8_t>(A, B, ops::OpAnd{}, o->b.data(), n);
      else ops::map2<uint8_t>(A, B, ops::OpOr{}, o->b.data(), n);
    });
  });
  return o;
}

DVP d_not(const DValue& a, size_t n) {
  if (a.kind != Value::B) throw Error("NOT requires a boolean operand");
  if (a.scalar) {
    auto o = new_dscalar(Value::B, n);
    o->sb = !a.sb;
    return o;
  }
  auto o = new_dvec(Value::B, n);
  if (n) ops::not_k<<<ops::grid_for(n), 256>>>(a.b.data(), o->b.data(), n);
  return o;
}

enum class CmpOp { Eq, Ne, Lt, Le, Gt, Ge };
CmpOp flip_cmp(CmpOp o) {
  switch (o) {
    case CmpOp::Lt: return CmpOp::Gt;
    case CmpOp::Le: return CmpOp::Ge;
    case CmpOp::Gt: return CmpOp::Lt;
    case CmpOp::Ge: return CmpOp::Le;
    default: return o;
  }
}
CmpOp cmp_of(const std::string& s) {
  if (s == "=") return CmpOp::Eq;
  if (s == "<>") return CmpOp::Ne;
  if (s == "<") return CmpOp::Lt;
  if (s == "<=") return CmpOp::Le;
  if (s == ">") return CmpOp::Gt;
  if (s == ">=") return CmpOp::Ge;
  throw Error("unknown comparison: " + s);
}

DVP cmp_dispatch(CmpOp op, const DValue& a, const DValue& b, size_t n) {
  switch (op) {
    case CmpOp::Eq: return bin_cmp(ops::OpEq{}, a, b, n);
    case CmpOp::Ne: return bin_cmp(ops::OpNe{}, a, b, n);
    case CmpOp::Lt: return bin_cmp(ops::OpLt{}, a, b, n);
    case CmpOp::Le: return bin_cmp(ops::OpLe{}, a, b, n);
    case CmpOp::Gt: return bin_cmp(ops::OpGt{}, a, b, n);
    case CmpOp::Ge: return bin_cmp(ops::OpGe{}, a, b, n);
  }
  throw Error("bad comparison");
}

// A string predicate is evaluated once per dictionary entry on the host, then looked up per row on the GPU.
DVP mask_lookup(const DValue& s, const std::vector<uint8_t>& mask, size_t n) {
  GpuVec<uint8_t> dm(mask.size());
  dm.from_host(mask.data(), mask.size());
  auto o = new_dvec(Value::B, n);
  if (n) ops::mask_lookup_k<<<ops::grid_for(n), 256>>>(s.i.data(), dm.data(), o->b.data(), n);
  WDB_CUDA_CHECK(cudaGetLastError());
  return o;
}

DVP cmp_str_lit(CmpOp op, const DValue& s, const std::string& lit, size_t n) {
  const auto& d = s.dict->dict;
  auto mask_p = cached_dict_mask(s.dict->uid, "cmp" + std::to_string(static_cast<int>(op)) + "|" + lit, d.size(),
                                 [&](std::vector<uint8_t>& mask) {
  for (size_t k = 0; k < d.size(); ++k) {
    int c = d[k].compare(lit);
    switch (op) {
      case CmpOp::Eq: mask[k] = c == 0; break;
      case CmpOp::Ne: mask[k] = c != 0; break;
      case CmpOp::Lt: mask[k] = c < 0; break;
      case CmpOp::Le: mask[k] = c <= 0; break;
      case CmpOp::Gt: mask[k] = c > 0; break;
      case CmpOp::Ge: mask[k] = c >= 0; break;
    }
  }
  });
  return mask_lookup(s, *mask_p, n);
}

DVP d_compare(CmpOp op, const DValue& a, const DValue& b, size_t n) {
  if (a.kind == Value::S && b.kind == Value::L) return cmp_str_lit(op, a, b.lit, n);
  if (a.kind == Value::L && b.kind == Value::S) return cmp_str_lit(flip_cmp(op), b, a.lit, n);
  if (a.kind == Value::S && b.kind == Value::S && a.dict.get() != b.dict.get())
    throw Error("comparing VARCHAR columns from different dictionaries is not supported yet");
  return cmp_dispatch(op, a, b, n);
}

DVP materialize(const DVP& v, size_t n) {
  if (!v->scalar) return v;
  auto o = new_dvec(v->kind, n);
  o->is_date = v->is_date;
  o->dict = v->dict;
  if (n) {
    if (v->kind == Value::I || v->kind == Value::S) ops::fill_k<int64_t><<<ops::grid_for(n), 256>>>(o->i.data(), n, v->si);
    else if (v->kind == Value::F) ops::fill_k<double><<<ops::grid_for(n), 256>>>(o->f.data(), n, v->sf);
    else if (v->kind == Value::B) ops::fill_k<uint8_t><<<ops::grid_for(n), 256>>>(o->b.data(), n, v->sb);
  }
  WDB_CUDA_CHECK(cudaGetLastError());
  return o;
}

DVP gather_dvalue(const DVP& v, const GpuVec<uint32_t>& idx, size_t k) {
  if (v->scalar) {
    auto c = std::make_shared<DValue>();
    c->kind = v->kind; c->is_date = v->is_date; c->scalar = true; c->n = k;
    c->si = v->si; c->sf = v->sf; c->sb = v->sb; c->dict = v->dict; c->lit = v->lit;
    return c;
  }
  auto o = new_dvec(v->kind, k);
  o->is_date = v->is_date;
  o->dict = v->dict;
  if (v->kind == Value::I || v->kind == Value::S) ops::gather<int64_t, int64_t>(v->i.data(), idx.data(), o->i.data(), k);
  else if (v->kind == Value::F) ops::gather<double, double>(v->f.data(), idx.data(), o->f.data(), k);
  else if (v->kind == Value::B) ops::gather<uint8_t, uint8_t>(v->b.data(), idx.data(), o->b.data(), k);
  return o;
}

Sel select_true(const DValue& b) {
  Sel s;
  const size_t n = b.n;
  if (b.kind != Value::B) throw Error("predicate must be boolean");
  if (b.scalar) {
    if (b.sb) {
      s.idx.allocate(n);
      s.k = n;
      if (n) ops::iota_k<<<ops::grid_for(n), 256>>>(s.idx.data(), n);
    }
    return s;
  }
  if (n == 0) return s;
  s.idx.allocate(n);
  GpuVec<uint8_t> ws(select_workspace_bytes(n));
  GpuVec<uint32_t> cnt(1);
  select_fused(MaskIndexGen{b.b.data()}, n, s.idx.data(), cnt.data(), ws.data());
  s.k = cnt.to_host(1)[0];  // synchronizes
  return s;
}

// ------------------------------------------------------------------------------------------------
// Expression evaluation
// ------------------------------------------------------------------------------------------------
DVP d_eval(const Expr& e, DScope& sc) {
  if (DVP v = sc.intercept(e)) return v;
  const size_t n = sc.n;
  switch (e.kind) {
    case Expr::Int: {
      auto v = new_dscalar(Value::I, n);
      v->si = e.ival;
      return v;
    }
    case Expr::Float: {
      auto v = new_dscalar(Value::F, n);
      v->sf = e.fval;
      return v;
    }
    case Expr::Date: {
      auto v = new_dscalar(Value::I, n);
      v->si = e.ival;
      v->is_date = true;
      return v;
    }
    case Expr::Str: {
      auto v = new_dscalar(Value::L, n);
      v->lit = e.name;
      return v;
    }
    case Expr::Bin: {
      const std::string& op = e.name;
      DVP a = d_eval(*e.args[0], sc);
      DVP b = d_eval(*e.args[1], sc);
      if (op == "and") return bin_logic(true, *a, *b, n);
      if (op == "or") return bin_logic(false, *a, *b, n);
      if (op == "=" || op == "<>" || op == "<" || op == "<=" || op == ">" || op == ">=")
        return d_compare(cmp_of(op), *a, *b, n);
      bool ints = a->kind == Value::I && b->kind == Value::I;
      DVP r;
      if (op == "+") r = bin_arith(ops::OpAdd{}, *a, *b, n, ints);
      else if (op == "-") r = bin_arith(ops::OpSub{}, *a, *b, n, ints);
      else if (op == "*") r = bin_arith(ops::OpMul{}, *a, *b, n, ints);
      else if (op == "/") r = bin_arith(ops::OpDiv{}, *a, *b, n, false);
      else if (op == "%") r = bin_arith(ops::OpMod{}, *a, *b, n, ints);
      else throw Error("unknown operator: " + op);
      if (ints && (op == "+" || op == "-")) {
        auto m = std::const_pointer_cast<DValue>(r);
        m->is_date = a->is_date != b->is_date && !(op == "-" && b->is_date);
      }
      return r;
    }
    case Expr::Not: return d_not(*d_eval(*e.args[0], sc), n);
    case Expr::Neg: {
      DVP a = d_eval(*e.args[0], sc);
      if (!is_numeric(*a)) throw Error("unary minus requires a numeric operand");
      if (a->scalar) {
        auto o = new_dscalar(a->kind, n);
        o->si = -a->si;
        o->sf = -a->sf;
        return o;
      }
      auto o = new_dvec(a->kind, n);
      if (n) {
        if (a->kind == Value::I) ops::unary_neg_k<int64_t, int64_t><<<ops::grid_for(n), 256>>>(a->i.data(), o->i.data(), n);
        else ops::unary_neg_k<double, double><<<ops::grid_for(n), 256>>>(a->f.data(), o->f.data(), n);
      }
      return o;
    }
    case Expr::Between: {
      DVP x = d_eval(*e.args[0], sc), lo = d_eval(*e.args[1], sc), hi = d_eval(*e.args[2], sc);
      DVP r = bin_logic(true, *d_compare(CmpOp::Ge, *x, *lo, n), *d_compare(CmpOp::Le, *x, *hi, n), n);
      return e.negated ? d_not(*r, n) : r;
    }
    case Expr::In: {
      DVP x = d_eval(*e.args[0], sc);
      DVP r;
      if (x->kind == Value::S) {
        std::set<std::string> set;
        for (size_t k = 1; k < e.args.size(); ++k) {
          DVP it = d_eval(*e.args[k], sc);
          if (it->kind != Value::L) throw Error("IN list for a VARCHAR column must contain string literals");
          set.insert(it->lit);
        }
        const auto& d = x->dict->dict;
        std::string key = "in";
        for (auto& v : set) key += "|" + v;
        auto mask = cached_dict_mask(x->dict->uid, key, d.size(), [&](std::vector<uint8_t>& m) {
          for (size_t k = 0; k < d.size(); ++k) m[k] = set.count(d[k]) != 0;
        });
        r = mask_lookup(*x, *mask, n);
      } else {
        for (size_t k = 1; k < e.args.size(); ++k) {
          DVP c = d_compare(CmpOp::Eq, *x, *d_eval(*e.args[k], sc), n);
          r = r ? bin_logic(false, *r, *c, n) : c;
        }
        if (!r) throw Error("empty IN list");
      }
      return e.negated ? d_not(*r, n) : r;
    }
    case Expr::Like: {
      DVP x = d_eval(*e.args[0], sc), p = d_eval(*e.args[1], sc);
      if (x->kind != Value::S || p->kind != Value::L) throw Error("LIKE requires a VARCHAR column and a string pattern");
      const auto& d = x->dict->dict;
      auto mask = cached_dict_mask(x->dict->uid, "like|" + p->lit, d.size(), [&](std::vector<uint8_t>& m) {
        for (size_t k = 0; k < d.size(); ++k) m[k] = like_match(d[k], p->lit);
      });
      DVP r = mask_lookup(*x, *mask, n);
      return e.negated ? d_not(*r, n) : r;
    }
    case Expr::Case: {
      size_t nw = (e.args.size() - (e.has_else ? 1 : 0)) / 2;
      std::vector<DVP> conds, thens;
      bool anyf = false;
      for (size_t k = 0; k < nw; ++k) {
        conds.push_back(d_eval(*e.args[2 * k], sc));
        thens.push_back(d_eval(*e.args[2 * k + 1], sc));
        if (conds.back()->kind != Value::B) throw Error("CASE WHEN condition must be boolean");
        if (!is_numeric(*thens.back())) throw Error("CASE branches must be numeric");
        anyf |= thens.back()->kind == Value::F;
      }
      DVP cur;
      if (e.has_else) {
        cur = d_eval(*e.args.back(), sc);
        if (!is_numeric(*cur)) throw Error("CASE branches must be numeric");
        anyf |= cur->kind == Value::F;
      } else {
        auto z = new_dscalar(Value::I, n);
        cur = z;
      }
      for (size_t k = nw; k-- > 0;) {
        auto o = new_dvec(anyf ? Value::F : Value::I, n);
        visit_bool(*conds[k], [&](auto C) {
          visit_num(*thens[k], [&](auto A) {
            visit_num(*cur, [&](auto B) {
              if (anyf) ops::where<double>(C, A, B, o->f.data(), n);
              else ops::where<int64_t>(C, A, B, o->i.data(), n);
            });
          });
        });
        WDB_CUDA_CHECK(cudaGetLastError());
        cur = o;
      }
      return cur;
    }
    case Expr::Func: {
      DVP a = d_eval(*e.args[0], sc);
      if (e.name == "abs") {
        if (!is_numeric(*a)) throw Error("abs requires a numeric operand");
        if (a->scalar) {
          auto o = new_dscalar(a->kind, n);
          o->si = a->si < 0 ? -a->si : a->si;
          o->sf = std::fabs(a->sf);
          return o;
        }
        auto o = new_dvec(a->kind, n);
        if (n) {
          if (a->kind == Value::I) ops::unary_abs_k<int64_t><<<ops::grid_for(n), 256>>>(a->i.data(), o->i.data(), n);
          else ops::unary_abs_k<double><<<ops::grid_for(n), 256>>>(a->f.data(), o->f.data(), n);
        }
        return o;
      }
      if (a->kind != Value::I || !a->is_date) throw Error(e.name + "() requires a DATE argument");
      int which = e.name == "year" ? 0 : e.name == "month" ? 1 : 2;
      if (a->scalar) {
        int y;
        unsigned m, d;
        civil_from_days(static_cast<int32_t>(a->si), y, m, d);
        auto o = new_dscalar(Value::I, n);
        o->si = which == 0 ? y : which == 1 ? static_cast<int64_t>(m) : static_cast<int64_t>(d);
        return o;
      }
      auto o = new_dvec(Value::I, n);
      if (n) ops::date_part_k<<<ops::grid_for(n), 256>>>(a->i.data(), o->i.data(), n, which);
      WDB_CUDA_CHECK(cudaGetLastError());
      return o;
    }
    case Expr::Col: throw Error("internal: unresolved column " + e.name);
    case Expr::Agg: throw Error("aggregate function " + e.name + "() is not allowed here");
    case Expr::Interval: throw Error("INTERVAL is only supported when added to / subtracted from a DATE literal");
    case Expr::Star: throw Error("* is not allowed here");
  }
  throw Error("unsupported expression");
}

// ------------------------------------------------------------------------------------------------
// Relations (row-id lists) and scopes
// ------------------------------------------------------------------------------------------------
struct DRel {
  std::vector<int> tbls;
  std::vector<GpuVec<uint32_t>> rid;
  size_t n = 0;
  int pos(int t) const {
    for (size_t i = 0; i < tbls.size(); ++i)
      if (tbls[i] == t) return static_cast<int>(i);
    return -1;
  }
};

DRel d_compose(const DRel& r, const GpuVec<uint32_t>& idx, size_t k) {
  DRel o;
  o.tbls = r.tbls;
  o.n = k;
  for (auto& src : r.rid) {
    GpuVec<uint32_t> dst(k);
    ops::gather<uint32_t, uint32_t>(src.data(), idx.data(), dst.data(), k);
    o.rid.push_back(std::move(dst));
  }
  return o;
}

struct GpuCtx {
  ColumnCache* cache;
  double upload_ms = 0;
  uint64_t upload_bytes = 0;
};

class DRelScope : public DScope {
 public:
  DRelScope(const TableVec& tabs, DRel r, GpuCtx* ctx) : tabs_(tabs), rel(std::move(r)), ctx_(ctx) { n = rel.n; }
  DVP intercept(const Expr& e) override {
    if (e.kind != Expr::Col) return nullptr;
    uint64_t key = (static_cast<uint64_t>(e.tbl) << 32) | static_cast<uint32_t>(e.col);
    auto it = cache_.find(key);
    if (it != cache_.end()) return it->second;
    int p = rel.pos(e.tbl);
    if (p < 0) throw Error("internal: column's table is not part of the relation");
    const auto& col = tabs_[e.tbl]->cols[e.col];
    bool was = ctx_->cache->resident(*col);
    uint64_t before = ctx_->cache->uploaded_bytes();
    Timer t;
    const void* dev = ctx_->cache->get(*col);
    if (!was) {
      WDB_CUDA_CHECK(cudaDeviceSynchronize());
      ctx_->upload_ms += t.ms();
      ctx_->upload_bytes += ctx_->cache->uploaded_bytes() - before;
    }
    DVP v;
    switch (col->type) {
      case Type::INT32:
      case Type::DATE:
      case Type::VARCHAR: {
        auto o = new_dvec(col->type == Type::VARCHAR ? Value::S : Value::I, n);
        o->is_date = col->type == Type::DATE;
        if (col->type == Type::VARCHAR) o->dict = col;
        ops::gather<int64_t, int32_t>(static_cast<const int32_t*>(dev), rel.rid[p].data(), o->i.data(), n);
        v = o;
        break;
      }
      case Type::INT64: {
        auto o = new_dvec(Value::I, n);
        ops::gather<int64_t, int64_t>(static_cast<const int64_t*>(dev), rel.rid[p].data(), o->i.data(), n);
        v = o;
        break;
      }
      case Type::FLOAT64: {
        auto o = new_dvec(Value::F, n);
        ops::gather<double, double>(static_cast<const double*>(dev), rel.rid[p].data(), o->f.data(), n);
        v = o;
        break;
      }
    }
    cache_[key] = v;
    return v;
  }
  std::unique_ptr<DScope> gather(const GpuVec<uint32_t>& idx, size_t k) override {
    return std::make_unique<DRelScope>(tabs_, d_compose(rel, idx, k), ctx_);
  }
  void set_rows(const GpuVec<uint32_t>& idx, size_t k) {
    rel = d_compose(rel, idx, k);
    n = k;
    cache_.clear();
  }
  DRel rel;

 private:
  const TableVec& tabs_;
  GpuCtx* ctx_;
  std::unordered_map<uint64_t, DVP> cache_;
};

class DAggScope : public DScope {
 public:
  std::unordered_map<std::string, DVP> m;
  DVP intercept(const Expr& e) override {
    auto it = m.find(e.str());
    if (it != m.end()) return it->second;
    if (e.kind == Expr::Agg) throw Error("internal: aggregate was not computed: " + e.str());
    if (e.kind == Expr::Col)
      throw Error("column \"" + e.name + "\" must appear in the GROUP BY clause or be used in an aggregate function");
    return nullptr;
  }
  std::unique_ptr<DScope> gather(const GpuVec<uint32_t>& idx, size_t k) override {
    auto s = std::make_unique<DAggScope>();
    for (auto& kv : m) s->m[kv.first] = gather_dvalue(kv.second, idx, k);
    s->n = k;
    return s;
  }
};

// ------------------------------------------------------------------------------------------------
// Hash join
// ------------------------------------------------------------------------------------------------
struct JoinPairs {
  GpuVec<uint32_t> l, r;
  size_t n = 0;
};

ops::KeyPtrs key_ptrs(const std::vector<DVP>& k) {
  if (k.size() > static_cast<size_t>(ops::MAX_KEYS)) throw Error("too many join / group keys (max 8 on the GPU)");
  ops::KeyPtrs kp{};
  kp.K = static_cast<int>(k.size());
  for (size_t j = 0; j < k.size(); ++j) kp.p[j] = k[j]->i.data();
  return kp;
}

JoinPairs d_hash_join(const std::vector<DVP>& lk, size_t ln, const std::vector<DVP>& rk, size_t rn) {
  JoinPairs out;
  if (ln == 0 || rn == 0) return out;
  const bool build_left = ln <= rn;
  const auto& bk = build_left ? lk : rk;
  const auto& pk = build_left ? rk : lk;
  const size_t bn = build_left ? ln : rn, pn = build_left ? rn : ln;
  ops::KeyPtrs bp = key_ptrs(bk), pp = key_ptrs(pk);

  size_t cap = 16;
  while (cap < bn * 2) cap <<= 1;
  const uint32_t mask = static_cast<uint32_t>(cap - 1);
  GpuVec<uint32_t> head(cap), next(bn);
  WDB_CUDA_CHECK(cudaMemset(head.data(), 0xFF, cap * 4));
  ops::join_build_k<<<ops::grid_for(bn), 256>>>(bp, bn, head.data(), next.data(), mask);

  GpuVec<uint32_t> counts(pn), offs(pn), total(1);
  ops::join_count_k<<<ops::grid_for(pn), 256>>>(bp, pp, pn, head.data(), next.data(), mask, counts.data());
  GpuVec<uint8_t> ws(scan_workspace_bytes(pn));
  scan_exclusive(counts.data(), offs.data(), pn, ws.data(), total.data());
  size_t m = total.to_host(1)[0];  // synchronizes
  out.n = m;
  GpuVec<uint32_t> probe_idx(m), build_idx(m);
  if (m) ops::join_write_k<<<ops::grid_for(pn), 256>>>(bp, pp, pn, head.data(), next.data(), mask, offs.data(),
                                                       probe_idx.data(), build_idx.data());
  WDB_CUDA_CHECK(cudaGetLastError());
  out.l = std::move(build_left ? build_idx : probe_idx);
  out.r = std::move(build_left ? probe_idx : build_idx);
  return out;
}

// Re-expresses VARCHAR codes in the code space of `target`'s dictionary (-1 = no match); both are sorted.
DVP d_remap_codes(const DValue& v, const Column& target) {
  const auto& src = v.dict->dict;
  const auto& dst = target.dict;
  std::vector<int64_t> m(src.size(), -1);
  size_t j = 0;
  for (size_t i = 0; i < src.size(); ++i) {
    while (j < dst.size() && dst[j] < src[i]) ++j;
    if (j < dst.size() && dst[j] == src[i]) m[i] = static_cast<int64_t>(j);
  }
  GpuVec<int64_t> dm(m.size());
  dm.from_host(m.data(), m.size());
  auto o = new_dvec(Value::I, v.n);
  if (v.n) ops::remap_k<<<ops::grid_for(v.n), 256>>>(v.i.data(), dm.data(), o->i.data(), v.n);
  WDB_CUDA_CHECK(cudaGetLastError());
  return o;
}

// ------------------------------------------------------------------------------------------------
// Hash aggregation
// ------------------------------------------------------------------------------------------------
enum AggFunc { SUM, COUNT, AVG, MIN, MAX };

struct AggInput {
  AggFunc func;
  DVP arg;  // null for count(*)
};

struct AggOutput {
  std::vector<DVP> keys, aggs;
  size_t n = 0;
};

template <int OP, class V, class Arg>
void launch_agg_update(Arg arg, const uint32_t* row_slot, size_t n, V* acc, bool warp_agg) {
  if (!n) return;
  ops::agg_update_k<OP, V, Arg><<<ops::grid_for(n), 256>>>(arg, row_slot, n, acc, warp_agg ? 1 : 0);
  WDB_CUDA_CHECK(cudaGetLastError());
}

template <int OP, class V>
void agg_update_dispatch(const DValue* arg, const uint32_t* row_slot, size_t n, V* acc, bool warp_agg) {
  if (!arg) {
    launch_agg_update<OP, V>(ops::Sca<int64_t>{1}, row_slot, n, acc, warp_agg);
    return;
  }
  visit_num(*arg, [&](auto A) { launch_agg_update<OP, V>(A, row_slot, n, acc, warp_agg); });
}

__global__ void avg_k(const double* sum, const int64_t* cnt, double* out, size_t n) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i < n) out[i] = cnt[i] ? sum[i] / static_cast<double>(cnt[i]) : 0.0;
}

AggOutput d_hash_aggregate(Engine& eng, const PlannedQuery& pq, const std::vector<DVP>& keys_in,
                           std::vector<AggInput>& ins, size_t nrows) {
  const size_t K = keys_in.size(), A = ins.size();
  if (K > static_cast<size_t>(ops::MAX_KEYS)) throw Error("too many GROUP BY keys (max 8 on the GPU)");
  // 64-bit key views: ints/codes as-is, doubles reinterpreted bit-for-bit, booleans widened.
  std::vector<DVP> keyv(K);
  std::vector<GpuVec<int64_t>> widened(K);
  ops::KeyPtrs kp{};
  kp.K = static_cast<int>(K);
  for (size_t j = 0; j < K; ++j) {
    keyv[j] = materialize(keys_in[j], nrows);
    const DValue& v = *keyv[j];
    if (v.kind == Value::I || v.kind == Value::S) kp.p[j] = v.i.data();
    else if (v.kind == Value::F) kp.p[j] = reinterpret_cast<const int64_t*>(v.f.data());
    else if (v.kind == Value::B) {
      widened[j].allocate(nrows);
      if (nrows) ops::widen_u8_k<<<ops::grid_for(nrows), 256>>>(v.b.data(), widened[j].data(), nrows);
      kp.p[j] = widened[j].data();
    } else throw Error("cannot GROUP BY this expression type");
  }

  // ---- group table (slot -> representative row) ----
  GpuVec<uint32_t> rep, row_slot;
  size_t cap = 1;
  bool warp_agg = true;
  if (K > 0) {
    double est = static_cast<double>(std::max<size_t>(nrows, 1));
    double prod = 1;
    for (size_t j = 0; j < K; ++j) prod *= eng.key_ndv(pq, pq.sel.group_by[j], nrows);
    est = std::min(est, prod);
    cap = 16;
    while (static_cast<double>(cap) < est * 2.0) cap <<= 1;
    row_slot.allocate(nrows);
    for (;;) {
      rep.allocate(cap);
      WDB_CUDA_CHECK(cudaMemset(rep.data(), 0xFF, cap * 4));
      GpuVec<uint32_t> overflow(1);
      WDB_CUDA_CHECK(cudaMemset(overflow.data(), 0, 4));
      if (nrows) ops::agg_insert_k<<<ops::grid_for(nrows), 256>>>(kp, nrows, rep.data(), static_cast<uint32_t>(cap - 1),
                                                                 row_slot.data(), overflow.data());
      WDB_CUDA_CHECK(cudaGetLastError());
      if (overflow.to_host(1)[0] == 0) break;
      cap <<= 2;  // underestimated the number of groups: retry with a bigger table
      if (cap > (size_t(1) << 30)) throw Error("GPU aggregation: too many groups");
    }
    warp_agg = cap <= 4096;
  }
  const uint32_t* rs = K > 0 ? row_slot.data() : nullptr;

  // ---- accumulate ----
  struct Acc {
    GpuVec<int64_t> i;
    GpuVec<double> f;
    GpuVec<int64_t> cnt;
  };
  std::vector<Acc> accs(A);
  for (size_t a = 0; a < A; ++a) {
    AggInput& in = ins[a];
    Acc& ac = accs[a];
    DVP keep = in.arg ? materialize(in.arg, nrows) : nullptr;  // owns the (possibly broadcast) argument
    const DValue* arg = keep.get();
    const bool argf = arg && arg->kind == Value::F;
    auto init_i = [&](GpuVec<int64_t>& v, int64_t x) {
      v.allocate(cap);
      ops::fill_k<int64_t><<<ops::grid_for(cap), 256>>>(v.data(), cap, x);
    };
    auto init_f = [&](GpuVec<double>& v, double x) {
      v.allocate(cap);
      ops::fill_k<double><<<ops::grid_for(cap), 256>>>(v.data(), cap, x);
    };
    switch (in.func) {
      case COUNT:
        init_i(ac.i, 0);
        agg_update_dispatch<ops::A_SUM, int64_t>(nullptr, rs, nrows, ac.i.data(), warp_agg);
        break;
      case SUM:
        if (argf) { init_f(ac.f, 0.0); agg_update_dispatch<ops::A_SUM, double>(arg, rs, nrows, ac.f.data(), warp_agg); }
        else { init_i(ac.i, 0); agg_update_dispatch<ops::A_SUM, int64_t>(arg, rs, nrows, ac.i.data(), warp_agg); }
        break;
      case AVG:
        init_f(ac.f, 0.0);
        init_i(ac.cnt, 0);
        agg_update_dispatch<ops::A_SUM, double>(arg, rs, nrows, ac.f.data(), warp_agg);
        agg_update_dispatch<ops::A_SUM, int64_t>(nullptr, rs, nrows, ac.cnt.data(), warp_agg);
        break;
      case MIN:
        if (argf) { init_f(ac.f, std::numeric_limits<double>::infinity()); agg_update_dispatch<ops::A_MIN, double>(arg, rs, nrows, ac.f.data(), warp_agg); }
        else { init_i(ac.i, INT64_MAX); agg_update_dispatch<ops::A_MIN, int64_t>(arg, rs, nrows, ac.i.data(), warp_agg); }
        break;
      case MAX:
        if (argf) { init_f(ac.f, -std::numeric_limits<double>::infinity()); agg_update_dispatch<ops::A_MAX, double>(arg, rs, nrows, ac.f.data(), warp_agg); }
        else { init_i(ac.i, INT64_MIN); agg_update_dispatch<ops::A_MAX, int64_t>(arg, rs, nrows, ac.i.data(), warp_agg); }
        break;
    }
    WDB_CUDA_CHECK(cudaGetLastError());
  }

  // ---- compact occupied slots into dense group arrays ----
  AggOutput out;
  GpuVec<uint32_t> slots;
  size_t G = 1;
  if (K > 0) {
    slots.allocate(cap);
    GpuVec<uint8_t> ws(select_workspace_bytes(cap));
    GpuVec<uint32_t> cnt(1);
    select_fused(ops::SlotGen{rep.data()}, cap, slots.data(), cnt.data(), ws.data());
    G = cnt.to_host(1)[0];
  }
  out.n = G;
  GpuVec<uint32_t> rep_rows;
  if (K > 0) {
    rep_rows.allocate(G);
    ops::gather<uint32_t, uint32_t>(rep.data(), slots.data(), rep_rows.data(), G);
  }
  for (size_t j = 0; j < K; ++j) {
    const DValue& src = *keyv[j];
    auto o = new_dvec(src.kind == Value::S ? Value::S : (src.kind == Value::F ? Value::F : (src.kind == Value::B ? Value::B : Value::I)), G);
    o->is_date = src.is_date;
    o->dict = src.dict;
    if (src.kind == Value::F) ops::gather<double, double>(src.f.data(), rep_rows.data(), o->f.data(), G);
    else if (src.kind == Value::B) ops::gather<uint8_t, uint8_t>(src.b.data(), rep_rows.data(), o->b.data(), G);
    else ops::gather<int64_t, int64_t>(src.i.data(), rep_rows.data(), o->i.data(), G);
    out.keys.push_back(o);
  }
  auto take_i = [&](const GpuVec<int64_t>& acc) {
    GpuVec<int64_t> r(G);
    if (K > 0) ops::gather<int64_t, int64_t>(acc.data(), slots.data(), r.data(), G);
    else WDB_CUDA_CHECK(cudaMemcpy(r.data(), acc.data(), 8, cudaMemcpyDeviceToDevice));
    return r;
  };
  auto take_f = [&](const GpuVec<double>& acc) {
    GpuVec<double> r(G);
    if (K > 0) ops::gather<double, double>(acc.data(), slots.data(), r.data(), G);
    else WDB_CUDA_CHECK(cudaMemcpy(r.data(), acc.data(), 8, cudaMemcpyDeviceToDevice));
    return r;
  };
  for (size_t a = 0; a < A; ++a) {
    const AggInput& in = ins[a];
    Acc& ac = accs[a];
    const bool argf = in.arg && in.arg->kind == Value::F;
    std::shared_ptr<DValue> o;
    switch (in.func) {
      case COUNT:
        o = new_dvec(Value::I, 0);
        o->n = G;
        o->i = take_i(ac.i);
        break;
      case SUM:
        if (argf) { o = new_dvec(Value::F, 0); o->n = G; o->f = take_f(ac.f); }
        else { o = new_dvec(Value::I, 0); o->n = G; o->i = take_i(ac.i); }
        break;
      case AVG: {
        auto s = take_f(ac.f);
        auto c = take_i(ac.cnt);
        o = new_dvec(Value::F, G);
        if (G) avg_k<<<ops::grid_for(G), 256>>>(s.data(), c.data(), o->f.data(), G);  // 0 for empty groups
        break;
      }
      case MIN:
      case MAX:
        if (argf) { o = new_dvec(Value::F, 0); o->n = G; o->f = take_f(ac.f); }
        else {
          o = new_dvec(in.arg->kind == Value::S ? Value::S : Value::I, 0);
          o->n = G;
          o->i = take_i(ac.i);
          o->is_date = in.arg->is_date;
          o->dict = in.arg->dict;
        }
        break;
    }
    out.aggs.push_back(o);
  }
  WDB_CUDA_CHECK(cudaGetLastError());
  return out;
}

// ------------------------------------------------------------------------------------------------
// Sort: stable LSD radix sort over (key bits, row id), one pass per ORDER BY key, last key first.
// ------------------------------------------------------------------------------------------------
GpuVec<uint32_t> d_sort_perm(const std::vector<DVP>& keys_in, const std::vector<bool>& desc, size_t n) {
  GpuVec<uint32_t> perm(n);
  if (n) ops::iota_k<<<ops::grid_for(n), 256>>>(perm.data(), n);
  if (n <= 1) return perm;
  GpuVec<uint64_t> kbits(n), kbits2(n);
  GpuVec<uint32_t> perm2(n);
  RadixWorkspace ws;
  for (size_t jj = keys_in.size(); jj-- > 0;) {
    DVP kv = materialize(keys_in[jj], n);
    if (kv->kind == Value::L) continue;
    const int dsc = desc[jj] ? 1 : 0;
    // gather the key in current permutation order, then convert to order-preserving unsigned bits
    if (kv->kind == Value::I || kv->kind == Value::S) {
      GpuVec<int64_t> g(n);
      ops::gather<int64_t, int64_t>(kv->i.data(), perm.data(), g.data(), n);
      ops::sortkey_i64_k<<<ops::grid_for(n), 256>>>(g.data(), kbits.data(), n, dsc);
    } else if (kv->kind == Value::F) {
      GpuVec<double> g(n);
      ops::gather<double, double>(kv->f.data(), perm.data(), g.data(), n);
      ops::sortkey_f64_k<<<ops::grid_for(n), 256>>>(g.data(), kbits.data(), n, dsc);
    } else {
      GpuVec<uint8_t> g(n);
      ops::gather<uint8_t, uint8_t>(kv->b.data(), perm.data(), g.data(), n);
      ops::sortkey_u8_k<<<ops::grid_for(n), 256>>>(g.data(), kbits.data(), n, dsc);
    }
    WDB_CUDA_CHECK(cudaGetLastError());
    bool in_b = radix_sort<uint64_t, uint32_t>(kbits.data(), kbits2.data(), perm.data(), perm2.data(), n, ws);
    if (in_b) std::swap(perm, perm2);
  }
  return perm;
}

std::string fmt_ms(double ms) {
  char b[32];
  std::snprintf(b, sizeof b, "%.2f ms", ms);
  return b;
}
StageStat make_stat(StageOp op, const std::string& label, double rin, double raux, double rout, int k1, int k2,
                    double ms, int k3 = 0) {
  StageStat s;
  s.op = op; s.gpu = true; s.label = label;
  s.rows_in = rin; s.rows_aux = raux; s.rows_out = rout;
  s.k1 = k1; s.k2 = k2; s.k3 = k3; s.ms = ms;
  return s;
}
double sync_ms(const Timer& t) {
  WDB_CUDA_CHECK(cudaDeviceSynchronize());
  return t.ms();
}

// device -> host result column
VP download(const DVP& vin, size_t n) {
  DVP v = materialize(vin, n);
  auto o = std::make_shared<Value>();
  o->kind = v->kind;
  o->is_date = v->is_date;
  o->n = n;
  o->dict = v->dict;
  if (v->kind == Value::I || v->kind == Value::S) o->i = v->i.to_host(n);
  else if (v->kind == Value::F) o->f = v->f.to_host(n);
  else if (v->kind == Value::B) o->b = v->b.to_host(n);
  else o->lit = v->lit;
  return o;
}

// ------------------------------------------------------------------------------------------------
// JIT fusion: whole pipelines compiled to one kernel (see gpu/jit/codegen.h).
// ------------------------------------------------------------------------------------------------
struct JitCtx {
  bool enabled = false;
  size_t min_rows = 200000;  // below this a not-yet-cached kernel is not worth an NVRTC compile
  size_t force_cap = 0;      // test hook: start fused group tables at this capacity to exercise overflow retry
  std::vector<std::string> notes;
  double compile_ms = 0;  // NVRTC / module-load time spent so far in this query
};

std::string group_digits(double v) {
  std::string s = std::to_string(static_cast<long long>(v)), out;
  for (size_t i = 0; i < s.size(); ++i) {
    if (i && (s.size() - i) % 3 == 0) out += ',';
    out += s[i];
  }
  return out;
}

// Device pointer of a base column, uploading it if necessary (upload time is accounted separately).
const void* jit_col_ptr(GpuCtx& ctx, const Column& col) {
  const bool was = ctx.cache->resident(col);
  const uint64_t before = ctx.cache->uploaded_bytes();
  Timer t;
  const void* p = ctx.cache->get(col);
  if (!was) {
    WDB_CUDA_CHECK(cudaDeviceSynchronize());
    ctx.upload_ms += t.ms();
    ctx.upload_bytes += ctx.cache->uploaded_bytes() - before;
  }
  return p;
}

std::vector<GpuVec<uint8_t>> jit_upload_masks(const JitPlan& plan) {
  std::vector<GpuVec<uint8_t>> out;
  for (auto& m : plan.masks) {
    auto hm = cached_dict_mask(m.dictcol->uid, m.key, m.dictcol->dict.size(), m.fill);
    GpuVec<uint8_t> d(hm->size());
    d.from_host(hm->data(), hm->size());
    out.push_back(std::move(d));
  }
  return out;
}

std::shared_ptr<JitKernel> jit_kernel(const JitPlan& plan, size_t rows, JitCtx& jc, const char** origin,
                                      double* ms) {
  auto& rt = JitRuntime::instance();
  if (rows < jc.min_rows && !rt.peek(plan.source, plan.entry))
    throw JitUnsupported("input is small and the kernel is not cached: compiling would cost more than it saves");
  Timer t;
  try {
    auto k = rt.get(plan.source, plan.entry, ms, origin);
    jc.compile_ms += *ms;  // one-off cost: excluded from the stage times the cost model learns from
    return k;
  } catch (...) {
    jc.compile_ms += t.ms();  // a failed compile is just as one-off: keep it out of the learned stage times
    throw;
  }
}

std::string jit_origin_note(const char* origin, double compile_ms, double kernel_ms) {
  std::string o = origin ? origin : "";
  std::string s = "kernel ";
  s += fmt_ms(kernel_ms);
  s += o == "compiled" ? ", NVRTC compile " + fmt_ms(compile_ms) : ", " + o;
  // when getting the kernel was unexpectedly slow, say where the time went (helps diagnose driver / disk stalls)
  if (compile_ms > 250 && o != "compiled") s += " [" + JitRuntime::instance().last_timing() + "]";
  return s;
}

// scan + all pushed-down predicates + stream compaction in one kernel. Returns false => use the interpreter.
bool jit_filter(GpuCtx& ctx, const PlannedQuery& pq, int tbl, JitCtx& jc, Sel& out, double* ms_out) {
  const auto& preds = pq.push[tbl];
  const size_t n = pq.tabs[tbl]->nrows;
  if (!jc.enabled || preds.empty() || n == 0) return false;
  try {
    JitPlan plan = gen_filter_plan(pq, tbl, preds);
    double cms = 0;
    const char* origin = "";
    auto kernel = jit_kernel(plan, n, jc, &origin, &cms);
    auto masks = jit_upload_masks(plan);
    ArgPack args;
    args.add<unsigned long long>(n);
    for (auto& tc : plan.cols) args.add<const void*>(jit_col_ptr(ctx, *pq.tabs[tc.first]->cols[tc.second]));
    for (auto& m : masks) args.add<const void*>(m.data());
    const size_t tiles = ceil_div(n, size_t(256) * 16);
    GpuVec<uint8_t> ws(tiles * 8 + 64);
    WDB_CUDA_CHECK(cudaMemset(ws.data(), 0, tiles * 8 + 4));
    out.idx.allocate(n);
    GpuVec<uint32_t> cnt(1);
    auto* status = reinterpret_cast<unsigned long long*>(ws.data());
    auto* counter = reinterpret_cast<unsigned int*>(status + tiles);
    args.add<void*>(out.idx.data());
    args.add<void*>(status);
    args.add<void*>(counter);
    args.add<unsigned int>(static_cast<unsigned int>(tiles - 1));
    args.add<void*>(cnt.data());
    Timer t;
    JitRuntime::instance().launch(*kernel, static_cast<unsigned>(tiles), 256, args);
    out.k = cnt.to_host(1)[0];  // synchronizes
    const double kms = t.ms();
    if (ms_out) *ms_out = kms;
    jc.notes.push_back("jit fused filter+compact on " + pq.tabs[tbl]->name + " (" + std::to_string(preds.size()) +
                       " predicates, " + jit_origin_note(origin, cms, kms) + ")");
    return true;
  } catch (const JitUnsupported&) {
    return false;
  } catch (const JitError& e) {
    jc.notes.push_back(std::string("jit filter fell back to interpreter: ") + e.what());
    return false;
  }
}

// Whole post-scan pipeline in one kernel: [filter] -> group keys -> group-table slot -> every aggregate update.
// `rel == nullptr` means the pipeline scans table `scan_tbl` directly (row i = row i of the table).
std::optional<AggOutput> jit_aggregate(Engine& eng, const PlannedQuery& pq, GpuCtx& ctx, JitCtx& jc, const DRel* rel,
                                       int scan_tbl, const std::vector<ExprP>& preds, size_t* kept_out,
                                       double* kernel_ms_out) {
  if (!jc.enabled) return std::nullopt;
  const bool identity = rel == nullptr;
  const size_t n = identity ? pq.tabs[scan_tbl]->nrows : rel->n;
  try {
    std::vector<JAggDesc> descs;
    for (auto& a : pq.aggs) {
      JAggDesc d;
      d.func = a->name == "sum" ? JAgg::Sum : a->name == "count" ? JAgg::Count : a->name == "avg" ? JAgg::Avg
               : a->name == "min" ? JAgg::Min : JAgg::Max;
      if (a->args[0]->kind != Expr::Star) d.arg = a->args[0];
      else if (d.func != JAgg::Count) throw JitUnsupported("* is only valid in count(*)");
      descs.push_back(d);
    }
    std::vector<int> rel_tbls = identity ? std::vector<int>{scan_tbl} : rel->tbls;
    JitPlan plan = gen_agg_plan(pq, rel_tbls, identity, preds, pq.sel.group_by, descs);
    double cms = 0;
    const char* origin = "";
    auto kernel = jit_kernel(plan, n, jc, &origin, &cms);
    auto masks = jit_upload_masks(plan);

    // group table sizing: estimated groups among the rows that survive the filters
    const size_t K = static_cast<size_t>(plan.K), A = static_cast<size_t>(plan.A);
    double est_n = static_cast<double>(std::max<size_t>(n, 1));
    if (identity && !preds.empty()) est_n = std::max(1.0, est_n * eng.sample_selectivity(pq, scan_tbl));
    size_t cap = 1;
    if (K > 0) {
      double prod = 1;
      for (auto& g : pq.sel.group_by) prod *= eng.key_ndv(pq, g, static_cast<size_t>(est_n));
      double est = std::min(est_n, prod);
      cap = 16;
      while (static_cast<double>(cap) < est * 2.0) cap <<= 1;
      if (jc.force_cap) cap = jc.force_cap;
    }

    GpuVec<uint32_t> state, overflow(1), kept(1);
    GpuVec<int64_t> skeys;
    struct Acc { GpuVec<int64_t> i; GpuVec<double> f; GpuVec<int64_t> cnt; };
    std::vector<Acc> accs(A);
    size_t acc_n = 1;
    double kernel_ms = 0;
    for (;;) {
      acc_n = K > 0 ? cap : 1;
      if (K > 0) {
        state.allocate(cap);
        WDB_CUDA_CHECK(cudaMemset(state.data(), 0, cap * 4));
        skeys.allocate(K * cap);
      }
      WDB_CUDA_CHECK(cudaMemset(overflow.data(), 0, 4));
      WDB_CUDA_CHECK(cudaMemset(kept.data(), 0, 4));
      for (size_t a = 0; a < A; ++a) {
        Acc& ac = accs[a];
        auto init_i = [&](GpuVec<int64_t>& v, int64_t x) {
          v.allocate(acc_n);
          ops::fill_k<int64_t><<<ops::grid_for(acc_n), 256>>>(v.data(), acc_n, x);
        };
        auto init_f = [&](GpuVec<double>& v, double x) {
          v.allocate(acc_n);
          ops::fill_k<double><<<ops::grid_for(acc_n), 256>>>(v.data(), acc_n, x);
        };
        const double inf = std::numeric_limits<double>::infinity();
        switch (plan.aggs[a].func) {
          case JAgg::Count: init_i(ac.i, 0); break;
          case JAgg::Sum: if (plan.acc_is_f[a]) init_f(ac.f, 0.0); else init_i(ac.i, 0); break;
          case JAgg::Avg: init_f(ac.f, 0.0); init_i(ac.cnt, 0); break;
          case JAgg::Min: if (plan.acc_is_f[a]) init_f(ac.f, inf); else init_i(ac.i, INT64_MAX); break;
          case JAgg::Max: if (plan.acc_is_f[a]) init_f(ac.f, -inf); else init_i(ac.i, INT64_MIN); break;
        }
      }
      ArgPack args;
      args.add<unsigned long long>(n);
      for (auto& tc : plan.cols) args.add<const void*>(jit_col_ptr(ctx, *pq.tabs[tc.first]->cols[tc.second]));
      if (!identity)
        for (size_t r = 0; r < rel->rid.size(); ++r) args.add<const void*>(rel->rid[r].data());
      for (auto& m : masks) args.add<const void*>(m.data());
      if (K > 0) {
        args.add<void*>(state.data());
        args.add<void*>(skeys.data());
        args.add<unsigned int>(static_cast<unsigned int>(cap));
        args.add<unsigned int>(static_cast<unsigned int>(cap - 1));
        args.add<void*>(overflow.data());
      }
      for (size_t a = 0; a < A; ++a) {
        Acc& ac = accs[a];
        if (plan.acc_is_f[a]) args.add<void*>(ac.f.data());
        else args.add<void*>(ac.i.data());
        if (plan.aggs[a].func == JAgg::Avg) args.add<void*>(ac.cnt.data());
      }
      args.add<void*>(kept.data());
      args.add<int>((K == 0 || cap <= 4096) ? 1 : 0);
      WDB_CUDA_CHECK(cudaDeviceSynchronize());
      Timer t;
      JitRuntime::instance().launch(*kernel, static_cast<unsigned>(ceil_div(std::max<size_t>(n, 1), 256)), 256, args);
      WDB_CUDA_CHECK(cudaDeviceSynchronize());
      kernel_ms += t.ms();
      if (K == 0 || overflow.to_host(1)[0] == 0) break;
      cap <<= 2;  // the group count was underestimated: rebuild a bigger table and rerun
      if (cap > (size_t(1) << 30)) throw Error("GPU aggregation: too many groups");
    }
    if (kept_out) *kept_out = kept.to_host(1)[0];
    if (kernel_ms_out) *kernel_ms_out = kernel_ms;

    // ---- compact occupied slots, gather keys and accumulators ----
    AggOutput out;
    GpuVec<uint32_t> slots;
    size_t G = 1;
    if (K > 0) {
      slots.allocate(cap);
      GpuVec<uint8_t> ws(select_workspace_bytes(cap));
      GpuVec<uint32_t> cnt(1);
      select_fused(ops::StateGen{state.data()}, cap, slots.data(), cnt.data(), ws.data());
      G = cnt.to_host(1)[0];
    }
    out.n = G;
    for (size_t j = 0; j < K; ++j) {
      const ValType& kt = plan.key_types[j];
      auto o = new_dvec(kt.kind == Value::F ? Value::F : (kt.kind == Value::S ? Value::S : Value::I), G);
      o->is_date = kt.is_date;
      o->dict = kt.dict;
      const int64_t* src = skeys.data() + j * cap;
      if (kt.kind == Value::F) ops::gather<double, double>(reinterpret_cast<const double*>(src), slots.data(), o->f.data(), G);
      else ops::gather<int64_t, int64_t>(src, slots.data(), o->i.data(), G);
      out.keys.push_back(o);
    }
    auto take_i = [&](const GpuVec<int64_t>& acc) {
      GpuVec<int64_t> r(G);
      if (K > 0) ops::gather<int64_t, int64_t>(acc.data(), slots.data(), r.data(), G);
      else WDB_CUDA_CHECK(cudaMemcpy(r.data(), acc.data(), 8, cudaMemcpyDeviceToDevice));
      return r;
    };
    auto take_f = [&](const GpuVec<double>& acc) {
      GpuVec<double> r(G);
      if (K > 0) ops::gather<double, double>(acc.data(), slots.data(), r.data(), G);
      else WDB_CUDA_CHECK(cudaMemcpy(r.data(), acc.data(), 8, cudaMemcpyDeviceToDevice));
      return r;
    };
    for (size_t a = 0; a < A; ++a) {
      Acc& ac = accs[a];
      const ValType& ot = plan.agg_out_types[a];
      std::shared_ptr<DValue> o;
      if (plan.aggs[a].func == JAgg::Avg) {
        auto s = take_f(ac.f);
        auto c = take_i(ac.cnt);
        o = new_dvec(Value::F, G);
        if (G) avg_k<<<ops::grid_for(G), 256>>>(s.data(), c.data(), o->f.data(), G);
      } else if (plan.acc_is_f[a]) {
        o = new_dvec(Value::F, 0);
        o->n = G;
        o->f = take_f(ac.f);
      } else {
        o = new_dvec(ot.kind == Value::S ? Value::S : Value::I, 0);
        o->n = G;
        o->i = take_i(ac.i);
        o->is_date = ot.is_date;
        o->dict = ot.dict;
      }
      out.aggs.push_back(o);
    }
    WDB_CUDA_CHECK(cudaGetLastError());
    jc.notes.push_back(std::string("jit fused ") + (identity ? "scan+filter+" : "") + "aggregate pipeline (" +
                       std::to_string(preds.size()) + " predicates, " + std::to_string(K) + " keys, " + std::to_string(A) +
                       " aggregates, " + jit_origin_note(origin, cms, kernel_ms) + ")");
    return out;
  } catch (const JitUnsupported&) {
    return std::nullopt;
  } catch (const JitError& e) {
    jc.notes.push_back(std::string("jit aggregate fell back to interpreter: ") + e.what());
    return std::nullopt;
  }
}

}  // namespace

// ------------------------------------------------------------------------------------------------
struct GpuBackend::Impl {
  ColumnCache cache;
  bool jit_enabled = true;
  size_t jit_min_rows = 200000;
  size_t jit_force_cap = 0;
  Impl() {
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    cache.set_budget(static_cast<size_t>(free_b * 0.6));
  }
};

GpuBackend::GpuBackend() : impl_(std::make_unique<Impl>()) {}
GpuBackend::~GpuBackend() = default;

GpuMemStats GpuBackend::mem_stats() const {
  GpuMemStats s;
  auto& a = GpuAllocator::instance();
  s.cache_bytes = impl_->cache.used();
  s.cache_budget = impl_->cache.budget();
  s.in_use = a.in_use();
  s.peak = a.peak();
  s.reserved = a.reserved();
  s.uploaded_total = impl_->cache.uploaded_bytes();
  return s;
}
void GpuBackend::drop_cache() { impl_->cache.clear(); }
void GpuBackend::set_jit(bool on) { impl_->jit_enabled = on; }
bool GpuBackend::jit_enabled() const { return impl_->jit_enabled && JitRuntime::instance().available(); }
void GpuBackend::set_jit_min_rows(size_t n) { impl_->jit_min_rows = n; }
size_t GpuBackend::jit_min_rows() const { return impl_->jit_min_rows; }
void GpuBackend::set_jit_force_capacity_for_test(size_t cap) { impl_->jit_force_cap = cap; }
JitStats GpuBackend::jit_stats() const { return JitRuntime::instance().stats(); }
bool GpuBackend::column_resident(const Column& c) const { return impl_->cache.resident(c); }
size_t GpuBackend::cache_budget_bytes() const { return impl_->cache.budget(); }
void GpuBackend::forget_columns(const std::vector<uint64_t>& uids) {
  for (uint64_t u : uids) impl_->cache.erase(u);
}
std::string GpuBackend::fingerprint() const {
  cudaDeviceProp p{};
  cudaGetDeviceProperties(&p, 0);
  return std::string(p.name) + "|sm" + std::to_string(p.major) + std::to_string(p.minor);
}

static __global__ void empty_kernel() {}

// PCIe bandwidth (pageable host memory, as our columns live in std::vector), kernel-launch latency, VRAM bandwidth.
bool GpuBackend::measure_hardware(HwProfile& hw) {
  const size_t bytes = size_t(64) << 20;
  std::vector<char> host(bytes, 1);
  GpuVec<uint8_t> dev(bytes), dev2(bytes);
  auto median = [](std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  };
  std::vector<double> h2d, d2h, d2d;
  for (int i = 0; i < 7; ++i) {
    Timer t;
    WDB_CUDA_CHECK(cudaMemcpy(dev.data(), host.data(), bytes, cudaMemcpyHostToDevice));
    h2d.push_back(t.ms());
    t.reset();
    WDB_CUDA_CHECK(cudaMemcpy(host.data(), dev.data(), bytes, cudaMemcpyDeviceToHost));
    d2h.push_back(t.ms());
    t.reset();
    WDB_CUDA_CHECK(cudaMemcpy(dev2.data(), dev.data(), bytes, cudaMemcpyDeviceToDevice));
    WDB_CUDA_CHECK(cudaDeviceSynchronize());
    d2d.push_back(t.ms());
  }
  hw.h2d_gbs = bytes / 1e6 / median(h2d);   // bytes / 1e6 = MB; MB per ms == GB/s
  hw.d2h_gbs = bytes / 1e6 / median(d2h);
  hw.gpu_membw_gbs = 2.0 * bytes / 1e6 / median(d2d);  // read + write
  for (int i = 0; i < 20; ++i) {  // warm up context / driver
    empty_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
  }
  Timer t;
  const int reps = 300;
  for (int i = 0; i < reps; ++i) {
    empty_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
  }
  hw.launch_us = t.ms() * 1000.0 / reps;
  hw.has_gpu = true;
  return true;
}

QueryResult GpuBackend::execute(Engine& eng, PlannedQuery& pq) {
  Timer total;
  Select& s = pq.sel;
  const TableVec& tabs = pq.tabs;
  const int T = static_cast<int>(tabs.size());
  QueryResult res;
  res.explain = false;
  auto trace = [&](const std::string& m) { res.trace.push_back(m); };
  impl_->cache.begin_query();
  GpuAllocator::instance().reset_peak();
  GpuCtx ctx{&impl_->cache};

  JitCtx jc;
  jc.enabled = impl_->jit_enabled && JitRuntime::instance().available();
  jc.min_rows = impl_->jit_min_rows;
  jc.force_cap = impl_->jit_force_cap;
  // A single-table aggregate query is compiled as ONE kernel (scan + filter + group + aggregate), so its scan stage
  // is skipped here. If that pipeline turns out to be unsupported the scan is performed in the aggregate branch.
  const bool fused_single = jc.enabled && T == 1 && pq.agg_mode;

  // ---- scans with pushed-down filters ----
  std::vector<DRel> rels(T);
  for (int k = 0; k < T; ++k) {
    if (fused_single) continue;
    Timer t;
    const double up0 = ctx.upload_ms;  // column uploads triggered inside a stage are accounted separately
    const double c0 = jc.compile_ms;   // so are one-off JIT compiles
    Sel fsel;
    const bool fused_scan = jit_filter(ctx, pq, k, jc, fsel, nullptr);
    if (fused_scan) {
      DRel r;
      r.tbls = {k};
      r.n = fsel.k;
      r.rid.push_back(std::move(fsel.idx));
      rels[k] = std::move(r);
    } else {
      DRel r;
      r.tbls = {k};
      r.n = tabs[k]->nrows;
      r.rid.emplace_back(r.n);
      if (r.n) ops::iota_k<<<ops::grid_for(r.n), 256>>>(r.rid[0].data(), r.n);
      DRelScope sc(tabs, std::move(r), &ctx);
      for (auto& c : pq.push[k]) {
        DVP b = d_eval(*c, sc);
        Sel sel = select_true(*b);
        sc.set_rows(sel.idx, sel.k);
      }
      rels[k] = std::move(sc.rel);
    }
    const double ms = sync_ms(t) - (ctx.upload_ms - up0) - (jc.compile_ms - c0);
    trace(std::string(fused_scan ? "gpu scan (jit-fused) " : "gpu scan ") + tabs[k]->name + ": " +
          std::to_string(tabs[k]->nrows) + " -> " + std::to_string(rels[k].n) + " rows (" +
          std::to_string(pq.push[k].size()) + " filters, " + fmt_ms(ms) + ")");
    res.stages.push_back(make_stat(StageOp::Scan, tabs[k]->name, static_cast<double>(tabs[k]->nrows), 0,
                                   static_cast<double>(rels[k].n), distinct_columns(pq.push[k]),
                                   expr_nodes(pq.push[k]), ms));
  }

  // ---- joins ----
  DRel cur;
  {
    std::vector<bool> joined(T, false);
    std::vector<size_t> rel_n(T);
    for (int k = 0; k < T; ++k) rel_n[k] = rels[k].n;
    int start = eng.pick_start(pq, rel_n);
    cur = std::move(rels[start]);
    joined[start] = true;
    for (int remaining = T - 1; remaining > 0; --remaining) {
      int best = eng.pick_next(pq, joined, cur.n, rel_n);
      if (best < 0) throw Error("cross joins are not supported: add an equality join condition between all tables");
      Timer t;
      const double up0 = ctx.upload_ms;
      std::vector<ExprP> kc, kb;
      for (auto& e : pq.edges) {
        if (joined[e.ta] && e.tb == best) { kc.push_back(e.ea); kb.push_back(e.eb); }
        else if (joined[e.tb] && e.ta == best) { kc.push_back(e.eb); kb.push_back(e.ea); }
      }
      DRelScope cs(tabs, std::move(cur), &ctx), bs(tabs, std::move(rels[best]), &ctx);
      std::vector<DVP> lk, rk;
      for (auto& e : kc) lk.push_back(materialize(d_eval(*e, cs), cs.n));
      for (auto& e : kb) rk.push_back(materialize(d_eval(*e, bs), bs.n));
      for (size_t j = 0; j < lk.size(); ++j) {
        bool ls = lk[j]->kind == Value::S, rs2 = rk[j]->kind == Value::S;
        if (ls != rs2) throw Error("cannot join a VARCHAR column with a non-VARCHAR column");
        if (lk[j]->kind != Value::I && !ls) throw Error("join keys must be integer, date or VARCHAR columns");
        if (ls && lk[j]->dict.get() != rk[j]->dict.get()) rk[j] = d_remap_codes(*rk[j], *lk[j]->dict);
      }
      size_t ln = cs.rel.n, rn = bs.rel.n;
      JoinPairs jp = d_hash_join(lk, ln, rk, rn);
      DRel a = d_compose(cs.rel, jp.l, jp.n), b = d_compose(bs.rel, jp.r, jp.n);
      DRel o;
      o.n = jp.n;
      o.tbls = a.tbls;
      o.rid = std::move(a.rid);
      for (size_t k = 0; k < b.tbls.size(); ++k) {
        o.tbls.push_back(b.tbls[k]);
        o.rid.push_back(std::move(b.rid[k]));
      }
      cur = std::move(o);
      joined[best] = true;
      const double jms = sync_ms(t) - (ctx.upload_ms - up0);
      trace("gpu hash join " + tabs[best]->name + " (" + std::to_string(kc.size()) + " keys): " + std::to_string(ln) +
            " x " + std::to_string(rn) + " -> " + std::to_string(cur.n) + " rows (" + fmt_ms(jms) + ")");
      res.stages.push_back(make_stat(StageOp::Join, tabs[best]->name, static_cast<double>(ln), static_cast<double>(rn),
                                     static_cast<double>(cur.n), static_cast<int>(kc.size()),
                                     static_cast<int>(cur.tbls.size()), jms));
    }
  }

  std::unique_ptr<DScope> scope;
  {
    auto rs = std::make_unique<DRelScope>(tabs, std::move(cur), &ctx);
    if (!fused_single)  // (single-table fused pipelines evaluate the residual predicates inside the kernel)
      for (auto& c : pq.residual) {
        DVP b = d_eval(*c, *rs);
        Sel sel = select_true(*b);
        rs->set_rows(sel.idx, sel.k);
      }
    if (!pq.residual.empty() && !fused_single) trace("gpu residual filter -> " + std::to_string(rs->n) + " rows");

    if (pq.agg_mode) {
      Timer t;
      const double up0 = ctx.upload_ms, c0 = jc.compile_ms;
      std::optional<AggOutput> fused;
      size_t kept = 0;
      double kernel_ms = 0;
      std::vector<ExprP> single_preds;
      if (fused_single) {
        single_preds = pq.push[0];
        single_preds.insert(single_preds.end(), pq.residual.begin(), pq.residual.end());
        fused = jit_aggregate(eng, pq, ctx, jc, nullptr, 0, single_preds, &kept, &kernel_ms);
        if (!fused) {  // unsupported / failed: do the scan now and continue with the interpreted operators
          DRel r;
          r.tbls = {0};
          r.n = tabs[0]->nrows;
          r.rid.emplace_back(r.n);
          if (r.n) ops::iota_k<<<ops::grid_for(r.n), 256>>>(r.rid[0].data(), r.n);
          rs = std::make_unique<DRelScope>(tabs, std::move(r), &ctx);
          for (auto& c : single_preds) {
            DVP b = d_eval(*c, *rs);
            Sel sel = select_true(*b);
            rs->set_rows(sel.idx, sel.k);
          }
        }
      } else if (jc.enabled) {
        fused = jit_aggregate(eng, pq, ctx, jc, &rs->rel, -1, {}, &kept, &kernel_ms);
      }

      AggOutput ao;
      size_t nin = fused_single && fused ? kept : rs->n;
      if (fused) {
        ao = std::move(*fused);
      } else {
        std::vector<DVP> keyv;
        for (auto& g : s.group_by) keyv.push_back(d_eval(*g, *rs));
        std::vector<AggInput> ins;
        for (auto& a : pq.aggs) {
          AggInput in;
          if (a->name == "sum") in.func = SUM;
          else if (a->name == "count") in.func = COUNT;
          else if (a->name == "avg") in.func = AVG;
          else if (a->name == "min") in.func = MIN;
          else in.func = MAX;
          if (a->args[0]->kind != Expr::Star) in.arg = d_eval(*a->args[0], *rs);
          else if (in.func != COUNT) throw Error("* is only valid in count(*)");
          ins.push_back(in);
        }
        ao = d_hash_aggregate(eng, pq, keyv, ins, nin);
      }
      auto as = std::make_unique<DAggScope>();
      as->n = ao.n;
      for (size_t j = 0; j < s.group_by.size(); ++j) as->m[s.group_by[j]->str()] = ao.keys[j];
      for (size_t a = 0; a < pq.aggs.size(); ++a) as->m[pq.aggs[a]->str()] = ao.aggs[a];
      const double ams = sync_ms(t) - (ctx.upload_ms - up0) - (jc.compile_ms - c0);
      trace(std::string(fused ? "gpu hash aggregate (jit-fused" + std::string(fused_single ? " with scan+filter" : "") + "): "
                              : "gpu hash aggregate: ") +
            std::to_string(nin) + " rows -> " + std::to_string(ao.n) + " groups (" + std::to_string(s.group_by.size()) +
            " keys, " + std::to_string(pq.aggs.size()) + " aggs, " + fmt_ms(ams) + ")");
      StageStat agg_stat = make_stat(StageOp::Aggregate, "", static_cast<double>(nin), 0, static_cast<double>(ao.n),
                                     static_cast<int>(s.group_by.size()), static_cast<int>(pq.aggs.size()), ams,
                                     agg_expr_nodes(pq));
      if (fused_single && fused) {
        // One kernel did the scan and the aggregation. Attribute its time to the two stages in proportion to what
        // the cost model predicts for each, so both models keep seeing consistent (fused) magnitudes.
        StageStat scan_stat = make_stat(StageOp::Scan, tabs[0]->name, static_cast<double>(tabs[0]->nrows), 0,
                                        static_cast<double>(kept), distinct_columns(pq.push[0]),
                                        expr_nodes(pq.push[0]), 0);
        double ps = std::max(eng.cost.predict(scan_stat), 1e-6), pa = std::max(eng.cost.predict(agg_stat), 1e-6);
        scan_stat.ms = ams * ps / (ps + pa);
        agg_stat.ms = ams * pa / (ps + pa);
        res.stages.push_back(scan_stat);
      }
      res.stages.push_back(agg_stat);
      scope = std::move(as);
      if (s.having) {
        DVP b = d_eval(*s.having, *scope);
        Sel sel = select_true(*b);
        scope = scope->gather(sel.idx, sel.k);
      }
    } else {
      scope = std::move(rs);
    }
  }

  // ---- ORDER BY / LIMIT ----
  if (!s.order_by.empty() || s.limit >= 0) {
    Timer t;
    const double up0 = ctx.upload_ms;
    GpuVec<uint32_t> perm;
    size_t k = scope->n;
    if (!s.order_by.empty()) {
      std::vector<DVP> keys;
      std::vector<bool> desc;
      for (auto& o : s.order_by) {
        keys.push_back(d_eval(*o.e, *scope));
        desc.push_back(o.desc);
      }
      perm = d_sort_perm(keys, desc, scope->n);
      if (s.limit >= 0) k = std::min<size_t>(k, static_cast<size_t>(s.limit));
      const double sms = sync_ms(t) - (ctx.upload_ms - up0);
      trace("gpu radix sort " + std::to_string(scope->n) + " rows by " + std::to_string(keys.size()) + " keys" +
            (s.limit >= 0 ? " (limit " + std::to_string(s.limit) + ")" : "") + " (" + fmt_ms(sms) + ")");
      res.stages.push_back(make_stat(StageOp::Sort, "", static_cast<double>(scope->n), 0,
                                     static_cast<double>(k), static_cast<int>(keys.size()), 0, sms));
    } else {
      k = std::min<size_t>(k, static_cast<size_t>(s.limit));
      perm.allocate(k);
      if (k) ops::iota_k<<<ops::grid_for(k), 256>>>(perm.data(), k);
    }
    scope = scope->gather(perm, k);
  }

  // ---- projection + download ----
  res.nrows = scope->n;
  Timer td;
  const double up0p = ctx.upload_ms;
  for (auto& it : s.items) {
    DVP v = d_eval(*it.e, *scope);
    res.cols.push_back(download(v, res.nrows));
    std::string name = !it.alias.empty() ? it.alias : (it.e->kind == Expr::Col ? it.e->name : it.text);
    res.names.push_back(name);
  }
  WDB_CUDA_CHECK(cudaDeviceSynchronize());
  const double pms = td.ms() - (ctx.upload_ms - up0p);
  trace("gpu projection + result download: " + std::to_string(res.nrows) + " rows (" + fmt_ms(pms) + ")");
  res.stages.push_back(make_stat(StageOp::Project, "", static_cast<double>(res.nrows), 0,
                                 static_cast<double>(res.nrows), static_cast<int>(s.items.size()), 0, pms));
  if (ctx.upload_bytes) {
    trace("gpu column upload (cold): " + std::to_string(ctx.upload_bytes >> 20) + " MB in " + fmt_ms(ctx.upload_ms));
    res.stages.push_back(make_stat(StageOp::Upload, "columns", static_cast<double>(ctx.upload_bytes) / 1e6, 0, 0, 0, 0,
                                   ctx.upload_ms));
  }
  auto& al = GpuAllocator::instance();
  trace("gpu memory: peak " + std::to_string(al.peak() >> 20) + " MB, column cache " +
        std::to_string(impl_->cache.used() >> 20) + " MB");
  for (auto& n : jc.notes) trace(n);
  if (jc.compile_ms > 0) trace("gpu jit: " + fmt_ms(jc.compile_ms) + " spent compiling / loading kernels (excluded from stage times)");
  res.ms = total.ms();
  trace("total: " + std::to_string(res.nrows) + " rows in " + fmt_ms(res.ms));
  return res;
}

std::shared_ptr<GpuBackend> make_gpu_backend() { return std::make_shared<GpuBackend>(); }

}  // namespace warpdb::gpu
