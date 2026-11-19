#include "gpu/jit/codegen.h"

#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <sstream>

namespace warpdb::gpu {
namespace {

// Same numbering as the interpreted GPU path (gpu_backend.cu) so that dictionary masks are shared via the cache.
enum class CmpOp { Eq = 0, Ne, Lt, Le, Gt, Ge };

CmpOp flip_cmp(CmpOp o) {
  switch (o) {
    case CmpOp::Lt: return CmpOp::Gt;
    case CmpOp::Le: return CmpOp::Ge;
    case CmpOp::Gt: return CmpOp::Lt;
    case CmpOp::Ge: return CmpOp::Le;
    default: return o;
  }
}
const char* cmp_symbol(CmpOp o) {
  switch (o) {
    case CmpOp::Eq: return "==";
    case CmpOp::Ne: return "!=";
    case CmpOp::Lt: return "<";
    case CmpOp::Le: return "<=";
    case CmpOp::Gt: return ">";
    case CmpOp::Ge: return ">=";
  }
  return "==";
}
bool cmp_of(const std::string& s, CmpOp& out) {
  if (s == "=") out = CmpOp::Eq;
  else if (s == "<>") out = CmpOp::Ne;
  else if (s == "<") out = CmpOp::Lt;
  else if (s == "<=") out = CmpOp::Le;
  else if (s == ">") out = CmpOp::Gt;
  else if (s == ">=") out = CmpOp::Ge;
  else return false;
  return true;
}

struct ConstScope : Scope {
  ConstScope() { n = 1; }
  VP intercept(const Expr&) override { return nullptr; }
  std::unique_ptr<Scope> gather(const std::vector<uint32_t>&) const override { return nullptr; }
};

bool has_col(const Expr& e) {
  if (e.kind == Expr::Col) return true;
  for (auto& a : e.args)
    if (has_col(*a)) return true;
  return false;
}

std::string int_lit(int64_t v) {
  if (v == INT64_MIN) return "(-9223372036854775807LL-1)";
  return "(" + std::to_string(v) + "LL)";
}
std::string dbl_lit(double x) {
  if (!std::isfinite(x)) throw JitUnsupported("non-finite constant");
  char b[48];
  std::snprintf(b, sizeof b, "%.17g", x);
  std::string s = b;
  if (s.find_first_of(".eE") == std::string::npos) s += ".0";
  return "(" + s + ")";
}

struct CE {  // a compiled expression: C++ text + static type
  std::string code;
  ValType t;
  std::string str;  // string literal value when t.kind == L
};

class Emitter {
 public:
  explicit Emitter(const PlannedQuery& pq) : pq_(pq) {}

  std::vector<std::pair<int, int>> cols;
  std::vector<MaskSpec> masks;

  int col_param(int tbl, int col) {
    auto key = std::make_pair(tbl, col);
    for (size_t i = 0; i < cols.size(); ++i)
      if (cols[i] == key) return static_cast<int>(i);
    cols.push_back(key);
    return static_cast<int>(cols.size() - 1);
  }
  const Column& column(int idx) const { return *pq_.tabs[cols[idx].first]->cols[cols[idx].second]; }

  // Statements that load every not-yet-loaded column referenced by `e` (they run inside an `if (keep)` block).
  std::string loads_for(const Expr& e) {
    std::string out;
    collect_loads(e, out);
    return out;
  }

  std::string decls() const {
    std::string out;
    for (size_t i = 0; i < cols.size(); ++i)
      out += std::string("  ") + (column(static_cast<int>(i)).type == Type::FLOAT64 ? "double" : "long long") + " c" +
             std::to_string(i) + ";\n";
    return out;
  }
  std::string param_decls() const {
    std::string out;
    for (size_t i = 0; i < cols.size(); ++i) {
      const char* t = "const int*";
      switch (column(static_cast<int>(i)).type) {
        case Type::INT64: t = "const long long*"; break;
        case Type::FLOAT64: t = "const double*"; break;
        default: break;
      }
      out += std::string("    ") + t + " P" + std::to_string(i) + ",\n";
    }
    return out;
  }

  CE gen(const Expr& e) {
    if (!has_col(e)) return const_ce(e);
    switch (e.kind) {
      case Expr::Col: return gen_col(e);
      case Expr::Bin: return gen_bin(e);
      case Expr::Not: {
        CE a = gen(*e.args[0]);
        if (a.t.kind != Value::B) throw JitUnsupported("NOT of non-boolean");
        return boolean("(!" + a.code + ")");
      }
      case Expr::Neg: {
        CE a = gen(*e.args[0]);
        if (a.t.kind != Value::I && a.t.kind != Value::F) throw JitUnsupported("negation of non-number");
        CE r;
        r.code = "(-" + a.code + ")";
        r.t.kind = a.t.kind;
        return r;
      }
      case Expr::Between: {
        CE x = gen(*e.args[0]), lo = gen(*e.args[1]), hi = gen(*e.args[2]);
        CE r = boolean("(" + cmp(CmpOp::Ge, x, lo).code + " && " + cmp(CmpOp::Le, x, hi).code + ")");
        if (e.negated) r.code = "(!" + r.code + ")";
        return r;
      }
      case Expr::In: return gen_in(e);
      case Expr::Like: return gen_like(e);
      case Expr::Case: return gen_case(e);
      case Expr::Func: return gen_func(e);
      default: throw JitUnsupported("unsupported expression kind in JIT");
    }
  }

 private:
  const PlannedQuery& pq_;
  std::set<int> loaded_;

  void collect_loads(const Expr& e, std::string& out) {
    if (e.kind == Expr::Col) {
      int idx = col_param(e.tbl, e.col);
      if (loaded_.insert(idx).second) {
        const bool f = column(idx).type == Type::FLOAT64;
        out += "    c" + std::to_string(idx) + " = " + (f ? "" : "(long long)") + "P" + std::to_string(idx) + "[t" +
               std::to_string(e.tbl) + "];\n";
      }
    }
    for (auto& a : e.args) collect_loads(*a, out);
  }

  static CE boolean(std::string code) {
    CE r;
    r.code = std::move(code);
    r.t.kind = Value::B;
    return r;
  }

  CE const_ce(const Expr& e) {
    ConstScope sc;
    VP v;
    try {
      v = eval(e, sc);
    } catch (const std::exception& ex) {
      throw JitUnsupported(std::string("cannot fold constant: ") + ex.what());
    }
    CE r;
    switch (v->kind) {
      case Value::I: r.code = int_lit(v->i[0]); r.t.kind = Value::I; r.t.is_date = v->is_date; break;
      case Value::F: r.code = dbl_lit(v->f[0]); r.t.kind = Value::F; break;
      case Value::B: r.code = v->b[0] ? "true" : "false"; r.t.kind = Value::B; break;
      case Value::L: r.t.kind = Value::L; r.str = v->lit; break;
      default: throw JitUnsupported("unsupported constant");
    }
    return r;
  }

  CE gen_col(const Expr& e) {
    int idx = col_param(e.tbl, e.col);
    const auto& colp = pq_.tabs[e.tbl]->cols[e.col];
    CE r;
    r.code = "c" + std::to_string(idx);
    switch (colp->type) {
      case Type::INT32:
      case Type::INT64: r.t.kind = Value::I; break;
      case Type::DATE: r.t.kind = Value::I; r.t.is_date = true; break;
      case Type::FLOAT64: r.t.kind = Value::F; break;
      case Type::VARCHAR: r.t.kind = Value::S; r.t.dict = colp; break;
    }
    return r;
  }

  std::string mask_ref(const std::shared_ptr<const Column>& dc, const std::string& key,
                       std::function<void(std::vector<uint8_t>&)> fill) {
    for (size_t i = 0; i < masks.size(); ++i)
      if (masks[i].dictcol->uid == dc->uid && masks[i].key == key) return "M" + std::to_string(i);
    masks.push_back({dc, key, std::move(fill)});
    return "M" + std::to_string(masks.size() - 1);
  }

  CE str_cmp(CmpOp op, const CE& s, const std::string& lit) {
    auto dc = s.t.dict;
    std::string m = mask_ref(dc, "cmp" + std::to_string(static_cast<int>(op)) + "|" + lit,
                             [dc, op, lit](std::vector<uint8_t>& mask) {
                               const auto& d = dc->dict;
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
    return boolean("(" + m + "[" + s.code + "] != 0)");
  }

  CE cmp(CmpOp op, const CE& a, const CE& b) {
    if (a.t.kind == Value::S && b.t.kind == Value::L) return str_cmp(op, a, b.str);
    if (a.t.kind == Value::L && b.t.kind == Value::S) return str_cmp(flip_cmp(op), b, a.str);
    auto ok = [](const ValType& t) { return t.kind == Value::I || t.kind == Value::F || t.kind == Value::S; };
    if (!ok(a.t) || !ok(b.t)) throw JitUnsupported("cannot compare these operand types");
    if (a.t.kind == Value::S && b.t.kind == Value::S && a.t.dict.get() != b.t.dict.get())
      throw JitUnsupported("comparing VARCHAR columns with different dictionaries");
    return boolean("(" + a.code + " " + cmp_symbol(op) + " " + b.code + ")");
  }

  CE gen_bin(const Expr& e) {
    const std::string& op = e.name;
    CE a = gen(*e.args[0]), b = gen(*e.args[1]);
    if (op == "and" || op == "or") {
      if (a.t.kind != Value::B || b.t.kind != Value::B) throw JitUnsupported("AND/OR need booleans");
      return boolean("(" + a.code + (op == "and" ? " && " : " || ") + b.code + ")");
    }
    CmpOp c;
    if (cmp_of(op, c)) return cmp(c, a, b);
    auto num = [](const ValType& t) { return t.kind == Value::I || t.kind == Value::F; };
    if (!num(a.t) || !num(b.t)) throw JitUnsupported("arithmetic on non-numbers");
    const bool ints = a.t.kind == Value::I && b.t.kind == Value::I && op != "/";
    CE r;
    r.t.kind = ints ? Value::I : Value::F;
    if (ints) {
      if (op == "%") r.code = "wdb_mod(" + a.code + ", " + b.code + ")";
      else if (op == "+" || op == "-" || op == "*") r.code = "(" + a.code + " " + op + " " + b.code + ")";
      else throw JitUnsupported("unknown integer operator " + op);
      if (op == "+" || op == "-") r.t.is_date = a.t.is_date != b.t.is_date && !(op == "-" && b.t.is_date);
    } else {
      std::string x = "((double)" + a.code + ")", y = "((double)" + b.code + ")";
      if (op == "%") r.code = "fmod(" + x + ", " + y + ")";
      else if (op == "+" || op == "-" || op == "*" || op == "/") r.code = "(" + x + " " + op + " " + y + ")";
      else throw JitUnsupported("unknown operator " + op);
    }
    return r;
  }

  CE gen_in(const Expr& e) {
    CE x = gen(*e.args[0]);
    CE r;
    if (x.t.kind == Value::S) {
      std::set<std::string> set;
      for (size_t k = 1; k < e.args.size(); ++k) {
        CE it = gen(*e.args[k]);
        if (it.t.kind != Value::L) throw JitUnsupported("IN list for VARCHAR must be string literals");
        set.insert(it.str);
      }
      std::string key = "in";
      for (auto& v : set) key += "|" + v;
      auto dc = x.t.dict;
      std::string m = mask_ref(dc, key, [dc, set](std::vector<uint8_t>& mask) {
        const auto& d = dc->dict;
        for (size_t k = 0; k < d.size(); ++k) mask[k] = set.count(d[k]) != 0;
      });
      r = boolean("(" + m + "[" + x.code + "] != 0)");
    } else {
      std::string code;
      for (size_t k = 1; k < e.args.size(); ++k) {
        std::string c = cmp(CmpOp::Eq, x, gen(*e.args[k])).code;
        code += code.empty() ? c : " || " + c;
      }
      if (code.empty()) throw JitUnsupported("empty IN list");
      r = boolean("(" + code + ")");
    }
    if (e.negated) r.code = "(!" + r.code + ")";
    return r;
  }

  CE gen_like(const Expr& e) {
    CE x = gen(*e.args[0]), p = gen(*e.args[1]);
    if (x.t.kind != Value::S || p.t.kind != Value::L) throw JitUnsupported("LIKE needs VARCHAR column and literal");
    auto dc = x.t.dict;
    std::string pat = p.str;
    std::string m = mask_ref(dc, "like|" + pat, [dc, pat](std::vector<uint8_t>& mask) {
      const auto& d = dc->dict;
      for (size_t k = 0; k < d.size(); ++k) mask[k] = like_match(d[k], pat);
    });
    CE r = boolean("(" + m + "[" + x.code + "] != 0)");
    if (e.negated) r.code = "(!" + r.code + ")";
    return r;
  }

  CE gen_case(const Expr& e) {
    size_t nw = (e.args.size() - (e.has_else ? 1 : 0)) / 2;
    std::vector<CE> conds, thens;
    bool anyf = false;
    auto num = [](const ValType& t) { return t.kind == Value::I || t.kind == Value::F; };
    for (size_t k = 0; k < nw; ++k) {
      conds.push_back(gen(*e.args[2 * k]));
      thens.push_back(gen(*e.args[2 * k + 1]));
      if (conds.back().t.kind != Value::B) throw JitUnsupported("CASE condition must be boolean");
      if (!num(thens.back().t)) throw JitUnsupported("CASE branches must be numeric");
      anyf |= thens.back().t.kind == Value::F;
    }
    CE els;
    if (e.has_else) {
      els = gen(*e.args.back());
      if (!num(els.t)) throw JitUnsupported("CASE branches must be numeric");
      anyf |= els.t.kind == Value::F;
    } else {
      els.code = "(0LL)";
      els.t.kind = Value::I;
    }
    const std::string ty = anyf ? "double" : "long long";
    std::string code = "((" + ty + ")" + els.code + ")";
    for (size_t k = nw; k-- > 0;)
      code = "(" + conds[k].code + " ? ((" + ty + ")" + thens[k].code + ") : " + code + ")";
    CE r;
    r.code = code;
    r.t.kind = anyf ? Value::F : Value::I;
    return r;
  }

  CE gen_func(const Expr& e) {
    CE a = gen(*e.args[0]);
    CE r;
    if (e.name == "abs") {
      if (a.t.kind == Value::I) { r.code = "((" + a.code + ") < 0 ? -(" + a.code + ") : (" + a.code + "))"; r.t.kind = Value::I; }
      else if (a.t.kind == Value::F) { r.code = "fabs(" + a.code + ")"; r.t.kind = Value::F; }
      else throw JitUnsupported("abs of non-number");
      return r;
    }
    if (a.t.kind != Value::I || !a.t.is_date) throw JitUnsupported(e.name + "() needs a DATE");
    int which = e.name == "year" ? 0 : e.name == "month" ? 1 : 2;
    r.code = "wdb_date_part(" + a.code + ", " + std::to_string(which) + ")";
    r.t.kind = Value::I;
    return r;
  }
};

std::string row_id_setup(const std::vector<int>& tbls, bool identity) {
  std::string out;
  if (identity) {
    out += "  unsigned int t" + std::to_string(tbls[0]) + " = (unsigned int)i;\n";
  } else {
    for (size_t r = 0; r < tbls.size(); ++r)
      out += "  unsigned int t" + std::to_string(tbls[r]) + " = 0;\n  if (valid) t" + std::to_string(tbls[r]) + " = R" +
             std::to_string(r) + "[i];\n";
  }
  return out;
}

std::string mask_param_decls(const std::vector<MaskSpec>& masks) {
  std::string out;
  for (size_t m = 0; m < masks.size(); ++m) out += "    const unsigned char* M" + std::to_string(m) + ",\n";
  return out;
}

const char* kHeader =
    "#include \"gpu/jit/jit_device.cuh\"\n"
    "using namespace warpdb::gpu;\n\n";

}  // namespace

// ---------------------------------------------------------------------------------------------------------------
JitPlan gen_filter_plan(const PlannedQuery& pq, int tbl, const std::vector<ExprP>& preds) {
  Emitter E(pq);
  std::ostringstream body;
  for (auto& p : preds) {
    std::string loads = E.loads_for(*p);
    CE c = E.gen(*p);
    if (c.t.kind != Value::B) throw JitUnsupported("predicate is not boolean");
    body << "  if (keep) {\n" << loads << "    keep = " << c.code << ";\n  }\n";
  }
  JitPlan plan;
  plan.cols = E.cols;
  plan.masks = E.masks;

  std::ostringstream src;
  src << kHeader;
  src << "struct FGen {\n  unsigned long long n;\n";
  for (size_t i = 0; i < plan.cols.size(); ++i) {
    const char* t = "const int*";
    switch (pq.tabs[plan.cols[i].first]->cols[plan.cols[i].second]->type) {
      case Type::INT64: t = "const long long*"; break;
      case Type::FLOAT64: t = "const double*"; break;
      default: break;
    }
    src << "  " << t << " P" << i << ";\n";
  }
  for (size_t m = 0; m < plan.masks.size(); ++m) src << "  const unsigned char* M" << m << ";\n";
  src << "  __device__ bool eval(size_t i, unsigned int& v) const {\n    v = (unsigned int)i;\n";
  src << "    const bool valid = true;\n    (void)valid;\n";
  src << "    unsigned int t" << tbl << " = (unsigned int)i;\n    bool keep = true;\n";
  src << E.decls() << body.str() << "    return keep;\n  }\n};\n\n";
  // body above uses 2-space indentation; fine for NVRTC
  src << "extern \"C\" __global__ void __launch_bounds__(256) k_main(\n    unsigned long long n,\n" << E.param_decls()
      << mask_param_decls(plan.masks)
      << "    unsigned int* out, unsigned long long* status, unsigned int* counter, unsigned int last_tile,\n"
         "    unsigned int* d_count) {\n  FGen g{n";
  for (size_t i = 0; i < plan.cols.size(); ++i) src << ", P" << i;
  for (size_t m = 0; m < plan.masks.size(); ++m) src << ", M" << m;
  src << "};\n  warpdb::gpu::detail::select_tile<256, 16, FGen, unsigned int>(g, (size_t)n, out, status, counter, "
         "last_tile, d_count);\n}\n";
  plan.source = src.str();
  return plan;
}

JitPlan gen_agg_plan(const PlannedQuery& pq, const std::vector<int>& rel_tbls, bool identity,
                     const std::vector<ExprP>& preds, const std::vector<ExprP>& keys,
                     const std::vector<JAggDesc>& aggs_in) {
  std::vector<JAggDesc> aggs = aggs_in;
  for (auto& d : aggs)
    if (d.func == JAgg::Count) d.arg = nullptr;  // no NULLs: count(x) == count(*), so x is never loaded
  if (keys.size() > 8) throw JitUnsupported("too many group keys");
  if (identity && rel_tbls.size() != 1) throw JitUnsupported("identity pipelines read exactly one table");
  Emitter E(pq);
  JitPlan plan;
  plan.K = static_cast<int>(keys.size());
  plan.A = static_cast<int>(aggs.size());
  plan.aggs = aggs;
  if (!identity) plan.rid_tables = rel_tbls;

  std::ostringstream body;
  for (auto& p : preds) {
    std::string loads = E.loads_for(*p);
    CE c = E.gen(*p);
    if (c.t.kind != Value::B) throw JitUnsupported("predicate is not boolean");
    body << "  if (keep) {\n" << loads << "    keep = " << c.code << ";\n  }\n";
  }

  // keys and aggregate arguments (computed only for rows that pass the filters)
  std::string loads;
  for (auto& k : keys) loads += E.loads_for(*k);
  for (auto& a : aggs)
    if (a.arg) loads += E.loads_for(*a.arg);
  std::vector<std::string> key_code;
  for (auto& k : keys) {
    CE c = E.gen(*k);
    ValType t = c.t;
    std::string code;
    switch (t.kind) {
      case Value::I:
      case Value::S: code = "(long long)" + c.code; break;
      case Value::F: code = "__double_as_longlong(" + c.code + ")"; break;
      case Value::B: code = "(long long)" + c.code; t.kind = Value::I; break;  // like the CPU: boolean keys become ints
      default: throw JitUnsupported("cannot group by this expression");
    }
    key_code.push_back(code);
    plan.key_types.push_back(t);
  }
  struct AggGen { std::string arg_code; bool acc_f = false; ValType out; };
  std::vector<AggGen> ag(aggs.size());
  for (size_t a = 0; a < aggs.size(); ++a) {
    const JAggDesc& d = aggs[a];
    AggGen& g = ag[a];
    ValType at;
    if (d.arg) {
      CE c = E.gen(*d.arg);
      at = c.t;
      g.arg_code = c.code;
      if (at.kind != Value::I && at.kind != Value::F && at.kind != Value::S) throw JitUnsupported("aggregate argument type");
    }
    switch (d.func) {
      case JAgg::Count: g.out.kind = Value::I; g.acc_f = false; break;
      case JAgg::Sum:
        if (at.kind == Value::S) throw JitUnsupported("SUM over VARCHAR");
        g.acc_f = at.kind == Value::F;
        g.out.kind = g.acc_f ? Value::F : Value::I;
        break;
      case JAgg::Avg:
        if (at.kind == Value::S) throw JitUnsupported("AVG over VARCHAR");
        g.acc_f = true;
        g.out.kind = Value::F;
        break;
      case JAgg::Min:
      case JAgg::Max:
        g.acc_f = at.kind == Value::F;
        g.out = at;
        break;
    }
    plan.agg_out_types.push_back(g.out);
    plan.acc_is_f.push_back(g.acc_f);
  }

  const int K = plan.K;
  body << "  unsigned int slot = EMPTY_SLOT;\n";
  for (size_t a = 0; a < aggs.size(); ++a) {
    if (!aggs[a].arg) continue;
    body << "  " << (ag[a].acc_f ? "double" : "long long") << " v" << a << " = 0;\n";
  }
  if (K > 0) {
    body << "  long long key[" << K << "];\n";
    for (int j = 0; j < K; ++j) body << "  key[" << j << "] = 0;\n";
  }
  body << "  if (keep) {\n" << loads;
  if (K > 0) {
    for (int j = 0; j < K; ++j) body << "    key[" << j << "] = " << key_code[j] << ";\n";
  } else {
    body << "    slot = 0u;\n";
  }
  for (size_t a = 0; a < aggs.size(); ++a)
    if (aggs[a].arg) body << "    v" << a << " = (" << (ag[a].acc_f ? "double" : "long long") << ")(" << ag[a].arg_code << ");\n";
  body << "  }\n";
  if (K > 0) {
    // Low-cardinality tables: lanes with identical key tuples (compared exactly, word by word) elect ONE lane to
    // probe the global group table and share the slot, so a warp issues one probe per distinct key, not 32.
    body << "  if (warp_agg) {\n"
            "    unsigned int peers = __ballot_sync(FULL_MASK, keep);\n";
    for (int j = 0; j < K; ++j) body << "    peers &= __match_any_sync(FULL_MASK, key[" << j << "]);\n";
    body << "    const int lane_id = (int)(threadIdx.x & 31);\n"
            "    const int src = peers ? (__ffs(peers) - 1) : lane_id;\n"
            "    unsigned int s2 = EMPTY_SLOT;\n"
            "    if (keep && lane_id == src) s2 = group_slot<" << K << ">(state, skeys, cap, hmask, key, overflow);\n"
            "    s2 = __shfl_sync(FULL_MASK, s2, src);\n"
            "    if (keep) slot = s2;\n"
            "  } else if (keep) {\n"
            "    slot = group_slot<" << K << ">(state, skeys, cap, hmask, key, overflow);\n"
            "  }\n";
  }
  body << "  { unsigned int b = __ballot_sync(FULL_MASK, keep);\n"
          "    if ((threadIdx.x & 31) == 0 && b) atomicAdd(kept, (unsigned int)__popc(b)); }\n";
  body << "  WarpGroups wg = {0, 0, 0};\n  if (warp_agg) wg = warp_groups(slot);\n";
  for (size_t a = 0; a < aggs.size(); ++a) {
    const std::string A = "A" + std::to_string(a), V = ag[a].acc_f ? "double" : "long long", v = "v" + std::to_string(a);
    switch (aggs[a].func) {
      case JAgg::Count: body << "  agg_apply<A_SUM, long long>(" << A << ", slot, 1LL, warp_agg, wg);\n"; break;
      case JAgg::Sum: body << "  agg_apply<A_SUM, " << V << ">(" << A << ", slot, " << v << ", warp_agg, wg);\n"; break;
      case JAgg::Avg:
        body << "  agg_apply<A_SUM, double>(" << A << ", slot, " << v << ", warp_agg, wg);\n"
             << "  agg_apply<A_SUM, long long>(N" << a << ", slot, 1LL, warp_agg, wg);\n";
        break;
      case JAgg::Min: body << "  agg_apply<A_MIN, " << V << ">(" << A << ", slot, " << v << ", warp_agg, wg);\n"; break;
      case JAgg::Max: body << "  agg_apply<A_MAX, " << V << ">(" << A << ", slot, " << v << ", warp_agg, wg);\n"; break;
    }
  }

  plan.cols = E.cols;
  plan.masks = E.masks;

  std::ostringstream src;
  src << kHeader << "extern \"C\" __global__ void __launch_bounds__(256) k_main(\n    unsigned long long n,\n"
      << E.param_decls();
  for (size_t r = 0; r < plan.rid_tables.size(); ++r) src << "    const unsigned int* R" << r << ",\n";
  src << mask_param_decls(plan.masks);
  if (K > 0) src << "    unsigned int* state, long long* skeys, unsigned int cap, unsigned int hmask, unsigned int* overflow,\n";
  for (size_t a = 0; a < aggs.size(); ++a) {
    src << "    " << (ag[a].acc_f ? "double" : "long long") << "* A" << a << ",\n";
    if (aggs[a].func == JAgg::Avg) src << "    long long* N" << a << ",\n";
  }
  src << "    unsigned int* kept, int warp_agg) {\n"
      << "  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;\n"
      << "  const bool valid = i < (size_t)n;\n  bool keep = valid;\n"
      << row_id_setup(rel_tbls, identity) << E.decls() << body.str() << "}\n";
  plan.source = src.str();
  return plan;
}

}  // namespace warpdb::gpu
