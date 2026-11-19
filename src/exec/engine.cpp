#include "exec/engine.h"

#include <intrin.h>

#include <cmath>
#include <cstring>
#include <functional>
#include <numeric>
#include <set>
#include <unordered_map>

namespace warpdb {
namespace {

constexpr uint32_t NONE = 0xFFFFFFFFu;

// ------------------------------------------------------------------------------------------
// Relations: late-materialized row-id lists over base tables.
// ------------------------------------------------------------------------------------------
struct Relation {
  std::vector<int> tbls;                  // FROM-clause indexes present
  std::vector<std::vector<uint32_t>> rid;  // rid[k][row] = base-table row of tbls[k]
  size_t n = 0;
  int pos(int t) const {
    for (size_t i = 0; i < tbls.size(); ++i)
      if (tbls[i] == t) return static_cast<int>(i);
    return -1;
  }
};

Relation compose(const Relation& r, const std::vector<uint32_t>& idx) {
  Relation o;
  o.tbls = r.tbls;
  o.n = idx.size();
  o.rid.resize(r.rid.size());
  for (size_t k = 0; k < r.rid.size(); ++k) {
    o.rid[k].resize(idx.size());
    const uint32_t* src = r.rid[k].data();
    uint32_t* dst = o.rid[k].data();
    const uint32_t* ix = idx.data();
    parallel_for(idx.size(), 1 << 15, [&](size_t, size_t b, size_t e) {
      for (size_t i = b; i < e; ++i) dst[i] = src[ix[i]];
    });
  }
  return o;
}

using TableVec = std::vector<std::shared_ptr<Table>>;

class RelScope : public Scope {
 public:
  RelScope(const TableVec& tabs, Relation r) : tabs_(tabs), rel(std::move(r)) { n = rel.n; }
  VP intercept(const Expr& e) override {
    if (e.kind != Expr::Col) return nullptr;
    uint64_t key = (static_cast<uint64_t>(e.tbl) << 32) | static_cast<uint32_t>(e.col);
    auto it = cache_.find(key);
    if (it != cache_.end()) return it->second;
    int p = rel.pos(e.tbl);
    if (p < 0) throw Error("internal: column's table is not part of the relation");
    VP v = gather_column(tabs_[e.tbl]->cols[e.col], rel.rid[p]);
    cache_[key] = v;
    return v;
  }
  std::unique_ptr<Scope> gather(const std::vector<uint32_t>& idx) const override {
    return std::make_unique<RelScope>(tabs_, compose(rel, idx));
  }
  void set_rows(const std::vector<uint32_t>& idx) {
    rel = compose(rel, idx);
    n = rel.n;
    cache_.clear();
  }
  Relation rel;

 private:
  const TableVec& tabs_;
  std::unordered_map<uint64_t, VP> cache_;
};

class AggScope : public Scope {
 public:
  std::unordered_map<std::string, VP> m;  // group-key / aggregate expression text -> column
  VP intercept(const Expr& e) override {
    auto it = m.find(e.str());
    if (it != m.end()) return it->second;
    if (e.kind == Expr::Agg) throw Error("internal: aggregate was not computed: " + e.str());
    if (e.kind == Expr::Col)
      throw Error("column \"" + e.name + "\" must appear in the GROUP BY clause or be used in an aggregate function");
    return nullptr;
  }
  std::unique_ptr<Scope> gather(const std::vector<uint32_t>& idx) const override {
    auto s = std::make_unique<AggScope>();
    for (auto& kv : m) s->m[kv.first] = gather_value(kv.second, idx);
    s->n = idx.size();
    return s;
  }
};

// Re-expresses VARCHAR codes of `v` in the code space of `target`'s dictionary (-1 = no match).
// Both dictionaries are sorted, so a two-pointer merge builds the mapping.
VP remap_codes(const Value& v, const Column& target) {
  const auto& src = v.dict->dict;
  const auto& dst = target.dict;
  std::vector<int64_t> m(src.size(), -1);
  size_t j = 0;
  for (size_t i = 0; i < src.size(); ++i) {
    while (j < dst.size() && dst[j] < src[i]) ++j;
    if (j < dst.size() && dst[j] == src[i]) m[i] = static_cast<int64_t>(j);
  }
  auto o = std::make_shared<Value>();
  o->kind = Value::I;
  o->n = v.n;
  o->scalar = v.scalar;
  o->i.resize(v.i.size());
  parallel_for(v.i.size(), 1 << 15, [&](size_t, size_t b, size_t e) {
    for (size_t k = b; k < e; ++k) o->i[k] = m[static_cast<size_t>(v.i[k])];
  });
  return o;
}

// ------------------------------------------------------------------------------------------
// Hash join (inner, equality on one or more int64 keys). Build side = smaller input.
// ------------------------------------------------------------------------------------------
inline uint64_t mix64(uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return x;
}

void hash_join(const std::vector<VP>& lk, size_t ln, const std::vector<VP>& rk, size_t rn,
               std::vector<uint32_t>& li, std::vector<uint32_t>& ri) {
  const size_t K = lk.size();
  for (auto* ks : {&lk, &rk})
    for (auto& v : *ks)
      if (v->kind != Value::I && v->kind != Value::S) throw Error("join keys must be integer, date or VARCHAR columns (DOUBLE keys unsupported)");
  bool build_left = ln <= rn;
  const auto& bk = build_left ? lk : rk;
  const auto& pk = build_left ? rk : lk;
  size_t bn = build_left ? ln : rn, pn = build_left ? rn : ln;

  std::vector<const int64_t*> bp(K), pp(K);
  std::vector<size_t> bs(K), ps(K);
  for (size_t j = 0; j < K; ++j) {
    bp[j] = bk[j]->i.data(); bs[j] = bk[j]->stride();
    pp[j] = pk[j]->i.data(); ps[j] = pk[j]->stride();
  }
  auto hrow = [K](const std::vector<const int64_t*>& p, const std::vector<size_t>& st, size_t r) {
    uint64_t h = 0x9E3779B97F4A7C15ULL;
    for (size_t j = 0; j < K; ++j) h = mix64(h ^ static_cast<uint64_t>(p[j][r * st[j]]));
    return h;
  };

  size_t cap = 16;
  while (cap < bn * 2) cap <<= 1;
  const size_t mask = cap - 1;
  std::vector<uint32_t> head(cap, NONE), next(bn);
  for (size_t r = bn; r-- > 0;) {  // reverse so chains are in ascending row order
    size_t b = hrow(bp, bs, r) & mask;
    next[r] = head[b];
    head[b] = static_cast<uint32_t>(r);
  }

  size_t nch = std::max<size_t>(1, num_chunks(pn, 1 << 15));
  std::vector<std::vector<uint32_t>> outp(nch), outb(nch);
  parallel_for(pn, 1 << 15, [&](size_t c, size_t s, size_t e) {
    auto& op = outp[c];
    auto& ob = outb[c];
    for (size_t r = s; r < e; ++r) {
      uint64_t h = hrow(pp, ps, r);
      for (uint32_t b = head[h & mask]; b != NONE; b = next[b]) {
        bool eq = true;
        for (size_t j = 0; j < K; ++j)
          if (bp[j][b * bs[j]] != pp[j][r * ps[j]]) { eq = false; break; }
        if (eq) {
          op.push_back(static_cast<uint32_t>(r));
          ob.push_back(b);
        }
      }
    }
  });
  size_t total = 0;
  std::vector<size_t> off(nch);
  for (size_t c = 0; c < nch; ++c) { off[c] = total; total += outp[c].size(); }
  auto& pout = build_left ? ri : li;
  auto& bout = build_left ? li : ri;
  pout.resize(total);
  bout.resize(total);
  parallel_for(nch, 1, [&](size_t, size_t s, size_t e) {
    for (size_t c = s; c < e; ++c) {
      std::memcpy(pout.data() + off[c], outp[c].data(), outp[c].size() * 4);
      std::memcpy(bout.data() + off[c], outb[c].data(), outb[c].size() * 4);
    }
  });
}

// ------------------------------------------------------------------------------------------
// Hash aggregation (thread-local tables merged in chunk order -> deterministic group order).
// ------------------------------------------------------------------------------------------
enum AggFunc { SUM, COUNT, AVG, MIN, MAX };

struct AggState {
  int64_t cnt = 0;
  int64_t si = 0;
  double sf = 0;
  int64_t imin = INT64_MAX, imax = INT64_MIN;
  double fmin = INFINITY, fmax = -INFINITY;
};

struct AggInput {
  AggFunc func;
  VP arg;  // null for count(*)
  bool isf = false;
  const int64_t* pi = nullptr;
  const double* pf = nullptr;
  size_t st = 1;
};

inline uint64_t hash_keys(const int64_t* k, size_t K) {
  uint64_t h = 0x9E3779B97F4A7C15ULL;
  for (size_t j = 0; j < K; ++j) h = mix64(h ^ static_cast<uint64_t>(k[j]));
  return h;
}

class AggTable {
 public:
  AggTable(size_t K, size_t A) : K(K), A(A), slots_(64, -1) {}
  size_t K, A, ngroups = 0;
  std::vector<int64_t> keys;
  std::vector<AggState> st;

  uint32_t find_or_insert(const int64_t* k, uint64_t h) {
    size_t mask = slots_.size() - 1, s = h & mask;
    for (;;) {
      int32_t g = slots_[s];
      if (g < 0) break;
      if (K == 0 || std::memcmp(keys.data() + size_t(g) * K, k, K * 8) == 0) return static_cast<uint32_t>(g);
      s = (s + 1) & mask;
    }
    if ((ngroups + 1) * 2 > slots_.size()) {
      grow();
      mask = slots_.size() - 1;
      s = h & mask;
      while (slots_[s] >= 0) s = (s + 1) & mask;
    }
    uint32_t g = static_cast<uint32_t>(ngroups++);
    keys.insert(keys.end(), k, k + K);
    st.resize(st.size() + A);
    slots_[s] = static_cast<int32_t>(g);
    return g;
  }

 private:
  std::vector<int32_t> slots_;
  void grow() {
    slots_.assign(slots_.size() * 2, -1);
    size_t mask = slots_.size() - 1;
    for (size_t g = 0; g < ngroups; ++g) {
      size_t s = hash_keys(keys.data() + g * K, K) & mask;
      while (slots_[s] >= 0) s = (s + 1) & mask;
      slots_[s] = static_cast<int32_t>(g);
    }
  }
};

struct AggOutput {
  std::vector<VP> keys;  // one column per group key
  std::vector<VP> aggs;  // one column per aggregate
  size_t n = 0;
};

AggOutput hash_aggregate(const std::vector<VP>& keyv, std::vector<AggInput>& ins, size_t nrows) {
  const size_t K = keyv.size(), A = ins.size();
  if (K > 16) throw Error("too many GROUP BY keys (max 16)");
  // int64 views of the keys (doubles are bit-cast)
  std::vector<std::vector<int64_t>> conv(K);
  std::vector<const int64_t*> kp(K);
  std::vector<size_t> ks(K);
  for (size_t j = 0; j < K; ++j) {
    const Value& v = *keyv[j];
    ks[j] = v.stride();
    if (v.kind == Value::I || v.kind == Value::S) {
      kp[j] = v.i.data();
    } else if (v.kind == Value::F) {
      size_t m = v.scalar ? 1 : nrows;
      conv[j].resize(m);
      std::memcpy(conv[j].data(), v.f.data(), m * 8);
      kp[j] = conv[j].data();
    } else if (v.kind == Value::B) {
      size_t m = v.scalar ? 1 : nrows;
      conv[j].resize(m);
      for (size_t i = 0; i < m; ++i) conv[j][i] = v.b[i];
      kp[j] = conv[j].data();
    } else throw Error("cannot GROUP BY this expression type");
  }
  for (auto& in : ins) {
    if (!in.arg) continue;
    const Value& v = *in.arg;
    in.st = v.stride();
    if (v.kind == Value::F) { in.isf = true; in.pf = v.f.data(); }
    else if (v.kind == Value::I || v.kind == Value::S) { in.pi = v.i.data(); }
    else throw Error("aggregate argument must be numeric");
    if (v.kind == Value::S && in.func != MIN && in.func != MAX && in.func != COUNT)
      throw Error("SUM/AVG over VARCHAR is not allowed");
  }

  auto process = [&](AggTable& t, size_t b, size_t e) {
    int64_t key[16];
    for (size_t r = b; r < e; ++r) {
      for (size_t j = 0; j < K; ++j) key[j] = kp[j][r * ks[j]];
      uint32_t g = t.find_or_insert(key, hash_keys(key, K));
      AggState* s = &t.st[size_t(g) * A];
      for (size_t a = 0; a < A; ++a) {
        const AggInput& in = ins[a];
        AggState& x = s[a];
        switch (in.func) {
          case COUNT: x.cnt++; break;
          case SUM:
            if (in.isf) x.sf += in.pf[r * in.st]; else x.si += in.pi[r * in.st];
            break;
          case AVG:
            x.cnt++;
            x.sf += in.isf ? in.pf[r * in.st] : double(in.pi[r * in.st]);
            break;
          case MIN:
            if (in.isf) x.fmin = std::min(x.fmin, in.pf[r * in.st]);
            else x.imin = std::min(x.imin, in.pi[r * in.st]);
            break;
          case MAX:
            if (in.isf) x.fmax = std::max(x.fmax, in.pf[r * in.st]);
            else x.imax = std::max(x.imax, in.pi[r * in.st]);
            break;
        }
      }
    }
  };

  constexpr size_t GR = 1 << 16;
  size_t nch = std::max<size_t>(1, num_chunks(nrows, GR));
  AggTable global(K, A);
  int64_t zero[1] = {0};
  if (K == 0) global.find_or_insert(zero, hash_keys(zero, 0));  // global aggregate always yields one row
  if (nch == 1) {
    process(global, 0, nrows);
  } else {
    std::vector<std::unique_ptr<AggTable>> locals(nch);
    parallel_for(nrows, GR, [&](size_t c, size_t b, size_t e) {
      locals[c] = std::make_unique<AggTable>(K, A);
      process(*locals[c], b, e);
    });
    for (size_t c = 0; c < nch; ++c) {
      if (!locals[c]) continue;
      AggTable& l = *locals[c];
      for (size_t g = 0; g < l.ngroups; ++g) {
        const int64_t* k = l.keys.data() + g * K;
        uint32_t gg = global.find_or_insert(k, hash_keys(k, K));
        AggState* d = &global.st[size_t(gg) * A];
        const AggState* s = &l.st[g * A];
        for (size_t a = 0; a < A; ++a) {
          d[a].cnt += s[a].cnt;
          d[a].si += s[a].si;
          d[a].sf += s[a].sf;
          d[a].imin = std::min(d[a].imin, s[a].imin);
          d[a].imax = std::max(d[a].imax, s[a].imax);
          d[a].fmin = std::min(d[a].fmin, s[a].fmin);
          d[a].fmax = std::max(d[a].fmax, s[a].fmax);
        }
      }
      locals[c].reset();
    }
  }

  AggOutput out;
  out.n = global.ngroups;
  const size_t G = global.ngroups;
  for (size_t j = 0; j < K; ++j) {
    const Value& src = *keyv[j];
    auto v = std::make_shared<Value>();
    v->n = G;
    v->is_date = src.is_date;
    v->dict = src.dict;
    if (src.kind == Value::F) {
      v->kind = Value::F;
      v->f.resize(G);
      for (size_t g = 0; g < G; ++g) std::memcpy(&v->f[g], &global.keys[g * K + j], 8);
    } else {
      v->kind = src.kind == Value::S ? Value::S : Value::I;
      v->i.resize(G);
      for (size_t g = 0; g < G; ++g) v->i[g] = global.keys[g * K + j];
    }
    out.keys.push_back(v);
  }
  for (size_t a = 0; a < A; ++a) {
    const AggInput& in = ins[a];
    auto v = std::make_shared<Value>();
    v->n = G;
    bool f_out = false;
    switch (in.func) {
      case COUNT: v->kind = Value::I; break;
      case AVG: v->kind = Value::F; f_out = true; break;
      case SUM: f_out = in.isf; v->kind = in.isf ? Value::F : Value::I; break;
      case MIN:
      case MAX:
        f_out = in.isf;
        v->kind = in.isf ? Value::F : in.arg->kind;
        v->is_date = in.arg->is_date;
        v->dict = in.arg->dict;
        break;
    }
    if (f_out) v->f.resize(G); else v->i.resize(G);
    for (size_t g = 0; g < G; ++g) {
      const AggState& s = global.st[g * A + a];
      switch (in.func) {
        case COUNT: v->i[g] = s.cnt; break;
        case SUM: if (in.isf) v->f[g] = s.sf; else v->i[g] = s.si; break;
        case AVG: v->f[g] = s.cnt ? s.sf / double(s.cnt) : 0.0; break;
        case MIN: if (in.isf) v->f[g] = s.fmin; else v->i[g] = s.imin; break;
        case MAX: if (in.isf) v->f[g] = s.fmax; else v->i[g] = s.imax; break;
      }
    }
    out.aggs.push_back(v);
  }
  return out;
}

// ------------------------------------------------------------------------------------------
// Sort / top-k
// ------------------------------------------------------------------------------------------
std::vector<uint32_t> sort_perm(const std::vector<VP>& keys, const std::vector<bool>& desc, size_t n, int64_t limit) {
  struct K { const int64_t* i; const double* f; const uint8_t* b; size_t st; bool desc; };
  std::vector<K> ks;
  for (size_t j = 0; j < keys.size(); ++j) {
    const Value& v = *keys[j];
    if (v.kind == Value::L) continue;
    K k{nullptr, nullptr, nullptr, v.stride(), bool(desc[j])};
    if (v.kind == Value::I || v.kind == Value::S) k.i = v.i.data();
    else if (v.kind == Value::F) k.f = v.f.data();
    else k.b = v.b.data();
    ks.push_back(k);
  }
  std::vector<uint32_t> perm(n);
  std::iota(perm.begin(), perm.end(), 0u);
  auto cmp = [&](uint32_t a, uint32_t b) {
    for (const K& k : ks) {
      int c;
      if (k.i) { int64_t x = k.i[a * k.st], y = k.i[b * k.st]; c = x < y ? -1 : x > y; }
      else if (k.f) { double x = k.f[a * k.st], y = k.f[b * k.st]; c = x < y ? -1 : x > y; }
      else { int x = k.b[a * k.st], y = k.b[b * k.st]; c = x < y ? -1 : x > y; }
      if (c) return k.desc ? c > 0 : c < 0;
    }
    return a < b;  // deterministic tie-break
  };
  if (limit >= 0 && static_cast<size_t>(limit) < n) {
    std::partial_sort(perm.begin(), perm.begin() + limit, perm.end(), cmp);
    perm.resize(static_cast<size_t>(limit));
  } else {
    std::sort(perm.begin(), perm.end(), cmp);
  }
  return perm;
}

// ------------------------------------------------------------------------------------------
// Binder helpers
// ------------------------------------------------------------------------------------------
void tables_of(const Expr& e, std::set<int>& out) {
  if (e.kind == Expr::Col) out.insert(e.tbl);
  for (auto& a : e.args) tables_of(*a, out);
}
bool has_agg(const Expr& e) {
  if (e.kind == Expr::Agg) return true;
  for (auto& a : e.args)
    if (has_agg(*a)) return true;
  return false;
}
void collect_aggs(const ExprP& e, std::vector<ExprP>& out) {
  if (e->kind == Expr::Agg) {
    for (auto& o : out)
      if (o->str() == e->str()) return;
    out.push_back(e);
    return;
  }
  for (auto& a : e->args) collect_aggs(a, out);
}
void split_and(const ExprP& e, std::vector<ExprP>& out) {
  if (e->kind == Expr::Bin && e->name == "and") {
    split_and(e->args[0], out);
    split_and(e->args[1], out);
  } else {
    out.push_back(e);
  }
}

StageStat make_stat(StageOp op, bool gpu, const std::string& label, double rin, double raux, double rout, int k1,
                    int k2, double ms, int k3 = 0) {
  StageStat s;
  s.op = op; s.gpu = gpu; s.label = label;
  s.rows_in = rin; s.rows_aux = raux; s.rows_out = rout;
  s.k1 = k1; s.k2 = k2; s.k3 = k3; s.ms = ms;
  return s;
}

std::string fmt_ms(double ms) {
  char b[32];
  std::snprintf(b, sizeof b, "%.2f ms", ms);
  return b;
}

}  // namespace

// ------------------------------------------------------------------------------------------
double Engine::column_ndv(const std::shared_ptr<Table>& t, int col) {
  uint64_t key = (reinterpret_cast<uint64_t>(t.get()) << 8) ^ static_cast<uint64_t>(col);
  auto it = ndv_cache_.find(key);
  if (it != ndv_cache_.end()) return it->second;
  const Column& c = *t->cols[col];
  const size_t n = t->nrows;
  constexpr size_t M = size_t(1) << 24;  // bits; accurate up to ~10M distinct values
  size_t nch = std::max<size_t>(1, num_chunks(n, 1 << 18));
  std::vector<std::vector<uint64_t>> maps(nch, std::vector<uint64_t>(M / 64, 0));
  parallel_for(n, 1 << 18, [&](size_t ch, size_t b, size_t e) {
    auto& m = maps[ch];
    for (size_t i = b; i < e; ++i) {
      int64_t v = c.type == Type::INT64 ? c.i64[i] : c.type == Type::FLOAT64 ? static_cast<int64_t>(c.f64[i] * 1e6) : c.i32[i];
      uint64_t h = mix64(static_cast<uint64_t>(v)) & (M - 1);
      m[h >> 6] |= uint64_t(1) << (h & 63);
    }
  });
  size_t set = 0;
  for (size_t w = 0; w < M / 64; ++w) {
    uint64_t x = 0;
    for (auto& m : maps) x |= m[w];
    set += static_cast<size_t>(__popcnt64(x));
  }
  double z = static_cast<double>(M - set) / static_cast<double>(M);
  double est = set >= M ? static_cast<double>(n) : -static_cast<double>(M) * std::log(std::max(z, 1e-12));
  est = std::min(est, static_cast<double>(n));
  est = std::max(est, 1.0);
  ndv_cache_[key] = est;
  return est;
}

double Engine::key_ndv(const PlannedQuery& pq, const ExprP& e, size_t n) {
  double nn = std::max<double>(1.0, static_cast<double>(n));
  if (e->kind == Expr::Col) return std::min(column_ndv(pq.tabs[e->tbl], e->col), nn);
  if (e->kind == Expr::Int || e->kind == Expr::Float || e->kind == Expr::Date || e->kind == Expr::Str) return 1.0;

  // General expression over one table (year(d), a + b, CASE ...): evaluate it on a strided sample and count
  // distinct values. Few distinct values in the sample => the domain is small (saturated); a nearly-unique
  // sample => scale up to the table size.
  std::set<int> ts;
  tables_of(*e, ts);
  if (ts.size() != 1) return nn;
  const int k = *ts.begin();
  const size_t rows = pq.tabs[k]->nrows;
  if (rows == 0) return 1.0;
  std::string key = std::to_string(pq.tabs[k]->cols[0]->uid) + "|ndv|" + e->str();
  double d_est;
  if (auto hit = sel_cache_.find(key); hit != sel_cache_.end()) {
    d_est = hit->second;
  } else {
    d_est = static_cast<double>(rows);
    try {
      const size_t S = std::min<size_t>(rows, 4096);
      Relation r;
      r.tbls = {k};
      r.n = S;
      r.rid.resize(1);
      r.rid[0].resize(S);
      for (size_t i = 0; i < S; ++i) r.rid[0][i] = static_cast<uint32_t>(static_cast<uint64_t>(i) * rows / S);
      RelScope sc(pq.tabs, std::move(r));
      VP v = materialize(eval(*e, sc), S);
      std::set<int64_t> seen;
      for (size_t i = 0; i < S; ++i) {
        int64_t bits = 0;
        if (v->kind == Value::I || v->kind == Value::S) bits = v->i[i];
        else if (v->kind == Value::F) std::memcpy(&bits, &v->f[i], 8);
        else if (v->kind == Value::B) bits = v->b[i];
        seen.insert(bits);
      }
      double d = static_cast<double>(seen.size());
      d_est = d < 0.7 * static_cast<double>(S) ? d : std::min(static_cast<double>(rows), d * static_cast<double>(rows) / static_cast<double>(S));
    } catch (const std::exception&) {
      d_est = static_cast<double>(rows);  // cannot evaluate on a sample: assume the worst
    }
    if (sel_cache_.size() > 4096) sel_cache_.clear();
    sel_cache_[key] = d_est;
  }
  return std::min(d_est, nn);
}

int Engine::pick_start(const PlannedQuery& pq, const std::vector<size_t>& rel_n) const {
  const int T = static_cast<int>(pq.tabs.size());
  int start = 0;
  size_t best = SIZE_MAX;
  for (int k = 0; k < T; ++k) {
    bool has_edge = T == 1;
    for (auto& e : pq.edges) has_edge |= (e.ta == k || e.tb == k);
    if (has_edge && rel_n[k] < best) { best = rel_n[k]; start = k; }
  }
  return start;
}

// Pick the connected table whose join with the current result is estimated to be smallest:
//   |cur| * |cand| / prod_over_edges(max(ndv(cur key), ndv(cand key)))
double Engine::join_estimate(const PlannedQuery& pq, const std::vector<bool>& joined, size_t cur_n, int k,
                             const std::vector<size_t>& rel_n, int* keys) {
  double est = static_cast<double>(cur_n) * static_cast<double>(rel_n[k]);
  int nk = 0;
  for (auto& e : pq.edges) {
    bool cur_is_a = joined[e.ta] && e.tb == k, cur_is_b = joined[e.tb] && e.ta == k;
    if (!cur_is_a && !cur_is_b) continue;
    ++nk;
    double nc = key_ndv(pq, cur_is_a ? e.ea : e.eb, cur_n), nd = key_ndv(pq, cur_is_a ? e.eb : e.ea, rel_n[k]);
    est /= std::max(1.0, std::max(nc, nd));
  }
  if (keys) *keys = nk;
  return est;
}

// Pick the connected table whose join with the current result is estimated to be smallest:
//   |cur| * |cand| / prod_over_edges(max(ndv(cur key), ndv(cand key)))
int Engine::pick_next(const PlannedQuery& pq, const std::vector<bool>& joined, size_t cur_n,
                      const std::vector<size_t>& rel_n) {
  const int T = static_cast<int>(pq.tabs.size());
  int best = -1;
  double best_est = 0;
  for (int k = 0; k < T; ++k) {
    if (joined[k]) continue;
    int nk = 0;
    double est = join_estimate(pq, joined, cur_n, k, rel_n, &nk);
    if (nk > 0 && (best < 0 || est < best_est)) { best = k; best_est = est; }
  }
  return best;
}

double Engine::sample_selectivity(const PlannedQuery& pq, int k) {
  const size_t n = pq.tabs[k]->nrows;
  if (n == 0) return 0.0;
  if (pq.push[k].empty()) return 1.0;
  std::string cache_key = std::to_string(pq.tabs[k]->cols[0]->uid) + "|" + std::to_string(n);
  for (auto& c : pq.push[k]) cache_key += "|" + c->str();
  if (auto hit = sel_cache_.find(cache_key); hit != sel_cache_.end()) return hit->second;
  const size_t S = std::min<size_t>(n, 4096);
  Relation r;
  r.tbls = {k};
  r.n = S;
  r.rid.resize(1);
  r.rid[0].resize(S);
  for (size_t i = 0; i < S; ++i) r.rid[0][i] = static_cast<uint32_t>(static_cast<uint64_t>(i) * n / S);
  RelScope sc(pq.tabs, std::move(r));
  for (auto& c : pq.push[k]) {
    VP b = eval(*c, sc);
    sc.set_rows(select_true(*b));
  }
  double passed = static_cast<double>(sc.n);
  if (passed == 0) passed = 0.5;  // nothing in the sample passed: assume a small but non-zero selectivity
  double sel = passed / static_cast<double>(S);
  if (sel_cache_.size() > 4096) sel_cache_.clear();
  sel_cache_[cache_key] = sel;
  return sel;
}

static void collect_cols(const Expr& e, std::set<std::pair<int, int>>& out) {
  if (e.kind == Expr::Col) out.insert({e.tbl, e.col});
  for (auto& a : e.args) collect_cols(*a, out);
}

static int count_nodes(const Expr& e) {
  int n = 0;
  switch (e.kind) {
    case Expr::Bin: case Expr::Not: case Expr::Neg: case Expr::Between: case Expr::In: case Expr::Like:
    case Expr::Case: case Expr::Func: n = 1; break;
    default: break;
  }
  for (auto& a : e.args) n += count_nodes(*a);
  return n;
}

int expr_nodes(const std::vector<ExprP>& exprs) {
  int n = 0;
  for (auto& e : exprs) n += count_nodes(*e);
  return n;
}

int agg_expr_nodes(const PlannedQuery& pq) {
  int n = expr_nodes(pq.sel.group_by);
  for (auto& a : pq.aggs)
    for (auto& arg : a->args)
      if (arg->kind != Expr::Star) n += count_nodes(*arg);
  return n;
}

int distinct_columns(const std::vector<ExprP>& exprs) {
  std::set<std::pair<int, int>> cols;
  for (auto& e : exprs) collect_cols(*e, cols);
  return static_cast<int>(cols.size());
}

QueryResult Engine::execute(const std::string& sql) {
  PlannedQuery pq = plan(sql);
  if (pq.sel.explain && !pq.sel.analyze) return explain_only(pq);  // estimates only, nothing is executed

  std::string dev = device;
  PlacementEstimate est;
  double decide_ms = 0;
  const bool automatic = dev == "auto";
  if (automatic) {
    Timer td;
    ensure_calibrated();
    td.reset();
    est = estimate_placement(pq);
    decide_ms = td.ms();
    dev = est.choice;
  }
  if (dev == "gpu" && !gpu_) throw Error("no GPU backend available");

  QueryResult r = (dev == "gpu") ? gpu_->execute(*this, pq) : execute_cpu(pq);
  r.explain = pq.sel.explain;
  r.device_used = dev;
  if (learn)
    for (auto& st : r.stages) cost.observe(st);
  if (automatic) {
    r.plan_lines = est.lines;
    decisions.push_back({sql, est.cpu_ms, est.gpu_ms, dev, r.ms, decide_ms});
  }
  return r;
}

PlannedQuery Engine::plan(const std::string& sql) {
  Select s = parse_sql(sql);

  // ---- FROM ----
  TableVec tabs;
  std::vector<std::string> aliases;
  for (auto& r : s.from) {
    auto t = catalog.get(r.name);
    if (!t) throw Error("table not found: " + r.name);
    tabs.push_back(t);
    aliases.push_back(to_lower(r.alias));
  }
  const int T = static_cast<int>(tabs.size());

  // ---- binding ----
  auto bind_col = [&](Expr& e) -> bool {
    if (!e.table.empty()) {
      std::string q = to_lower(e.table);
      for (int k = 0; k < T; ++k) {
        if (aliases[k] == q || to_lower(tabs[k]->name) == q) {
          int c = tabs[k]->find(e.name);
          if (c < 0) throw Error("column not found: " + e.table + "." + e.name);
          e.tbl = k; e.col = c;
          return true;
        }
      }
      throw Error("unknown table or alias: " + e.table);
    }
    int found = -1, fc = -1;
    for (int k = 0; k < T; ++k) {
      int c = tabs[k]->find(e.name);
      if (c >= 0) {
        if (found >= 0) throw Error("ambiguous column: " + e.name);
        found = k; fc = c;
      }
    }
    if (found < 0) return false;
    e.tbl = found; e.col = fc;
    return true;
  };
  std::function<void(const ExprP&)> bind = [&](const ExprP& e) {
    if (e->kind == Expr::Col) {
      if (!bind_col(*e)) throw Error("column not found: " + e->name);
      return;
    }
    for (auto& a : e->args) bind(a);
  };

  if (s.star) {
    for (int k = 0; k < T; ++k)
      for (size_t c = 0; c < tabs[k]->names.size(); ++c) {
        SelectItem it;
        it.e = make_expr(Expr::Col);
        it.e->name = tabs[k]->names[c];
        it.e->tbl = k;
        it.e->col = static_cast<int>(c);
        it.alias = tabs[k]->names[c];
        s.items.push_back(it);
      }
  } else {
    for (auto& it : s.items) bind(it.e);
  }
  std::unordered_map<std::string, size_t> alias_map;
  for (size_t i = 0; i < s.items.size(); ++i)
    if (!s.items[i].alias.empty()) alias_map[to_lower(s.items[i].alias)] = i;

  std::function<ExprP(const ExprP&)> subst = [&](const ExprP& e) -> ExprP {
    if (e->kind == Expr::Col && e->table.empty() && e->tbl < 0) {
      auto it = alias_map.find(e->name);
      if (it != alias_map.end()) return s.items[it->second].e;
      return e;
    }
    if (e->kind == Expr::Int) return e;
    for (auto& a : e->args) a = subst(a);
    return e;
  };
  auto ordinal = [&](const ExprP& e) -> ExprP {
    if (e->kind == Expr::Int) {
      if (e->ival < 1 || e->ival > static_cast<int64_t>(s.items.size())) throw Error("ORDER/GROUP BY position out of range");
      return s.items[e->ival - 1].e;
    }
    return e;
  };

  if (s.distinct && s.group_by.empty())
    for (auto& it : s.items) s.group_by.push_back(it.e);
  else
    for (auto& g : s.group_by) {
      g = ordinal(g);
      if (g->kind == Expr::Col && g->table.empty() && g->tbl < 0) {
        if (!bind_col(*g)) {  // fall back to a select-list alias
          auto it = alias_map.find(g->name);
          if (it == alias_map.end()) throw Error("column not found: " + g->name);
          g = s.items[it->second].e;
        }
      } else {
        bind(g);
      }
    }
  if (s.having) { s.having = subst(s.having); bind(s.having); }
  for (auto& o : s.order_by) {
    o.e = ordinal(o.e);
    o.e = subst(o.e);
    bind(o.e);
  }

  std::vector<ExprP> aggs;
  for (auto& it : s.items) collect_aggs(it.e, aggs);
  if (s.having) collect_aggs(s.having, aggs);
  for (auto& o : s.order_by) collect_aggs(o.e, aggs);
  const bool agg_mode = !aggs.empty() || !s.group_by.empty();

  // ---- classify predicates ----
  std::vector<ExprP> conj;
  if (s.where) split_and(s.where, conj);
  for (auto& c : s.on_conds) split_and(c, conj);
  for (auto& c : conj) {
    bind(c);
    if (has_agg(*c)) throw Error("aggregates are not allowed in WHERE / ON");
  }
  std::vector<std::vector<ExprP>> push(T);
  std::vector<Edge> edges;
  std::vector<ExprP> residual;
  for (auto& c : conj) {
    std::set<int> ts;
    tables_of(*c, ts);
    if (ts.size() == 1) {
      push[*ts.begin()].push_back(c);
    } else if (ts.size() == 2 && c->kind == Expr::Bin && c->name == "=") {
      std::set<int> l, r;
      tables_of(*c->args[0], l);
      tables_of(*c->args[1], r);
      if (l.size() == 1 && r.size() == 1 && *l.begin() != *r.begin())
        edges.push_back({*l.begin(), *r.begin(), c->args[0], c->args[1]});
      else
        residual.push_back(c);
    } else {
      residual.push_back(c);
    }
  }

  PlannedQuery pq;
  pq.sel = std::move(s);
  pq.tabs = std::move(tabs);
  pq.aggs = std::move(aggs);
  pq.agg_mode = agg_mode;
  pq.push = std::move(push);
  pq.edges = std::move(edges);
  pq.residual = std::move(residual);
  return pq;
}

QueryResult Engine::execute_cpu(PlannedQuery& pq) {
  Timer total;
  Select& s = pq.sel;
  TableVec& tabs = pq.tabs;
  auto& aggs = pq.aggs;
  const bool agg_mode = pq.agg_mode;
  auto& push = pq.push;
  auto& edges = pq.edges;
  auto& residual = pq.residual;
  const int T = static_cast<int>(tabs.size());
  QueryResult res;
  res.explain = s.explain;
  auto trace = [&](const std::string& m) { res.trace.push_back(m); };

  // ---- scans with pushed-down filters ----
  std::vector<Relation> rels(T);
  for (int k = 0; k < T; ++k) {
    Timer t;
    Relation r;
    r.tbls = {k};
    r.n = tabs[k]->nrows;
    r.rid.resize(1);
    r.rid[0].resize(r.n);
    std::iota(r.rid[0].begin(), r.rid[0].end(), 0u);
    RelScope sc(tabs, std::move(r));
    for (auto& c : push[k]) {
      VP b = eval(*c, sc);
      sc.set_rows(select_true(*b));
    }
    rels[k] = std::move(sc.rel);
    trace("scan " + tabs[k]->name + ": " + std::to_string(tabs[k]->nrows) + " -> " + std::to_string(rels[k].n) +
          " rows (" + std::to_string(push[k].size()) + " filters, " + fmt_ms(t.ms()) + ")");
    res.stages.push_back(make_stat(StageOp::Scan, false, tabs[k]->name, static_cast<double>(tabs[k]->nrows), 0,
                                   static_cast<double>(rels[k].n), distinct_columns(push[k]), expr_nodes(push[k]), t.ms()));
  }

  // ---- joins (greedy: smallest first, always join the smallest connected table) ----
  Relation cur;
  {
    std::vector<bool> joined(T, false);
    std::vector<size_t> rel_n(T);
    for (int k = 0; k < T; ++k) rel_n[k] = rels[k].n;
    int start = pick_start(pq, rel_n);
    cur = std::move(rels[start]);
    joined[start] = true;
    for (int remaining = T - 1; remaining > 0; --remaining) {
      int best = pick_next(pq, joined, cur.n, rel_n);
      if (best < 0) throw Error("cross joins are not supported: add an equality join condition between all tables");
      Timer t;
      std::vector<ExprP> kc, kb;
      for (auto& e : edges) {
        if (joined[e.ta] && e.tb == best) { kc.push_back(e.ea); kb.push_back(e.eb); }
        else if (joined[e.tb] && e.ta == best) { kc.push_back(e.eb); kb.push_back(e.ea); }
      }
      RelScope cs(tabs, std::move(cur)), bs(tabs, std::move(rels[best]));
      std::vector<VP> lk, rk;
      for (auto& e : kc) lk.push_back(eval(*e, cs));
      for (auto& e : kb) rk.push_back(eval(*e, bs));
      for (size_t j = 0; j < lk.size(); ++j) {
        bool ls = lk[j]->kind == Value::S, rs2 = rk[j]->kind == Value::S;
        if (ls != rs2) throw Error("cannot join a VARCHAR column with a non-VARCHAR column");
        if (ls && lk[j]->dict.get() != rk[j]->dict.get()) rk[j] = remap_codes(*rk[j], *lk[j]->dict);
      }
      std::vector<uint32_t> li, ri;
      size_t ln = cs.rel.n, rn = bs.rel.n;
      hash_join(lk, ln, rk, rn, li, ri);
      Relation o;
      o.n = li.size();
      Relation a = compose(cs.rel, li), b = compose(bs.rel, ri);
      o.tbls = a.tbls;
      o.rid = std::move(a.rid);
      for (size_t k = 0; k < b.tbls.size(); ++k) {
        o.tbls.push_back(b.tbls[k]);
        o.rid.push_back(std::move(b.rid[k]));
      }
      cur = std::move(o);
      joined[best] = true;
      trace("hash join " + tabs[best]->name + " (" + std::to_string(kc.size()) + " keys): " + std::to_string(ln) +
            " x " + std::to_string(rn) + " -> " + std::to_string(cur.n) + " rows (" + fmt_ms(t.ms()) + ")");
      res.stages.push_back(make_stat(StageOp::Join, false, tabs[best]->name, static_cast<double>(ln),
                                     static_cast<double>(rn), static_cast<double>(cur.n), static_cast<int>(kc.size()),
                                     static_cast<int>(cur.tbls.size()), t.ms()));
    }
  }

  std::unique_ptr<Scope> scope;
  {
    auto rs = std::make_unique<RelScope>(tabs, std::move(cur));
    for (auto& c : residual) {
      VP b = eval(*c, *rs);
      rs->set_rows(select_true(*b));
    }
    if (!residual.empty()) trace("residual filter -> " + std::to_string(rs->n) + " rows");

    if (agg_mode) {
      Timer t;
      std::vector<VP> keyv;
      for (auto& g : s.group_by) keyv.push_back(eval(*g, *rs));
      std::vector<AggInput> ins;
      for (auto& a : aggs) {
        AggInput in;
        if (a->name == "sum") in.func = SUM;
        else if (a->name == "count") in.func = COUNT;
        else if (a->name == "avg") in.func = AVG;
        else if (a->name == "min") in.func = MIN;
        else in.func = MAX;
        if (a->args[0]->kind != Expr::Star) in.arg = eval(*a->args[0], *rs);
        else if (in.func != COUNT) throw Error("* is only valid in count(*)");
        ins.push_back(in);
      }
      size_t nin = rs->n;
      AggOutput ao = hash_aggregate(keyv, ins, nin);
      auto as = std::make_unique<AggScope>();
      as->n = ao.n;
      for (size_t j = 0; j < s.group_by.size(); ++j) as->m[s.group_by[j]->str()] = ao.keys[j];
      for (size_t a = 0; a < aggs.size(); ++a) as->m[aggs[a]->str()] = ao.aggs[a];
      trace("hash aggregate: " + std::to_string(nin) + " rows -> " + std::to_string(ao.n) + " groups (" +
            std::to_string(s.group_by.size()) + " keys, " + std::to_string(aggs.size()) + " aggs, " + fmt_ms(t.ms()) + ")");
      res.stages.push_back(make_stat(StageOp::Aggregate, false, "", static_cast<double>(nin), 0,
                                     static_cast<double>(ao.n), static_cast<int>(s.group_by.size()),
                                     static_cast<int>(aggs.size()), t.ms(), agg_expr_nodes(pq)));
      scope = std::move(as);
      if (s.having) {
        VP b = eval(*s.having, *scope);
        scope = scope->gather(select_true(*b));
      }
    } else {
      scope = std::move(rs);
    }
  }

  // ---- ORDER BY / LIMIT ----
  if (!s.order_by.empty() || s.limit >= 0) {
    Timer t;
    std::vector<uint32_t> idx;
    if (!s.order_by.empty()) {
      std::vector<VP> keys;
      std::vector<bool> desc;
      for (auto& o : s.order_by) {
        keys.push_back(eval(*o.e, *scope));
        desc.push_back(o.desc);
      }
      idx = sort_perm(keys, desc, scope->n, s.limit);
      trace("sort " + std::to_string(scope->n) + " rows by " + std::to_string(keys.size()) + " keys" +
            (s.limit >= 0 ? " (top-" + std::to_string(s.limit) + ")" : "") + " (" + fmt_ms(t.ms()) + ")");
      res.stages.push_back(make_stat(StageOp::Sort, false, "", static_cast<double>(scope->n), 0,
                                     static_cast<double>(idx.size()), static_cast<int>(keys.size()), 0, t.ms()));
    } else {
      size_t k = std::min<size_t>(scope->n, static_cast<size_t>(s.limit));
      idx.resize(k);
      std::iota(idx.begin(), idx.end(), 0u);
    }
    scope = scope->gather(idx);
  }

  // ---- projection ----
  res.nrows = scope->n;
  Timer tproj;
  for (auto& it : s.items) {
    VP v = eval(*it.e, *scope);
    res.cols.push_back(materialize(v, res.nrows));
    std::string name = !it.alias.empty() ? it.alias : (it.e->kind == Expr::Col ? it.e->name : it.text);
    res.names.push_back(name);
  }
  res.stages.push_back(make_stat(StageOp::Project, false, "", static_cast<double>(res.nrows), 0,
                                 static_cast<double>(res.nrows), static_cast<int>(s.items.size()), 0, tproj.ms()));
  res.ms = total.ms();
  trace("total: " + std::to_string(res.nrows) + " rows in " + fmt_ms(res.ms));
  return res;
}

}  // namespace warpdb
