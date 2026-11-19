#pragma once
// CUDA kernels backing the GPU relational operators (operator-at-a-time, unfused baseline; the JIT
// fusion work in a later phase is measured against this).
#include <cmath>
#include <type_traits>

#include "gpu/memory.cuh"
#include "gpu/prims/hash_table.cuh"
#include "gpu/prims/radix_sort.cuh"
#include "gpu/prims/scan.cuh"
#include "gpu/prims/select.cuh"

namespace warpdb::gpu::ops {

constexpr uint32_t EMPTY_SLOT = 0xFFFFFFFFu;
inline unsigned grid_for(size_t n, unsigned block = 256) { return static_cast<unsigned>(ceil_div(n ? n : 1, block)); }

// ---- operands: a vector in device memory, or a broadcast scalar -----------------------------------
template <class T>
struct Vec {
  const T* p;
  __device__ __forceinline__ T operator[](size_t i) const { return p[i]; }
};
template <class T>
struct Sca {
  T v;
  __host__ __device__ __forceinline__ T operator[](size_t) const { return v; }
};

// ---- operators ----------------------------------------------------------------------------------------
struct OpAdd { template <class X, class Y> __host__ __device__ auto operator()(X x, Y y) const { return x + y; } };
struct OpSub { template <class X, class Y> __host__ __device__ auto operator()(X x, Y y) const { return x - y; } };
struct OpMul { template <class X, class Y> __host__ __device__ auto operator()(X x, Y y) const { return x * y; } };
struct OpDiv { template <class X, class Y> __host__ __device__ double operator()(X x, Y y) const { return static_cast<double>(x) / static_cast<double>(y); } };
struct OpMod {
  template <class X, class Y>
  __host__ __device__ auto operator()(X x, Y y) const {
    if constexpr (std::is_integral_v<X> && std::is_integral_v<Y>) return y ? static_cast<int64_t>(x % y) : int64_t(0);
    else return std::fmod(static_cast<double>(x), static_cast<double>(y));
  }
};
struct OpEq { template <class X, class Y> __host__ __device__ uint8_t operator()(X x, Y y) const { return x == y; } };
struct OpNe { template <class X, class Y> __host__ __device__ uint8_t operator()(X x, Y y) const { return x != y; } };
struct OpLt { template <class X, class Y> __host__ __device__ uint8_t operator()(X x, Y y) const { return x < y; } };
struct OpLe { template <class X, class Y> __host__ __device__ uint8_t operator()(X x, Y y) const { return x <= y; } };
struct OpGt { template <class X, class Y> __host__ __device__ uint8_t operator()(X x, Y y) const { return x > y; } };
struct OpGe { template <class X, class Y> __host__ __device__ uint8_t operator()(X x, Y y) const { return x >= y; } };
struct OpAnd { __host__ __device__ uint8_t operator()(uint8_t x, uint8_t y) const { return x & y; } };
struct OpOr { __host__ __device__ uint8_t operator()(uint8_t x, uint8_t y) const { return x | y; } };

template <class A, class B, class Op, class R>
__global__ void map2_k(A a, B b, Op op, R* out, size_t n) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i < n) out[i] = static_cast<R>(op(a[i], b[i]));
}
template <class R, class A, class B, class Op>
void map2(A a, B b, Op op, R* out, size_t n) {
  if (!n) return;
  map2_k<A, B, Op, R><<<grid_for(n), 256>>>(a, b, op, out, n);
  WDB_CUDA_CHECK(cudaGetLastError());
}

template <class T, class S>
__global__ void unary_neg_k(const S* in, T* out, size_t n) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i < n) out[i] = -static_cast<T>(in[i]);
}
template <class T>
__global__ void unary_abs_k(const T* in, T* out, size_t n) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i < n) out[i] = in[i] < 0 ? -in[i] : in[i];
}
static __global__ void not_k(const uint8_t* in, uint8_t* out, size_t n) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i < n) out[i] = !in[i];
}
// mask-by-dictionary-code (string predicates are evaluated on the dictionary on the host, then looked up)
static __global__ void mask_lookup_k(const int64_t* codes, const uint8_t* mask, uint8_t* out, size_t n) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i < n) out[i] = mask[codes[i]];
}
static __global__ void remap_k(const int64_t* codes, const int64_t* map, int64_t* out, size_t n) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i < n) out[i] = map[codes[i]];
}
template <class C, class A, class B, class R>
__global__ void where_k(C c, A a, B b, R* out, size_t n) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i < n) out[i] = c[i] ? static_cast<R>(a[i]) : static_cast<R>(b[i]);
}
// out[i] = c[i] ? a[i] : b[i]. Launched from a plain function template (nvcc does not reliably emit
// device code for kernels launched directly inside generic lambdas).
template <class R, class C, class A, class B>
void where(C c, A a, B b, R* out, size_t n) {
  if (!n) return;
  where_k<C, A, B, R><<<grid_for(n), 256>>>(c, a, b, out, n);
  WDB_CUDA_CHECK(cudaGetLastError());
}

__device__ inline void civil_from_days_d(int32_t z, int& y, unsigned& m, unsigned& d) {
  z += 719468;
  const int era = (z >= 0 ? z : z - 146096) / 146097;
  const unsigned doe = static_cast<unsigned>(z - era * 146097);
  const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  y = static_cast<int>(yoe) + era * 400;
  const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  const unsigned mp = (5 * doy + 2) / 153;
  d = doy - (153 * mp + 2) / 5 + 1;
  m = mp < 10 ? mp + 3 : mp - 9;
  y += (m <= 2);
}
static __global__ void date_part_k(const int64_t* in, int64_t* out, size_t n, int which) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i >= n) return;
  int y;
  unsigned m, d;
  civil_from_days_d(static_cast<int32_t>(in[i]), y, m, d);
  out[i] = which == 0 ? y : which == 1 ? static_cast<int64_t>(m) : static_cast<int64_t>(d);
}

// ---- gathers --------------------------------------------------------------------------------------------
template <class T, class S>
__global__ void gather_k(const S* __restrict__ src, const uint32_t* __restrict__ idx, T* __restrict__ out, size_t n) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i < n) out[i] = static_cast<T>(src[idx[i]]);
}
template <class T, class S>
void gather(const S* src, const uint32_t* idx, T* out, size_t n) {
  if (!n) return;
  gather_k<T, S><<<grid_for(n), 256>>>(src, idx, out, n);
  WDB_CUDA_CHECK(cudaGetLastError());
}
static __global__ void iota_k(uint32_t* out, size_t n) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i < n) out[i] = static_cast<uint32_t>(i);
}
static __global__ void widen_u8_k(const uint8_t* in, int64_t* out, size_t n) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i < n) out[i] = in[i];
}
template <class T>
__global__ void fill_k(T* p, size_t n, T v) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i < n) p[i] = v;
}

// ---- hash join (chained buckets; build side = smaller input) -----------------------------------------
constexpr int MAX_KEYS = 8;
struct KeyPtrs {
  const int64_t* p[MAX_KEYS];
  int K;
};
__device__ __forceinline__ uint64_t hash_row(const KeyPtrs& k, size_t r) {
  uint64_t h = 0x9E3779B97F4A7C15ULL;
  for (int j = 0; j < k.K; ++j) h = hash64(h ^ static_cast<uint64_t>(k.p[j][r]));
  return h;
}
__device__ __forceinline__ bool keys_eq(const KeyPtrs& a, size_t ra, const KeyPtrs& b, size_t rb) {
  for (int j = 0; j < a.K; ++j)
    if (a.p[j][ra] != b.p[j][rb]) return false;
  return true;
}
static __global__ void join_build_k(KeyPtrs bk, size_t bn, uint32_t* head, uint32_t* next, uint32_t mask) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i >= bn) return;
  uint32_t b = static_cast<uint32_t>(hash_row(bk, i)) & mask;
  next[i] = atomicExch(&head[b], static_cast<uint32_t>(i));
}
static __global__ void join_count_k(KeyPtrs bk, KeyPtrs pk, size_t pn, const uint32_t* head, const uint32_t* next,
                                    uint32_t mask, uint32_t* counts) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i >= pn) return;
  uint32_t c = 0;
  for (uint32_t b = head[static_cast<uint32_t>(hash_row(pk, i)) & mask]; b != EMPTY_SLOT; b = next[b])
    c += keys_eq(bk, b, pk, i);
  counts[i] = c;
}
static __global__ void join_write_k(KeyPtrs bk, KeyPtrs pk, size_t pn, const uint32_t* head, const uint32_t* next,
                                    uint32_t mask, const uint32_t* offs, uint32_t* out_probe, uint32_t* out_build) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i >= pn) return;
  uint32_t pos = offs[i];
  for (uint32_t b = head[static_cast<uint32_t>(hash_row(pk, i)) & mask]; b != EMPTY_SLOT; b = next[b])
    if (keys_eq(bk, b, pk, i)) {
      out_probe[pos] = static_cast<uint32_t>(i);
      out_build[pos] = b;
      ++pos;
    }
}

// ---- hash aggregation -----------------------------------------------------------------------------------
// rep[slot] = representative row of the group that owns the slot (EMPTY_SLOT if free). A row claims a slot
// with atomicCAS(EMPTY -> row); other rows with equal keys find that representative and reuse the slot.
static __global__ void agg_insert_k(KeyPtrs keys, size_t n, uint32_t* rep, uint32_t mask, uint32_t* row_slot,
                                    uint32_t* overflow) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i >= n) return;
  uint32_t s = static_cast<uint32_t>(hash_row(keys, i)) & mask;
  for (uint32_t probes = 0; probes <= mask; ++probes) {
    uint32_t cur = rep[s];
    if (cur == EMPTY_SLOT) {
      uint32_t old = atomicCAS(&rep[s], EMPTY_SLOT, static_cast<uint32_t>(i));
      cur = (old == EMPTY_SLOT) ? static_cast<uint32_t>(i) : old;
    }
    if (cur == i || keys_eq(keys, cur, keys, i)) {
      row_slot[i] = s;
      return;
    }
    s = (s + 1) & mask;
  }
  atomicExch(overflow, 1u);
  row_slot[i] = 0;
}

enum AggOp { A_SUM = 0, A_MIN = 1, A_MAX = 2 };

template <int OP, class V>
__device__ __forceinline__ V agg_combine(V a, V b) {
  if constexpr (OP == A_SUM) return a + b;
  else if constexpr (OP == A_MIN) return b < a ? b : a;
  else return b > a ? b : a;
}
template <int OP, class V>
__device__ __forceinline__ void agg_atomic(V* p, V v) {
  if constexpr (std::is_same_v<V, int64_t>) {
    if constexpr (OP == A_SUM) atomicAdd(reinterpret_cast<unsigned long long*>(p), static_cast<unsigned long long>(v));
    else if constexpr (OP == A_MIN) atomicMin(reinterpret_cast<long long*>(p), static_cast<long long>(v));
    else atomicMax(reinterpret_cast<long long*>(p), static_cast<long long>(v));
  } else {
    if constexpr (OP == A_SUM) {
      atomicAdd(p, v);
    } else {
      auto* ull = reinterpret_cast<unsigned long long*>(p);
      unsigned long long old = *ull, assumed;
      do {
        assumed = old;
        double cur = __longlong_as_double(static_cast<long long>(assumed));
        if (OP == A_MIN ? !(v < cur) : !(v > cur)) break;
        old = atomicCAS(ull, assumed, static_cast<unsigned long long>(__double_as_longlong(v)));
      } while (old != assumed);
    }
  }
}

// acc[slot] (op)= value(row). With warp_agg, lanes of a warp that hit the same slot are pre-combined with
// shuffles so only one atomic per (warp, slot) reaches memory -- essential for low-cardinality GROUP BY
// and global aggregates, where all lanes would otherwise serialize on a handful of addresses.
template <int OP, class V, class Arg>
__global__ void agg_update_k(Arg arg, const uint32_t* row_slot, size_t n, V* acc, int warp_agg) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const bool valid = i < n;
  uint32_t slot = valid ? (row_slot ? row_slot[i] : 0u) : EMPTY_SLOT;
  V v = valid ? static_cast<V>(arg[i]) : V(0);
  if (!warp_agg) {
    if (valid) agg_atomic<OP, V>(acc + slot, v);
    return;
  }
  const uint32_t lane = threadIdx.x & 31;
  uint32_t peers = __match_any_sync(FULL_MASK, slot);
  V r = v;  // the lowest lane of each peer group ends up holding the combination of the whole group
  for (int j = 0; j < 32; ++j) {
    V t = __shfl_sync(FULL_MASK, v, j);  // executed by all lanes, every iteration (uniform control flow)
    if (((peers >> j) & 1u) && j != static_cast<int>(lane)) r = agg_combine<OP, V>(r, t);
  }
  if (valid && static_cast<uint32_t>(__ffs(peers) - 1) == lane) agg_atomic<OP, V>(acc + slot, r);
}

struct SlotGen {  // enumerates occupied hash slots for compaction
  const uint32_t* rep;
  __device__ __forceinline__ bool eval(size_t i, uint32_t& v) const {
    v = static_cast<uint32_t>(i);
    return rep[i] != EMPTY_SLOT;
  }
};

// ---- sort-key preparation ----------------------------------------------------------------------------------
static __global__ void sortkey_i64_k(const int64_t* in, uint64_t* out, size_t n, int desc) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i >= n) return;
  uint64_t b = static_cast<uint64_t>(in[i]) ^ 0x8000000000000000ull;
  out[i] = desc ? ~b : b;
}
static __global__ void sortkey_f64_k(const double* in, uint64_t* out, size_t n, int desc) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i >= n) return;
  uint64_t b = RadixTraits<double>::to_bits(in[i]);
  out[i] = desc ? ~b : b;
}
static __global__ void sortkey_u8_k(const uint8_t* in, uint64_t* out, size_t n, int desc) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i >= n) return;
  uint64_t b = in[i];
  out[i] = desc ? ~b : b;
}

}  // namespace warpdb::gpu::ops
