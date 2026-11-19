#pragma once
// Device-only building blocks of hash aggregation, shared by the AOT-compiled operators and by JIT-fused
// aggregate pipelines.
#include "gpu/prims/device_defs.cuh"

namespace warpdb::gpu {

enum AggOp { A_SUM = 0, A_MIN = 1, A_MAX = 2 };

template <int OP, class V>
__device__ __forceinline__ V agg_combine(V a, V b) {
  if constexpr (OP == A_SUM) return a + b;
  else if constexpr (OP == A_MIN) return b < a ? b : a;
  else return b > a ? b : a;
}

template <int OP, class V>
__device__ __forceinline__ void agg_atomic(V* p, V v) {
  if constexpr (is_same_type<V, int64_t>::value) {
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

// Which lanes of the warp share a group slot. Computed once per row and reused by every aggregate of the row.
struct WarpGroups {
  uint32_t peers;   // lanes (incl. this one) with the same slot as this lane
  uint32_t leaders; // one lane per distinct active slot in the warp (identical in all lanes)
  int ngroups;      // popcount(leaders)
};

// Must be called by every lane of the warp.
__device__ __forceinline__ WarpGroups warp_groups(uint32_t slot) {
  const uint32_t lane = threadIdx.x & 31;
  WarpGroups g;
  g.peers = __match_any_sync(FULL_MASK, slot);
  const bool leader = static_cast<uint32_t>(__ffs(g.peers) - 1) == lane && slot != EMPTY_SLOT;
  g.leaders = __ballot_sync(FULL_MASK, leader);
  g.ngroups = __popc(g.leaders);
  return g;
}

template <int OP, class V>
__device__ __forceinline__ V agg_identity() {
  if constexpr (OP == A_SUM) {
    return V(0);
  } else if constexpr (is_same_type<V, double>::value) {
    return OP == A_MIN ? V(1.7976931348623157e308) : V(-1.7976931348623157e308);
  } else {
    return OP == A_MIN ? V(0x7fffffffffffffffLL) : V(-0x7fffffffffffffffLL - 1);
  }
}

// acc[slot] (op)= v for lanes whose slot != EMPTY_SLOT. Every lane of the warp MUST call this (warp-wide
// intrinsics). Lanes hitting the same slot are pre-combined so that only one atomic per (warp, slot) reaches
// memory -- essential for low-cardinality GROUP BY and global aggregates, where all lanes would otherwise
// serialize on a handful of addresses. Each distinct group in the warp is reduced with a 5-step butterfly; when a
// warp holds many distinct groups the same-address contention is already low and plain atomics are used instead.
template <int OP, class V>
__device__ __forceinline__ void agg_apply(V* acc, uint32_t slot, V v, int warp_agg, const WarpGroups& wg) {
  if (!warp_agg || wg.ngroups > 4) {
    if (slot != EMPTY_SLOT) agg_atomic<OP, V>(acc + slot, v);
    return;
  }
  const uint32_t lane = threadIdx.x & 31;
  uint32_t todo = wg.leaders;  // uniform across the warp, so the loop below is convergent
  while (todo) {
    const int gl = __ffs(todo) - 1;
    todo &= todo - 1;
    const uint32_t gmask = __shfl_sync(FULL_MASK, wg.peers, gl);
    V x = ((gmask >> lane) & 1u) ? v : agg_identity<OP, V>();
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) x = agg_combine<OP, V>(x, __shfl_xor_sync(FULL_MASK, x, o));
    const uint32_t gslot = __shfl_sync(FULL_MASK, slot, gl);
    if (static_cast<int>(lane) == gl) agg_atomic<OP, V>(acc + gslot, x);
  }
}

// Convenience form for single-aggregate kernels.
template <int OP, class V>
__device__ __forceinline__ void agg_apply(V* acc, uint32_t slot, V v, int warp_agg) {
  WarpGroups wg = {0, 0, 0};
  if (warp_agg) wg = warp_groups(slot);
  agg_apply<OP, V>(acc, slot, v, warp_agg, wg);
}

// Multi-word-key group table used by fused aggregate kernels. Slot lifecycle in state[]: 0 empty -> 1 being
// published -> 2 ready. The thread that wins the CAS writes the K key words, fences, then marks the slot ready;
// other threads wait for "ready" and compare keys. Returns EMPTY_SLOT (and sets *overflow) if the table is full.
// Keys of word j live at skeys[j * cap + slot].
template <int K>
__device__ __forceinline__ uint32_t group_slot(uint32_t* state, int64_t* skeys, uint32_t cap, uint32_t mask,
                                               const int64_t* key, uint32_t* overflow) {
  uint64_t h = 0x9E3779B97F4A7C15ULL;
#pragma unroll
  for (int j = 0; j < K; ++j) h = hash64(h ^ static_cast<uint64_t>(key[j]));
  uint32_t s = static_cast<uint32_t>(h) & mask;
  volatile uint32_t* vstate = state;
  for (uint32_t probes = 0; probes <= mask; ++probes) {
    uint32_t st = vstate[s];
    if (st == 0u) {
      if (atomicCAS(&state[s], 0u, 1u) == 0u) {
#pragma unroll
        for (int j = 0; j < K; ++j) skeys[static_cast<size_t>(j) * cap + s] = key[j];
        __threadfence();
        atomicExch(&state[s], 2u);
        return s;
      }
      st = vstate[s];
    }
    while (st == 1u) st = vstate[s];  // another thread is publishing this slot's keys
    __threadfence();
    bool eq = true;
#pragma unroll
    for (int j = 0; j < K; ++j) eq = eq && (__ldcg(&skeys[static_cast<size_t>(j) * cap + s]) == key[j]);
    if (eq) return s;
    s = (s + 1) & mask;
  }
  atomicExch(overflow, 1u);
  return EMPTY_SLOT;
}

}  // namespace warpdb::gpu
