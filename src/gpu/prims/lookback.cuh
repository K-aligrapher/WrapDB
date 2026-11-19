#pragma once
// Decoupled look-back (Merrill & Garland, "Single-pass Parallel Prefix Scan with Decoupled Look-back",
// NVIDIA TR NVR-2016-002). Shared by scan and compaction.
//
// Each tile publishes a 64-bit status word {flag:32 | value:32} with ONE atomic store, so flag and
// value can never be observed out of sync:
//   X (0) = nothing yet,  A (1) = tile aggregate only,  P (2) = inclusive prefix of tiles 0..tile.
// A tile finds its exclusive prefix by walking backwards over predecessors (a warp at a time) until it
// meets a P. Tiles take work via an atomic ticket, so every tile a block waits on is already resident
// -> no deadlock regardless of scheduling order.
#include <cstdint>

#include "gpu/cuda_util.cuh"

namespace warpdb::gpu::detail {

constexpr uint32_t FLAG_X = 0, FLAG_A = 1, FLAG_P = 2;

__device__ __forceinline__ uint64_t ld_relaxed(const uint64_t* p) {
  uint64_t v;
  asm volatile("ld.relaxed.gpu.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_relaxed(uint64_t* p, uint64_t v) {
  asm volatile("st.relaxed.gpu.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ uint64_t pack(uint32_t flag, uint32_t value) {
  return (static_cast<uint64_t>(flag) << 32) | value;
}

// Executed by ONE full warp. Returns the exclusive prefix (sum of aggregates of tiles [0, tile)).
// `stride` is the distance between consecutive tiles' status words (1 for scan; 256 for per-digit sort).
__device__ __forceinline__ uint32_t warp_lookback(const uint64_t* status, uint32_t tile, uint32_t lane,
                                                  uint32_t stride = 1) {
  uint32_t prefix = 0;
  int32_t cur = static_cast<int32_t>(tile);
  for (;;) {
    int32_t j = cur - 1 - static_cast<int32_t>(lane);  // lane 0 = nearest predecessor
    uint32_t flag = FLAG_P, val = 0;                    // tiles before 0 count as an empty prefix
    if (j >= 0) {
      uint64_t s;
      do { s = ld_relaxed(status + static_cast<size_t>(j) * stride); } while ((s >> 32) == FLAG_X);
      flag = static_cast<uint32_t>(s >> 32);
      val = static_cast<uint32_t>(s);
    }
    __syncwarp();
    uint32_t pmask = __ballot_sync(FULL_MASK, flag == FLAG_P);
    int first_p = pmask ? (__ffs(pmask) - 1) : 32;  // nearest P among the 32 examined
    uint32_t contrib = (static_cast<int>(lane) <= first_p) ? val : 0;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) contrib += __shfl_xor_sync(FULL_MASK, contrib, o);
    prefix += contrib;
    if (pmask) return prefix;
    cur -= 32;
  }
}

}  // namespace warpdb::gpu::detail
