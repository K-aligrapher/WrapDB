#pragma once
// Device-only block/tile scan used by the scan and compaction kernels (and by JIT-generated kernels).
#include "gpu/prims/lookback.cuh"

namespace warpdb::gpu::detail {

// Block-wide: given each thread's total, computes exclusive offsets inside the block, publishes the tile
// aggregate and resolves the tile's global prefix with look-back. Returns via out-params.
//   thread_total -> (excl_in_block, tile_prefix, tile_total)
template <int BLOCK>
__device__ __forceinline__ void block_tile_scan(uint32_t thread_total, uint32_t tile, uint64_t* status,
                                                uint32_t last_tile, uint32_t* d_total, uint32_t& excl_in_block,
                                                uint32_t& tile_prefix, uint32_t& tile_total) {
  constexpr int NW = BLOCK / 32;
  __shared__ uint32_t s_warp[NW];
  __shared__ uint32_t s_prefix, s_total;
  const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5;

  uint32_t incl = thread_total;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    uint32_t t = __shfl_up_sync(FULL_MASK, incl, o);
    if (lane >= static_cast<uint32_t>(o)) incl += t;
  }
  if (lane == 31) s_warp[warp] = incl;
  __syncthreads();

  if (warp == 0) {
    uint32_t wv = lane < NW ? s_warp[lane] : 0;
    uint32_t wi = wv;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      uint32_t t = __shfl_up_sync(FULL_MASK, wi, o);
      if (lane >= static_cast<uint32_t>(o)) wi += t;
    }
    uint32_t total = __shfl_sync(FULL_MASK, wi, NW - 1);
    if (lane < NW) s_warp[lane] = wi - wv;  // exclusive warp offsets
    uint32_t prefix = 0;
    if (tile == 0) {
      if (lane == 0) st_relaxed(status, pack(FLAG_P, total));
    } else {
      if (lane == 0) st_relaxed(status + tile, pack(FLAG_A, total));
      prefix = warp_lookback(status, tile, lane);
      if (lane == 0) st_relaxed(status + tile, pack(FLAG_P, prefix + total));
    }
    if (lane == 0) {
      s_prefix = prefix;
      s_total = total;
      if (tile == last_tile && d_total) *d_total = prefix + total;
    }
  }
  __syncthreads();
  excl_in_block = s_warp[warp] + (incl - thread_total);
  tile_prefix = s_prefix;
  tile_total = s_total;
}

}  // namespace warpdb::gpu::detail
