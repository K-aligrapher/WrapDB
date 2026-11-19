#pragma once
// Device-only body of the fused filter / stream-compaction kernel. The host-compiled select_kernel and
// JIT-generated filter kernels both call select_tile with a different "generator" (the predicate).
//
// Gen contract:   __device__ bool eval(size_t i, Out& value) const
//   returns whether row i survives and writes the value to emit for it.
#include "gpu/prims/tile_scan.cuh"

namespace warpdb::gpu::detail {

template <int BLOCK, int ITEMS, class Gen, class Out>
__device__ __forceinline__ void select_tile(const Gen& gen, size_t n, Out* __restrict__ out, uint64_t* status,
                                            uint32_t* counter, uint32_t last_tile, uint32_t* d_count) {
  __shared__ uint32_t s_tile;
  if (threadIdx.x == 0) s_tile = atomicAdd(counter, 1u);
  __syncthreads();
  const uint32_t tile = s_tile;
  const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  // Warp-striped layout: iteration k of a warp covers 32 consecutive elements -> perfectly coalesced
  // loads, and (k, lane) order == input order, so ranking by ballot keeps the output stable.
  const size_t wbase = static_cast<size_t>(tile) * BLOCK * ITEMS + static_cast<size_t>(warp) * 32 * ITEMS;

  uint32_t ballots[ITEMS];
  Out vals[ITEMS];
  uint32_t warp_count = 0;
#pragma unroll
  for (int k = 0; k < ITEMS; ++k) {
    size_t i = wbase + static_cast<size_t>(k) * 32 + lane;
    bool keep = false;
    if (i < n) keep = gen.eval(i, vals[k]);
    ballots[k] = __ballot_sync(FULL_MASK, keep);
    warp_count += __popc(ballots[k]);
  }

  // Only lane 0 contributes the warp's count, so the block scan below is a scan over warps; lane 0's
  // exclusive offset is then the warp's offset inside the tile.
  uint32_t excl_in_block, tile_prefix, tile_total;
  block_tile_scan<BLOCK>(lane == 0 ? warp_count : 0u, tile, status, last_tile, d_count, excl_in_block, tile_prefix,
                         tile_total);
  uint32_t warp_off = __shfl_sync(FULL_MASK, excl_in_block, 0);
  uint32_t pos = tile_prefix + warp_off;
  const uint32_t lt = (1u << lane) - 1u;
#pragma unroll
  for (int k = 0; k < ITEMS; ++k) {
    if (ballots[k] & (1u << lane)) out[pos + __popc(ballots[k] & lt)] = vals[k];
    pos += __popc(ballots[k]);
  }
}

}  // namespace warpdb::gpu::detail
