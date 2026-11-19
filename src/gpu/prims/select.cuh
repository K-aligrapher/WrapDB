#pragma once
// Stream compaction (the GPU "filter").
//
// select_fused: ONE kernel does predicate evaluation, warp-ballot ranking, block scan and the
// decoupled-look-back for the tile's output offset, then scatters survivors. Input is read once, output
// written once. Output order is stable (same as input order).
//
// select_threepass: the textbook flags -> scan -> scatter pipeline, kept as a baseline to quantify the
// benefit of fusion.
#include "gpu/prims/scan.cuh"

namespace warpdb::gpu {

// ---- element generators: eval(i, out) returns whether row i survives and fills the value to write ----
template <class T, class Pred>
struct PredGen {  // keep values v = in[i] with pred(v)
  const T* in;
  Pred pred;
  __device__ __forceinline__ bool eval(size_t i, T& v) const {
    v = in[i];
    return pred(v);
  }
};

struct MaskIndexGen {  // keep row ids i where mask[i] != 0  (the engine's "selection vector")
  const uint8_t* mask;
  __device__ __forceinline__ bool eval(size_t i, uint32_t& v) const {
    v = static_cast<uint32_t>(i);
    return mask[i] != 0;
  }
};

namespace detail {

constexpr int SEL_BLOCK = 256;
constexpr int SEL_ITEMS = 16;  // per thread; tile = 4096 elements

template <int BLOCK, int ITEMS, class Gen, class Out>
__global__ void __launch_bounds__(BLOCK) select_kernel(Gen gen, size_t n, Out* __restrict__ out, uint64_t* status,
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

  // block-level: exclusive scan of per-warp counts, then publish + look-back (warp 0)
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

template <class T>
__global__ void flag_kernel(const T* in, uint8_t* flags, size_t n, T lo, T hi) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i < n) flags[i] = (in[i] >= lo && in[i] <= hi);
}

static __global__ void widen_kernel(const uint8_t* f, uint32_t* w, size_t n) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i < n) w[i] = f[i];
}

template <class T>
__global__ void scatter_kernel(const T* in, const uint8_t* flags, const uint32_t* pos, T* out, size_t n) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i < n && flags[i]) out[pos[i]] = in[i];
}

}  // namespace detail

inline size_t select_workspace_bytes(size_t n) {
  size_t tiles = ceil_div(n, static_cast<size_t>(detail::SEL_BLOCK) * detail::SEL_ITEMS);
  return tiles * sizeof(uint64_t) + 64;
}

// Compacts the rows accepted by `gen` into `out`; the number of survivors is written to *d_count.
template <class Gen, class Out>
void select_fused(Gen gen, size_t n, Out* out, uint32_t* d_count, void* workspace, cudaStream_t stream = 0) {
  using namespace detail;
  if (n == 0) {
    WDB_CUDA_CHECK(cudaMemsetAsync(d_count, 0, 4, stream));
    return;
  }
  size_t tiles = ceil_div(n, static_cast<size_t>(SEL_BLOCK) * SEL_ITEMS);
  auto* status = static_cast<uint64_t*>(workspace);
  auto* counter = reinterpret_cast<uint32_t*>(status + tiles);
  WDB_CUDA_CHECK(cudaMemsetAsync(workspace, 0, tiles * sizeof(uint64_t) + 4, stream));
  select_kernel<SEL_BLOCK, SEL_ITEMS><<<static_cast<unsigned>(tiles), SEL_BLOCK, 0, stream>>>(
      gen, n, out, status, counter, static_cast<uint32_t>(tiles - 1), d_count);
  WDB_CUDA_CHECK(cudaGetLastError());
}

// Baseline: keeps lo <= v <= hi using three kernels and 2 extra temporary arrays (n bytes + 4n bytes).
template <class T>
void select_threepass_range(const T* in, size_t n, T lo, T hi, T* out, uint32_t* d_count, uint8_t* tmp_flags,
                            uint32_t* tmp_pos, void* scan_workspace, cudaStream_t stream = 0) {
  using namespace detail;
  if (n == 0) return;
  unsigned blocks = static_cast<unsigned>(ceil_div(n, 256));
  flag_kernel<T><<<blocks, 256, 0, stream>>>(in, tmp_flags, n, lo, hi);
  widen_kernel<<<blocks, 256, 0, stream>>>(tmp_flags, tmp_pos, n);  // the scan consumes uint32
  scan_exclusive(tmp_pos, tmp_pos, n, scan_workspace, d_count, stream);
  scatter_kernel<T><<<blocks, 256, 0, stream>>>(in, tmp_flags, tmp_pos, out, n);
  WDB_CUDA_CHECK(cudaGetLastError());
}

}  // namespace warpdb::gpu
