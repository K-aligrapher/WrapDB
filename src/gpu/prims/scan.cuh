#pragma once
// Single-pass exclusive prefix sum over uint32 with decoupled look-back.
//   out[i] = in[0] + ... + in[i-1]      (in-place is allowed: out == in)
// Reads the input once and writes the output once (2N memory traffic, the lower bound).
#include "gpu/prims/lookback.cuh"

namespace warpdb::gpu {

namespace detail {

constexpr int SCAN_BLOCK = 256;
constexpr int SCAN_ITEMS = 16;  // per thread; tile = 4096 elements

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

template <int BLOCK, int ITEMS>
__global__ void __launch_bounds__(BLOCK) scan_kernel(const uint32_t* __restrict__ in, uint32_t* __restrict__ out,
                                                     size_t n, uint64_t* status, uint32_t* counter,
                                                     uint32_t last_tile, uint32_t* d_total) {
  __shared__ uint32_t s_tile;
  if (threadIdx.x == 0) s_tile = atomicAdd(counter, 1u);
  __syncthreads();
  const uint32_t tile = s_tile;
  const size_t base = static_cast<size_t>(tile) * BLOCK * ITEMS;
  const size_t mine = base + static_cast<size_t>(threadIdx.x) * ITEMS;
  const bool full = base + static_cast<size_t>(BLOCK) * ITEMS <= n;

  uint32_t v[ITEMS];
  if (full) {
    const uint4* p = reinterpret_cast<const uint4*>(in + mine);
#pragma unroll
    for (int i = 0; i < ITEMS / 4; ++i) {
      uint4 q = p[i];
      v[4 * i] = q.x; v[4 * i + 1] = q.y; v[4 * i + 2] = q.z; v[4 * i + 3] = q.w;
    }
  } else {
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) v[i] = (mine + i < n) ? in[mine + i] : 0u;
  }
  uint32_t sum = 0;
#pragma unroll
  for (int i = 0; i < ITEMS; ++i) {
    uint32_t t = v[i];
    v[i] = sum;
    sum += t;
  }
  uint32_t excl, prefix, total;
  block_tile_scan<BLOCK>(sum, tile, status, last_tile, d_total, excl, prefix, total);
  const uint32_t add = prefix + excl;
  if (full) {
    uint4* p = reinterpret_cast<uint4*>(out + mine);
#pragma unroll
    for (int i = 0; i < ITEMS / 4; ++i)
      p[i] = make_uint4(v[4 * i] + add, v[4 * i + 1] + add, v[4 * i + 2] + add, v[4 * i + 3] + add);
  } else {
#pragma unroll
    for (int i = 0; i < ITEMS; ++i)
      if (mine + i < n) out[mine + i] = v[i] + add;
  }
}

}  // namespace detail

// Scratch space needed by scan_exclusive (tile status words + ticket counter).
inline size_t scan_workspace_bytes(size_t n) {
  size_t tiles = ceil_div(n, static_cast<size_t>(detail::SCAN_BLOCK) * detail::SCAN_ITEMS);
  return tiles * sizeof(uint64_t) + 64;
}

// `workspace` must hold scan_workspace_bytes(n) bytes of device memory (8-byte aligned). If `d_total` is
// non-null it receives the sum of all inputs.
inline void scan_exclusive(const uint32_t* in, uint32_t* out, size_t n, void* workspace,
                           uint32_t* d_total = nullptr, cudaStream_t stream = 0) {
  if (n == 0) {
    if (d_total) WDB_CUDA_CHECK(cudaMemsetAsync(d_total, 0, 4, stream));
    return;
  }
  using namespace detail;
  size_t tiles = ceil_div(n, static_cast<size_t>(SCAN_BLOCK) * SCAN_ITEMS);
  auto* status = static_cast<uint64_t*>(workspace);
  auto* counter = reinterpret_cast<uint32_t*>(status + tiles);
  WDB_CUDA_CHECK(cudaMemsetAsync(workspace, 0, tiles * sizeof(uint64_t) + 4, stream));
  scan_kernel<SCAN_BLOCK, SCAN_ITEMS><<<static_cast<unsigned>(tiles), SCAN_BLOCK, 0, stream>>>(
      in, out, n, status, counter, static_cast<uint32_t>(tiles - 1), d_total);
  WDB_CUDA_CHECK(cudaGetLastError());
}

}  // namespace warpdb::gpu
