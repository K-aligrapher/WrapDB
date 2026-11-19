#pragma once
// Stable LSD radix sort, "onesweep" style (Adinets & Merrill, 2022): one kernel per 8-bit digit pass.
//
//  1. hist_kernel      : one read of the keys builds the histograms of ALL digit passes at once.
//  2. offsets_kernel   : exclusive scan of each pass's 256 bin counts -> global digit offsets.
//  3. sort_pass_kernel : per tile: rank keys with warp-level match_any (stable), derive the tile's
//                        per-digit prefix by decoupled look-back over earlier tiles, then scatter.
//
// Keys of type u32/i32/f32/u64/i64/f64 are supported; signed/float keys are mapped to an order-preserving
// unsigned bit pattern on the fly (no extra pass). Optional value payload (e.g. row ids).
#include <cstring>
#include <type_traits>

#include "gpu/prims/lookback.cuh"

namespace warpdb::gpu {

template <class K>
struct RadixTraits;
template <> struct RadixTraits<uint32_t> {
  using U = uint32_t;
  __host__ __device__ static U to_bits(uint32_t k) { return k; }
};
template <> struct RadixTraits<int32_t> {
  using U = uint32_t;
  __host__ __device__ static U to_bits(int32_t k) { return static_cast<uint32_t>(k) ^ 0x80000000u; }
};
template <> struct RadixTraits<float> {
  using U = uint32_t;
  __host__ __device__ static U to_bits(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return u ^ ((u >> 31) ? 0xFFFFFFFFu : 0x80000000u);
  }
};
template <> struct RadixTraits<uint64_t> {
  using U = uint64_t;
  __host__ __device__ static U to_bits(uint64_t k) { return k; }
};
template <> struct RadixTraits<int64_t> {
  using U = uint64_t;
  __host__ __device__ static U to_bits(int64_t k) { return static_cast<uint64_t>(k) ^ 0x8000000000000000ull; }
};
template <> struct RadixTraits<double> {
  using U = uint64_t;
  __host__ __device__ static U to_bits(double f) {
    uint64_t u;
    memcpy(&u, &f, 8);
    return u ^ ((u >> 63) ? 0xFFFFFFFFFFFFFFFFull : 0x8000000000000000ull);
  }
};

namespace detail {

constexpr int RS_BLOCK = 256;  // == number of digit values, so thread d owns digit d in the look-back
constexpr int RS_ITEMS = 16;   // per thread; tile = 4096 keys
constexpr int RS_MAX_PASSES = 8;

struct NoValue {};

template <class K, int BLOCK>
__global__ void __launch_bounds__(BLOCK) hist_kernel(const K* __restrict__ keys, size_t n, uint32_t* g_hist,
                                                     int begin_bit, int end_bit, int passes) {
  __shared__ uint32_t s_hist[RS_MAX_PASSES * 256];
  for (int i = threadIdx.x; i < passes * 256; i += BLOCK) s_hist[i] = 0;
  __syncthreads();
  size_t stride = static_cast<size_t>(gridDim.x) * BLOCK;
  for (size_t i = blockIdx.x * static_cast<size_t>(BLOCK) + threadIdx.x; i < n; i += stride) {
    auto bits = RadixTraits<K>::to_bits(keys[i]);
    for (int p = 0; p < passes; ++p) {
      int shift = begin_bit + 8 * p;
      int nb = min(8, end_bit - shift);
      atomicAdd(&s_hist[p * 256 + static_cast<int>((bits >> shift) & ((1u << nb) - 1u))], 1u);
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < passes * 256; i += BLOCK)
    if (s_hist[i]) atomicAdd(&g_hist[i], s_hist[i]);
}

// One block per pass: exclusive scan of 256 counters.
static __global__ void offsets_kernel(const uint32_t* g_hist, uint32_t* g_off) {
  __shared__ uint32_t s_w[8];
  const uint32_t t = threadIdx.x, lane = t & 31, warp = t >> 5;
  uint32_t v = g_hist[blockIdx.x * 256 + t], incl = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    uint32_t x = __shfl_up_sync(FULL_MASK, incl, o);
    if (lane >= static_cast<uint32_t>(o)) incl += x;
  }
  if (lane == 31) s_w[warp] = incl;
  __syncthreads();
  uint32_t woff = 0;
  for (uint32_t w = 0; w < warp; ++w) woff += s_w[w];
  g_off[blockIdx.x * 256 + t] = woff + incl - v;
}

// Keys wider than 4 bytes use smaller tiles so the staging buffers fit in static shared memory.
template <class K>
struct SortCfg {
  static constexpr int ITEMS = sizeof(K) > 4 ? 8 : 16;
  static constexpr int TILE = RS_BLOCK * ITEMS;
};

template <class K, class V, bool HAS_V, int BLOCK, int ITEMS>
__global__ void __launch_bounds__(BLOCK) sort_pass_kernel(const K* __restrict__ keys_in, K* __restrict__ keys_out,
                                                          const V* __restrict__ vals_in, V* __restrict__ vals_out,
                                                          size_t n, const uint32_t* __restrict__ digit_off,
                                                          uint64_t* status, uint32_t* counter, int shift, int nbits) {
  constexpr int NW = BLOCK / 32;
  constexpr int TILE = BLOCK * ITEMS;
  __shared__ uint32_t s_wc[NW][256];  // per-warp digit counts, later: exclusive prefix over warps
  __shared__ uint32_t s_dstart[256];  // first in-tile position of each digit (exclusive scan of tile histogram)
  __shared__ uint32_t s_dest[256];    // (global output offset of this tile's first element of digit d) - s_dstart[d]
  __shared__ uint32_t s_wsum[NW];
  __shared__ K s_keys[TILE];          // tile staged in digit order so global writes are coalesced runs
  __shared__ V s_vals[HAS_V ? TILE : 1];
  __shared__ uint32_t s_tile;
  if (threadIdx.x == 0) s_tile = atomicAdd(counter, 1u);
  const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int j = 0; j < 8; ++j) s_wc[warp][lane + 32 * j] = 0;
  __syncthreads();
  const uint32_t tile = s_tile;
  const size_t wbase = static_cast<size_t>(tile) * BLOCK * ITEMS + static_cast<size_t>(warp) * 32 * ITEMS;
  const uint32_t digit_mask = (1u << nbits) - 1u;
  const uint32_t lt = (1u << lane) - 1u;

  K key[ITEMS];
  V val[ITEMS];
  uint32_t dig[ITEMS], rank[ITEMS];
#pragma unroll
  for (int k = 0; k < ITEMS; ++k) {
    size_t i = wbase + static_cast<size_t>(k) * 32 + lane;
    bool valid = i < n;
    uint32_t d = 0xFFFFFFFFu;  // invalid lanes form their own match group and are ignored
    if (valid) {
      key[k] = keys_in[i];
      if constexpr (HAS_V) val[k] = vals_in[i];
      d = static_cast<uint32_t>((RadixTraits<K>::to_bits(key[k]) >> shift) & digit_mask);
    }
    dig[k] = d;
    uint32_t peers = __match_any_sync(FULL_MASK, d);
    uint32_t before = valid ? s_wc[warp][d] : 0u;
    __syncwarp();
    if (valid && (static_cast<uint32_t>(__ffs(peers) - 1) == lane)) s_wc[warp][d] = before + __popc(peers);
    __syncwarp();
    rank[k] = before + __popc(peers & lt);
  }
  __syncthreads();

  // Thread t owns digit t: prefix over warps, tile histogram, then look-back over earlier tiles.
  {
    const uint32_t d = threadIdx.x;
    uint32_t running = 0;
#pragma unroll
    for (int w = 0; w < NW; ++w) {
      uint32_t c = s_wc[w][d];
      s_wc[w][d] = running;
      running += c;
    }
    // exclusive scan of the tile histogram across digits -> where each digit's run starts inside the tile
    uint32_t incl = running;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      uint32_t x = __shfl_up_sync(FULL_MASK, incl, o);
      if (lane >= static_cast<uint32_t>(o)) incl += x;
    }
    if (lane == 31) s_wsum[warp] = incl;
    __syncthreads();
    uint32_t woff = 0;
    for (uint32_t w = 0; w < warp; ++w) woff += s_wsum[w];
    s_dstart[d] = woff + incl - running;
    uint64_t* my = status + static_cast<size_t>(tile) * 256 + d;
    uint32_t prefix = 0;
    if (tile == 0) {
      st_relaxed(my, pack(FLAG_P, running));
    } else {
      st_relaxed(my, pack(FLAG_A, running));
      for (int32_t j = static_cast<int32_t>(tile) - 1; j >= 0; --j) {
        uint64_t s;
        do { s = ld_relaxed(status + static_cast<size_t>(j) * 256 + d); } while ((s >> 32) == FLAG_X);
        prefix += static_cast<uint32_t>(s);
        if ((s >> 32) == FLAG_P) break;
      }
      st_relaxed(my, pack(FLAG_P, prefix + running));
    }
    s_dest[d] = digit_off[d] + prefix - s_dstart[d];  // modular arithmetic; final sum is < n
  }
  __syncthreads();

  // 1) scatter into shared memory at the element's position inside the digit-sorted tile
#pragma unroll
  for (int k = 0; k < ITEMS; ++k) {
    if (dig[k] != 0xFFFFFFFFu) {
      uint32_t t = s_dstart[dig[k]] + s_wc[warp][dig[k]] + rank[k];
      s_keys[t] = key[k];
      if constexpr (HAS_V) s_vals[t] = val[k];
    }
  }
  __syncthreads();

  // 2) write the tile out in order: consecutive threads hold consecutive elements of the same digit
  //    run, which land on consecutive global addresses -> coalesced stores.
  const size_t tile_base = static_cast<size_t>(tile) * TILE;
  const uint32_t cnt = static_cast<uint32_t>(min(static_cast<size_t>(TILE), n - tile_base));
#pragma unroll
  for (int j = 0; j < ITEMS; ++j) {
    uint32_t t = threadIdx.x + j * BLOCK;
    if (t < cnt) {
      K kk = s_keys[t];
      uint32_t d = static_cast<uint32_t>((RadixTraits<K>::to_bits(kk) >> shift) & digit_mask);
      size_t pos = static_cast<size_t>(static_cast<uint32_t>(s_dest[d] + t));
      keys_out[pos] = kk;
      if constexpr (HAS_V) vals_out[pos] = s_vals[t];
    }
  }
}

}  // namespace detail

// Reusable scratch memory so that sorting never allocates in the hot path.
class RadixWorkspace {
 public:
  void ensure(size_t n, size_t tile_elems = static_cast<size_t>(detail::RS_BLOCK) * detail::RS_ITEMS) {
    size_t tiles = ceil_div(n, tile_elems);
    size_t status_bytes = tiles * 256 * sizeof(uint64_t);
    size_t need = status_bytes + detail::RS_MAX_PASSES * 256 * 4 * 2 + detail::RS_MAX_PASSES * 4 + 64;
    buf_.resize(need);
    tiles_ = tiles;
    status_ = reinterpret_cast<uint64_t*>(buf_.data());
    hist_ = reinterpret_cast<uint32_t*>(buf_.data() + status_bytes);
    off_ = hist_ + detail::RS_MAX_PASSES * 256;
    counters_ = off_ + detail::RS_MAX_PASSES * 256;
  }
  uint64_t* status() { return status_; }
  uint32_t* hist() { return hist_; }
  uint32_t* off() { return off_; }
  uint32_t* counters() { return counters_; }
  size_t tiles() const { return tiles_; }
  size_t status_bytes() const { return tiles_ * 256 * sizeof(uint64_t); }

 private:
  DeviceBuffer<uint8_t> buf_;
  uint64_t* status_ = nullptr;
  uint32_t *hist_ = nullptr, *off_ = nullptr, *counters_ = nullptr;
  size_t tiles_ = 0;
};

// Sorts (keys_a[, vals_a]) ascending; keys_b/vals_b are same-sized ping-pong buffers. Only bits
// [begin_bit, end_bit) of the (order-mapped) key are compared. Returns true if the sorted result is in
// the *_b buffers, false if it is in the *_a buffers.
template <class K, class V = detail::NoValue, bool HAS_V = !std::is_same_v<V, detail::NoValue>>
bool radix_sort(K* keys_a, K* keys_b, V* vals_a, V* vals_b, size_t n, RadixWorkspace& ws, int begin_bit = 0,
                int end_bit = static_cast<int>(sizeof(K) * 8), cudaStream_t stream = 0) {
  using namespace detail;
  if (n <= 1) return false;
  const int passes = (end_bit - begin_bit + 7) / 8;
  if (passes <= 0 || passes > RS_MAX_PASSES) throw std::runtime_error("radix_sort: bad bit range");
  constexpr int ITEMS = SortCfg<K>::ITEMS;
  ws.ensure(n, SortCfg<K>::TILE);
  WDB_CUDA_CHECK(cudaMemsetAsync(ws.hist(), 0, passes * 256 * 4, stream));
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  unsigned hblocks = static_cast<unsigned>(std::min<size_t>(ceil_div(n, 256 * 8), static_cast<size_t>(sms) * 8));
  hist_kernel<K, 256><<<hblocks, 256, 0, stream>>>(keys_a, n, ws.hist(), begin_bit, end_bit, passes);
  offsets_kernel<<<passes, 256, 0, stream>>>(ws.hist(), ws.off());

  K *src_k = keys_a, *dst_k = keys_b;
  V *src_v = vals_a, *dst_v = vals_b;
  for (int p = 0; p < passes; ++p) {
    int shift = begin_bit + 8 * p;
    int nb = std::min(8, end_bit - shift);
    WDB_CUDA_CHECK(cudaMemsetAsync(ws.status(), 0, ws.status_bytes(), stream));
    WDB_CUDA_CHECK(cudaMemsetAsync(ws.counters() + p, 0, 4, stream));
    sort_pass_kernel<K, V, HAS_V, RS_BLOCK, ITEMS><<<static_cast<unsigned>(ws.tiles()), RS_BLOCK, 0, stream>>>(
        src_k, dst_k, src_v, dst_v, n, ws.off() + p * 256, ws.status(), ws.counters() + p, shift, nb);
    std::swap(src_k, dst_k);
    std::swap(src_v, dst_v);
  }
  WDB_CUDA_CHECK(cudaGetLastError());
  return passes % 2 == 1;  // odd number of swaps -> result sits in the b buffers
}

// Keys-only convenience overload.
template <class K>
bool radix_sort_keys(K* keys_a, K* keys_b, size_t n, RadixWorkspace& ws, int begin_bit = 0,
                     int end_bit = static_cast<int>(sizeof(K) * 8), cudaStream_t stream = 0) {
  return radix_sort<K, detail::NoValue, false>(keys_a, keys_b, nullptr, nullptr, n, ws, begin_bit, end_bit, stream);
}

}  // namespace warpdb::gpu
