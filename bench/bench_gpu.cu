// Micro-benchmarks: WarpDB's hand-written GPU primitives vs NVIDIA CUB.
// CUB is used ONLY here, as the reference; the engine never links against it.
//   warpdb_bench [--quick]
#include <cub/cub.cuh>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "gpu/prims/hash_table.cuh"
#include "gpu/prims/radix_sort.cuh"
#include "gpu/prims/select.cuh"

using namespace warpdb::gpu;

namespace {

double g_peak_gbs = 0;

struct ModLess {
  uint32_t p;
  __host__ __device__ bool operator()(uint32_t v) const { return v % 100 < p; }
};

// median over `iters` timed runs; `prep` (untimed) runs before each.
template <class Prep, class Run>
double time_ms(Prep&& prep, Run&& run, int iters = 15, int warm = 3) {
  GpuTimer t;
  std::vector<float> ms;
  for (int i = 0; i < warm + iters; ++i) {
    prep();
    WDB_CUDA_CHECK(cudaDeviceSynchronize());
    t.start();
    run();
    float m = t.stop();
    if (i >= warm) ms.push_back(m);
  }
  std::sort(ms.begin(), ms.end());
  return ms[ms.size() / 2];
}

void header(const char* title, const char* cols) {
  std::printf("\n### %s\n\n%s\n", title, cols);
}

std::vector<uint32_t> rand_u32(size_t n, uint32_t mod, uint64_t seed) {
  std::mt19937_64 g(seed);
  std::vector<uint32_t> v(n);
  for (auto& x : v) x = mod ? static_cast<uint32_t>(g() % mod) : static_cast<uint32_t>(g());
  return v;
}

template <class T>
void fill_device(DeviceBuffer<T>& d, const std::vector<T>& h) { d.upload(h); }

// ------------------------------------------------------------------------------------------
void bench_scan(size_t n) {
  header("Prefix scan (exclusive, uint32)", "| impl | ms | GB/s | % of peak BW | vs CUB |\n|---|---|---|---|---|");
  DeviceBuffer<uint32_t> in, out(n);
  in.upload(rand_u32(n, 100, 1));
  DeviceBuffer<uint8_t> ws(scan_workspace_bytes(n));
  size_t cub_bytes = 0;
  cub::DeviceScan::ExclusiveSum(nullptr, cub_bytes, in.data(), out.data(), static_cast<int>(n));
  DeviceBuffer<uint8_t> cub_tmp(cub_bytes);

  double bytes = 2.0 * 4 * n;
  double mine = time_ms([] {}, [&] { scan_exclusive(in.data(), out.data(), n, ws.data()); });
  double ref = time_ms([] {}, [&] {
    cub::DeviceScan::ExclusiveSum(cub_tmp.data(), cub_bytes, in.data(), out.data(), static_cast<int>(n));
  });
  auto row = [&](const char* name, double ms, double vs) {
    double gbs = bytes / (ms * 1e6);
    std::printf("| %s | %.3f | %.1f | %.0f%% | %s |\n", name, ms, gbs, 100 * gbs / g_peak_gbs, vs > 0 ? (std::to_string(vs).substr(0, 4) + "x").c_str() : "-");
  };
  row("WarpDB decoupled look-back", mine, ref / mine);
  row("CUB DeviceScan", ref, -1);
}

void bench_select(size_t n) {
  header("Stream compaction / filter (uint32, predicate v%100 < p)",
         "| selectivity | impl | ms | GB/s (read+write) | vs CUB |\n|---|---|---|---|---|");
  auto h = rand_u32(n, 0, 2);
  DeviceBuffer<uint32_t> in, out(n), cnt(1), pos(n);
  DeviceBuffer<uint8_t> flags(n);
  in.upload(h);
  DeviceBuffer<uint8_t> ws(select_workspace_bytes(n)), ws2(scan_workspace_bytes(n));
  for (uint32_t p : {1u, 50u, 99u}) {
    size_t cub_bytes = 0;
    cub::DeviceSelect::If(nullptr, cub_bytes, in.data(), out.data(), cnt.data(), static_cast<int>(n), ModLess{p});
    DeviceBuffer<uint8_t> cub_tmp(cub_bytes);
    double sel = p / 100.0;
    double bytes = 4.0 * n + 4.0 * n * sel;
    double fused = time_ms([] {}, [&] {
      select_fused(PredGen<uint32_t, ModLess>{in.data(), ModLess{p}}, n, out.data(), cnt.data(), ws.data());
    });
    double three = time_ms([] {}, [&] {
      select_threepass_range<uint32_t>(in.data(), n, 0, 0, out.data(), cnt.data(), flags.data(), pos.data(), ws2.data());
    });
    double ref = time_ms([] {}, [&] {
      cub::DeviceSelect::If(cub_tmp.data(), cub_bytes, in.data(), out.data(), cnt.data(), static_cast<int>(n), ModLess{p});
    });
    auto row = [&](const char* name, double ms, bool with_vs) {
      std::printf("| %u%% | %s | %.3f | %.1f | %s |\n", p, name, ms, bytes / (ms * 1e6),
                  with_vs ? (std::to_string(ref / ms).substr(0, 4) + "x").c_str() : "-");
    };
    row("WarpDB fused (ballot + look-back)", fused, true);
    row("WarpDB 3-pass (flags, scan, scatter)", three, true);
    row("CUB DeviceSelect::If", ref, false);
  }
}

template <class K, class V = detail::NoValue>
void bench_sort_one(const char* label, size_t n, const std::vector<K>& h) {
  constexpr bool HAS_V = !std::is_same_v<V, detail::NoValue>;
  DeviceBuffer<K> orig_k, a_k(n), b_k(n), c_k(n), d_k(n);
  orig_k.upload(h);
  DeviceBuffer<V> orig_v, a_v, b_v, c_v, d_v;
  if constexpr (HAS_V) {
    std::vector<V> iv(n);
    std::iota(iv.begin(), iv.end(), V(0));
    orig_v.upload(iv);
    a_v.resize(n); b_v.resize(n); c_v.resize(n); d_v.resize(n);
  }
  RadixWorkspace ws;
  ws.ensure(n, detail::SortCfg<K>::TILE);
  auto restore = [&] {
    WDB_CUDA_CHECK(cudaMemcpy(a_k.data(), orig_k.data(), n * sizeof(K), cudaMemcpyDeviceToDevice));
    WDB_CUDA_CHECK(cudaMemcpy(c_k.data(), orig_k.data(), n * sizeof(K), cudaMemcpyDeviceToDevice));
    if constexpr (HAS_V) {
      WDB_CUDA_CHECK(cudaMemcpy(a_v.data(), orig_v.data(), n * sizeof(V), cudaMemcpyDeviceToDevice));
      WDB_CUDA_CHECK(cudaMemcpy(c_v.data(), orig_v.data(), n * sizeof(V), cudaMemcpyDeviceToDevice));
    }
  };
  double mine = time_ms(restore, [&] {
    radix_sort<K, V, HAS_V>(a_k.data(), b_k.data(), a_v.data(), b_v.data(), n, ws);
  });
  size_t cub_bytes = 0;
  if constexpr (HAS_V) cub::DeviceRadixSort::SortPairs(nullptr, cub_bytes, c_k.data(), d_k.data(), c_v.data(), d_v.data(), static_cast<int>(n));
  else cub::DeviceRadixSort::SortKeys(nullptr, cub_bytes, c_k.data(), d_k.data(), static_cast<int>(n));
  DeviceBuffer<uint8_t> tmp(cub_bytes);
  double ref = time_ms(restore, [&] {
    if constexpr (HAS_V) cub::DeviceRadixSort::SortPairs(tmp.data(), cub_bytes, c_k.data(), d_k.data(), c_v.data(), d_v.data(), static_cast<int>(n));
    else cub::DeviceRadixSort::SortKeys(tmp.data(), cub_bytes, c_k.data(), d_k.data(), static_cast<int>(n));
  });
  std::printf("| %s | %.2f | %.0f | %.2f | %.0f | %.2fx |\n", label, mine, n / (mine * 1e3), ref, n / (ref * 1e3), ref / mine);
}

void bench_sort(size_t n) {
  header("Radix sort (random keys)", "| keys | WarpDB ms | WarpDB Mkeys/s | CUB ms | CUB Mkeys/s | WarpDB vs CUB |\n|---|---|---|---|---|---|");
  std::mt19937_64 g(5);
  std::vector<uint32_t> k32(n);
  std::vector<uint64_t> k64(n);
  std::vector<int64_t> s64(n);
  for (size_t i = 0; i < n; ++i) { k32[i] = static_cast<uint32_t>(g()); k64[i] = g(); s64[i] = static_cast<int64_t>(g() % 1000000) - 500000; }
  bench_sort_one<uint32_t>("uint32", n, k32);
  bench_sort_one<uint32_t, uint32_t>("uint32 + uint32 row id", n, k32);
  bench_sort_one<uint64_t>("uint64", n, k64);
  bench_sort_one<uint64_t, uint32_t>("uint64 + uint32 row id", n, k64);
  bench_sort_one<int64_t, uint32_t>("int64 (small range) + row id", n, s64);
}

// ---- hash table vs sort-based distinct counting ----
void bench_hash(size_t n) {
  header("Hash table insert (int64 keys) vs sort-based distinct", "| distinct keys | hash insert ms | M inserts/s | sort+RLE ms | speedup |\n|---|---|---|---|---|");
  for (uint64_t distinct : {uint64_t(1000), uint64_t(1) << 20, uint64_t(1) << 24}) {
    std::mt19937_64 g(distinct);
    std::vector<int64_t> h(n);
    for (auto& k : h) k = static_cast<int64_t>(g() % distinct);
    DeviceBuffer<int64_t> keys, sorted(n), tmp_in(n);
    keys.upload(h);
    DeviceBuffer<uint32_t> slots(n);
    HashTable t(distinct);
    double mine = time_ms([&] { t.clear(); }, [&] { t.insert(keys.data(), n, slots.data()); });

    DeviceBuffer<int64_t> uniq(n);
    DeviceBuffer<int> counts(n), nruns(1);
    size_t b1 = 0, b2 = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, b1, tmp_in.data(), sorted.data(), static_cast<int>(n));
    cub::DeviceRunLengthEncode::Encode(nullptr, b2, sorted.data(), uniq.data(), counts.data(), nruns.data(), static_cast<int>(n));
    DeviceBuffer<uint8_t> t1(b1), t2(b2);
    double ref = time_ms(
        [&] { WDB_CUDA_CHECK(cudaMemcpy(tmp_in.data(), keys.data(), n * 8, cudaMemcpyDeviceToDevice)); },
        [&] {
          cub::DeviceRadixSort::SortKeys(t1.data(), b1, tmp_in.data(), sorted.data(), static_cast<int>(n));
          cub::DeviceRunLengthEncode::Encode(t2.data(), b2, sorted.data(), uniq.data(), counts.data(), nruns.data(), static_cast<int>(n));
        });
    std::printf("| %llu | %.2f | %.0f | %.2f | %.2fx |\n", static_cast<unsigned long long>(distinct), mine, n / (mine * 1e3), ref, ref / mine);
  }
}

}  // namespace

int main(int argc, char** argv) {
  bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
  cudaDeviceProp p{};
  cudaGetDeviceProperties(&p, 0);
  int mem_khz = 0, bus = 0;
  cudaDeviceGetAttribute(&mem_khz, cudaDevAttrMemoryClockRate, 0);
  cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, 0);
  g_peak_gbs = 2.0 * mem_khz * 1e3 * (bus / 8.0) / 1e9;
  std::printf("# WarpDB GPU primitive benchmarks\n\nGPU: %s (sm_%d%d), %.0f GB/s peak DRAM bandwidth, median of 15 runs\n",
              p.name, p.major, p.minor, g_peak_gbs);

  size_t n_stream = quick ? (1u << 24) : (1u << 26);
  size_t n_sort = quick ? (1u << 22) : (1u << 25);
  size_t n_hash = quick ? (1u << 22) : (1u << 24);
  std::printf("\nsizes: stream ops n=%zu, sort n=%zu, hash n=%zu\n", n_stream, n_sort, n_hash);
  bench_scan(n_stream);
  bench_select(n_stream);
  bench_sort(n_sort);
  bench_hash(n_hash);
  return 0;
}
