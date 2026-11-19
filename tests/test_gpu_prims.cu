// Correctness tests for the hand-written GPU primitives (scan, select, radix sort, hash table).
#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <random>
#include <unordered_set>
#include <vector>

#include "gpu/prims/hash_table.cuh"
#include "gpu/prims/radix_sort.cuh"
#include "gpu/prims/select.cuh"

using namespace warpdb::gpu;

namespace {

// Sizes chosen around tile boundaries (4096), warp boundaries and "big".
const std::vector<size_t> kSizes = {0, 1, 2, 31, 32, 33, 255, 256, 257, 4095, 4096, 4097, 8191, 8192, 8193,
                                    100000, (1u << 20) + 5, 3 * 4096 * 17 + 11, 1u << 24};

std::vector<uint32_t> rand_u32(size_t n, uint32_t mod, uint64_t seed) {
  std::mt19937_64 g(seed);
  std::vector<uint32_t> v(n);
  for (auto& x : v) x = mod ? static_cast<uint32_t>(g() % mod) : static_cast<uint32_t>(g());
  return v;
}

// ---------------------------------------------------------------- scan
TEST(Scan, MatchesHostForManySizes) {
  for (size_t n : kSizes) {
    auto h = rand_u32(n, 100, n + 1);
    DeviceBuffer<uint32_t> in, out(n), total(1);
    in.upload(h);
    DeviceBuffer<uint8_t> ws(scan_workspace_bytes(n));
    scan_exclusive(in.data(), out.data(), n, ws.data(), total.data());
    WDB_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint32_t> want(n);
    uint32_t run = 0;
    for (size_t i = 0; i < n; ++i) { want[i] = run; run += h[i]; }
    EXPECT_EQ(out.download(n), want) << "n=" << n;
    EXPECT_EQ(total.download(1)[0], run) << "n=" << n;
  }
}

TEST(Scan, InPlaceAndAllOnes) {
  size_t n = 5 * 4096 + 17;
  std::vector<uint32_t> h(n, 1);
  DeviceBuffer<uint32_t> buf;
  buf.upload(h);
  DeviceBuffer<uint8_t> ws(scan_workspace_bytes(n));
  scan_exclusive(buf.data(), buf.data(), n, ws.data());
  WDB_CUDA_CHECK(cudaDeviceSynchronize());
  auto got = buf.download(n);
  for (size_t i = 0; i < n; ++i) ASSERT_EQ(got[i], i);
}

TEST(Scan, RepeatedRunsAreDeterministic) {  // look-back races would show up as flaky results
  size_t n = 1 << 22;
  auto h = rand_u32(n, 7, 99);
  DeviceBuffer<uint32_t> in, out(n);
  in.upload(h);
  DeviceBuffer<uint8_t> ws(scan_workspace_bytes(n));
  std::vector<uint32_t> first;
  for (int r = 0; r < 20; ++r) {
    scan_exclusive(in.data(), out.data(), n, ws.data());
    WDB_CUDA_CHECK(cudaDeviceSynchronize());
    auto got = out.download(n);
    if (r == 0) first = got;
    else ASSERT_EQ(got, first) << "run " << r;
  }
}

// ---------------------------------------------------------------- select
struct ModLess {
  uint32_t p;
  __device__ bool operator()(uint32_t v) const { return v % 100 < p; }
};

struct InRange {
  uint32_t lo, hi;
  __device__ bool operator()(uint32_t v) const { return v >= lo && v <= hi; }
};

TEST(Select, FusedValuesStableAcrossSelectivities) {
  for (size_t n : kSizes) {
    auto h = rand_u32(n, 0, n + 7);
    DeviceBuffer<uint32_t> in, out(n + 1), cnt(1);
    in.upload(h);
    DeviceBuffer<uint8_t> ws(select_workspace_bytes(n));
    for (uint32_t p : {0u, 1u, 50u, 99u, 100u}) {
      select_fused(PredGen<uint32_t, ModLess>{in.data(), ModLess{p}}, n, out.data(), cnt.data(), ws.data());
      WDB_CUDA_CHECK(cudaDeviceSynchronize());
      std::vector<uint32_t> want;
      for (auto v : h) if (v % 100 < p) want.push_back(v);
      uint32_t got_n = cnt.download(1)[0];
      ASSERT_EQ(got_n, want.size()) << "n=" << n << " p=" << p;
      ASSERT_EQ(out.download(got_n), want) << "n=" << n << " p=" << p;
    }
  }
}

TEST(Select, MaskToRowIds) {
  size_t n = (1 << 20) + 333;
  std::mt19937 g(5);
  std::vector<uint8_t> mask(n);
  for (auto& m : mask) m = (g() % 10) == 0;
  DeviceBuffer<uint8_t> dm;
  dm.upload(mask);
  DeviceBuffer<uint32_t> out(n), cnt(1);
  DeviceBuffer<uint8_t> ws(select_workspace_bytes(n));
  select_fused(MaskIndexGen{dm.data()}, n, out.data(), cnt.data(), ws.data());
  WDB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint32_t> want;
  for (size_t i = 0; i < n; ++i) if (mask[i]) want.push_back(static_cast<uint32_t>(i));
  uint32_t got_n = cnt.download(1)[0];
  ASSERT_EQ(got_n, want.size());
  EXPECT_EQ(out.download(got_n), want);
}

TEST(Select, ThreePassBaselineAgreesWithFused) {
  size_t n = 3 * 4096 * 5 + 99;
  auto h = rand_u32(n, 1000, 3);
  DeviceBuffer<uint32_t> in, a(n), b(n), ca(1), cb(1), pos(n);
  DeviceBuffer<uint8_t> flags(n), ws(select_workspace_bytes(n)), ws2(scan_workspace_bytes(n));
  in.upload(h);
  select_fused(PredGen<uint32_t, InRange>{in.data(), InRange{100, 400}}, n, a.data(), ca.data(), ws.data());
  select_threepass_range<uint32_t>(in.data(), n, 100, 400, b.data(), cb.data(), flags.data(), pos.data(), ws2.data());
  WDB_CUDA_CHECK(cudaDeviceSynchronize());
  uint32_t na = ca.download(1)[0], nb = cb.download(1)[0];
  ASSERT_EQ(na, nb);
  EXPECT_EQ(a.download(na), b.download(nb));
}

// ---------------------------------------------------------------- radix sort
template <class K>
void check_sort(std::vector<K> h, int begin_bit = 0, int end_bit = sizeof(K) * 8) {
  size_t n = h.size();
  DeviceBuffer<K> a, b(n);
  a.upload(h);
  RadixWorkspace ws;
  bool in_b = radix_sort_keys(a.data(), b.data(), n, ws, begin_bit, end_bit);
  WDB_CUDA_CHECK(cudaDeviceSynchronize());
  auto got = (in_b ? b : a).download(n);
  std::sort(h.begin(), h.end());
  ASSERT_EQ(got, h) << "n=" << n;
}

const std::vector<size_t> kSortSizes = {0, 1, 2, 100, 4095, 4096, 4097, 100003, (1u << 20) + 13, 3u << 20};

TEST(RadixSort, UInt32) {
  for (size_t n : kSortSizes) check_sort(rand_u32(n, 0, n));
}
TEST(RadixSort, Int32WithNegatives) {
  for (size_t n : kSortSizes) {
    std::mt19937_64 g(n);
    std::vector<int32_t> h(n);
    for (auto& x : h) x = static_cast<int32_t>(g());
    check_sort(h);
  }
}
TEST(RadixSort, UInt64AndInt64) {
  for (size_t n : kSortSizes) {
    std::mt19937_64 g(n + 1);
    std::vector<uint64_t> u(n);
    std::vector<int64_t> s(n);
    for (size_t i = 0; i < n; ++i) { u[i] = g(); s[i] = static_cast<int64_t>(g()); }
    check_sort(u);
    check_sort(s);
  }
}
TEST(RadixSort, FloatAndDouble) {
  for (size_t n : kSortSizes) {
    std::mt19937_64 g(n + 2);
    std::uniform_real_distribution<double> d(-1e6, 1e6);
    std::vector<float> f(n);
    std::vector<double> db(n);
    for (size_t i = 0; i < n; ++i) { f[i] = static_cast<float>(d(g)); db[i] = d(g); }
    check_sort(f);
    check_sort(db);
  }
}
TEST(RadixSort, DuplicatesSortedAndReversed) {
  size_t n = (1 << 20) + 7;
  check_sort(rand_u32(n, 5, 1));  // only 5 distinct values -> heavy digit collisions
  std::vector<uint32_t> asc(n), desc(n);
  std::iota(asc.begin(), asc.end(), 0u);
  std::iota(desc.rbegin(), desc.rend(), 0u);
  check_sort(asc);
  check_sort(desc);
  check_sort(std::vector<uint32_t>(n, 42));
}
TEST(RadixSort, BitRangeOnlyComparesRequestedBits) {
  size_t n = (1 << 18) + 5;
  auto h = rand_u32(n, 0, 11);
  // sort by bits [4, 20) only, expect stable order for equal bit-fields
  DeviceBuffer<uint32_t> a, b(n), va, vb(n);
  a.upload(h);
  std::vector<uint32_t> idx(n);
  std::iota(idx.begin(), idx.end(), 0u);
  va.upload(idx);
  RadixWorkspace ws;
  bool in_b = radix_sort<uint32_t, uint32_t>(a.data(), b.data(), va.data(), vb.data(), n, ws, 4, 20);
  WDB_CUDA_CHECK(cudaDeviceSynchronize());
  auto gk = (in_b ? b : a).download(n);
  auto gv = (in_b ? vb : va).download(n);
  std::vector<uint32_t> order(n);
  std::iota(order.begin(), order.end(), 0u);
  auto field = [&](uint32_t i) { return (h[i] >> 4) & 0xFFFFu; };
  std::stable_sort(order.begin(), order.end(), [&](uint32_t x, uint32_t y) { return field(x) < field(y); });
  EXPECT_EQ(gv, order);
  for (size_t i = 0; i < n; ++i) ASSERT_EQ(gk[i], h[gv[i]]);
}
TEST(RadixSort, PairsAreStable) {
  size_t n = (1 << 20) + 19;
  auto h = rand_u32(n, 1000, 21);  // many equal keys
  std::vector<uint32_t> idx(n);
  std::iota(idx.begin(), idx.end(), 0u);
  DeviceBuffer<uint32_t> a, b(n), va, vb(n);
  a.upload(h);
  va.upload(idx);
  RadixWorkspace ws;
  bool in_b = radix_sort<uint32_t, uint32_t>(a.data(), b.data(), va.data(), vb.data(), n, ws);
  WDB_CUDA_CHECK(cudaDeviceSynchronize());
  auto gv = (in_b ? vb : va).download(n);
  std::vector<uint32_t> order = idx;
  std::stable_sort(order.begin(), order.end(), [&](uint32_t x, uint32_t y) { return h[x] < h[y]; });
  EXPECT_EQ(gv, order);
}
TEST(RadixSort, Int64KeyUInt32Payload) {
  size_t n = (1 << 19) + 3;
  std::mt19937_64 g(8);
  std::vector<int64_t> h(n);
  for (auto& x : h) x = static_cast<int64_t>(g() % 20000) - 10000;
  std::vector<uint32_t> idx(n);
  std::iota(idx.begin(), idx.end(), 0u);
  DeviceBuffer<int64_t> a, b(n);
  DeviceBuffer<uint32_t> va, vb(n);
  a.upload(h);
  va.upload(idx);
  RadixWorkspace ws;
  bool in_b = radix_sort<int64_t, uint32_t>(a.data(), b.data(), va.data(), vb.data(), n, ws);
  WDB_CUDA_CHECK(cudaDeviceSynchronize());
  auto gv = (in_b ? vb : va).download(n);
  std::vector<uint32_t> order = idx;
  std::stable_sort(order.begin(), order.end(), [&](uint32_t x, uint32_t y) { return h[x] < h[y]; });
  EXPECT_EQ(gv, order);
}
TEST(RadixSort, RepeatedRunsDeterministic) {
  size_t n = 1 << 21;
  auto h = rand_u32(n, 300, 77);
  std::vector<uint32_t> idx(n);
  std::iota(idx.begin(), idx.end(), 0u);
  RadixWorkspace ws;
  std::vector<uint32_t> first;
  for (int r = 0; r < 10; ++r) {
    DeviceBuffer<uint32_t> a, b(n), va, vb(n);
    a.upload(h);
    va.upload(idx);
    bool in_b = radix_sort<uint32_t, uint32_t>(a.data(), b.data(), va.data(), vb.data(), n, ws);
    WDB_CUDA_CHECK(cudaDeviceSynchronize());
    auto gv = (in_b ? vb : va).download(n);
    if (r == 0) first = gv;
    else ASSERT_EQ(gv, first) << "run " << r;
  }
}

// ---------------------------------------------------------------- hash table
TEST(HashTable, InsertFindDistinct) {
  for (size_t n : {size_t(1), size_t(1000), size_t(1) << 20}) {
    for (int64_t range : {int64_t(10), int64_t(5000), int64_t(1) << 40}) {
      std::mt19937_64 g(n + range);
      std::vector<int64_t> keys(n);
      for (auto& k : keys) k = static_cast<int64_t>(g() % static_cast<uint64_t>(range)) - range / 2;
      std::unordered_set<int64_t> uniq(keys.begin(), keys.end());

      DeviceBuffer<int64_t> dk;
      dk.upload(keys);
      DeviceBuffer<uint32_t> slots(n);
      HashTable t(uniq.size());
      t.insert(dk.data(), n, slots.data());
      WDB_CUDA_CHECK(cudaDeviceSynchronize());
      EXPECT_EQ(t.size(), uniq.size()) << "n=" << n << " range=" << range;

      auto s = slots.download(n);
      auto stored = DeviceBuffer<int64_t>();  // verify slot <-> key bijection via host copy of the table
      std::vector<int64_t> table(t.capacity());
      WDB_CUDA_CHECK(cudaMemcpy(table.data(), t.keys(), table.size() * 8, cudaMemcpyDeviceToHost));
      std::unordered_map<int64_t, uint32_t> seen;
      for (size_t i = 0; i < n; ++i) {
        ASSERT_NE(s[i], HT_NOT_FOUND);
        ASSERT_EQ(table[s[i]], keys[i]);  // the slot really holds this key
        auto [it, fresh] = seen.emplace(keys[i], s[i]);
        if (!fresh) ASSERT_EQ(it->second, s[i]);  // equal keys -> equal slot
      }

      // lookups: present keys hit the same slot, absent keys miss
      std::vector<int64_t> probe = keys;
      for (size_t i = 0; i < n; i += 2) probe[i] = range * 4 + static_cast<int64_t>(i);  // absent
      DeviceBuffer<int64_t> dp;
      dp.upload(probe);
      DeviceBuffer<uint32_t> found(n);
      t.find(dp.data(), n, found.data());
      WDB_CUDA_CHECK(cudaDeviceSynchronize());
      auto f = found.download(n);
      for (size_t i = 0; i < n; ++i) {
        if (i % 2 == 0) ASSERT_EQ(f[i], HT_NOT_FOUND);
        else ASSERT_EQ(f[i], s[i]);
      }
    }
  }
}

TEST(HashTable, ReportsOverflowInsteadOfHanging) {
  HashTable t(1);  // capacity 16
  std::vector<int64_t> keys(1000);
  std::iota(keys.begin(), keys.end(), 1);
  DeviceBuffer<int64_t> dk;
  dk.upload(keys);
  DeviceBuffer<uint32_t> slots(keys.size());
  t.insert(dk.data(), keys.size(), slots.data());
  WDB_CUDA_CHECK(cudaDeviceSynchronize());
  auto s = slots.download(keys.size());
  size_t overflow = std::count(s.begin(), s.end(), HT_NOT_FOUND);
  EXPECT_EQ(overflow, keys.size() - t.capacity());
}

}  // namespace
