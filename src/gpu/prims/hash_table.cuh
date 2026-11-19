#pragma once
// Open-addressing (linear probing) hash table for 64-bit keys, built for GROUP BY / DISTINCT:
//   insert_or_find(key) -> slot index (stable for the table's lifetime); equal keys map to equal slots.
// Insertion is lock-free (atomicCAS on the key word). Aggregation operators accumulate into
// per-slot payload arrays addressed by the returned slot.
//
// Constraints: capacity is a power of two and must exceed the number of DISTINCT keys (the table never
// grows on the device; the host sizes it from an NDV estimate and retries bigger on overflow).
// The key INT64_MIN is reserved as the empty marker (the engine remaps it).
#include <cstdint>

#include "gpu/cuda_util.cuh"

namespace warpdb::gpu {

constexpr int64_t HT_EMPTY = INT64_MIN;
constexpr uint32_t HT_NOT_FOUND = 0xFFFFFFFFu;

__host__ __device__ __forceinline__ uint64_t hash64(uint64_t x) {  // murmur3 finalizer
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return x;
}

struct HashTableView {
  int64_t* keys;
  uint32_t mask;  // capacity - 1

  // Returns the slot holding `key`, inserting it if absent. Returns HT_NOT_FOUND if the table is full.
  __device__ __forceinline__ uint32_t insert_or_find(int64_t key) const {
    uint32_t s = static_cast<uint32_t>(hash64(static_cast<uint64_t>(key))) & mask;
    for (uint32_t probes = 0; probes <= mask; ++probes) {
      int64_t cur = keys[s];
      if (cur == key) return s;
      if (cur == HT_EMPTY) {
        auto old = static_cast<int64_t>(
            atomicCAS(reinterpret_cast<unsigned long long*>(keys + s), static_cast<unsigned long long>(HT_EMPTY),
                      static_cast<unsigned long long>(key)));
        if (old == HT_EMPTY || old == key) return s;
      }
      s = (s + 1) & mask;
    }
    return HT_NOT_FOUND;
  }
  __device__ __forceinline__ uint32_t find(int64_t key) const {
    uint32_t s = static_cast<uint32_t>(hash64(static_cast<uint64_t>(key))) & mask;
    for (uint32_t probes = 0; probes <= mask; ++probes) {
      int64_t cur = keys[s];
      if (cur == key) return s;
      if (cur == HT_EMPTY) return HT_NOT_FOUND;
      s = (s + 1) & mask;
    }
    return HT_NOT_FOUND;
  }
};

namespace detail {

static __global__ void ht_clear_kernel(int64_t* keys, size_t cap) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i < cap) keys[i] = HT_EMPTY;
}
static __global__ void ht_insert_kernel(HashTableView t, const int64_t* keys, size_t n, uint32_t* slots) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i < n) slots[i] = t.insert_or_find(keys[i]);
}
static __global__ void ht_find_kernel(HashTableView t, const int64_t* keys, size_t n, uint32_t* slots) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i < n) slots[i] = t.find(keys[i]);
}
static __global__ void ht_count_kernel(const int64_t* keys, size_t cap, uint32_t* out) {
  uint32_t local = 0;
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < cap;
       i += static_cast<size_t>(gridDim.x) * blockDim.x)
    local += keys[i] != HT_EMPTY;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) local += __shfl_xor_sync(FULL_MASK, local, o);
  if ((threadIdx.x & 31) == 0 && local) atomicAdd(out, local);
}

}  // namespace detail

class HashTable {
 public:
  // capacity = smallest power of two >= 2 * expected_distinct (load factor <= 0.5).
  explicit HashTable(size_t expected_distinct) {
    size_t cap = 16;
    while (cap < expected_distinct * 2) cap <<= 1;
    keys_.resize(cap);
    cap_ = cap;
    clear();
  }
  void clear(cudaStream_t s = 0) {
    detail::ht_clear_kernel<<<static_cast<unsigned>(ceil_div(cap_, 256)), 256, 0, s>>>(keys_.data(), cap_);
    WDB_CUDA_CHECK(cudaGetLastError());
  }
  HashTableView view() { return {keys_.data(), static_cast<uint32_t>(cap_ - 1)}; }
  size_t capacity() const { return cap_; }
  const int64_t* keys() const { return keys_.data(); }

  // slots[i] = slot of keys[i] (inserting). HT_NOT_FOUND marks overflow.
  void insert(const int64_t* d_keys, size_t n, uint32_t* d_slots, cudaStream_t s = 0) {
    if (!n) return;
    detail::ht_insert_kernel<<<static_cast<unsigned>(ceil_div(n, 256)), 256, 0, s>>>(view(), d_keys, n, d_slots);
    WDB_CUDA_CHECK(cudaGetLastError());
  }
  void find(const int64_t* d_keys, size_t n, uint32_t* d_slots, cudaStream_t s = 0) {
    if (!n) return;
    detail::ht_find_kernel<<<static_cast<unsigned>(ceil_div(n, 256)), 256, 0, s>>>(view(), d_keys, n, d_slots);
    WDB_CUDA_CHECK(cudaGetLastError());
  }
  // Number of occupied slots (= distinct keys inserted). Synchronizes.
  uint32_t size(cudaStream_t s = 0) {
    DeviceBuffer<uint32_t> out(1);
    WDB_CUDA_CHECK(cudaMemsetAsync(out.data(), 0, 4, s));
    detail::ht_count_kernel<<<static_cast<unsigned>(std::min<size_t>(ceil_div(cap_, 256), 1024)), 256, 0, s>>>(
        keys_.data(), cap_, out.data());
    WDB_CUDA_CHECK(cudaGetLastError());
    WDB_CUDA_CHECK(cudaStreamSynchronize(s));
    return out.download(1)[0];
  }

 private:
  DeviceBuffer<int64_t> keys_;
  size_t cap_ = 0;
};

}  // namespace warpdb::gpu
