#pragma once
// GPU memory management:
//   * GpuAllocator : caching device allocator (size classes, ~12% worst-case waste). cudaMalloc/cudaFree
//                    are slow and implicitly synchronize, so freed blocks are kept in per-size free lists and
//                    reused. On cudaMalloc failure it returns every cached block to the driver and retries.
//   * GpuVec<T>    : RAII array on top of the allocator.
//   * ColumnCache  : keeps base-table columns resident in VRAM across queries (LRU eviction under a byte
//                    budget). Residency is what the CPU/GPU placement optimizer reasons about.
#include <cstring>
#include <mutex>
#include <unordered_map>

#include "gpu/cuda_util.cuh"
#include "storage/table.h"

namespace warpdb::gpu {

class GpuAllocator {
 public:
  // Intentionally leaked: GpuVec / ColumnCache objects with static storage duration free through this
  // allocator during process teardown, so it must outlive every other static. The driver reclaims the
  // device memory at exit anyway.
  static GpuAllocator& instance() {
    static GpuAllocator* a = new GpuAllocator();
    return *a;
  }

  void* alloc(size_t bytes) {
    std::lock_guard<std::mutex> g(mu_);
    size_t cs = class_size(bytes);
    auto& fl = free_[cs];
    void* p = nullptr;
    if (!fl.empty()) {
      p = fl.back();
      fl.pop_back();
      cached_ -= cs;
    } else {
      cudaError_t e = cudaMalloc(&p, cs);
      if (e != cudaSuccess) {  // out of memory: drop the cache and retry once
        cudaGetLastError();
        release_cached_locked();
        e = cudaMalloc(&p, cs);
        if (e != cudaSuccess) {
          cudaGetLastError();
          throw std::runtime_error("GPU out of memory: requested " + std::to_string(cs >> 20) + " MB, " +
                                   std::to_string(in_use_ >> 20) + " MB in use");
        }
      }
      reserved_ += cs;
    }
    live_[p] = cs;
    in_use_ += cs;
    peak_ = std::max(peak_, in_use_);
    return p;
  }

  void free(void* p) {
    if (!p) return;
    std::lock_guard<std::mutex> g(mu_);
    auto it = live_.find(p);
    if (it == live_.end()) return;
    size_t cs = it->second;
    live_.erase(it);
    in_use_ -= cs;
    cached_ += cs;
    free_[cs].push_back(p);
  }

  void release_cached() {
    std::lock_guard<std::mutex> g(mu_);
    release_cached_locked();
  }
  size_t in_use() const { return in_use_; }
  size_t peak() const { return peak_; }
  size_t reserved() const { return reserved_; }
  void reset_peak() { peak_ = in_use_; }

 private:
  GpuAllocator() = default;
  ~GpuAllocator() = default;  // the process is ending; the driver reclaims everything
  static size_t class_size(size_t b) {
    if (b <= 256) return 256;
    size_t p = 256;
    while ((p << 1) < b) p <<= 1;  // p < b <= 2p
    size_t step = p / 4;
    return p + ((b - p + step - 1) / step) * step;
  }
  void release_cached_locked() {
    for (auto& kv : free_) {
      for (void* p : kv.second) {
        cudaFree(p);
        reserved_ -= kv.first;
      }
      kv.second.clear();
    }
    cached_ = 0;
  }
  std::mutex mu_;
  std::unordered_map<size_t, std::vector<void*>> free_;
  std::unordered_map<void*, size_t> live_;
  size_t in_use_ = 0, peak_ = 0, reserved_ = 0, cached_ = 0;
};

template <class T>
class GpuVec {
 public:
  GpuVec() = default;
  explicit GpuVec(size_t n) { allocate(n); }
  ~GpuVec() { GpuAllocator::instance().free(p_); }
  GpuVec(const GpuVec&) = delete;
  GpuVec& operator=(const GpuVec&) = delete;
  GpuVec(GpuVec&& o) noexcept : p_(o.p_), n_(o.n_) { o.p_ = nullptr; o.n_ = 0; }
  GpuVec& operator=(GpuVec&& o) noexcept {
    if (this != &o) {
      GpuAllocator::instance().free(p_);
      p_ = o.p_; n_ = o.n_;
      o.p_ = nullptr; o.n_ = 0;
    }
    return *this;
  }
  void allocate(size_t n) {
    GpuAllocator::instance().free(p_);
    p_ = nullptr;
    n_ = n;
    p_ = static_cast<T*>(GpuAllocator::instance().alloc((n ? n : 1) * sizeof(T)));
  }
  T* data() { return p_; }
  const T* data() const { return p_; }
  size_t size() const { return n_; }
  std::vector<T> to_host(size_t n) const {
    std::vector<T> h(n);
    if (n) WDB_CUDA_CHECK(cudaMemcpy(h.data(), p_, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
  }
  void from_host(const T* src, size_t n) {
    if (n) WDB_CUDA_CHECK(cudaMemcpy(p_, src, n * sizeof(T), cudaMemcpyHostToDevice));
  }

 private:
  T* p_ = nullptr;
  size_t n_ = 0;
};

// Resident copies of base-table columns in their native (narrow) layout.
class ColumnCache {
 public:
  explicit ColumnCache(size_t budget_bytes = 0) : budget_(budget_bytes) {}
  void set_budget(size_t b) { budget_ = b; }
  size_t budget() const { return budget_; }
  size_t used() const { return used_; }
  uint64_t uploaded_bytes() const { return uploaded_; }
  void begin_query() { ++clock_; }

  bool resident(const Column& c) const { return map_.count(&c) != 0; }

  // Device pointer to the column's data (int32* / int64* / double* depending on type), uploading if needed.
  const void* get(const Column& c) {
    auto it = map_.find(&c);
    if (it != map_.end()) {
      it->second.stamp = clock_;
      return it->second.buf.data();
    }
    size_t bytes = c.size() * elem_size(c);
    while (budget_ && used_ + bytes > budget_ && evict_one()) {
    }
    if (budget_ && used_ + bytes > budget_)
      throw std::runtime_error("GPU column cache budget exceeded: working set of one query does not fit in VRAM");
    Entry e;
    e.buf.allocate(bytes);
    const void* src = c.type == Type::INT64 ? static_cast<const void*>(c.i64.data())
                      : c.type == Type::FLOAT64 ? static_cast<const void*>(c.f64.data())
                                                : static_cast<const void*>(c.i32.data());
    if (bytes) WDB_CUDA_CHECK(cudaMemcpy(e.buf.data(), src, bytes, cudaMemcpyHostToDevice));
    e.bytes = bytes;
    e.stamp = clock_;
    used_ += bytes;
    uploaded_ += bytes;
    auto& slot = map_.emplace(&c, std::move(e)).first->second;
    return slot.buf.data();
  }

  void clear() {
    map_.clear();
    used_ = 0;
  }

  static size_t elem_size(const Column& c) {
    return (c.type == Type::INT64 || c.type == Type::FLOAT64) ? 8 : 4;
  }

 private:
  struct Entry {
    GpuVec<uint8_t> buf;
    size_t bytes = 0;
    uint64_t stamp = 0;
  };
  // Evicts the least-recently-used column that is not part of the current query.
  bool evict_one() {
    auto victim = map_.end();
    for (auto it = map_.begin(); it != map_.end(); ++it)
      if (it->second.stamp < clock_ && (victim == map_.end() || it->second.stamp < victim->second.stamp)) victim = it;
    if (victim == map_.end()) return false;
    used_ -= victim->second.bytes;
    map_.erase(victim);
    return true;
  }
  std::unordered_map<const Column*, Entry> map_;
  size_t budget_ = 0, used_ = 0;
  uint64_t clock_ = 1, uploaded_ = 0;
};

}  // namespace warpdb::gpu
