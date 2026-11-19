#pragma once
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#define WDB_CUDA_CHECK(call)                                                                        \
  do {                                                                                              \
    cudaError_t e_ = (call);                                                                        \
    if (e_ != cudaSuccess)                                                                          \
      throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e_) + " at " +      \
                               __FILE__ + ":" + std::to_string(__LINE__));                          \
  } while (0)

namespace warpdb::gpu {

// Owning, move-only device array.
template <class T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t n) { resize(n); }
  ~DeviceBuffer() { release(); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& o) noexcept : p_(o.p_), n_(o.n_) { o.p_ = nullptr; o.n_ = 0; }
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
    if (this != &o) { release(); p_ = o.p_; n_ = o.n_; o.p_ = nullptr; o.n_ = 0; }
    return *this;
  }
  // Grows (never shrinks) the allocation; contents are not preserved.
  void resize(size_t n) {
    if (n <= n_) return;
    release();
    WDB_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&p_), (n ? n : 1) * sizeof(T)));
    n_ = n;
  }
  void upload(const std::vector<T>& h) {
    resize(h.size());
    WDB_CUDA_CHECK(cudaMemcpy(p_, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
  std::vector<T> download(size_t n) const {
    std::vector<T> h(n);
    if (n) WDB_CUDA_CHECK(cudaMemcpy(h.data(), p_, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
  }
  T* data() { return p_; }
  const T* data() const { return p_; }
  size_t size() const { return n_; }

 private:
  void release() {
    if (p_) cudaFree(p_);
    p_ = nullptr;
    n_ = 0;
  }
  T* p_ = nullptr;
  size_t n_ = 0;
};

class GpuTimer {
 public:
  GpuTimer() { cudaEventCreate(&a_); cudaEventCreate(&b_); }
  ~GpuTimer() { cudaEventDestroy(a_); cudaEventDestroy(b_); }
  void start(cudaStream_t s = 0) { cudaEventRecord(a_, s); }
  float stop(cudaStream_t s = 0) {  // milliseconds
    cudaEventRecord(b_, s);
    cudaEventSynchronize(b_);
    float ms = 0;
    cudaEventElapsedTime(&ms, a_, b_);
    return ms;
  }

 private:
  cudaEvent_t a_, b_;
};

constexpr uint32_t FULL_MASK = 0xFFFFFFFFu;

inline size_t ceil_div(size_t a, size_t b) { return (a + b - 1) / b; }

}  // namespace warpdb::gpu
