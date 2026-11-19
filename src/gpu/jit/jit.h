#pragma once
// Runtime kernel compilation: CUDA C++ source -> (NVRTC) CUBIN -> (driver API) loaded function.
//
// Compilation takes 100s of ms, so compiled kernels are cached
//   1. in memory, keyed by a hash of (NVRTC version, GPU architecture, source text), and
//   2. on disk (%LOCALAPPDATA%\warpdb\jit\<key>.cubin), so a later *process* skips NVRTC entirely.
// If NVRTC or the driver API cannot be used, available() is false and callers fall back to the AOT kernels.
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace warpdb::gpu {

struct JitError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct JitStats {
  size_t compiles = 0;    // NVRTC invocations
  size_t mem_hits = 0;    // served from the in-process cache
  size_t disk_hits = 0;   // loaded from a CUBIN written by an earlier run
  size_t failures = 0;    // compilation errors (callers fall back to unfused kernels)
  double compile_ms = 0;  // total time inside NVRTC
  double load_ms = 0;     // total time loading modules (cuModuleLoadData)
};

class JitKernel;  // opaque: a loaded CUfunction

// Collects kernel arguments by value (each at most 8 bytes: pointers, sizes, ints).
class ArgPack {
 public:
  template <class T>
  ArgPack& add(T v) {
    static_assert(sizeof(T) <= 8, "kernel arguments are packed in 8-byte slots");
    store_.emplace_back(0);
    std::memcpy(&store_.back(), &v, sizeof(T));
    ptrs_.push_back(&store_.back());
    return *this;
  }
  void** data() { return ptrs_.data(); }
  size_t size() const { return ptrs_.size(); }

 private:
  std::deque<uint64_t> store_;  // stable addresses
  std::vector<void*> ptrs_;
};

class JitRuntime {
 public:
  static JitRuntime& instance();

  bool available();  // NVRTC + driver API usable on this machine
  // Compiles `source` (or fetches it from a cache) and returns the function `entry`.
  // *compile_ms is the time this call spent compiling/loading (0 on an in-memory hit).
  std::shared_ptr<JitKernel> get(const std::string& source, const std::string& entry, double* ms = nullptr,
                                 const char** origin = nullptr);
  // Where the time of the most recent get() went, e.g. "disk read 0.1 ms, module load 2.9 ms" (for diagnostics).
  std::string last_timing() const;
  // True if get() would be cheap: the kernel is in the in-process cache or on disk (no NVRTC run needed).
  bool peek(const std::string& source, const std::string& entry);
  void launch(JitKernel& k, unsigned grid, unsigned block, ArgPack& args);

  JitStats stats() const;
  void set_disk_cache_dir(const std::string& dir);  // "" disables the on-disk cache
  void clear_memory_cache();

 private:
  JitRuntime();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace warpdb::gpu
