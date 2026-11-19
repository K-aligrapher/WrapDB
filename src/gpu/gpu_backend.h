#pragma once
#include <memory>

#include "exec/engine.h"
#include "gpu/jit/jit.h"

namespace warpdb::gpu {

struct GpuMemStats {
  size_t cache_bytes = 0;      // base-table columns resident in VRAM
  size_t cache_budget = 0;
  size_t in_use = 0;           // allocator: live bytes
  size_t peak = 0;             // allocator: high-water mark of the last query
  size_t reserved = 0;         // allocator: live + cached free blocks
  uint64_t uploaded_total = 0; // bytes ever copied host -> device for columns
};

// CUDA execution backend: runs a PlannedQuery with GPU kernels (see Engine::set_gpu_backend).
class GpuBackend : public Backend {
 public:
  GpuBackend();
  ~GpuBackend() override;
  const char* name() const override { return "gpu"; }
  QueryResult execute(Engine& eng, PlannedQuery& pq) override;

  GpuMemStats mem_stats() const;
  void drop_cache() override;  // evict all resident columns (to measure cold queries)
  bool column_resident(const Column& c) const override;
  size_t cache_budget_bytes() const override;
  void forget_columns(const std::vector<uint64_t>& uids) override;
  std::string fingerprint() const override;
  bool measure_hardware(HwProfile& hw) override;

  // JIT pipeline fusion (NVRTC). Falls back to the interpreted kernels whenever it is unavailable or unsupported.
  void set_jit(bool on);
  bool jit_enabled() const;             // requested AND NVRTC usable
  void set_jit_min_rows(size_t n);      // compile unseen kernels only for inputs at least this large
  size_t jit_min_rows() const;
  JitStats jit_stats() const;
  size_t jit_preload(size_t max_kernels = 64);  // warm the JIT from its on-disk cache
  void set_jit_force_capacity_for_test(size_t cap);  // 0 = size group tables from estimates (default)

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

std::shared_ptr<GpuBackend> make_gpu_backend();

}  // namespace warpdb::gpu
