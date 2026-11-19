#pragma once
#include <memory>

#include "exec/engine.h"

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

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

std::shared_ptr<GpuBackend> make_gpu_backend();

}  // namespace warpdb::gpu
