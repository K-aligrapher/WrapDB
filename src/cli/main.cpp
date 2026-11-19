#include <cstdio>

#include "gpu/device.h"

int main() {
  using namespace warpdb::gpu;
  DeviceInfo d;
  if (!query_device(d)) {
    std::puts("warpdb: no CUDA device found");
    return 1;
  }
  std::printf("WarpDB v0.0.1 (Phase 0)\n");
  std::printf("GPU        : %s (sm_%d%d, %d SMs)\n", d.name.c_str(), d.sm_major, d.sm_minor,
              d.sm_count);
  std::printf("VRAM       : %.2f GB total, %.2f GB free\n", d.total_mem / 1073741824.0,
              d.free_mem / 1073741824.0);
  std::printf("Peak mem BW: %.1f GB/s (%d-bit bus)\n", d.peak_bandwidth_gbs(), d.mem_bus_bits);
  bool ok = run_hello_kernel();
  std::printf("Hello kernel: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 2;
}
