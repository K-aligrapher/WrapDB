#pragma once
#include <cstddef>
#include <string>

namespace warpdb::gpu {

struct DeviceInfo {
  std::string name;
  int sm_major = 0, sm_minor = 0;
  size_t total_mem = 0, free_mem = 0;
  int sm_count = 0;
  int clock_khz = 0;
  int mem_clock_khz = 0;
  int mem_bus_bits = 0;
  double peak_bandwidth_gbs() const {
    return 2.0 * mem_clock_khz * 1e3 * (mem_bus_bits / 8.0) / 1e9;
  }
};

// Returns false if no CUDA device is available.
bool query_device(DeviceInfo& out, int ordinal = 0);

// Launches a trivial kernel (y[i] = a*x[i] + b) and verifies on the host.
// Returns true if every element matched.
bool run_hello_kernel(size_t n = 1 << 20);

}  // namespace warpdb::gpu
