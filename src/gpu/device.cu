#include "gpu/device.h"

#include <cuda_runtime.h>

namespace warpdb::gpu {

bool query_device(DeviceInfo& out, int ordinal) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || ordinal >= count) return false;
  cudaDeviceProp p{};
  if (cudaGetDeviceProperties(&p, ordinal) != cudaSuccess) return false;
  cudaSetDevice(ordinal);
  size_t free_b = 0, total_b = 0;
  cudaMemGetInfo(&free_b, &total_b);
  out.name = p.name;
  out.sm_major = p.major;
  out.sm_minor = p.minor;
  out.total_mem = total_b;
  out.free_mem = free_b;
  out.sm_count = p.multiProcessorCount;
  int v = 0;
  cudaDeviceGetAttribute(&v, cudaDevAttrClockRate, ordinal);
  out.clock_khz = v;
  cudaDeviceGetAttribute(&v, cudaDevAttrMemoryClockRate, ordinal);
  out.mem_clock_khz = v;
  cudaDeviceGetAttribute(&v, cudaDevAttrGlobalMemoryBusWidth, ordinal);
  out.mem_bus_bits = v;
  return true;
}

}  // namespace warpdb::gpu
