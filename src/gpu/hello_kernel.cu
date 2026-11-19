#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

#include "gpu/device.h"

namespace warpdb::gpu {

__global__ void saxpy_kernel(const float* x, float* y, float a, float b, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + b;
}

bool run_hello_kernel(size_t n) {
  std::vector<float> hx(n), hy(n, 0.f);
  for (size_t i = 0; i < n; ++i) hx[i] = static_cast<float>(i % 1000);

  float *dx = nullptr, *dy = nullptr;
  if (cudaMalloc(&dx, n * sizeof(float)) != cudaSuccess) return false;
  if (cudaMalloc(&dy, n * sizeof(float)) != cudaSuccess) {
    cudaFree(dx);
    return false;
  }
  cudaMemcpy(dx, hx.data(), n * sizeof(float), cudaMemcpyHostToDevice);

  const int block = 256;
  const int grid = static_cast<int>((n + block - 1) / block);
  saxpy_kernel<<<grid, block>>>(dx, dy, 2.f, 1.f, n);
  cudaError_t launch_err = cudaGetLastError();
  cudaError_t sync_err = cudaDeviceSynchronize();
  bool ok = launch_err == cudaSuccess && sync_err == cudaSuccess;
  if (!ok) std::fprintf(stderr, "hello_kernel: launch=%s sync=%s\n", cudaGetErrorString(launch_err), cudaGetErrorString(sync_err));

  if (ok) {
    cudaMemcpy(hy.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < n && ok; ++i) ok = (hy[i] == 2.f * hx[i] + 1.f);
  }
  cudaFree(dx);
  cudaFree(dy);
  return ok;
}

}  // namespace warpdb::gpu
