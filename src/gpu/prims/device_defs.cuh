#pragma once
// Definitions shared by host-compiled (nvcc) and runtime-compiled (NVRTC) device code. This header must
// not include anything that NVRTC cannot see (no <cuda_runtime.h>, no standard library).
#ifdef __CUDACC_RTC__
typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef short int16_t;
typedef unsigned short uint16_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;
#ifndef INT64_MIN
#define INT64_MIN (-9223372036854775807LL - 1)
#endif
#ifndef INT64_MAX
#define INT64_MAX 9223372036854775807LL
#endif
#else
#include <cstdint>
#endif

namespace warpdb::gpu {

constexpr uint32_t FULL_MASK = 0xFFFFFFFFu;
constexpr uint32_t EMPTY_SLOT = 0xFFFFFFFFu;

__host__ __device__ __forceinline__ uint64_t hash64(uint64_t x) {  // murmur3 finalizer
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return x;
}

template <class A, class B>
struct is_same_type {
  static constexpr bool value = false;
};
template <class A>
struct is_same_type<A, A> {
  static constexpr bool value = true;
};

}  // namespace warpdb::gpu
