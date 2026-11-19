#pragma once
// Device-side runtime for JIT-generated kernels. Included by generated source through NVRTC (-I<src>), so
// everything it pulls in must be NVRTC-clean (no host headers).
#include "gpu/prims/agg_device.cuh"
#include "gpu/prims/select_tile.cuh"

namespace warpdb::gpu {

__device__ __forceinline__ long long wdb_mod(long long x, long long y) { return y ? x % y : 0LL; }

// year / month / day of a date given as days since 1970-01-01 (Howard Hinnant's civil_from_days).
__device__ __forceinline__ long long wdb_date_part(long long days, int which) {
  int z = static_cast<int>(days) + 719468;
  const int era = (z >= 0 ? z : z - 146096) / 146097;
  const unsigned doe = static_cast<unsigned>(z - era * 146097);
  const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  int y = static_cast<int>(yoe) + era * 400;
  const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  const unsigned mp = (5 * doy + 2) / 153;
  const unsigned d = doy - (153 * mp + 2) / 5 + 1;
  const unsigned m = mp < 10 ? mp + 3 : mp - 9;
  y += (m <= 2);
  return which == 0 ? y : which == 1 ? static_cast<long long>(m) : static_cast<long long>(d);
}

}  // namespace warpdb::gpu
