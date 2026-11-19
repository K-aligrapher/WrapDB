#include <gtest/gtest.h>

#include "gpu/device.h"

using namespace warpdb::gpu;

TEST(Device, QueryReportsBlackwellWithMemory) {
  DeviceInfo d;
  ASSERT_TRUE(query_device(d));
  EXPECT_GT(d.total_mem, 1ull << 30);
  EXPECT_GE(d.sm_major, 7);
  EXPECT_GT(d.sm_count, 0);
}

TEST(Device, HelloKernelSmallAndLarge) {
  EXPECT_TRUE(run_hello_kernel(1));
  EXPECT_TRUE(run_hello_kernel(1000));        // not a multiple of block size
  EXPECT_TRUE(run_hello_kernel(1 << 22));
}
