#include "hello_cuda.h"
#include <gtest/gtest.h>

TEST(CudaTest, KernelTest) {
  // Example test case
  EXPECT_NO_THROW(launchKernel());
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
