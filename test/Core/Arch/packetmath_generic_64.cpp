// Force the generic clang vector backend with 64-byte vectors.
#define EIGEN_VECTORIZE_GENERIC 1
#define EIGEN_GENERIC_VECTOR_SIZE_BYTES 64
#include "packetmath_test_helpers.h"

TEST(PacketmathGeneric64Test, Basic) {
  g_first_pass = true;
  for (int i = 0; i < g_repeat; i++) {
    test::runner<float>::run();
    test::runner<double>::run();
    g_first_pass = false;
  }
}
