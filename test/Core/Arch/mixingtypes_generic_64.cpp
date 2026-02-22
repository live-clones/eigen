// Force the generic clang vector backend with 64-byte vectors.
#define EIGEN_VECTORIZE_GENERIC 1
#define EIGEN_GENERIC_VECTOR_SIZE_BYTES 64

#include "../mixingtypes_helpers.h"

// =============================================================================
// Tests for mixingtypes_generic_64
// =============================================================================
TEST(MixingTypesGeneric64Test, Basic) {
  g_called = false;  // Silence -Wunneeded-internal-declaration.
  for (int i = 0; i < g_repeat; i++) {
    mixingtypes<3>();
    mixingtypes<4>();
    mixingtypes<Dynamic>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE));
  }
}
