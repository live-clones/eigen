// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.

#define EIGEN_VECTORIZE_SME
#define EIGEN_ARM64_USE_SME

#include <arm_neon.h>
#include "product.h"

EIGEN_DECLARE_TEST(product_sme) {
  // Explicit edge cases for boundary conditions (assuming VL=16 floats = 512 bits)
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(1, 1)));
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(15, 15)));
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(16, 16)));
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(17, 17)));
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(31, 31)));
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(33, 33)));
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(63, 63)));
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(64, 64)));
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(65, 65)));
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(32, 1)));
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(1, 32)));
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(1, 64)));
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(64, 1)));

  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE),
                                                           internal::random<int>(1, EIGEN_TEST_MAX_SIZE))));
  }
}
