// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// SME GEMM kernel tests.
// Requires compiler flags: -march=armv9.2-a+sme2+nosve -msve-vector-bits=512
// and the define: -DEIGEN_ARM64_USE_SME

#include "product.h"

EIGEN_DECLARE_TEST(product_sme) {
  // Square edge cases around tile boundaries (VL=16 floats, NR=64)
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(1, 1)));
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(15, 15)));
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(16, 16)));
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(17, 17)));
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(31, 31)));
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(33, 33)));
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(63, 63)));
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(64, 64)));
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(65, 65)));

  // Thin / wide rectangular cases (M x 1, 1 x N)
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(32, 1)));
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(1, 32)));
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(1, 64)));
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(64, 1)));

  // Non-square cases that exercise tail paths for both M and N
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(17, 65)));
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(65, 17)));
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(15, 63)));
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(33, 7)));
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(7, 33)));
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(128, 3)));
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(3, 128)));

  // Random sizes
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE),
                                                           internal::random<int>(1, EIGEN_TEST_MAX_SIZE))));
  }
}
