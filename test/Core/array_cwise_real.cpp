// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// array_cwise split: real math functions and complex math functions.

#include "array_cwise_helpers.h"

// =============================================================================
// Tests for array_cwise_real
// =============================================================================
TEST(ArrayCwiseRealTest, RealFixed) {
  for (int i = 0; i < g_repeat; i++) {
    array_real(Array<float, 1, 1>());
    array_real(Array22f());
    array_real(Array44d());
  }
}

TEST(ArrayCwiseRealTest, RealDynamic) {
  for (int i = 0; i < g_repeat; i++) {
    array_real(ArrayXXf(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
    array_real(Array<Eigen::half, 32, 32>());
    array_real(Array<Eigen::bfloat16, 32, 32>());
  }
}

TEST(ArrayCwiseRealTest, Complex) {
  for (int i = 0; i < g_repeat; i++) {
    array_complex(
        ArrayXXcf(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
    array_complex(
        ArrayXXcd(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
  }
}
