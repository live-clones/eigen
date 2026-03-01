// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// array_cwise split: complex math functions only.
// Split from array_cwise_real.cpp to reduce per-TU memory usage
// under ASAN+UBSAN.

#include "array_cwise_helpers.h"

TEST(ArrayCwiseComplexTest, Complex) {
  for (int i = 0; i < g_repeat; i++) {
    array_complex(
        ArrayXXcf(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
    array_complex(
        ArrayXXcd(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
  }
}
