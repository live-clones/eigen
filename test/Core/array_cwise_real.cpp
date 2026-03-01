// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// array_cwise split: real math functions only.
// Complex math function tests are in array_cwise_complex.cpp to reduce
// per-TU memory usage under ASAN+UBSAN.

#include "array_cwise_helpers.h"

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
