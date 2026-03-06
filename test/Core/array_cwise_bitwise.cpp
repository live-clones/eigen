// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// array_cwise split: shift and typed logical operations.

#include "array_cwise_helpers.h"

// =============================================================================
// Tests for array_cwise_bitwise
// =============================================================================
TEST(ArrayCwiseBitwiseTest, Shift) {
  for (int i = 0; i < g_repeat; i++) {
    shift_test(ArrayXXi(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
    shift_test(Array<Index, Dynamic, Dynamic>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE),
                                              internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
  }
}

TEST(ArrayCwiseBitwiseTest, TypedLogicalsReal) {
  for (int i = 0; i < g_repeat; i++) {
    typed_logicals_test(ArrayX<int>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
    typed_logicals_test(ArrayX<float>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
    typed_logicals_test(ArrayX<double>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
  }
}

TEST(ArrayCwiseBitwiseTest, TypedLogicalsComplex) {
  for (int i = 0; i < g_repeat; i++) {
    typed_logicals_test(ArrayX<std::complex<float>>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
    typed_logicals_test(ArrayX<std::complex<double>>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
  }
}
