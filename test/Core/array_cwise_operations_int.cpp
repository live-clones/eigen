// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// array_cwise split: generic operations, comparisons, and min/max for
// double, complex, and integer types.

#include "array_cwise_helpers.h"

// =============================================================================
// Tests for array_cwise_operations_int
// =============================================================================
TEST(ArrayCwiseOperationsIntTest, GenericMixed) {
  for (int i = 0; i < g_repeat; i++) {
    array_generic(Array44d());
    array_generic(
        ArrayXXcf(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
    array_generic(
        ArrayXXi(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
  }
}

TEST(ArrayCwiseOperationsIntTest, GenericIntegerTypes) {
  for (int i = 0; i < g_repeat; i++) {
    array_generic(Array<Index, Dynamic, Dynamic>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE),
                                                 internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
    array_generic(Array<uint32_t, Dynamic, Dynamic>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE),
                                                    internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
    array_generic(Array<uint64_t, Dynamic, Dynamic>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE),
                                                    internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
  }
}

TEST(ArrayCwiseOperationsIntTest, Comparisons) {
  for (int i = 0; i < g_repeat; i++) {
    comparisons(Array44d());
    comparisons(ArrayXXi(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
  }
}

TEST(ArrayCwiseOperationsIntTest, MinMax) {
  for (int i = 0; i < g_repeat; i++) {
    min_max(Array44d());
    min_max(ArrayXXi(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
  }
}
