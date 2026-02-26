// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2021 Kolja Brix <kolja.brix@rwth-aachen.de>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// random_matrix split: real scalar types (float and double).

#include "random_matrix_helpers.h"

// =============================================================================
// Typed test suite for random_matrix_real
// =============================================================================
template <typename T>
class RandomMatrixRealTest : public ::testing::Test {};

using RandomMatrixRealTypes = ::testing::Types<Matrix<float, 1, 1>, Matrix<float, 4, 4>, Matrix<float, 2, 3>,
                                               Matrix<float, 7, 4>, Matrix<double, 1, 1>, Matrix<double, 6, 6>,
                                               Matrix<double, 5, 3>, Matrix<double, 4, 9>, MatrixXf, MatrixXd>;
EIGEN_TYPED_TEST_SUITE(RandomMatrixRealTest, RandomMatrixRealTypes);

TYPED_TEST(RandomMatrixRealTest, RandomMatrix) {
  for (int i = 0; i < g_repeat; i++) {
    check_random_matrix(make_test_matrix<TypeParam>());
  }
}
