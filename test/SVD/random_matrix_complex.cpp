// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2021 Kolja Brix <kolja.brix@rwth-aachen.de>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// random_matrix split: complex scalar types.

#include "random_matrix_helpers.h"

// =============================================================================
// Typed test suite for random_matrix_complex
// =============================================================================
template <typename T>
class RandomMatrixComplexTest : public ::testing::Test {};

using RandomMatrixComplexTypes =
    ::testing::Types<Matrix<std::complex<float>, 12, 12>, Matrix<std::complex<float>, 7, 14>,
                     Matrix<std::complex<double>, 15, 11>, Matrix<std::complex<double>, 6, 9>, MatrixXcf, MatrixXcd>;
EIGEN_TYPED_TEST_SUITE(RandomMatrixComplexTest, RandomMatrixComplexTypes);

TYPED_TEST(RandomMatrixComplexTest, RandomMatrix) {
  for (int i = 0; i < g_repeat; i++) {
    check_random_matrix(make_test_matrix<TypeParam>());
  }
}
