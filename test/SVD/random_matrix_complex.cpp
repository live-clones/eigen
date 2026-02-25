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

TEST(RandomMatrixTest, Complex) {
  for (int i = 0; i < g_repeat; i++) {
    check_random_matrix(Matrix<std::complex<float>, 12, 12>());
    check_random_matrix(Matrix<std::complex<float>, 7, 14>());
    check_random_matrix(Matrix<std::complex<double>, 15, 11>());
    check_random_matrix(Matrix<std::complex<double>, 6, 9>());

    check_random_matrix(
        MatrixXcf(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
    check_random_matrix(
        MatrixXcd(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
  }
}
