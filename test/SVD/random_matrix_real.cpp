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

TEST(RandomMatrixTest, Real) {
  for (int i = 0; i < g_repeat; i++) {
    check_random_matrix(Matrix<float, 1, 1>());
    check_random_matrix(Matrix<float, 4, 4>());
    check_random_matrix(Matrix<float, 2, 3>());
    check_random_matrix(Matrix<float, 7, 4>());

    check_random_matrix(Matrix<double, 1, 1>());
    check_random_matrix(Matrix<double, 6, 6>());
    check_random_matrix(Matrix<double, 5, 3>());
    check_random_matrix(Matrix<double, 4, 9>());

    check_random_matrix(
        MatrixXf(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
    check_random_matrix(
        MatrixXd(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
  }
}
