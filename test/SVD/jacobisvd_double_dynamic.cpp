// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/

// jacobisvd split: thin/full option checks for dynamic double types.

#include "jacobisvd_helpers.h"

TEST(JacobisvdDoubleDynamicTest, Basic) {
  for (int i = 0; i < g_repeat; i++) {
    int r = internal::random<int>(1, 30), c = internal::random<int>(1, 30);

    TEST_SET_BUT_UNUSED_VARIABLE(r)
    TEST_SET_BUT_UNUSED_VARIABLE(c)

    (jacobisvd_thin_options<Matrix<double, Dynamic, 5>>(Matrix<double, Dynamic, 5>(r, 5)));
    (jacobisvd_full_options<Matrix<double, Dynamic, 5>>(Matrix<double, Dynamic, 5>(r, 5)));
    (jacobisvd_thin_options<Matrix<double, 5, Dynamic>>(Matrix<double, 5, Dynamic>(5, c)));
    (jacobisvd_full_options<Matrix<double, 5, Dynamic>>(Matrix<double, 5, Dynamic>(5, c)));
    (jacobisvd_thin_options<MatrixXd>(MatrixXd(r, c)));
    (jacobisvd_full_options<MatrixXd>(MatrixXd(r, c)));
  }

  (jacobisvd_thin_options<MatrixXd>(MatrixXd(internal::random<int>(EIGEN_TEST_MAX_SIZE / 4, EIGEN_TEST_MAX_SIZE / 2),
                                             internal::random<int>(EIGEN_TEST_MAX_SIZE / 4, EIGEN_TEST_MAX_SIZE / 2))));
  (jacobisvd_full_options<MatrixXd>(MatrixXd(internal::random<int>(EIGEN_TEST_MAX_SIZE / 4, EIGEN_TEST_MAX_SIZE / 2),
                                             internal::random<int>(EIGEN_TEST_MAX_SIZE / 4, EIGEN_TEST_MAX_SIZE / 2))));
}
