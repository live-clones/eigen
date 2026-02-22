// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/

// jacobisvd split: thin/full option checks for complex types.

#include "jacobisvd_helpers.h"

TEST(JacobisvdComplexTest, Basic) {
  for (int i = 0; i < g_repeat; i++) {
    int r = internal::random<int>(1, 30), c = internal::random<int>(1, 30);

    TEST_SET_BUT_UNUSED_VARIABLE(r)
    TEST_SET_BUT_UNUSED_VARIABLE(c)

    (jacobisvd_thin_options<MatrixXcd>(MatrixXcd(r, c)));
    (jacobisvd_full_options<MatrixXcd>(MatrixXcd(r, c)));

    MatrixXcd noQRTest = MatrixXcd(r, r);
    svd_fill_random(noQRTest);
    (svd_thin_option_checks<MatrixXcd, NoQRPreconditioner>(noQRTest));
    (svd_option_checks_full_only<MatrixXcd, NoQRPreconditioner>(noQRTest));
  }

  (jacobisvd_thin_options<MatrixXcd>(
      MatrixXcd(internal::random<int>(EIGEN_TEST_MAX_SIZE / 4, EIGEN_TEST_MAX_SIZE / 3),
                internal::random<int>(EIGEN_TEST_MAX_SIZE / 4, EIGEN_TEST_MAX_SIZE / 3))));
  (jacobisvd_full_options<MatrixXcd>(
      MatrixXcd(internal::random<int>(EIGEN_TEST_MAX_SIZE / 4, EIGEN_TEST_MAX_SIZE / 3),
                internal::random<int>(EIGEN_TEST_MAX_SIZE / 4, EIGEN_TEST_MAX_SIZE / 3))));
}
