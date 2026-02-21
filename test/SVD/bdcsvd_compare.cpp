// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/

// bdcsvd split: compare, inf/nan, and misc tests.

#include "bdcsvd_helpers.h"

EIGEN_DECLARE_TEST(bdcsvd_compare) {
  for (int i = 0; i < g_repeat; i++) {
    int r = internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 2), c = internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 2);

    TEST_SET_BUT_UNUSED_VARIABLE(r)
    TEST_SET_BUT_UNUSED_VARIABLE(c)

    (compare_bdc_jacobi<MatrixXf>(MatrixXf(r, c)));
    (compare_bdc_jacobi<MatrixXd>(MatrixXd(r, c)));
    (compare_bdc_jacobi<MatrixXcd>(MatrixXcd(r, c)));
    // Test on inf/nan matrix
    (svd_inf_nan<MatrixXf>());
    (svd_inf_nan<MatrixXd>());
  }

  // Test problem size constructors
  BDCSVD<MatrixXf>(10, 10);

  svd_underoverflow<void>();

  // Without total deflation issues.
  (compare_bdc_jacobi_instance(true));
  (compare_bdc_jacobi_instance(false));

  // With total deflation issues before, when it shouldn't be triggered.
  (compare_bdc_jacobi_instance(true, 3));
  (compare_bdc_jacobi_instance(false, 3));

  // Convergence for large constant matrix (https://gitlab.com/libeigen/eigen/-/issues/2491)
  bdcsvd_check_convergence<MatrixXf>(MatrixXf::Constant(500, 500, 1));
}
