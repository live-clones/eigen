// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/

// bdcsvd split: compare, inf/nan, and misc tests.

#include "bdcsvd_helpers.h"

TEST(BDCSVDCompareTest, CompareWithJacobi) {
  for (int i = 0; i < g_repeat; i++) {
    int r = internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 2), c = internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 2);

    TEST_SET_BUT_UNUSED_VARIABLE(r)
    TEST_SET_BUT_UNUSED_VARIABLE(c)

    compare_bdc_jacobi<MatrixXf>(MatrixXf(r, c));
    compare_bdc_jacobi<MatrixXd>(MatrixXd(r, c));
    compare_bdc_jacobi<MatrixXcd>(MatrixXcd(r, c));
  }
}

TEST(BDCSVDCompareTest, InfNan) {
  for (int i = 0; i < g_repeat; i++) {
    svd_inf_nan<MatrixXf>();
    svd_inf_nan<MatrixXd>();
  }
}

TEST(BDCSVDCompareTest, ProblemSizeConstructor) { BDCSVD<MatrixXf>(10, 10); }

TEST(BDCSVDCompareTest, Underoverflow) { svd_underoverflow<void>(); }

TEST(BDCSVDCompareTest, TotalDeflation) {
  // Without total deflation issues.
  compare_bdc_jacobi_instance(true);
  compare_bdc_jacobi_instance(false);

  // With total deflation issues before, when it shouldn't be triggered.
  compare_bdc_jacobi_instance(true, 3);
  compare_bdc_jacobi_instance(false, 3);
}

// Convergence for large constant matrix (https://gitlab.com/libeigen/eigen/-/issues/2491)
TEST(BDCSVDCompareTest, Convergence) { bdcsvd_check_convergence<MatrixXf>(MatrixXf::Constant(500, 500, 1)); }
