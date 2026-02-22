// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/

// jacobisvd split: vector verify_assert, inf/nan, method, triangular, misc.

#include "jacobisvd_helpers.h"

TEST(JacobisvdMiscTest, Basic) {
  for (int i = 0; i < g_repeat; i++) {
    int r = internal::random<int>(1, 30), c = internal::random<int>(1, 30);

    TEST_SET_BUT_UNUSED_VARIABLE(r)
    TEST_SET_BUT_UNUSED_VARIABLE(c)

    // Test on inf/nan matrix
    (svd_inf_nan<MatrixXf>());
    (svd_inf_nan<MatrixXd>());

    (jacobisvd_verify_assert<Matrix<double, 6, 1>>());
    (jacobisvd_verify_assert<Matrix<double, 1, 6>>());
    (jacobisvd_verify_assert<Matrix<double, Dynamic, 1>>(Matrix<double, Dynamic, 1>(r)));
    (jacobisvd_verify_assert<Matrix<double, 1, Dynamic>>(Matrix<double, 1, Dynamic>(c)));
  }

  // test matrixbase method
  (jacobisvd_method<Matrix2cd>());
  (jacobisvd_method<Matrix3f>());

  // Test problem size constructors
  JacobiSVD<MatrixXf>(10, 10);

  // Check that preallocation avoids subsequent mallocs
  svd_preallocate<void>();

  svd_underoverflow<void>();

  // Check that the TriangularBase constructor works
  (svd_triangular_matrix<Matrix3d>());
  (svd_triangular_matrix<Matrix4f>());
  (svd_triangular_matrix<Matrix<double, 10, 10>>());

  msvc_workaround();
}
