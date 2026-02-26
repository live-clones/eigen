// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/

// jacobisvd split: vector verify_assert, inf/nan, method, triangular, misc.

#include "jacobisvd_helpers.h"

TEST(JacobisvdMiscTest, InfNan) {
  for (int i = 0; i < g_repeat; i++) {
    svd_inf_nan<MatrixXf>();
    svd_inf_nan<MatrixXd>();
  }
}

TEST(JacobisvdMiscTest, VerifyAssert) {
  for (int i = 0; i < g_repeat; i++) {
    int r = internal::random<int>(1, 30), c = internal::random<int>(1, 30);

    TEST_SET_BUT_UNUSED_VARIABLE(r)
    TEST_SET_BUT_UNUSED_VARIABLE(c)

    jacobisvd_verify_assert<Matrix<double, 6, 1>>();
    jacobisvd_verify_assert<Matrix<double, 1, 6>>();
    jacobisvd_verify_assert<Matrix<double, Dynamic, 1>>(Matrix<double, Dynamic, 1>(r));
    jacobisvd_verify_assert<Matrix<double, 1, Dynamic>>(Matrix<double, 1, Dynamic>(c));
  }
}

TEST(JacobisvdMiscTest, Method) {
  jacobisvd_method<Matrix2cd>();
  jacobisvd_method<Matrix3f>();
}

TEST(JacobisvdMiscTest, ProblemSizeConstructor) { JacobiSVD<MatrixXf>(10, 10); }

TEST(JacobisvdMiscTest, Preallocation) { svd_preallocate<void>(); }

TEST(JacobisvdMiscTest, Underoverflow) { svd_underoverflow<void>(); }

TEST(JacobisvdMiscTest, TriangularMatrix) {
  svd_triangular_matrix<Matrix3d>();
  svd_triangular_matrix<Matrix4f>();
  svd_triangular_matrix<Matrix<double, 10, 10>>();
}

TEST(JacobisvdMiscTest, MsvcWorkaround) { msvc_workaround(); }
