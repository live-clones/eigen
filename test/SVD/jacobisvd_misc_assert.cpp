// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/

// jacobisvd split: vector verify_assert and triangular tests.

#include "jacobisvd_helpers.h"

TEST(JacobisvdMiscAssertTest, Basic) {
  for (int i = 0; i < g_repeat; i++) {
    int r = internal::random<int>(1, 30), c = internal::random<int>(1, 30);

    TEST_SET_BUT_UNUSED_VARIABLE(r);
    TEST_SET_BUT_UNUSED_VARIABLE(c);

    (jacobisvd_verify_assert<Matrix<double, 6, 1>>());
    (jacobisvd_verify_assert<Matrix<double, 1, 6>>());
    (jacobisvd_verify_assert<Matrix<double, Dynamic, 1>>(Matrix<double, Dynamic, 1>(r)));
    (jacobisvd_verify_assert<Matrix<double, 1, Dynamic>>(Matrix<double, 1, Dynamic>(c)));
  }

  // Check that the TriangularBase constructor works
  (svd_triangular_matrix<Matrix3d>());
  (svd_triangular_matrix<Matrix4f>());
  (svd_triangular_matrix<Matrix<double, 10, 10>>());
}
