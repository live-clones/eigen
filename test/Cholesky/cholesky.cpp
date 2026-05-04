// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// cholesky split: real scalar types, edge cases, and assertions.
// Complex types are in cholesky_complex.cpp; blocking boundary is in
// cholesky_blocking.cpp.

#include "cholesky_helpers.h"

TEST(CholeskyTest, Basic) {
  int s = 0;
  for (int i = 0; i < g_repeat; i++) {
    cholesky(Matrix<double, 1, 1>());
    cholesky(Matrix2d());
    cholesky_bug241(Matrix2d());
    cholesky_definiteness(Matrix2d());
    cholesky(Matrix3f());
    cholesky(Matrix4d());

    s = internal::random<int>(1, EIGEN_TEST_MAX_SIZE);
    cholesky(MatrixXd(s, s));
    TEST_SET_BUT_UNUSED_VARIABLE(s);
  }
  // empty matrix, regression test for Bug 785:
  cholesky(MatrixXd(0, 0));

  cholesky_verify_assert<Matrix3f>();
  cholesky_verify_assert<Matrix3d>();
  cholesky_verify_assert<MatrixXf>();
  cholesky_verify_assert<MatrixXd>();

  // Test problem size constructors
  LLT<MatrixXf>(10);
  LDLT<MatrixXf>(10);

  cholesky_faillure_cases<void>();

  TEST_SET_BUT_UNUSED_VARIABLE(nb_temporaries);
}
