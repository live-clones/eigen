// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// eigensolver_selfadjoint split: real dynamic-size types.

#include "eigensolver_selfadjoint_helpers.h"

TEST(EigensolverSelfadjointDynamicTest, Real) {
  int s = 0;
  for (int i = 0; i < g_repeat; i++) {
    s = internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 4);
    selfadjointeigensolver(MatrixXf(s, s));
    selfadjointeigensolver(MatrixXd(s, s));

    // some trivial but implementation-wise tricky cases
    selfadjointeigensolver(MatrixXd(1, 1));
    selfadjointeigensolver(MatrixXd(2, 2));
    TEST_SET_BUT_UNUSED_VARIABLE(s);
  }

  // Test problem size constructors
  s = internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 4);
  SelfAdjointEigenSolver<MatrixXf> tmp1(s);
  Tridiagonalization<MatrixXf> tmp2(s);
  TEST_SET_BUT_UNUSED_VARIABLE(s);
}
