// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// eigensolver_selfadjoint split: small fixed-size types (1x1, 2x2) and bug regressions.

#include "eigensolver_selfadjoint_helpers.h"

EIGEN_DECLARE_TEST(eigensolver_selfadjoint_fixed) {
  for (int i = 0; i < g_repeat; i++) {
    // trivial test for 1x1 matrices:
    selfadjointeigensolver(Matrix<float, 1, 1>());
    selfadjointeigensolver(Matrix<double, 1, 1>());
    selfadjointeigensolver(Matrix<std::complex<double>, 1, 1>());

    // very important to test 2x2 matrices since we provide special paths for them
    selfadjointeigensolver(Matrix2f());
    selfadjointeigensolver(Matrix2d());
    selfadjointeigensolver(Matrix2cd());

    selfadjointeigensolver(Matrix<double, 1, 1>());
    selfadjointeigensolver(Matrix<double, 2, 2>());
  }

  bug_1204<0>();
}
