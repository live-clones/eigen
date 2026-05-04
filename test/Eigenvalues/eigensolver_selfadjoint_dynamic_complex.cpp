// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// eigensolver_selfadjoint split: complex dynamic-size types.

#include "eigensolver_selfadjoint_helpers.h"

TEST(EigensolverSelfadjointDynamicTest, Complex) {
  int s = 0;
  for (int i = 0; i < g_repeat; i++) {
    s = internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 4);
    selfadjointeigensolver(MatrixXcd(s, s));
    selfadjointeigensolver(Matrix<std::complex<double>, Dynamic, Dynamic, RowMajor>(s, s));

    // some trivial but implementation-wise tricky cases
    selfadjointeigensolver(MatrixXcd(1, 1));
    selfadjointeigensolver(MatrixXcd(2, 2));
    TEST_SET_BUT_UNUSED_VARIABLE(s);
  }
}
