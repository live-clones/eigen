// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// eigensolver_selfadjoint split: small fixed-size complex types (1x1, 2x2).

#include "eigensolver_selfadjoint_helpers.h"

TEST(EigensolverSelfadjointFixedTest, Complex) {
  for (int i = 0; i < g_repeat; i++) {
    selfadjointeigensolver(Matrix<std::complex<double>, 1, 1>());
    selfadjointeigensolver(Matrix2cd());
  }
}
