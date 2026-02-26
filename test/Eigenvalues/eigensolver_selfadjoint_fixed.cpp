// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// eigensolver_selfadjoint split: small fixed-size types (1x1, 2x2) and bug regressions.

#include "eigensolver_selfadjoint_helpers.h"

// =============================================================================
// Typed test suite for eigensolver_selfadjoint_fixed (1x1, 2x2)
// =============================================================================
template <typename T>
class EigensolverSelfadjointFixedTest : public ::testing::Test {};

using EigensolverSelfadjointFixedTypes =
    ::testing::Types<Matrix<float, 1, 1>, Matrix<double, 1, 1>, Matrix<std::complex<double>, 1, 1>, Matrix2f, Matrix2d,
                     Matrix2cd>;
TYPED_TEST_SUITE(EigensolverSelfadjointFixedTest, EigensolverSelfadjointFixedTypes);

TYPED_TEST(EigensolverSelfadjointFixedTest, SelfAdjoint) {
  for (int i = 0; i < g_repeat; i++) {
    selfadjointeigensolver(TypeParam());
  }
}

TEST(EigensolverSelfadjointFixedRegressionTest, Bug1204) { bug_1204<0>(); }
