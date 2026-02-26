// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// eigensolver_selfadjoint split: 3x3 and 4x4 fixed-size types.

#include "eigensolver_selfadjoint_helpers.h"

// =============================================================================
// Typed test suite for eigensolver_selfadjoint 3x3/4x4
// =============================================================================
template <typename T>
class EigensolverSelfadjoint3x34x4Test : public ::testing::Test {};

using EigensolverSelfadjoint3x34x4Types = ::testing::Types<Matrix3f, Matrix3d, Matrix3cd, Matrix4d, Matrix4cd>;
EIGEN_TYPED_TEST_SUITE(EigensolverSelfadjoint3x34x4Test, EigensolverSelfadjoint3x34x4Types);

TYPED_TEST(EigensolverSelfadjoint3x34x4Test, SelfAdjoint) {
  for (int i = 0; i < g_repeat; i++) {
    selfadjointeigensolver(TypeParam());
  }
}

TEST(EigensolverSelfadjoint3x34x4RegressionTest, Bug854) { bug_854<0>(); }
TEST(EigensolverSelfadjoint3x34x4RegressionTest, Bug1014) { bug_1014<0>(); }
TEST(EigensolverSelfadjoint3x34x4RegressionTest, Bug1225) { bug_1225<0>(); }
