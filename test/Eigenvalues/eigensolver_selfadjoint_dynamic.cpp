// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// eigensolver_selfadjoint split: dynamic-size types.

#include "eigensolver_selfadjoint_helpers.h"

// =============================================================================
// Typed test suite for eigensolver_selfadjoint_dynamic
// =============================================================================
template <typename T>
class EigensolverSelfadjointDynamicTest : public ::testing::Test {};

using EigensolverSelfadjointDynamicTypes =
    ::testing::Types<MatrixXf, MatrixXd, MatrixXcd, Matrix<std::complex<double>, Dynamic, Dynamic, RowMajor>>;
TYPED_TEST_SUITE(EigensolverSelfadjointDynamicTest, EigensolverSelfadjointDynamicTypes);

TYPED_TEST(EigensolverSelfadjointDynamicTest, Dynamic) {
  for (int i = 0; i < g_repeat; i++) {
    selfadjointeigensolver(make_square_test_matrix<TypeParam>(EIGEN_TEST_MAX_SIZE / 4));
  }
}

TEST(EigensolverSelfadjointDynamicRegressionTest, TrivialCases) {
  for (int i = 0; i < g_repeat; i++) {
    selfadjointeigensolver(MatrixXd(1, 1));
    selfadjointeigensolver(MatrixXd(2, 2));
    selfadjointeigensolver(MatrixXcd(1, 1));
    selfadjointeigensolver(MatrixXcd(2, 2));
  }
}

TEST(EigensolverSelfadjointDynamicRegressionTest, ProblemSizeConstructors) {
  int s = internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 4);
  SelfAdjointEigenSolver<MatrixXf> tmp1(s);
  Tridiagonalization<MatrixXf> tmp2(s);
  TEST_SET_BUT_UNUSED_VARIABLE(s)
}
