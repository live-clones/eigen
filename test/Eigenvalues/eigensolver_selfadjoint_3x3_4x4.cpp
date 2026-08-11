// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// eigensolver_selfadjoint split: 3x3 and 4x4 real fixed-size types.

#include "eigensolver_selfadjoint_helpers.h"

TEST(EigensolverSelfadjoint3x34x4Test, Real) {
  for (int i = 0; i < g_repeat; i++) {
    selfadjointeigensolver(Matrix3f());
    selfadjointeigensolver(Matrix3d());
    selfadjointeigensolver(Matrix4d());
  }

  bug_854<0>();
  bug_1014<0>();
  bug_1225<0>();
}

TEST(EigensolverSelfadjoint3x34x4Test, Direct2x2Stress) { direct_2x2_stress<0>(); }

TEST(EigensolverSelfadjoint3x34x4Test, Direct3x3Stress) { direct_3x3_stress<0>(); }

TEST(EigensolverSelfadjoint3x34x4Test, SelfadjointeigensolverDiagonal) {
  for (int i = 0; i < g_repeat; i++) {
    int s = internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 4);
    selfadjointeigensolver_diagonal(Matrix3d());
    selfadjointeigensolver_diagonal(MatrixXd(s, s));
  }
}

TEST(EigensolverSelfadjoint3x34x4Test, SelfadjointeigensolverExtremeEigenvalues) {
  for (int i = 0; i < g_repeat; i++) {
    int s = internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 4);
    selfadjointeigensolver_extreme_eigenvalues(Matrix3d());
    selfadjointeigensolver_extreme_eigenvalues(Matrix4d());
    selfadjointeigensolver_extreme_eigenvalues(MatrixXd(s, s));
    selfadjointeigensolver_extreme_eigenvalues(Matrix3f());
    selfadjointeigensolver_extreme_eigenvalues(MatrixXf(s, s));
    selfadjointeigensolver_extreme_eigenvalues(MatrixXd(128, 128));
  }
}

TEST(EigensolverSelfadjoint3x34x4Test, SelfadjointeigensolverInf) { selfadjointeigensolver_inf<0>(); }

TEST(EigensolverSelfadjoint3x34x4Test, SelfadjointeigensolverInverseSqrt) {
  for (int i = 0; i < g_repeat; i++) {
    int s = internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 4);
    selfadjointeigensolver_inverse_sqrt(Matrix3d());
    selfadjointeigensolver_inverse_sqrt(Matrix4d());
    selfadjointeigensolver_inverse_sqrt(MatrixXd(s, s));
    selfadjointeigensolver_inverse_sqrt(Matrix3f());
  }
}

TEST(EigensolverSelfadjoint3x34x4Test, SelfadjointeigensolverNonfinite) { selfadjointeigensolver_nonfinite<0>(); }

TEST(EigensolverSelfadjoint3x34x4Test, SelfadjointeigensolverRepeatedEigenvalues) {
  for (int i = 0; i < g_repeat; i++) {
    int s = internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 4);
    selfadjointeigensolver_repeated_eigenvalues(Matrix3d());
    selfadjointeigensolver_repeated_eigenvalues(Matrix2d());
    selfadjointeigensolver_repeated_eigenvalues(Matrix4d());
    selfadjointeigensolver_repeated_eigenvalues(MatrixXd(s, s));
    selfadjointeigensolver_repeated_eigenvalues(Matrix3f());
    selfadjointeigensolver_repeated_eigenvalues(Matrix2f());
    selfadjointeigensolver_repeated_eigenvalues(Matrix3cd());
    selfadjointeigensolver_repeated_eigenvalues(MatrixXd(128, 128));
    selfadjointeigensolver_repeated_eigenvalues(MatrixXf(128, 128));
    selfadjointeigensolver_repeated_eigenvalues(MatrixXcd(128, 128));
  }
}

TEST(EigensolverSelfadjoint3x34x4Test, SelfadjointeigensolverRowmajor) {
  for (int i = 0; i < g_repeat; i++) {
    selfadjointeigensolver_rowmajor<0>();
  }
}

TEST(EigensolverSelfadjoint3x34x4Test, SelfadjointeigensolverStructuredTridiagonal) {
  for (int i = 0; i < g_repeat; i++) {
    selfadjointeigensolver_structured_tridiagonal<double>();
    selfadjointeigensolver_structured_tridiagonal<float>();
  }
}

TEST(EigensolverSelfadjoint3x34x4Test, SelfadjointeigensolverTridiagonalScaled) {
  for (int i = 0; i < g_repeat; i++) {
    int s = internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 4);
    selfadjointeigensolver_tridiagonal_scaled(MatrixXd(s, s));
    selfadjointeigensolver_tridiagonal_scaled(MatrixXf(s, s));
  }
}

TEST(EigensolverSelfadjoint3x34x4Test, SelfadjointeigensolverTridiagonalWideRange) {
  for (int i = 0; i < g_repeat; i++) {
    selfadjointeigensolver_tridiagonal_wide_range<double>();
    selfadjointeigensolver_tridiagonal_wide_range<float>();
  }
}

TEST(EigensolverSelfadjoint3x34x4Test, SelfadjointeigensolverTridiagonalZerosized) {
  selfadjointeigensolver_tridiagonal_zerosized<0>();
}
