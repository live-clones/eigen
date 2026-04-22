// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Gael Guennebaud <g.gael@free.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "sparse_solver.h"
#include <Eigen/IterativeLinearSolvers>

template <typename T, typename I_>
void test_conjugate_gradient_T() {
  typedef SparseMatrix<T, 0, I_> SparseMatrixType;
  ConjugateGradient<SparseMatrixType, Lower> cg_colmajor_lower_diag;
  ConjugateGradient<SparseMatrixType, Upper> cg_colmajor_upper_diag;
  ConjugateGradient<SparseMatrixType, Lower | Upper> cg_colmajor_loup_diag;
  ConjugateGradient<SparseMatrixType, Lower, IdentityPreconditioner> cg_colmajor_lower_I;
  ConjugateGradient<SparseMatrixType, Upper, IdentityPreconditioner> cg_colmajor_upper_I;

  CALL_SUBTEST(check_sparse_spd_solving(cg_colmajor_lower_diag));
  CALL_SUBTEST(check_sparse_spd_solving(cg_colmajor_upper_diag));
  CALL_SUBTEST(check_sparse_spd_solving(cg_colmajor_loup_diag));
  CALL_SUBTEST(check_sparse_spd_solving(cg_colmajor_lower_I));
  CALL_SUBTEST(check_sparse_spd_solving(cg_colmajor_upper_I));
}

// https://gitlab.com/libeigen/eigen/-/issues/1704
void test_fixed_size_conjugate_gradient_construction() {
  ConjugateGradient<MatrixXd> cg_dynamic;
  ConjugateGradient<Matrix3d> cg_fixed;

  Matrix3d A = Matrix3d::Identity();
  Vector3d b = Vector3d::Ones();
  cg_fixed.compute(A);
  VERIFY_IS_APPROX(cg_fixed.solve(b), b);

  (void)cg_dynamic;
}

EIGEN_DECLARE_TEST(conjugate_gradient) {
  CALL_SUBTEST_1((test_conjugate_gradient_T<double, int>()));
  CALL_SUBTEST_2((test_conjugate_gradient_T<std::complex<double>, int>()));
  CALL_SUBTEST_3((test_conjugate_gradient_T<double, long int>()));
  CALL_SUBTEST_4(test_fixed_size_conjugate_gradient_construction());
}
