// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Gael Guennebaud <g.gael@free.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

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

  check_sparse_spd_solving(cg_colmajor_lower_diag);
  check_sparse_spd_solving(cg_colmajor_upper_diag);
  check_sparse_spd_solving(cg_colmajor_loup_diag);
  check_sparse_spd_solving(cg_colmajor_lower_I);
  check_sparse_spd_solving(cg_colmajor_upper_I);
}

TEST(ConjugateGradientTest, Basic) {
  (test_conjugate_gradient_T<double, int>());
  (test_conjugate_gradient_T<std::complex<double>, int>());
  (test_conjugate_gradient_T<double, long int>());
}
