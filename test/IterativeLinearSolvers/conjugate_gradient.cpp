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

// =============================================================================
// Config struct + typed test suite for ConjugateGradient
// =============================================================================
template <typename T_, typename I__>
struct ConjugateGradientConfig {
  using Scalar = T_;
  using Index = I__;
};

template <typename T>
class ConjugateGradientTest : public ::testing::Test {};

using ConjugateGradientTypes =
    ::testing::Types<ConjugateGradientConfig<double, int>, ConjugateGradientConfig<std::complex<double>, int>,
                     ConjugateGradientConfig<double, long int>>;
EIGEN_TYPED_TEST_SUITE(ConjugateGradientTest, ConjugateGradientTypes);

TYPED_TEST(ConjugateGradientTest, Basic) {
  test_conjugate_gradient_T<typename TypeParam::Scalar, typename TypeParam::Index>();
}
