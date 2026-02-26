// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015-2016 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// #define EIGEN_DONT_VECTORIZE
// #define EIGEN_MAX_ALIGN_BYTES 0
#include "sparse_solver.h"
#include <Eigen/IterativeLinearSolvers>

template <typename T, typename I_>
void test_incomplete_cholesky_T() {
  typedef SparseMatrix<T, 0, I_> SparseMatrixType;
  ConjugateGradient<SparseMatrixType, Lower, IncompleteCholesky<T, Lower, AMDOrdering<I_>>> cg_illt_lower_amd;
  ConjugateGradient<SparseMatrixType, Lower, IncompleteCholesky<T, Lower, NaturalOrdering<I_>>> cg_illt_lower_nat;
  ConjugateGradient<SparseMatrixType, Upper, IncompleteCholesky<T, Upper, AMDOrdering<I_>>> cg_illt_upper_amd;
  ConjugateGradient<SparseMatrixType, Upper, IncompleteCholesky<T, Upper, NaturalOrdering<I_>>> cg_illt_upper_nat;
  ConjugateGradient<SparseMatrixType, Upper | Lower, IncompleteCholesky<T, Lower, AMDOrdering<I_>>> cg_illt_uplo_amd;

  check_sparse_spd_solving(cg_illt_lower_amd);
  check_sparse_spd_solving(cg_illt_lower_nat);
  check_sparse_spd_solving(cg_illt_upper_amd);
  check_sparse_spd_solving(cg_illt_upper_nat);
  check_sparse_spd_solving(cg_illt_uplo_amd);
}

// =============================================================================
// Config struct + typed test suite for IncompleteCholesky
// =============================================================================
template <typename T_, typename I__>
struct IncompleteCholeskyConfig {
  using Scalar = T_;
  using Index = I__;
};

template <typename T>
class IncompleteCholeskyTest : public ::testing::Test {};

using IncompleteCholeskyTypes =
    ::testing::Types<IncompleteCholeskyConfig<double, int>, IncompleteCholeskyConfig<std::complex<double>, int>,
                     IncompleteCholeskyConfig<double, long int>>;
TYPED_TEST_SUITE(IncompleteCholeskyTest, IncompleteCholeskyTypes);

TYPED_TEST(IncompleteCholeskyTest, Basic) {
  test_incomplete_cholesky_T<typename TypeParam::Scalar, typename TypeParam::Index>();
}

// =============================================================================
// Regression tests
// =============================================================================

// regression for bug 1150
TEST(IncompleteCholeskyRegressionTest, Bug1150) {
  for (int N = 1; N < 20; ++N) {
    Eigen::MatrixXd b(N, N);
    b.setOnes();

    Eigen::SparseMatrix<double> m(N, N);
    m.reserve(Eigen::VectorXi::Constant(N, 4));
    for (int i = 0; i < N; ++i) {
      m.insert(i, i) = 1;
      m.coeffRef(i, i / 2) = 2;
      m.coeffRef(i, i / 3) = 2;
      m.coeffRef(i, i / 4) = 2;
    }

    Eigen::SparseMatrix<double> A;
    A = m * m.transpose();

    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper,
                             Eigen::IncompleteCholesky<double>>
        solver(A);
    VERIFY(solver.preconditioner().info() == Eigen::Success);
    VERIFY(solver.info() == Eigen::Success);
  }
}

TEST(IncompleteCholeskyRegressionTest, NonSPD) {
  Eigen::SparseMatrix<double> A(2, 2);
  A.insert(0, 0) = 0;
  A.insert(1, 1) = 3;

  Eigen::IncompleteCholesky<double> solver(A);

  // Recover original matrix.
  Eigen::MatrixXd M = solver.permutationP().transpose() *
                      (solver.scalingS().asDiagonal().inverse() *
                       (solver.matrixL() * solver.matrixL().transpose() -
                        solver.shift() * Eigen::MatrixXd::Identity(A.rows(), A.cols())) *
                       solver.scalingS().asDiagonal().inverse()) *
                      solver.permutationP();
  VERIFY_IS_APPROX(A.toDense(), M);
}
