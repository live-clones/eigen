// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) Essex Edwards <essex.edwards@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_RUNTIME_NO_MALLOC

#include "main.h"
#include <unsupported/Eigen/NNLS>

/// Check that 'x' solves the NNLS optimization problem `min ||A*x-b|| s.t. 0 <= x`.
template <typename MatrixType, typename VectorB, typename VectorX, typename Scalar>
void verify_nnls_optimality(const MatrixType &A, const VectorB &b, const VectorX &x, const Scalar tolerance) {
  const VectorX lambda = A.transpose() * (A * x - b);
  EXPECT_LE(0, x.minCoeff());
  EXPECT_LE(-tolerance, lambda.minCoeff());
  VERIFY(((x.array() == Scalar(0)) || (lambda.array() <= tolerance)).all());
}

template <typename MatrixType>
void test_nnls_random_problem(const MatrixType &) {
  Index cols = MatrixType::ColsAtCompileTime;
  if (cols == Dynamic) cols = internal::random<Index>(1, EIGEN_TEST_MAX_SIZE);
  Index rows = MatrixType::RowsAtCompileTime;
  if (rows == Dynamic) rows = internal::random<Index>(cols, EIGEN_TEST_MAX_SIZE);
  EXPECT_LE(cols, rows);

  using std::pow;
  using Scalar = typename MatrixType::Scalar;
  const Scalar sqrtConditionNumber = pow(Scalar(10), internal::random<Scalar>(Scalar(0), Scalar(2)));
  const Scalar scaleA = pow(Scalar(10), internal::random<Scalar>(Scalar(-3), Scalar(3)));
  const Scalar minSingularValue = scaleA / sqrtConditionNumber;
  const Scalar maxSingularValue = scaleA * sqrtConditionNumber;
  MatrixType A(rows, cols);
  generateRandomMatrixSvs(setupRangeSvs<Matrix<Scalar, Dynamic, 1>>(cols, minSingularValue, maxSingularValue), rows,
                          cols, A);

  using VectorB = decltype(A.col(0).eval());
  const Scalar scaleB = pow(Scalar(10), internal::random<Scalar>(Scalar(-3), Scalar(3)));
  const VectorB b = scaleB * VectorB::Random(A.rows());

  using std::sqrt;
  const Scalar tolerance =
      sqrt(Eigen::GenericNumTraits<Scalar>::epsilon()) * b.cwiseAbs().maxCoeff() * A.cwiseAbs().maxCoeff();
  Index max_iter = 5 * A.cols();
  NNLS<MatrixType> nnls(A, max_iter, tolerance);
  const typename NNLS<MatrixType>::SolutionVectorType &x = nnls.solve(b);

  VERIFY_IS_EQUAL(nnls.info(), ComputationInfo::Success);
  verify_nnls_optimality(A, b, x, tolerance);
}

void test_nnls_with_half_precision() {
  using Mat = Matrix<half, 8, 2>;
  using VecB = Matrix<half, 8, 1>;
  using VecX = Matrix<half, 2, 1>;
  Mat A = Mat::Random();
  VecB b = VecB::Random();

  NNLS<Mat> nnls(A, 20, half(1e-2f));
  const VecX x = nnls.solve(b);

  VERIFY_IS_EQUAL(nnls.info(), ComputationInfo::Success);
  verify_nnls_optimality(A, b, x, half(1e-1));
}

void test_nnls_handles_dependent_columns() {
  const Index rank = internal::random<Index>(1, EIGEN_TEST_MAX_SIZE / 2);
  const Index cols = 2 * rank;
  const Index rows = internal::random<Index>(cols, EIGEN_TEST_MAX_SIZE);
  const MatrixXd A = MatrixXd::Random(rows, rank) * MatrixXd::Random(rank, cols);
  const VectorXd b = VectorXd::Random(rows);

  const double tolerance = 1e-8;
  NNLS<MatrixXd> nnls(A);
  const VectorXd &x = nnls.solve(b);

  if (nnls.info() == ComputationInfo::Success) {
    verify_nnls_optimality(A, b, x, tolerance);
  }
}

void test_nnls_handles_wide_matrix() {
  const Index cols = internal::random<Index>(2, EIGEN_TEST_MAX_SIZE);
  const Index rows = internal::random<Index>(2, cols - 1);
  const MatrixXd A = MatrixXd::Random(rows, cols);
  const VectorXd b = VectorXd::Random(rows);

  const double tolerance = 1e-8;
  NNLS<MatrixXd> nnls(A);
  const VectorXd &x = nnls.solve(b);

  if (nnls.info() == ComputationInfo::Success) {
    verify_nnls_optimality(A, b, x, tolerance);
  }
}

void test_nnls_special_case_solves_in_zero_iterations() {
  const Index n = 10;
  const Index m = 3 * n;
  const VectorXd b = VectorXd::Random(m);
  MatrixXd A = MatrixXd::Random(m, n);
  const VectorXd alignment = -(A.transpose() * b).cwiseSign();
  A = A * alignment.asDiagonal();

  NNLS<MatrixXd> nnls(A);
  nnls.solve(b);

  VERIFY_IS_EQUAL(nnls.info(), ComputationInfo::Success);
  VERIFY(nnls.iterations() == 0);
}

void test_nnls_special_case_solves_in_n_iterations() {
  const Index n = 10;
  const Index m = 3 * n;
  const MatrixXd A = MatrixXd::Random(m, n);
  const VectorXd x = VectorXd::Random(n).cwiseAbs().array() + 1;
  const VectorXd b = A * x;

  NNLS<MatrixXd> nnls(A);
  nnls.solve(b);

  VERIFY_IS_EQUAL(nnls.info(), ComputationInfo::Success);
  VERIFY(nnls.iterations() == n);
}

void test_nnls_returns_NoConvergence_when_maxIterations_is_too_low() {
  const Index n = 10;
  const Index m = 3 * n;
  const MatrixXd A = MatrixXd::Random(m, n);
  const VectorXd x = VectorXd::Random(n).cwiseAbs().array() + 1;
  const VectorXd b = A * x;

  NNLS<MatrixXd> nnls(A);
  const Index max_iters = n - 1;
  nnls.setMaxIterations(max_iters);
  nnls.solve(b);

  VERIFY_IS_EQUAL(nnls.info(), ComputationInfo::NoConvergence);
  VERIFY(nnls.iterations() == max_iters);
}

void test_nnls_default_maxIterations_is_twice_column_count() {
  const Index cols = internal::random<Index>(1, EIGEN_TEST_MAX_SIZE);
  const Index rows = internal::random<Index>(cols, EIGEN_TEST_MAX_SIZE);
  const MatrixXd A = MatrixXd::Random(rows, cols);

  NNLS<MatrixXd> nnls(A);

  VERIFY_IS_EQUAL(nnls.maxIterations(), 2 * cols);
}

void test_nnls_repeated_calls_to_compute_and_solve() {
  NNLS<MatrixXd> nnls;

  for (int i = 0; i < 4; ++i) {
    const Index cols = internal::random<Index>(1, EIGEN_TEST_MAX_SIZE);
    const Index rows = internal::random<Index>(cols, EIGEN_TEST_MAX_SIZE);
    const MatrixXd A = MatrixXd::Random(rows, cols);

    nnls.compute(A);
    VERIFY_IS_EQUAL(nnls.info(), ComputationInfo::Success);

    for (int j = 0; j < 3; ++j) {
      const VectorXd b = VectorXd::Random(rows);
      const VectorXd x = nnls.solve(b);
      VERIFY_IS_EQUAL(nnls.info(), ComputationInfo::Success);
      verify_nnls_optimality(A, b, x, 1e-4);
    }
  }
}

TEST(NnlsTest, Random) {
  for (int i = 0; i < g_repeat; i++) {
    test_nnls_random_problem(MatrixXf());
    test_nnls_random_problem(MatrixXd());
    test_nnls_random_problem(Matrix<double, 12, 5>());
    test_nnls_with_half_precision();

    test_nnls_handles_dependent_columns();
    test_nnls_handles_wide_matrix();

    test_nnls_special_case_solves_in_zero_iterations();
    test_nnls_special_case_solves_in_n_iterations();
    test_nnls_returns_NoConvergence_when_maxIterations_is_too_low();
    test_nnls_default_maxIterations_is_twice_column_count();
    test_nnls_repeated_calls_to_compute_and_solve();
  }
}
