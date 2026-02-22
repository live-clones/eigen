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

template <typename MatrixType, typename VectorB, typename VectorX>
void test_nnls_known_solution(const MatrixType &A, const VectorB &b, const VectorX &x_expected) {
  using Scalar = typename MatrixType::Scalar;

  using std::sqrt;
  const Scalar tolerance = sqrt(Eigen::GenericNumTraits<Scalar>::epsilon());
  Index max_iter = 5 * A.cols();
  NNLS<MatrixType> nnls(A, max_iter, tolerance);
  const VectorX x = nnls.solve(b);

  VERIFY_IS_EQUAL(nnls.info(), ComputationInfo::Success);
  VERIFY_IS_APPROX(x, x_expected);
  verify_nnls_optimality(A, b, x, tolerance);
}

void test_nnls_known_1() {
  Matrix<double, 4, 2> A(4, 2);
  Matrix<double, 4, 1> b(4);
  Matrix<double, 2, 1> x(2);
  A << 1, 1, 2, 4, 3, 9, 4, 16;
  b << 0.6, 2.2, 4.8, 8.4;
  x << 0.1, 0.5;
  return test_nnls_known_solution(A, b, x);
}

void test_nnls_known_2() {
  Matrix<double, 4, 3> A(4, 3);
  Matrix<double, 4, 1> b(4);
  Matrix<double, 3, 1> x(3);
  A << 1, 1, 1, 2, 4, 8, 3, 9, 27, 4, 16, 64;
  b << 0.73, 3.24, 8.31, 16.72;
  x << 0.1, 0.5, 0.13;
  test_nnls_known_solution(A, b, x);
}

void test_nnls_known_3() {
  Matrix<double, 4, 4> A(4, 4);
  Matrix<double, 4, 1> b(4);
  Matrix<double, 4, 1> x(4);
  A << 1, 1, 1, 1, 2, 4, 8, 16, 3, 9, 27, 81, 4, 16, 64, 256;
  b << 0.73, 3.24, 8.31, 16.72;
  x << 0.1, 0.5, 0.13, 0;
  test_nnls_known_solution(A, b, x);
}

void test_nnls_known_4() {
  Matrix<double, 4, 3> A(4, 3);
  Matrix<double, 4, 1> b(4);
  Matrix<double, 3, 1> x(3);
  A << 1, 1, 1, 2, 4, 8, 3, 9, 27, 4, 16, 64;
  b << 0.23, 1.24, 3.81, 8.72;
  x << 0.1, 0, 0.13;
  test_nnls_known_solution(A, b, x);
}

void test_nnls_known_5() {
  Matrix<double, 4, 3> A(4, 3);
  Matrix<double, 4, 1> b(4);
  Matrix<double, 3, 1> x(3);
  A << 1, 1, 1, 2, 4, 8, 3, 9, 27, 4, 16, 64;
  b << 0.13, 0.84, 2.91, 7.12;
  x << 0.0, 0.0, 0.1106544;
  test_nnls_known_solution(A, b, x);
}

void test_nnls_handles_zero_rhs() {
  const Index cols = internal::random<Index>(1, EIGEN_TEST_MAX_SIZE);
  const Index rows = internal::random<Index>(cols, EIGEN_TEST_MAX_SIZE);
  const MatrixXd A = MatrixXd::Random(rows, cols);
  const VectorXd b = VectorXd::Zero(rows);
  NNLS<MatrixXd> nnls(A);
  const VectorXd x = nnls.solve(b);
  VERIFY_IS_EQUAL(nnls.info(), ComputationInfo::Success);
  EXPECT_LE(nnls.iterations(), 1);
  VERIFY_IS_EQUAL(x, VectorXd::Zero(cols));
}

void test_nnls_handles_Mx0_matrix() {
  const Index rows = internal::random<Index>(1, EIGEN_TEST_MAX_SIZE);
  const MatrixXd A(rows, 0);
  const VectorXd b = VectorXd::Random(rows);
  NNLS<MatrixXd> nnls(A);
  const VectorXd x = nnls.solve(b);
  VERIFY_IS_EQUAL(nnls.info(), ComputationInfo::Success);
  EXPECT_LE(nnls.iterations(), 0);
  VERIFY_IS_EQUAL(x.size(), 0);
}

void test_nnls_handles_0x0_matrix() {
  const MatrixXd A(0, 0);
  const VectorXd b(0);
  NNLS<MatrixXd> nnls(A);
  const VectorXd x = nnls.solve(b);
  VERIFY_IS_EQUAL(nnls.info(), ComputationInfo::Success);
  EXPECT_LE(nnls.iterations(), 0);
  VERIFY_IS_EQUAL(x.size(), 0);
}

TEST(NnlsTest, Known) {
  test_nnls_known_1();
  test_nnls_known_2();
  test_nnls_known_3();
  test_nnls_known_4();
  test_nnls_known_5();
  test_nnls_handles_Mx0_matrix();
  test_nnls_handles_0x0_matrix();

  for (int i = 0; i < g_repeat; i++) {
    test_nnls_handles_zero_rhs();
  }
}
