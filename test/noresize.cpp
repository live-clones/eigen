// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Test that EIGEN_NO_AUTOMATIC_RESIZING properly allows assignment to
// default-constructed (empty) matrices while preventing size-mismatched
// assignments.
#define EIGEN_NO_AUTOMATIC_RESIZING

#include "main.h"

template <typename MatrixType>
void noresize_assign_to_empty(const MatrixType& m) {
  // Assigning to a default-constructed (empty) destination should resize it.
  const Index rows = m.rows();
  const Index cols = m.cols();

  MatrixType src(rows, cols);
  src.setRandom();

  // Assignment to empty matrix should work: resize from 0x0 to rows x cols.
  MatrixType dst;
  VERIFY(dst.rows() == 0 || (MatrixType::RowsAtCompileTime != Dynamic));
  VERIFY(dst.cols() == 0 || (MatrixType::ColsAtCompileTime != Dynamic));

  dst = src;
  VERIFY_IS_EQUAL(dst.rows(), rows);
  VERIFY_IS_EQUAL(dst.cols(), cols);
  VERIFY_IS_APPROX(dst, src);
}

template <typename MatrixType>
void noresize_assign_expression_to_empty(const MatrixType& m) {
  // Assigning an expression to a default-constructed destination should work.
  const Index rows = m.rows();
  const Index cols = m.cols();

  MatrixType a(rows, cols), b(rows, cols);
  a.setRandom();
  b.setRandom();

  MatrixType dst;
  dst = a + b;
  VERIFY_IS_EQUAL(dst.rows(), rows);
  VERIFY_IS_EQUAL(dst.cols(), cols);
  VERIFY_IS_APPROX(dst, a + b);
}

template <typename MatrixType>
void noresize_col_access(const MatrixType& m) {
  // Verify that col() works on a matrix that was assigned from empty.
  const Index rows = m.rows();
  const Index cols = m.cols();
  if (cols == 0) return;

  MatrixType src(rows, cols);
  src.setRandom();

  MatrixType dst;
  dst = src;
  VERIFY_IS_EQUAL(dst.rows(), rows);
  VERIFY_IS_EQUAL(dst.cols(), cols);

  // Access each column.
  for (Index j = 0; j < cols; ++j) {
    VERIFY_IS_EQUAL(dst.col(j).rows(), rows);
    VERIFY_IS_APPROX(dst.col(j), src.col(j));
  }
}

template <typename MatrixType>
void noresize_construct_from_expression(const MatrixType& m) {
  // Construction from an expression should also work.
  const Index rows = m.rows();
  const Index cols = m.cols();

  MatrixType a(rows, cols);
  a.setRandom();

  MatrixType dst = a * 2;
  VERIFY_IS_EQUAL(dst.rows(), rows);
  VERIFY_IS_EQUAL(dst.cols(), cols);
  VERIFY_IS_APPROX(dst, a * 2);
}

template <typename MatrixType>
void noresize_size_mismatch(const MatrixType& m) {
  // Assigning to a non-empty matrix with mismatched dimensions should assert.
  const Index rows = m.rows();
  const Index cols = m.cols();
  if (rows < 2 || cols < 2) return;
  if (MatrixType::RowsAtCompileTime != Dynamic && MatrixType::ColsAtCompileTime != Dynamic) return;

  MatrixType src(rows, cols);
  src.setRandom();

  // Create a destination with different dimensions.
  MatrixType dst(rows - 1, cols - 1);
  dst.setRandom();

  VERIFY_RAISES_ASSERT(dst = src);
}

EIGEN_DECLARE_TEST(noresize) {
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(noresize_assign_to_empty(MatrixXf(10, 10)));
    CALL_SUBTEST_1(noresize_assign_to_empty(MatrixXd(7, 13)));
    CALL_SUBTEST_1(noresize_assign_to_empty(MatrixXcf(5, 5)));
    CALL_SUBTEST_1(noresize_assign_to_empty(MatrixXcd(8, 12)));
    CALL_SUBTEST_1(noresize_assign_to_empty(ArrayXXd(10, 10)));
    CALL_SUBTEST_1(noresize_assign_to_empty(ArrayXXcd(10, 30)));
    CALL_SUBTEST_1(noresize_assign_to_empty(VectorXf(20)));
    CALL_SUBTEST_1(noresize_assign_to_empty(RowVectorXd(15)));

    CALL_SUBTEST_2(noresize_assign_expression_to_empty(MatrixXd(10, 10)));
    CALL_SUBTEST_2(noresize_assign_expression_to_empty(ArrayXXcd(8, 12)));
    CALL_SUBTEST_2(noresize_assign_expression_to_empty(VectorXf(20)));

    CALL_SUBTEST_3(noresize_col_access(MatrixXd(10, 30)));
    CALL_SUBTEST_3(noresize_col_access(MatrixXcd(8, 12)));
    CALL_SUBTEST_3(noresize_col_access(ArrayXXd(5, 20)));
    CALL_SUBTEST_3(noresize_col_access(ArrayXXcd(10, 30)));

    CALL_SUBTEST_4(noresize_construct_from_expression(MatrixXd(10, 10)));
    CALL_SUBTEST_4(noresize_construct_from_expression(ArrayXXcd(5, 15)));

    CALL_SUBTEST_5(noresize_size_mismatch(MatrixXd(10, 10)));
    CALL_SUBTEST_5(noresize_size_mismatch(ArrayXXcd(8, 12)));
    CALL_SUBTEST_5(noresize_size_mismatch(VectorXf(20)));
  }
}
