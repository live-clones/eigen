// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// discard stack allocation as that too bypasses malloc
#define EIGEN_STACK_ALLOCATION_LIMIT 0
// heap allocation will raise an assert if enabled at runtime
#define EIGEN_RUNTIME_NO_MALLOC

#include "main.h"
using namespace std;
template <typename MatrixType>
void diagonalview(const MatrixType& m) {
  typedef typename MatrixType::Scalar Scalar;
  enum { Rows = MatrixType::RowsAtCompileTime, Cols = MatrixType::ColsAtCompileTime };
  typedef Matrix<Scalar, Rows, 1> VectorType;
  typedef Matrix<Scalar, 1, Cols> RowVectorType;
  typedef Matrix<Scalar, Rows, Rows> SquareMatrixType;
  typedef Matrix<Scalar, Dynamic, Dynamic> DynMatrixType;
  typedef DiagonalMatrix<Scalar, Rows> LeftDiagonalMatrix;
  typedef DiagonalMatrix<Scalar, Cols> RightDiagonalMatrix;
  typedef Matrix<Scalar, Rows == Dynamic ? Dynamic : 2 * Rows, Cols == Dynamic ? Dynamic : 2 * Cols> BigMatrix;
  Index rows = m.rows();
  Index cols = m.cols();

  MatrixType m1 = MatrixType::Random(rows, cols), m2 = MatrixType::Random(rows, cols);
  VectorType v1 = VectorType::Random(rows), v2 = VectorType::Random(rows);
  RowVectorType rv1 = RowVectorType::Random(cols), rv2 = RowVectorType::Random(cols);

  LeftDiagonalMatrix ldm1(v1), ldm2(v2);
  RightDiagonalMatrix rdm1(rv1), rdm2(rv2);

  Scalar s1 = internal::random<Scalar>();

  SquareMatrixType sq_m1(v1.asDiagonal());
  VERIFY_IS_APPROX(sq_m1, v1.asDiagonal().toDenseMatrix());
  sq_m1 = v1.asDiagonal();
  VERIFY_IS_APPROX(sq_m1, v1.asDiagonal().toDenseMatrix());
  SquareMatrixType sq_m2 = v1.asDiagonal();
  VERIFY_IS_APPROX(sq_m1, sq_m2);

  Index i = internal::random<Index>(0, rows - 1);
  Index j = internal::random<Index>(0, cols - 1);

  // scalar multiple
  VERIFY_IS_APPROX(LeftDiagonalMatrix(ldm1 * s1).diagonal(), ldm1.diagonal() * s1);
  VERIFY_IS_APPROX(LeftDiagonalMatrix(s1 * ldm1).diagonal(), s1 * ldm1.diagonal());

  VERIFY_IS_APPROX(m1 * (rdm1 * s1), (m1 * rdm1) * s1);
  VERIFY_IS_APPROX(m1 * (s1 * rdm1), (m1 * rdm1) * s1);

  // Zero and Identity
  LeftDiagonalMatrix zero = LeftDiagonalMatrix::Zero(rows);
  LeftDiagonalMatrix identity = LeftDiagonalMatrix::Identity(rows);
  VERIFY_IS_APPROX(identity.diagonal().sum(), Scalar(rows));
  VERIFY_IS_APPROX(zero.diagonal().sum(), Scalar(0));
}


EIGEN_DECLARE_TEST(diagonalview) {
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(diagonalview(Matrix<float, 1, 1>()));
  }
}
