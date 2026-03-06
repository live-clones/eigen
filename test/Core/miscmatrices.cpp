// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

template <typename MatrixType>
void miscMatrices(const MatrixType& m) {
  /* this test covers the following files:
     DiagonalMatrix.h Ones.h
  */
  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> VectorType;
  Index rows = m.rows();
  Index cols = m.cols();

  Index r = internal::random<Index>(0, rows - 1), r2 = internal::random<Index>(0, rows - 1),
        c = internal::random<Index>(0, cols - 1);
  VERIFY_IS_APPROX(MatrixType::Ones(rows, cols)(r, c), static_cast<Scalar>(1));
  MatrixType m1 = MatrixType::Ones(rows, cols);
  VERIFY_IS_APPROX(m1(r, c), static_cast<Scalar>(1));
  VectorType v1 = VectorType::Random(rows);
  v1[0];
  Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime> square(v1.asDiagonal());
  if (r == r2)
    VERIFY_IS_APPROX(square(r, r2), v1[r]);
  else
    VERIFY_IS_MUCH_SMALLER_THAN(square(r, r2), static_cast<Scalar>(1));
  square = MatrixType::Zero(rows, rows);
  square.diagonal() = VectorType::Ones(rows);
  VERIFY_IS_APPROX(square, MatrixType::Identity(rows, rows));
}

template <typename MatrixType>
MatrixType make_test_matrix() {
  const int rows = (MatrixType::RowsAtCompileTime == Dynamic) ? internal::random<int>(1, EIGEN_TEST_MAX_SIZE)
                                                              : MatrixType::RowsAtCompileTime;
  const int cols = (MatrixType::ColsAtCompileTime == Dynamic) ? internal::random<int>(1, EIGEN_TEST_MAX_SIZE)
                                                              : MatrixType::ColsAtCompileTime;
  return MatrixType(rows, cols);
}

// =============================================================================
// Typed test suite for miscMatrices
// =============================================================================
template <typename T>
class MiscMatricesTest : public ::testing::Test {};

using MiscMatricesTypes = ::testing::Types<Matrix<float, 1, 1>, Matrix4d, MatrixXcf, MatrixXi, MatrixXcd>;
EIGEN_TYPED_TEST_SUITE(MiscMatricesTest, MiscMatricesTypes);

TYPED_TEST(MiscMatricesTest, MiscMatrices) {
  for (int i = 0; i < g_repeat; i++) {
    miscMatrices(make_test_matrix<TypeParam>());
  }
}
