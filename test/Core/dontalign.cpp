// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_DONT_ALIGN

#include "main.h"
#include <Eigen/Dense>

template <typename MatrixType>
void dontalign(const MatrixType& m) {
  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> VectorType;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime> SquareMatrixType;

  Index rows = m.rows();
  Index cols = m.cols();

  MatrixType a = MatrixType::Random(rows, cols);
  SquareMatrixType square = SquareMatrixType::Random(rows, rows);
  VectorType v = VectorType::Random(rows);

  VERIFY_IS_APPROX(v, square * square.colPivHouseholderQr().solve(v));
  square = square.inverse().eval();
  a = square * a;
  square = square * square;
  v = square * v;
  v = a.adjoint() * v;
  VERIFY(square.determinant() != Scalar(0));

  // bug 219: MapAligned() was giving an assert with EIGEN_DONT_ALIGN, because Map Flags were miscomputed
  Scalar* array = internal::aligned_new<Scalar>(rows);
  v = VectorType::MapAligned(array, rows);
  internal::aligned_delete(array, rows);
}


// =============================================================================
// Typed test suite for dontalign
// =============================================================================
template <typename T>
class DontAlignTest : public ::testing::Test {};

using DontAlignTypes = ::testing::Types<Matrix3d, Matrix4f, Matrix3cd, Matrix4cf, Matrix<float, 32, 32>,
                                        Matrix<std::complex<float>, 32, 32>, MatrixXd, MatrixXcf>;
TYPED_TEST_SUITE(DontAlignTest, DontAlignTypes);

TYPED_TEST(DontAlignTest, DontAlign) {
  for (int i = 0; i < g_repeat; i++) {
    // Cap dynamic sizes at 32 to avoid determinant overflow/underflow in inverse tests.
    dontalign(make_square_test_matrix<TypeParam>(32));
  }
}
