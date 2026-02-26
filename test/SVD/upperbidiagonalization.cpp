// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <Eigen/SVD>

template <typename MatrixType>
void upperbidiag(const MatrixType& m) {
  const Index rows = m.rows();
  const Index cols = m.cols();

  typedef Matrix<typename MatrixType::RealScalar, MatrixType::RowsAtCompileTime, MatrixType::ColsAtCompileTime>
      RealMatrixType;
  typedef Matrix<typename MatrixType::Scalar, MatrixType::ColsAtCompileTime, MatrixType::RowsAtCompileTime>
      TransposeMatrixType;

  MatrixType a = MatrixType::Random(rows, cols);
  internal::UpperBidiagonalization<MatrixType> ubd(a);
  RealMatrixType b(rows, cols);
  b.setZero();
  b.block(0, 0, cols, cols) = ubd.bidiagonal();
  MatrixType c = ubd.householderU() * b * ubd.householderV().adjoint();
  VERIFY_IS_APPROX(a, c);
  TransposeMatrixType d = ubd.householderV() * b.adjoint() * ubd.householderU().adjoint();
  VERIFY_IS_APPROX(a.adjoint(), d);
}

// =============================================================================
// Typed test suite for upperbidiagonalization
// =============================================================================
template <typename T>
class UpperbidiagonalizationTest : public ::testing::Test {};

using UpperbidiagonalizationTypes =
    ::testing::Types<MatrixXf, MatrixXd, MatrixXcf, Matrix<std::complex<double>, Dynamic, Dynamic, RowMajor>,
                     Matrix<float, 6, 4>, Matrix<float, 5, 5>, Matrix<double, 4, 3>>;
TYPED_TEST_SUITE(UpperbidiagonalizationTest, UpperbidiagonalizationTypes);

TYPED_TEST(UpperbidiagonalizationTest, Upperbidiag) {
  for (int i = 0; i < g_repeat; i++) {
    // UpperBidiagonalization requires rows >= cols, so generate rows >= cols for dynamic types.
    if (TypeParam::RowsAtCompileTime == Eigen::Dynamic) {
      int cols = Eigen::internal::random<int>(1, 20);
      int rows = Eigen::internal::random<int>(cols, 20);
      upperbidiag(TypeParam(rows, cols));
    } else {
      upperbidiag(TypeParam());
    }
  }
}
