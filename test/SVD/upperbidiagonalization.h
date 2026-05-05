// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_TEST_UPPERBIDIAGONALIZATION_H
#define EIGEN_TEST_UPPERBIDIAGONALIZATION_H

#include "main.h"
#include <Eigen/SVD>

// Verify that U * B * V^T ≈ A for the bidiagonal decomposition.
template <typename MatrixType>
void upperbidiag_verify(const MatrixType& a) {
  const Index rows = a.rows();
  const Index cols = a.cols();

  typedef Matrix<typename MatrixType::RealScalar, MatrixType::RowsAtCompileTime, MatrixType::ColsAtCompileTime>
      RealMatrixType;
  typedef Matrix<typename MatrixType::Scalar, MatrixType::ColsAtCompileTime, MatrixType::RowsAtCompileTime>
      TransposeMatrixType;

  internal::UpperBidiagonalization<MatrixType> ubd(a);
  RealMatrixType b(rows, cols);
  b.setZero();
  b.block(0, 0, cols, cols) = ubd.bidiagonal();
  MatrixType c = ubd.householderU() * b * ubd.householderV().adjoint();
  VERIFY_IS_APPROX(a, c);
  TransposeMatrixType d = ubd.householderV() * b.adjoint() * ubd.householderU().adjoint();
  VERIFY_IS_APPROX(a.adjoint(), d);
}

// Verify the unblocked path directly.
template <typename MatrixType>
void upperbidiag_verify_unblocked(const MatrixType& a) {
  const Index rows = a.rows();
  const Index cols = a.cols();

  typedef Matrix<typename MatrixType::RealScalar, MatrixType::RowsAtCompileTime, MatrixType::ColsAtCompileTime>
      RealMatrixType;

  internal::UpperBidiagonalization<MatrixType> ubd(rows, cols);
  ubd.computeUnblocked(a);
  RealMatrixType b(rows, cols);
  b.setZero();
  b.block(0, 0, cols, cols) = ubd.bidiagonal();
  MatrixType c = ubd.householderU() * b * ubd.householderV().adjoint();
  VERIFY_IS_APPROX(a, c);
}

// Test with a random matrix of given dimensions.
template <typename MatrixType>
void upperbidiag(const MatrixType& m) {
  MatrixType a = MatrixType::Random(m.rows(), m.cols());
  upperbidiag_verify(a);
}

// Test with structured/ill-conditioned matrices.
template <typename Scalar>
void upperbidiag_structured(Index rows, Index cols) {
  typedef Matrix<Scalar, Dynamic, Dynamic> MatrixType;

  // Zero matrix.
  {
    MatrixType a = MatrixType::Zero(rows, cols);
    upperbidiag_verify(a);
  }

  // Identity (padded if non-square).
  {
    MatrixType a = MatrixType::Zero(rows, cols);
    a.block(0, 0, cols, cols).setIdentity();
    upperbidiag_verify(a);
  }

  // Matrix with large dynamic range (graded singular values).
  {
    MatrixType U = MatrixType::Random(rows, rows);
    MatrixType V = MatrixType::Random(cols, cols);
    // Orthogonalize via QR.
    Eigen::HouseholderQR<MatrixType> qrU(U), qrV(V);
    MatrixType Q_U = qrU.householderQ() * MatrixType::Identity(rows, rows);
    MatrixType Q_V = qrV.householderQ() * MatrixType::Identity(cols, cols);
    // Graded singular values: 1, 1e-2, 1e-4, ...
    MatrixType sigma = MatrixType::Zero(rows, cols);
    typedef typename NumTraits<Scalar>::Real RealScalar;
    for (Index i = 0; i < cols; ++i) {
      sigma(i, i) = static_cast<RealScalar>(std::pow(RealScalar(10), -RealScalar(2) * i / (cols > 1 ? cols - 1 : 1)));
    }
    MatrixType a = Q_U * sigma * Q_V.adjoint();
    upperbidiag_verify(a);
  }

  // Rank-1 matrix.
  {
    Matrix<Scalar, Dynamic, 1> u = Matrix<Scalar, Dynamic, 1>::Random(rows);
    Matrix<Scalar, 1, Dynamic> v = Matrix<Scalar, 1, Dynamic>::Random(cols);
    MatrixType a = u * v;
    upperbidiag_verify(a);
  }
}

#endif  // EIGEN_TEST_UPPERBIDIAGONALIZATION_H
