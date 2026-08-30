// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) European Organization for Nuclear Research (CERN)
//
// Adapted from ROOT::Math::CholeskyDecomp and CERNLIB cholesky inversion routines.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_CHOLESKY_INVERSE_IMPL_H
#define EIGEN_CHOLESKY_INVERSE_IMPL_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

/**
 * \internal
 * Computes the inverse of a positive-definite Hermitian/symmetric matrix using
 * Cholesky decomposition on the data in the UPPER triangle.
 */

template <typename MatrixType, typename ResultType>
struct compute_inverse_cholesky {
  EIGEN_DEVICE_FUNC static inline void run(const MatrixType& m, ResultType& result) {
    using Scalar = typename MatrixType::Scalar;
    using RealScalar = typename NumTraits<Scalar>::Real;
    using ScratchMatrix =
        Matrix<Scalar, MatrixType::RowsAtCompileTime != Dynamic ? MatrixType::RowsAtCompileTime + 1 : Dynamic,
               MatrixType::ColsAtCompileTime, MatrixType::ColsAtCompileTime != 1 ? RowMajor : ColMajor>;

    const Index n = m.rows();
    eigen_assert(m.rows() == m.cols() && "Input must be square");
    ScratchMatrix buffer(n + 1, n);

    auto upper = buffer.topRows(fix<MatrixType::RowsAtCompileTime>(int(n)));
    auto scratch = buffer.row(fix<MatrixType::RowsAtCompileTime>(int(n))).transpose();

    for (Index i = 0; i < n; ++i) {
      const auto block = upper.topRows(i);
      scratch.head(i).array() = block.diagonal().array() * block.col(i).array();

      const RealScalar d = numext::real(m(i, i)) - numext::real(scratch.head(i).dot(block.col(i)));
      eigen_assert(d > RealScalar(0) && "Input must be positive-definite");

      upper(i, i) = Scalar(d);
      for (Index j = i + 1; j < n; ++j) upper(i, j) = (m(i, j) - scratch.head(i).dot(block.col(j))) / Scalar(d);
    }

    for (Index i = n - 1; i >= 0; --i) {
      const auto u = upper.row(i).tail(n - i);
      for (Index j = n - 1; j >= i; --j) {
        Scalar rhs = i == j ? Scalar(1) / u(0) : Scalar(0);
        if (i < n - 1) rhs -= (u.tail(n - i - 1) * result.col(j).tail(n - i - 1)).value();
        result(i, j) = rhs;
        if (j > i) result(j, i) = numext::conj(result(i, j));
      }
    }
  }
};

template <typename MatrixType>
EIGEN_DEVICE_FUNC inline typename MatrixType::PlainObject inverse_cholesky(const MatrixType& matrix) {
  eigen_assert(matrix.rows() == matrix.cols() && "You can't take the inverse of a non-square matrix!");

  typename MatrixType::PlainObject result(matrix.rows(), matrix.cols());
  compute_inverse_cholesky<MatrixType, typename MatrixType::PlainObject>::run(matrix, result);
  return result;
}

}  // end namespace internal

}  // namespace Eigen

#endif  // EIGEN_CHOLESKY_INVERSE_IMPL_H
