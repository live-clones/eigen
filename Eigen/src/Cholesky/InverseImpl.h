// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026, The Eigen authors.
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

// Size at which the blocked sweep begins to outperform the unblocked algorithm.
constexpr int cholesky_inverse_blocked_threshold = 48;

// Packed factorization buffer, with scratch column for dot products.
template <typename MatrixType>
using cholesky_inverse_scratch_t =
    Matrix<typename MatrixType::Scalar, MatrixType::RowsAtCompileTime,
           MatrixType::ColsAtCompileTime != Dynamic ? MatrixType::ColsAtCompileTime + 1 : Dynamic>;

// In-place LDLT factorization (as `RTDR`), where `diag(upper) = diag(D)`, `upper(i, j) = R(i, j)`
// for `j > i`.  Only reads upper triangle of `m`.
template <typename MatrixType, typename FactorType, typename ScratchType>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void cholesky_inverse_factorize(const MatrixType& m, FactorType& upper,
                                                                      ScratchType& scratch) {
  using Scalar = typename FactorType::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;
  const Index n = upper.rows();
  eigen_assert(upper.cols() == n && m.rows() == n && m.cols() == n && scratch.size() >= n);
  for (Index i = 0; i < n; ++i) {
    const auto block = upper.topRows(i);
    scratch.head(i).array() = block.diagonal().array() * block.col(i).array();

    const RealScalar d = numext::real(m(i, i)) - numext::real(scratch.head(i).dot(block.col(i)));
    eigen_assert(d > RealScalar(0) && "Input must be positive-definite");

    upper(i, i) = Scalar(d);
    for (Index j = i + 1; j < n; ++j) upper(i, j) = (m(i, j) - scratch.head(i).dot(block.col(j))) / Scalar(d);
  }
}

// Algorithm from [arXiv:1111.4144], itself an application of [takahashi].
//
// [arXiv:1111.4144]: <https://arxiv.org/abs/1111.4144>
// [takahashi]: <https://cir.nii.ac.jp/crid/1574231875636049024>
template <typename FactorType, typename ResultType, typename ScratchType>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void cholesky_inverse_unblocked(const FactorType& upper, ResultType& result,
                                                                      ScratchType& scratch) {
  using Scalar = typename FactorType::Scalar;
  const Index k = upper.rows();
  eigen_assert(upper.cols() == k && result.rows() == k && result.cols() == k && scratch.size() >= k);
  for (Index i = k - 1; i >= 0; --i) {
    const Index len = k - i - 1;
    const Scalar d = upper(i, i);
    scratch.head(len) = upper.row(i).tail(len).transpose();
    const auto u = scratch.head(len).transpose();
    for (Index j = k - 1; j >= i; --j) {
      Scalar rhs = i == j ? Scalar(1) / d : Scalar(0);
      if (len > 0) rhs -= (u * result.col(j).tail(len)).value();
      result(i, j) = rhs;
      if (j > i) result(j, i) = numext::conj(result(i, j));
    }
  }
}

// NOTE: Keep the factorization prelude here separate from that in `solve_cholesky_inverse_blocked`.
// Merging them and branching at runtime in the dynamic-size case introduces a ~5% slowdown for
// `n=16..48` since the workspace must be saved to memory to accommodate the exception handler for
// the outlined call to the blocked solver.  As written, the calls to
// `cholesky_inverse_{un,}blocked` are instead tail calls.
template <typename MatrixType, typename ResultType>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void solve_cholesky_inverse_unblocked(const MatrixType& m, ResultType& result) {
  const Index n = m.rows();
  cholesky_inverse_scratch_t<MatrixType> buffer(n, n + 1);
  auto upper = buffer.leftCols(fix<MatrixType::ColsAtCompileTime>(int(n)));
  auto scratch = buffer.col(fix<MatrixType::ColsAtCompileTime>(int(n)));

  cholesky_inverse_factorize(m, upper, scratch);
  cholesky_inverse_unblocked(upper, result, scratch);
}

// Standard block inverse with Hermitian input assumption.
template <typename FactorType, typename ResultType, typename ScratchType>
EIGEN_DEVICE_FUNC void cholesky_inverse_blocked(const FactorType& upper, ResultType& result, ScratchType& scratch) {
  using Scalar = typename FactorType::Scalar;
  const Index n = upper.rows();
  eigen_assert(upper.cols() == n && result.rows() == n && result.cols() == n && scratch.size() >= n);

  constexpr Index PacketSize = packet_traits<Scalar>::size;
  // NOTE: Empirically determined by scanning `b={8, 12, 14..19, 22..27, 32, 48}`, as well as
  // proportional block sizes.  Packet size scaling is to accommodate architectures with large
  // packets.
  constexpr Index b = numext::div_ceil(Index(24), PacketSize) * PacketSize;

  // NOTE: Workspace for computing upper right sub-block.  Solving in-place leads to poor
  // performance at large power-of-2 `n`.
  Matrix<Scalar, int(b), Dynamic> sbuf(b, n);

  Index p = n;
  while (p > 0) {
    const Index bb = numext::mini(b, p);
    p -= bb;
    const Index m = n - p - bb;

    auto x11 = result.block(p, p, bb, bb);
    const auto r11 = upper.block(p, p, bb, bb);

    cholesky_inverse_unblocked(r11, x11, scratch);
    if (m == 0) continue;

    auto s = sbuf.topLeftCorner(bb, m);
    s = -upper.block(p, p + bb, bb, m);
    r11.template triangularView<UnitUpper>().solveInPlace(s);

    auto x12 = result.block(p, p + bb, bb, m);
    x12.noalias() = s * result.block(p + bb, p + bb, m, m);

    x11.noalias() += x12 * s.adjoint();

    result.block(p + bb, p, m, bb) = x12.adjoint();
  }
}

template <typename MatrixType, typename ResultType>
EIGEN_DEVICE_FUNC void solve_cholesky_inverse_blocked(const MatrixType& m, ResultType& result) {
  const Index n = m.rows();
  cholesky_inverse_scratch_t<MatrixType> buffer(n, n + 1);
  auto upper = buffer.leftCols(fix<MatrixType::ColsAtCompileTime>(int(n)));
  auto scratch = buffer.col(fix<MatrixType::ColsAtCompileTime>(int(n)));

  cholesky_inverse_factorize(m, upper, scratch);
  cholesky_inverse_blocked(upper, result, scratch);
}

template <typename MatrixType, typename ResultType>
struct compute_inverse_cholesky {
  template <int R = MatrixType::RowsAtCompileTime, std::enable_if_t<R == Dynamic, int> = 0>
  EIGEN_DEVICE_FUNC static EIGEN_ALWAYS_INLINE void dispatch(const MatrixType& m, ResultType& result) {
    if (m.rows() >= cholesky_inverse_blocked_threshold)
      solve_cholesky_inverse_blocked(m, result);
    else
      solve_cholesky_inverse_unblocked(m, result);
  }

  template <int R = MatrixType::RowsAtCompileTime,
            std::enable_if_t<R != Dynamic && R<cholesky_inverse_blocked_threshold, int> = 0> EIGEN_DEVICE_FUNC static
            EIGEN_ALWAYS_INLINE void dispatch(const MatrixType& m, ResultType& result) {
    solve_cholesky_inverse_unblocked(m, result);
  }

  template <int R = MatrixType::RowsAtCompileTime,
            std::enable_if_t<R != Dynamic && R >= cholesky_inverse_blocked_threshold, int> = 0>
  EIGEN_DEVICE_FUNC static EIGEN_ALWAYS_INLINE void dispatch(const MatrixType& m, ResultType& result) {
    solve_cholesky_inverse_blocked(m, result);
  }

  EIGEN_DEVICE_FUNC static inline void run(const MatrixType& m, ResultType& result) {
    eigen_assert(m.rows() == m.cols() && "Input must be square");
    dispatch(m, result);
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
