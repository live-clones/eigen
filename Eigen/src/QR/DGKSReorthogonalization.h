// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Pavel Guzenfeld
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_DGKS_REORTHOGONALIZATION_H
#define EIGEN_DGKS_REORTHOGONALIZATION_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

template <bool AccumCoeffs, typename MatrixType, typename VectorType, typename CoeffVectorType>
typename NumTraits<typename VectorType::Scalar>::Real dgks_orthogonalize_impl(const MatrixType& Q, VectorType& v,
                                                                               Index numCols, CoeffVectorType& coeffs,
                                                                               Index maxIter) {
  using Scalar = typename VectorType::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;

  eigen_assert(numCols >= 0 && numCols <= Q.cols());
  eigen_assert(v.size() == Q.rows());
  eigen_assert(maxIter >= 0);

  constexpr RealScalar kappa = RealScalar(0.70710678118654752440084);  // 1/sqrt(2)

  RealScalar normBefore = v.stableNorm();
  Vector<Scalar, Dynamic> h;

  for (Index iter = 0; iter < maxIter; ++iter) {
    auto Qblock = Q.leftCols(numCols);
    h.noalias() = Qblock.adjoint() * v;
    v -= Qblock * h;

    EIGEN_IF_CONSTEXPR(AccumCoeffs) { coeffs += h; }

    RealScalar normAfter = v.stableNorm();
    if (normAfter >= kappa * normBefore) return normAfter;
    normBefore = normAfter;
  }

  return normBefore;
}

/** \internal
 * \brief DGKS (Daniel-Gragg-Kaufman-Stewart) iterative reorthogonalization.
 *
 * Orthogonalizes a vector \a v against the orthonormal columns of \a Q
 * using Classical Gram-Schmidt with the DGKS reorthogonalization criterion.
 *
 * After each CGS pass, the algorithm checks whether the norm dropped by
 * more than a factor of kappa = 1/sqrt(2). If so, it performs another
 * orthogonalization pass. This is repeated up to \a maxIter times.
 *
 * When Q columns are orthonormal, CGS with DGKS reorthogonalization
 * achieves the same numerical stability as MGS while using BLAS-2
 * operations (GEMV) that are significantly faster on modern hardware.
 *
 * \param[in]  Q        Matrix whose first \a numCols columns are orthonormal
 * \param[in,out] v     Vector to orthogonalize (modified in place)
 * \param[in]  numCols  Number of columns of Q to orthogonalize against
 * \param[out] coeffs   Vector that accumulates the projection coefficients (Q^* v)
 * \param[in]  maxIter  Maximum number of reorthogonalization passes (default 2)
 *
 * \returns The stable norm of v after orthogonalization.
 *
 * Reference: J.W. Daniel, W.B. Gragg, L. Kaufman, G.W. Stewart,
 *            "Reorthogonalization and stable algorithms for updating the
 *            Gram-Schmidt QR factorization", Math. Comp. 30 (1976), 772-795.
 */
template <typename MatrixType, typename VectorType, typename CoeffVectorType>
typename NumTraits<typename VectorType::Scalar>::Real dgks_orthogonalize(const MatrixType& Q, VectorType& v,
                                                                          Index numCols, CoeffVectorType& coeffs,
                                                                          Index maxIter = 2) {
  eigen_assert(coeffs.size() == numCols);
  return dgks_orthogonalize_impl<true>(Q, v, numCols, coeffs, maxIter);
}

/** \internal
 * \brief Simplified overload without coefficient accumulation.
 */
template <typename MatrixType, typename VectorType>
typename NumTraits<typename VectorType::Scalar>::Real dgks_orthogonalize(const MatrixType& Q, VectorType& v,
                                                                          Index numCols, Index maxIter = 2) {
  using Scalar = typename VectorType::Scalar;
  Vector<Scalar, Dynamic> dummy;
  return dgks_orthogonalize_impl<false>(Q, v, numCols, dummy, maxIter);
}

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_DGKS_REORTHOGONALIZATION_H
