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
 * \param[out] coeffs   Optional pointer to a vector that accumulates the
 *                       projection coefficients (Q^* v). May be null.
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
                                                                         Index numCols, CoeffVectorType* coeffs,
                                                                         Index maxIter = 2) {
  typedef typename VectorType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;

  // DGKS threshold: 1/sqrt(2) ~= 0.7071
  const RealScalar kappa = RealScalar(1) / numext::sqrt(RealScalar(2));

  RealScalar normBefore = v.stableNorm();

  for (Index iter = 0; iter < maxIter; ++iter) {
    // CGS pass: compute projections and subtract.
    auto Qblock = Q.leftCols(numCols);
    Matrix<Scalar, Dynamic, 1> h = Qblock.adjoint() * v;
    v -= Qblock * h;

    if (coeffs) *coeffs += h;

    RealScalar normAfter = v.stableNorm();

    // If the norm did not drop significantly, orthogonality is sufficient.
    if (normAfter >= kappa * normBefore) return normAfter;

    normBefore = normAfter;
  }

  return normBefore;
}

/** \internal
 * \brief Simplified overload without coefficient accumulation.
 */
template <typename MatrixType, typename VectorType>
typename NumTraits<typename VectorType::Scalar>::Real dgks_orthogonalize(const MatrixType& Q, VectorType& v,
                                                                         Index numCols, Index maxIter = 2) {
  return dgks_orthogonalize(Q, v, numCols, static_cast<Matrix<typename VectorType::Scalar, Dynamic, 1>*>(nullptr),
                            maxIter);
}

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_DGKS_REORTHOGONALIZATION_H
