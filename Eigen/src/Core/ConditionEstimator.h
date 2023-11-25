// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Rasmus Munk Larsen (rmlarsen@google.com)
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CONDITIONESTIMATOR_H
#define EIGEN_CONDITIONESTIMATOR_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

// compute parallel columns of matrix, where all entries are +/-1
// for any pair of parallel columns, at least one is returned as being parallel
inline VectorX<bool> rcond_get_parallel_cols(const MatrixXi &m) {
  const Index rows = m.rows();
  const MatrixXi m_T_m = (m.transpose() * m).triangularView<StrictlyLower>();

  // fails for some reason:
  // const ArrayXi norms=m_T_m.colwise().lpNorm<Infinity>();
  // workaround:
  const Index cols = m.cols();
  ArrayXi norms = ArrayXi(cols);
  for (Index i = 0; i < cols; ++i) norms[i] = m_T_m.col(i).cwiseAbs().maxCoeff();
  const VectorX<bool> parallel_cols = (norms == rows);
  return parallel_cols;
}

// compute columns of m1 with parallel column in m2
// entries of m1 and m2 are all +/-1
inline VectorX<bool> rcond_get_parallel_cols(const MatrixXi &m1, const MatrixXi &m2) {
  const Index rows = m1.rows();
  const MatrixXi m2_T_m1 = (m2.transpose() * m1);

  // fails for some reason:
  // const ArrayXi norms{m2_T_m1.colwise().template lpNorm<Infinity>()};
  // workaround:
  const Index cols = m1.cols();
  ArrayXi norms = ArrayXi(cols);
  for (Index i = 0; i < cols; ++i) norms[i] = m2_T_m1.col(i).cwiseAbs().maxCoeff();
  const VectorX<bool> parallel_cols = (norms == rows);
  return parallel_cols;
}

// return sorted indices of v in descending order
template <typename Vector>
VectorX<Index> rcond_argsort(const Vector &v) {
  VectorX<Index> idxs = VectorX<Index>::LinSpaced(v.rows(), 0, v.rows() - 1);
  std::sort(std::begin(idxs), std::end(idxs), [&](Index i1, Index i2) { return v[i1] > v[i2]; });

  return idxs;
}

/**
 * \returns an estimate of ||inv(matrix)||_1 given a decomposition of
 * \a matrix that implements .solve() and .adjoint().solve() methods.
 *
 * This function implements Algorithm 2.4 from
 *   http://eprints.ma.man.ac.uk/321/1/35608.pdf
 * which needs O(tn^2) operations for t<<n.
 *
 * The most common usage is in estimating the condition number
 * ||matrix||_1 * ||inv(matrix)||_1. The first term ||matrix||_1 can be
 * computed directly in O(n^2) operations.
 *
 * Supports the following decompositions: FullPivLU, PartialPivLU, LDLT, and
 * LLT.
 *
 * \sa FullPivLU, PartialPivLU, LDLT, LLT.
 */
template <typename Decomposition>
typename Decomposition::RealScalar rcond_invmatrix_L1_norm_estimate(const Decomposition &dec, const Index t = 1,
                                                                    const Index it_max = 10) {
  typedef typename Decomposition::MatrixType MatrixType;
  typedef typename Decomposition::Scalar Scalar;
  typedef VectorX<Scalar> Vector;

  eigen_assert(dec.rows() == dec.cols());
  const Index n = dec.rows();
  if (n == 0) return 0;

  eigen_assert(1 <= t && t <= n);

  // starting matrix with unit vector columns
  MatrixType X = MatrixType::Identity(n, t);

  // vector recording indices of used unit vectors
  VectorX<bool> ind_hist = VectorX<bool>::Constant(n, false);
  for (Index i{0}; i < t; ++i) ind_hist[i] = true;

  MatrixXi S = MatrixXi(n, t);

  Scalar old_est = 0.;
  Index ind_best = 0;

  Index k = 0;
  for (;;) {
    ++k;

    MatrixType Y = dec.solve(X);
    Index max_idx;
    Scalar est = Y.colwise().template lpNorm<1>().maxCoeff(&max_idx);
    if (est > old_est || k == 2) ind_best = max_idx;

    // (1)
    if (k >= 2 && est <= old_est) break;
    old_est = est;
    const MatrixXi S_old = S;
    if (k > it_max) return est;
    S = 2 * MatrixX<bool>(Y.array() >= 0).cast<int>() - MatrixXi::Ones(n, t);

    // (2)
    const VectorX<bool> parallel_cols_S = internal::rcond_get_parallel_cols(S);
    Index col_ctr = parallel_cols_S.cast<Index>().sum();
    if (col_ctr == t) return est;
    if (t > 1) {
      const ArrayX<bool> parallel_cols_S_S_old{internal::rcond_get_parallel_cols(S, S_old)};
      for (Index i = 0; i < t; ++i)
        if (parallel_cols_S[i] || parallel_cols_S_S_old[i]) {
          S.col(i) = 2 * VectorX<bool>(VectorX<bool>::Random(n)).cast<int>() - VectorX<int>::Ones(n);
        }
    }

    // (3)
    const MatrixType Z = dec.adjoint().solve(S.cast<Scalar>());
    const Vector h = Z.rowwise().template lpNorm<Infinity>();

    // (4)
    VectorX<Index> sorted_ind = internal::rcond_argsort<Vector>(h);
    if (k >= 2 && sorted_ind[0] == ind_best) return est;

    // (5)
    if (t > 1) {
      Index ind_in_hist_ctr = 0;
      for (Index i = 0; i < t; ++i) ind_in_hist_ctr += ind_hist[sorted_ind[i]];
      if (ind_in_hist_ctr == t) return est;
    }
    X = MatrixType::Zero(n, t);

    Index col_ind = 0;
    for (Index i = 0; i < n; ++i) {
      if (!ind_hist[sorted_ind[i]]) {
        X(sorted_ind[i], col_ind++) = Scalar(1);
        ind_hist[sorted_ind[i]] = true;
      }
      if (col_ind == t) break;
    }
  }
  return old_est;
}

/** \brief Reciprocal condition number estimator.
 *
 * Computing a decomposition of a dense matrix takes O(n^3) operations, while
 * this method estimates the condition number quickly and reliably in O(n^2)
 * operations.
 *
 * \returns an estimate of the reciprocal condition number
 * (1 / (||matrix||_1 * ||inv(matrix)||_1)) of matrix, given ||matrix||_1 and
 * its decomposition. Supports the following decompositions: FullPivLU,
 * PartialPivLU, LDLT, and LLT.
 *
 * \sa FullPivLU, PartialPivLU, LDLT, LLT.
 */
template <typename Decomposition>
typename Decomposition::RealScalar rcond_estimate_helper(typename Decomposition::RealScalar matrix_norm,
                                                         const Decomposition &dec) {
  typedef typename Decomposition::RealScalar RealScalar;
  eigen_assert(dec.rows() == dec.cols());
  if (dec.rows() == 0) return NumTraits<RealScalar>::infinity();
  if (is_exactly_zero(matrix_norm)) return RealScalar(0);
  if (dec.rows() == 1) return RealScalar(1);
  const RealScalar inverse_matrix_norm = rcond_invmatrix_L1_norm_estimate(dec);
  return (is_exactly_zero(inverse_matrix_norm) ? RealScalar(0) : (RealScalar(1) / inverse_matrix_norm) / matrix_norm);
}

}  // namespace internal

}  // namespace Eigen

#endif