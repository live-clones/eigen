// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_RANDCOLPIVOTINGHOUSEHOLDERQR_H
#define EIGEN_RANDCOLPIVOTINGHOUSEHOLDERQR_H

#include <random>

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

template <typename MatrixType_, typename PermutationIndex_>
struct traits<RandColPivHouseholderQR<MatrixType_, PermutationIndex_>> : traits<MatrixType_> {
  typedef MatrixXpr XprKind;
  typedef SolverStorage StorageKind;
  typedef PermutationIndex_ PermutationIndex;
  enum { Flags = 0 };
};

// Fill `mat` with iid samples drawn from a real standard normal distribution.
template <typename Derived, typename Engine>
EIGEN_STRONG_INLINE std::enable_if_t<!NumTraits<typename Derived::Scalar>::IsComplex> fill_gaussian(
    MatrixBase<Derived>& mat, Engine& engine) {
  typedef typename Derived::Scalar Scalar;
  std::normal_distribution<Scalar> dist(Scalar(0), Scalar(1));
  for (Index j = 0; j < mat.cols(); ++j)
    for (Index i = 0; i < mat.rows(); ++i) mat.coeffRef(i, j) = dist(engine);
}

// Fill `mat` with iid samples drawn from a complex standard normal
// distribution (real and imaginary parts sampled independently).
template <typename Derived, typename Engine>
EIGEN_STRONG_INLINE std::enable_if_t<NumTraits<typename Derived::Scalar>::IsComplex> fill_gaussian(
    MatrixBase<Derived>& mat, Engine& engine) {
  typedef typename Derived::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  std::normal_distribution<RealScalar> dist(RealScalar(0), RealScalar(1));
  for (Index j = 0; j < mat.cols(); ++j)
    for (Index i = 0; i < mat.rows(); ++i) mat.coeffRef(i, j) = Scalar(dist(engine), dist(engine));
}

// Stable LAWN-176 column-norm downdate after a Householder reflector has
// been applied. `pivot_entry` is the entry of the just-pivoted row above
// column `j`; `tail_norm_fn` recomputes the trailing-column norm from
// scratch when the running estimate has lost too much precision (see
// http://www.netlib.org/lapack/lawnspdf/lawn176.pdf).
template <typename RealScalar, typename Scalar, typename RecomputeFn>
EIGEN_STRONG_INLINE void lawn176_norm_downdate(RealScalar& norm_updated, RealScalar& norm_direct, Scalar pivot_entry,
                                               RealScalar downdate_threshold, RecomputeFn&& tail_norm_fn) {
  using std::abs;
  if (numext::is_exactly_zero(norm_updated)) return;
  RealScalar t = abs(pivot_entry) / norm_updated;
  t = (RealScalar(1) + t) * (RealScalar(1) - t);
  if (t < RealScalar(0)) t = RealScalar(0);
  RealScalar t2 = t * numext::abs2<RealScalar>(norm_updated / norm_direct);
  if (t2 <= downdate_threshold) {
    norm_direct = tail_norm_fn();
    norm_updated = norm_direct;
  } else {
    norm_updated *= numext::sqrt(t);
  }
}

// Run `nsteps` iterations of column-pivoted Householder QR on `sketch`
// (shape: (b+p) x ncols). Records the absolute column index selected at
// each step into `pivots[step]`. The contents of `sketch` are destroyed.
template <typename SketchDerived, typename PivotsDerived>
void determine_pivots_on_sketch(MatrixBase<SketchDerived>& sketch, Index nsteps, MatrixBase<PivotsDerived>& pivots) {
  typedef typename SketchDerived::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;

  const Index sketch_rows = sketch.rows();
  const Index ncols = sketch.cols();
  eigen_assert(nsteps <= ncols);

  Matrix<RealScalar, 1, Dynamic> norms_updated(ncols);
  Matrix<RealScalar, 1, Dynamic> norms_direct(ncols);
  for (Index j = 0; j < ncols; ++j) {
    norms_direct.coeffRef(j) = sketch.col(j).norm();
    norms_updated.coeffRef(j) = norms_direct.coeffRef(j);
  }
  const RealScalar downdate_threshold = numext::sqrt(NumTraits<RealScalar>::epsilon());
  Matrix<Scalar, 1, Dynamic> temp(ncols);

  for (Index k = 0; k < nsteps; ++k) {
    Index biggest;
    norms_updated.tail(ncols - k).maxCoeff(&biggest);
    biggest += k;
    pivots.coeffRef(k) = biggest;

    if (k != biggest) {
      sketch.col(k).swap(sketch.col(biggest));
      std::swap(norms_updated.coeffRef(k), norms_updated.coeffRef(biggest));
      std::swap(norms_direct.coeffRef(k), norms_direct.coeffRef(biggest));
    }

    // Once all sketch rows have been triangularized further reflectors
    // would be no-ops; record any remaining pivots as identity.
    if (k >= sketch_rows - 1) {
      for (Index j = k + 1; j < nsteps; ++j) pivots.coeffRef(j) = j;
      break;
    }

    Scalar tau;
    RealScalar beta;
    sketch.col(k).tail(sketch_rows - k).makeHouseholderInPlace(tau, beta);
    sketch.coeffRef(k, k) = beta;
    if (k + 1 < ncols) {
      sketch.bottomRightCorner(sketch_rows - k, ncols - k - 1)
          .applyHouseholderOnTheLeft(sketch.col(k).tail(sketch_rows - k - 1), tau, &temp.coeffRef(k + 1));
    }

    for (Index j = k + 1; j < ncols; ++j) {
      lawn176_norm_downdate(norms_updated.coeffRef(j), norms_direct.coeffRef(j), sketch.coeff(k, j), downdate_threshold,
                            [&] { return sketch.col(j).tail(sketch_rows - k - 1).norm(); });
    }
  }
}

}  // end namespace internal

/** \ingroup QR_Module
 *
 * \class RandColPivHouseholderQR
 *
 * \brief Randomized blocked Householder rank-revealing QR with column pivoting
 *
 * \tparam MatrixType_ the type of the matrix being decomposed.
 * \tparam PermutationIndex_ the type of the permutation indices.
 *
 * Computes \f$ \mathbf{A} \mathbf{P} = \mathbf{Q} \mathbf{R} \f$ using the
 * randomized block-pivoted Householder algorithm published independently by
 * Duersch and Gu (RQRCP, SIAM J. Sci. Comput. 39(4):C263–C291, 2017,
 * arXiv:1509.06820) and by Martinsson, Quintana-Ortí, Heavner, and van de
 * Geijn (HQRRP, SIAM J. Sci. Comput. 39(2):C96–C115, 2017,
 * arXiv:1512.02671 / FLAME Working Note #78).
 *
 * The classical Businger–Golub pivot scan is replaced by selecting blocks of
 * \c b pivots from a small Gaussian sketch \f$ \mathbf{Y} = \mathbf{G}
 * \mathbf{A} \f$, with \f$ \mathbf{G} \in \mathbb{R}^{(b+p) \times m} \f$
 * having i.i.d.\ standard normal entries. After each block step \f$
 * \mathbf{Y} \f$ is downdated rather than recomputed, so the asymptotic flop
 * count matches that of an unpivoted blocked Householder QR
 * (\f$ 2mn^2 - \tfrac{2}{3}n^3 \f$ for \f$ m \ge n \f$). Almost all
 * computation is BLAS-3, which lifts the BLAS-2 ceiling that limits
 * ColPivHouseholderQR.
 *
 * The pivot-quality tradeoff is empirically minor: on the matrices in the
 * cited papers, the rank-revealing behavior is comparable to classical column
 * pivoting (LAPACK \c geqp3).
 *
 * The block size \c b and oversampling \c p can be configured via
 * setBlockSize() and setOversampling(); the seed for the internal RNG can be
 * fixed with setSeed() for reproducible output.
 *
 * This class supports the \link InplaceDecomposition inplace decomposition
 * \endlink mechanism. The same public API as ColPivHouseholderQR is provided.
 *
 * \sa ColPivHouseholderQR, MatrixBase::randColPivHouseholderQr()
 */
template <typename MatrixType_, typename PermutationIndex_>
class RandColPivHouseholderQR : public SolverBase<RandColPivHouseholderQR<MatrixType_, PermutationIndex_>>,
                                public RankRevealingBase<RandColPivHouseholderQR<MatrixType_, PermutationIndex_>> {
 public:
  typedef MatrixType_ MatrixType;
  typedef SolverBase<RandColPivHouseholderQR> Base;
  typedef RankRevealingBase<RandColPivHouseholderQR> RankRevealingBase_;
  friend class SolverBase<RandColPivHouseholderQR>;
  friend class RankRevealingBase<RandColPivHouseholderQR>;
  using RankRevealingBase_::dimensionOfKernel;
  using RankRevealingBase_::isInjective;
  using RankRevealingBase_::isInvertible;
  using RankRevealingBase_::isSurjective;
  using RankRevealingBase_::maxPivot;
  using RankRevealingBase_::nonzeroPivots;
  using RankRevealingBase_::rank;
  using RankRevealingBase_::setThreshold;
  using RankRevealingBase_::threshold;
  typedef PermutationIndex_ PermutationIndex;
  EIGEN_GENERIC_PUBLIC_INTERFACE(RandColPivHouseholderQR)

  enum {
    MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
  };
  typedef typename internal::plain_diag_type<MatrixType>::type HCoeffsType;
  typedef PermutationMatrix<ColsAtCompileTime, MaxColsAtCompileTime, PermutationIndex> PermutationType;
  typedef typename internal::plain_row_type<MatrixType, PermutationIndex>::type IntRowVectorType;
  typedef typename internal::plain_row_type<MatrixType>::type RowVectorType;
  typedef typename internal::plain_row_type<MatrixType, RealScalar>::type RealRowVectorType;
  typedef HouseholderSequence<MatrixType, internal::remove_all_t<typename HCoeffsType::ConjugateReturnType>>
      HouseholderSequenceType;
  typedef typename MatrixType::PlainObject PlainObject;

 private:
  // Defaults tuned empirically on Eigen's GEMM kernel via single- and
  // multi-thread sweeps over b in {32..128}, p in {0..20}, n in
  // {1000, 2000, 4000} on Intel/AMD x86. b=48 was best or tied at every
  // size and thread count tested, and matches the panel size used by
  // Eigen's unpivoted blocked HouseholderQR. p=5 matches the HQRRP
  // paper's "very good" oversampling (Remark 1).
  static constexpr Index kDefaultBlockSize = 48;
  static constexpr Index kDefaultOversampling = 5;

  void init(Index rows, Index cols) {
    Index diag = numext::mini(rows, cols);
    m_hCoeffs.resize(diag);
    m_colsPermutation.resize(cols);
    m_temp.resize(cols);
    m_isInitialized = false;
  }

 public:
  /** \brief Default constructor. */
  RandColPivHouseholderQR() = default;

  /** \brief Constructor with memory preallocation. */
  RandColPivHouseholderQR(Index rows, Index cols) : m_qr(rows, cols) { init(rows, cols); }

  /** \brief Constructs and computes a QR factorization from \a matrix. */
  template <typename InputType>
  explicit RandColPivHouseholderQR(const EigenBase<InputType>& matrix) : m_qr(matrix.rows(), matrix.cols()) {
    init(matrix.rows(), matrix.cols());
    compute(matrix.derived());
  }

  /** \brief Inplace constructor: takes a Ref and decomposes in place. */
  template <typename InputType>
  explicit RandColPivHouseholderQR(EigenBase<InputType>& matrix) : m_qr(matrix.derived()) {
    init(matrix.rows(), matrix.cols());
    computeInPlace();
  }

#ifdef EIGEN_PARSED_BY_DOXYGEN
  template <typename Rhs>
  inline Solve<RandColPivHouseholderQR, Rhs> solve(const MatrixBase<Rhs>& b) const;
#endif

  HouseholderSequenceType householderQ() const;
  HouseholderSequenceType matrixQ() const { return householderQ(); }

  const MatrixType& matrixQR() const {
    eigen_assert(m_isInitialized && "RandColPivHouseholderQR is not initialized.");
    return m_qr;
  }

  const MatrixType& matrixR() const {
    eigen_assert(m_isInitialized && "RandColPivHouseholderQR is not initialized.");
    return m_qr;
  }

  template <typename InputType>
  RandColPivHouseholderQR& compute(const EigenBase<InputType>& matrix);

  const PermutationType& colsPermutation() const {
    eigen_assert(m_isInitialized && "RandColPivHouseholderQR is not initialized.");
    return m_colsPermutation;
  }

  typename MatrixType::Scalar determinant() const;
  typename MatrixType::RealScalar absDeterminant() const;
  typename MatrixType::RealScalar logAbsDeterminant() const;
  typename MatrixType::Scalar signDeterminant() const;

  RealScalar pivotCoeff(Index i) const {
    using std::abs;
    return abs(m_qr.coeff(i, i));
  }

  inline Inverse<RandColPivHouseholderQR> inverse() const {
    eigen_assert(m_isInitialized && "RandColPivHouseholderQR is not initialized.");
    return Inverse<RandColPivHouseholderQR>(*this);
  }

  inline Index rows() const { return m_qr.rows(); }
  inline Index cols() const { return m_qr.cols(); }

  const HCoeffsType& hCoeffs() const { return m_hCoeffs; }

  ComputationInfo info() const {
    eigen_assert(m_isInitialized && "Decomposition is not initialized.");
    return Success;
  }

  /** \brief Sets the panel block size \c b (default 48).
   *
   * Larger \c b increases the proportion of work performed in level-3 BLAS
   * but raises the per-iteration overhead of the sketch QR. For small
   * matrices (where \c min(m,n) < 2b) this class transparently falls back to
   * the unblocked column-pivoted scalar loop.
   */
  RandColPivHouseholderQR& setBlockSize(Index b) {
    eigen_assert(b > 0 && "Block size must be positive.");
    m_blockSize = b;
    return *this;
  }

  /** \brief Sets the oversampling parameter \c p (default 5).
   *
   * The Gaussian sketch has \c b+p rows. Larger \c p slightly improves the
   * quality of the pivot selection at small extra cost.
   */
  RandColPivHouseholderQR& setOversampling(Index p) {
    eigen_assert(p >= 0 && "Oversampling must be non-negative.");
    m_oversampling = p;
    return *this;
  }

  /** \brief Fixes the seed of the internal RNG for reproducible factorization.
   *
   * If never called, each call to compute() draws a fresh seed from
   * \c std::random_device.
   */
  RandColPivHouseholderQR& setSeed(uint64_t seed) {
    m_seed = seed;
    m_seedSet = true;
    return *this;
  }

  Index blockSize() const { return m_blockSize; }
  Index oversampling() const { return m_oversampling; }

#ifndef EIGEN_PARSED_BY_DOXYGEN
  template <typename RhsType, typename DstType>
  void _solve_impl(const RhsType& rhs, DstType& dst) const;

  template <bool Conjugate, typename RhsType, typename DstType>
  void _solve_impl_transposed(const RhsType& rhs, DstType& dst) const;
#endif

 protected:
  EIGEN_STATIC_ASSERT_NON_INTEGER(Scalar)

  void computeInPlace();

  // Unblocked column-pivoted Householder QR on rows [row0, m) and columns
  // [col0, col0 + ncols), using full-column swaps. Updates m_hCoeffs in
  // [col0, col0 + ncols), m_colsPermutation, and the maxpivot tracker.
  // Returns the number of column transpositions applied.
  //
  // `threshold_helper` is the global rank-detection scale (ColPivHouseholderQR
  // computes this once from the original column norms; the blocked path here
  // does the same and threads it through every panel call so rank revelation
  // is consistent across blocks). This is the inner-panel routine HQR_P_UNB;
  // it also serves as the unblocked tail loop and small-matrix fallback path.
  Index unblocked_pivoted_qr(Index row0, Index col0, Index ncols, RealScalar threshold_helper);

  MatrixType m_qr;
  HCoeffsType m_hCoeffs;
  PermutationType m_colsPermutation;
  RowVectorType m_temp;
  Index m_blockSize = kDefaultBlockSize;
  Index m_oversampling = kDefaultOversampling;
  uint64_t m_seed = 0;
  bool m_seedSet = false;
  bool m_isInitialized = false;
  Index m_det_p = 1;
};

template <typename MatrixType, typename PermutationIndex>
typename MatrixType::Scalar RandColPivHouseholderQR<MatrixType, PermutationIndex>::determinant() const {
  eigen_assert(m_isInitialized && "RandColPivHouseholderQR is not initialized.");
  eigen_assert(m_qr.rows() == m_qr.cols() && "You can't take the determinant of a non-square matrix!");
  Scalar detQ;
  internal::householder_determinant<HCoeffsType, Scalar, NumTraits<Scalar>::IsComplex>::run(m_hCoeffs, detQ);
  return isInjective() ? (detQ * Scalar(m_det_p)) * m_qr.diagonal().prod() : Scalar(0);
}

template <typename MatrixType, typename PermutationIndex>
typename MatrixType::RealScalar RandColPivHouseholderQR<MatrixType, PermutationIndex>::absDeterminant() const {
  using std::abs;
  eigen_assert(m_isInitialized && "RandColPivHouseholderQR is not initialized.");
  eigen_assert(m_qr.rows() == m_qr.cols() && "You can't take the determinant of a non-square matrix!");
  return isInjective() ? abs(m_qr.diagonal().prod()) : RealScalar(0);
}

template <typename MatrixType, typename PermutationIndex>
typename MatrixType::RealScalar RandColPivHouseholderQR<MatrixType, PermutationIndex>::logAbsDeterminant() const {
  eigen_assert(m_isInitialized && "RandColPivHouseholderQR is not initialized.");
  eigen_assert(m_qr.rows() == m_qr.cols() && "You can't take the determinant of a non-square matrix!");
  return isInjective() ? m_qr.diagonal().cwiseAbs().array().log().sum() : -NumTraits<RealScalar>::infinity();
}

template <typename MatrixType, typename PermutationIndex>
typename MatrixType::Scalar RandColPivHouseholderQR<MatrixType, PermutationIndex>::signDeterminant() const {
  eigen_assert(m_isInitialized && "RandColPivHouseholderQR is not initialized.");
  eigen_assert(m_qr.rows() == m_qr.cols() && "You can't take the determinant of a non-square matrix!");
  Scalar detQ;
  internal::householder_determinant<HCoeffsType, Scalar, NumTraits<Scalar>::IsComplex>::run(m_hCoeffs, detQ);
  return isInjective() ? (detQ * Scalar(m_det_p)) * m_qr.diagonal().array().sign().prod() : Scalar(0);
}

template <typename MatrixType, typename PermutationIndex>
template <typename InputType>
RandColPivHouseholderQR<MatrixType, PermutationIndex>& RandColPivHouseholderQR<MatrixType, PermutationIndex>::compute(
    const EigenBase<InputType>& matrix) {
  m_qr = matrix.derived();
  computeInPlace();
  return *this;
}

template <typename MatrixType, typename PermutationIndex>
Index RandColPivHouseholderQR<MatrixType, PermutationIndex>::unblocked_pivoted_qr(Index row0, Index col0, Index ncols,
                                                                                  RealScalar threshold_helper) {
  using std::abs;
  const Index rows = m_qr.rows();
  const Index col_end = col0 + ncols;
  const Index sub_rows = rows - row0;
  Index num_transpositions = 0;

  Matrix<RealScalar, 1, Dynamic> norms_updated(ncols);
  Matrix<RealScalar, 1, Dynamic> norms_direct(ncols);
  for (Index j = 0; j < ncols; ++j) {
    norms_direct.coeffRef(j) = m_qr.col(col0 + j).tail(sub_rows).norm();
    norms_updated.coeffRef(j) = norms_direct.coeffRef(j);
  }
  const RealScalar downdate_threshold = numext::sqrt(NumTraits<RealScalar>::epsilon());

  const Index size = (std::min)(sub_rows, ncols);
  for (Index k = 0; k < size; ++k) {
    Index biggest;
    RealScalar biggest_sq = numext::abs2(norms_updated.tail(ncols - k).maxCoeff(&biggest));
    biggest += k;

    if (this->m_nonzero_pivots == m_qr.diagonalSize() && biggest_sq < threshold_helper * RealScalar(rows - row0 - k))
      this->m_nonzero_pivots = col0 + k;

    if (k != biggest) {
      m_qr.col(col0 + k).swap(m_qr.col(col0 + biggest));
      std::swap(norms_updated.coeffRef(k), norms_updated.coeffRef(biggest));
      std::swap(norms_direct.coeffRef(k), norms_direct.coeffRef(biggest));
      m_colsPermutation.applyTranspositionOnTheRight(col0 + k, col0 + biggest);
      ++num_transpositions;
    }

    RealScalar beta;
    m_qr.col(col0 + k).tail(sub_rows - k).makeHouseholderInPlace(m_hCoeffs.coeffRef(col0 + k), beta);
    m_qr.coeffRef(row0 + k, col0 + k) = beta;
    if (abs(beta) > this->m_maxpivot) this->m_maxpivot = abs(beta);

    // Apply the reflector only to the remaining panel columns. Columns at
    // indices >= col_end are owned by the outer caller and updated via a
    // BLAS-3 trailing update; in the tail/fallback path col_end equals
    // m_qr.cols(), so this naturally degrades to a full apply.
    const Index trail_cols = col_end - col0 - k - 1;
    if (trail_cols > 0) {
      m_qr.block(row0 + k, col0 + k + 1, sub_rows - k, trail_cols)
          .applyHouseholderOnTheLeft(m_qr.col(col0 + k).tail(sub_rows - k - 1), m_hCoeffs.coeff(col0 + k),
                                     &m_temp.coeffRef(col0 + k + 1));
    }

    for (Index j = k + 1; j < ncols; ++j) {
      internal::lawn176_norm_downdate(norms_updated.coeffRef(j), norms_direct.coeffRef(j),
                                      m_qr.coeff(row0 + k, col0 + j), downdate_threshold,
                                      [&] { return m_qr.col(col0 + j).tail(sub_rows - k - 1).norm(); });
    }
  }
  return num_transpositions;
}

template <typename MatrixType, typename PermutationIndex>
void RandColPivHouseholderQR<MatrixType, PermutationIndex>::computeInPlace() {
  eigen_assert(m_qr.cols() <= NumTraits<PermutationIndex>::highest());

  const Index rows = m_qr.rows();
  const Index cols = m_qr.cols();
  const Index size = (std::min)(rows, cols);

  m_hCoeffs.resize(size);
  m_temp.resize(cols);
  m_colsPermutation.resize(cols);
  m_colsPermutation.setIdentity();
  this->m_nonzero_pivots = size;
  this->m_maxpivot = RealScalar(0);
  Index num_transpositions = 0;

  // Global rank-detection scale, derived from the original column norms
  // exactly as ColPivHouseholderQR does (LAWN-176). All panel calls share
  // this threshold so rank revelation is consistent across blocks.
  RealScalar max_initial_norm = RealScalar(0);
  for (Index j = 0; j < cols; ++j) max_initial_norm = numext::maxi(max_initial_norm, m_qr.col(j).norm());
  const RealScalar threshold_helper =
      cols == 0 ? RealScalar(0)
                : numext::abs2<RealScalar>(max_initial_norm * NumTraits<RealScalar>::epsilon()) / RealScalar(rows);

  const Index b = (std::min)(m_blockSize, size);
  const Index sketch_rows = b + m_oversampling;

  // For very small problems the sketch overhead dominates; use the unblocked
  // pivoted loop for the whole matrix instead.
  const bool can_block = (b > 0) && (sketch_rows < rows) && (size >= 2 * b);
  if (!can_block) {
    num_transpositions += unblocked_pivoted_qr(0, 0, cols, threshold_helper);
    m_det_p = (num_transpositions % 2) ? -1 : 1;
    m_isInitialized = true;
    return;
  }

  typedef Matrix<Scalar, Dynamic, Dynamic, ColMajor> WorkMatrix;
  WorkMatrix G(sketch_rows, rows);
  WorkMatrix Y(sketch_rows, cols);
  WorkMatrix Y_work(sketch_rows, cols);  // scratch for DeterminePivots
  Matrix<PermutationIndex, 1, Dynamic> sketch_pivots(b);
  Matrix<Scalar, Dynamic, Dynamic, RowMajor> T(b, b);  // hoisted; fixed b x b each iter

  // Initialize G and the sketch Y = G * A. G is held fixed thereafter:
  // column pivoting on A does not affect it.
  uint64_t seed = m_seed;
  if (!m_seedSet) {
    std::random_device rd;
    seed = (uint64_t(rd()) << 32) | uint64_t(rd());
  }
  std::mt19937_64 engine(seed);
  internal::fill_gaussian(G, engine);
  Y.noalias() = G * m_qr;

  Index k = 0;
  while (k + b <= size && cols - k > b) {
    const Index n_remain = cols - k;        // columns from k to end
    const Index trail_cols = n_remain - b;  // columns to the right of the panel
    const Index sub_rows = rows - k;        // rows from k to end
    const Index sub_below = sub_rows - b;   // rows strictly below the panel

    // 1. DeterminePivots: copy current Y_R into scratch (the sketch QR is
    // destructive), run b-step pivoted QR, capture pivot indices.
    Y_work.leftCols(n_remain) = Y.rightCols(n_remain);
    auto sketch_work = Y_work.leftCols(n_remain);
    auto pivots = sketch_pivots.head(b);
    internal::determine_pivots_on_sketch(sketch_work, b, pivots);

    // 2 & 3. Apply sketch pivots to A and Y (full-column swaps).
    for (Index i = 0; i < b; ++i) {
      Index abs_target = k + sketch_pivots.coeff(i);
      if (abs_target != k + i) {
        m_qr.col(k + i).swap(m_qr.col(abs_target));
        Y.col(k + i).swap(Y.col(abs_target));
        m_colsPermutation.applyTranspositionOnTheRight(k + i, abs_target);
        ++num_transpositions;
      }
    }

    // 4. Unblocked column-pivoted Householder QR on the b-column panel
    // (HQR_P_UNB). Full-column swaps also re-permute the previously
    // triangularized rows above (paper's "A_01 := SwapCols(s_1', A_01)").
    num_transpositions += unblocked_pivoted_qr(k, k, b, threshold_helper);

    if (trail_cols == 0) {
      k += b;
      continue;
    }

    // 5. Build the compact-WY triangular factor for the panel, matching
    // Eigen's apply_block_householder_on_the_left "backward" convention so
    // that I - V * T^H * V^H equals H_{b-1} ... H_0 (the trailing-update
    // operator). The paper's T_P (UT transform) satisfies
    // I - U * T_P^{-H} * U^H = same operator, so T_P^{-1} = T.
    auto V = m_qr.block(k, k, sub_rows, b);  // U_11 stacked on U_21
    auto V_top = V.topRows(b);               // U_11 (b x b unit-lower)
    auto hCoeffsSegment = m_hCoeffs.segment(k, b);
    internal::make_block_householder_triangular_factor(T, V, hCoeffsSegment.conjugate());

    // 6. BLAS-3 trailing update: A_trailing -= V * T^H * (V^H * A_trailing).
    auto A_trailing = m_qr.block(k, k + b, sub_rows, trail_cols);
    WorkMatrix tmp = V_top.template triangularView<UnitLower>().adjoint() * A_trailing.topRows(b);
    if (sub_below > 0) tmp.noalias() += V.bottomRows(sub_below).adjoint() * A_trailing.bottomRows(sub_below);
    tmp = (T.template triangularView<Upper>().adjoint() * tmp).eval();
    A_trailing.topRows(b) -= V_top.template triangularView<UnitLower>() * tmp;
    if (sub_below > 0) A_trailing.bottomRows(sub_below).noalias() -= V.bottomRows(sub_below) * tmp;

    // After the trailing update, R_12 sits in the top b rows of A_trailing.
    auto R_12 = m_qr.block(k, k + b, b, trail_cols);

    // 7. Downdate Y_2: Y_2 -= (G_1 - GU * T_P^{-1} * U_11^H) * R_12, with
    // T_P^{-1} = T in the convention above. G is fixed; G_1, G_2 partition
    // G's columns to match A's row blocks at this iteration.
    auto G1 = G.middleCols(k, b);
    WorkMatrix GU = G1 * V_top.template triangularView<UnitLower>();
    if (sub_below > 0) GU.noalias() += G.middleCols(k + b, sub_below) * V.bottomRows(sub_below);
    WorkMatrix Z = GU * T.template triangularView<Upper>();
    WorkMatrix factor = G1 - Z * V_top.template triangularView<UnitLower>().adjoint();
    Y.rightCols(trail_cols).noalias() -= factor * R_12;

    k += b;
  }

  // Tail loop: remaining columns handled by the unblocked pivoted QR.
  if (k < size) num_transpositions += unblocked_pivoted_qr(k, k, cols - k, threshold_helper);

  m_det_p = (num_transpositions % 2) ? -1 : 1;
  m_isInitialized = true;
}

#ifndef EIGEN_PARSED_BY_DOXYGEN
template <typename MatrixType_, typename PermutationIndex_>
template <typename RhsType, typename DstType>
void RandColPivHouseholderQR<MatrixType_, PermutationIndex_>::_solve_impl(const RhsType& rhs, DstType& dst) const {
  const Index nonzero_pivots = nonzeroPivots();

  if (nonzero_pivots == 0) {
    dst.setZero();
    return;
  }

  typename RhsType::PlainObject c(rhs);

  c.applyOnTheLeft(householderQ().setLength(nonzero_pivots).adjoint());

  m_qr.topLeftCorner(nonzero_pivots, nonzero_pivots)
      .template triangularView<Upper>()
      .solveInPlace(c.topRows(nonzero_pivots));

  for (Index i = 0; i < nonzero_pivots; ++i) dst.row(m_colsPermutation.indices().coeff(i)) = c.row(i);
  for (Index i = nonzero_pivots; i < cols(); ++i) dst.row(m_colsPermutation.indices().coeff(i)).setZero();
}

template <typename MatrixType_, typename PermutationIndex_>
template <bool Conjugate, typename RhsType, typename DstType>
void RandColPivHouseholderQR<MatrixType_, PermutationIndex_>::_solve_impl_transposed(const RhsType& rhs,
                                                                                     DstType& dst) const {
  const Index nonzero_pivots = nonzeroPivots();

  if (nonzero_pivots == 0) {
    dst.setZero();
    return;
  }

  typename RhsType::PlainObject c(m_colsPermutation.transpose() * rhs);

  m_qr.topLeftCorner(nonzero_pivots, nonzero_pivots)
      .template triangularView<Upper>()
      .transpose()
      .template conjugateIf<Conjugate>()
      .solveInPlace(c.topRows(nonzero_pivots));

  dst.topRows(nonzero_pivots) = c.topRows(nonzero_pivots);
  dst.bottomRows(rows() - nonzero_pivots).setZero();

  dst.applyOnTheLeft(householderQ().setLength(nonzero_pivots).template conjugateIf<!Conjugate>());
}
#endif

namespace internal {

template <typename DstXprType, typename MatrixType, typename PermutationIndex>
struct Assignment<DstXprType, Inverse<RandColPivHouseholderQR<MatrixType, PermutationIndex>>,
                  internal::assign_op<typename DstXprType::Scalar,
                                      typename RandColPivHouseholderQR<MatrixType, PermutationIndex>::Scalar>,
                  Dense2Dense> {
  typedef RandColPivHouseholderQR<MatrixType, PermutationIndex> QrType;
  typedef Inverse<QrType> SrcXprType;
  static void run(DstXprType& dst, const SrcXprType& src,
                  const internal::assign_op<typename DstXprType::Scalar, typename QrType::Scalar>&) {
    dst = src.nestedExpression().solve(MatrixType::Identity(src.rows(), src.cols()));
  }
};

}  // end namespace internal

template <typename MatrixType, typename PermutationIndex>
typename RandColPivHouseholderQR<MatrixType, PermutationIndex>::HouseholderSequenceType
RandColPivHouseholderQR<MatrixType, PermutationIndex>::householderQ() const {
  eigen_assert(m_isInitialized && "RandColPivHouseholderQR is not initialized.");
  return HouseholderSequenceType(m_qr, m_hCoeffs.conjugate());
}

/** \return the randomized column-pivoted Householder QR decomposition of \c *this.
 *
 * \sa class RandColPivHouseholderQR
 */
template <typename Derived>
template <typename PermutationIndexType>
RandColPivHouseholderQR<typename MatrixBase<Derived>::PlainObject, PermutationIndexType>
MatrixBase<Derived>::randColPivHouseholderQr() const {
  return RandColPivHouseholderQR<PlainObject, PermutationIndexType>(eval());
}

}  // end namespace Eigen

#endif  // EIGEN_RANDCOLPIVOTINGHOUSEHOLDERQR_H
