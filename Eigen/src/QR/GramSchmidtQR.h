// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Pavel Guzenfeld
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_GRAM_SCHMIDT_QR_H
#define EIGEN_GRAM_SCHMIDT_QR_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {
template <typename MatrixType_>
struct traits<GramSchmidtQR<MatrixType_>> : traits<MatrixType_> {
  typedef MatrixXpr XprKind;
  typedef SolverStorage StorageKind;
  typedef int StorageIndex;
  enum { Flags = 0 };
};
}  // end namespace internal

/** \ingroup QR_Module
 *
 * \class GramSchmidtQR
 *
 * \brief Modified Gram-Schmidt QR decomposition of a matrix
 *
 * \tparam MatrixType_ the type of the matrix of which we are computing the QR decomposition
 *
 * This class performs a QR decomposition of a matrix \b A into matrices \b Q and \b R
 * such that
 * \f[
 *  \mathbf{A} = \mathbf{Q} \, \mathbf{R}
 * \f]
 * using the Modified Gram-Schmidt (MGS) algorithm. Here, \b Q is a unitary matrix and
 * \b R is an upper triangular matrix.
 *
 * Unlike HouseholderQR, this decomposition stores Q and R as explicit dense matrices,
 * making Q directly usable in subsequent matrix operations without special syntax.
 *
 * The Modified Gram-Schmidt algorithm has better numerical stability than the classical
 * Gram-Schmidt, producing a Q factor with orthogonality close to machine precision for
 * well-conditioned matrices.
 *
 * Note that no pivoting is performed. This is \b not a rank-revealing decomposition.
 *
 * \sa MatrixBase::gramSchmidtQr(), HouseholderQR, ColPivHouseholderQR
 */
template <typename MatrixType_>
class GramSchmidtQR : public SolverBase<GramSchmidtQR<MatrixType_>> {
 public:
  typedef MatrixType_ MatrixType;
  typedef SolverBase<GramSchmidtQR> Base;
  friend class SolverBase<GramSchmidtQR>;

  EIGEN_GENERIC_PUBLIC_INTERFACE(GramSchmidtQR)
  enum {
    MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
  };
  typedef Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime, (MatrixType::Flags & RowMajorBit) ? RowMajor : ColMajor,
                 MaxRowsAtCompileTime, MaxRowsAtCompileTime>
      MatrixQType;

  /** \brief Reports whether the QR factorization was successful.
   *
   * \note This function always returns \c Success. It is provided for compatibility
   * with other factorization routines.
   * \returns \c Success
   */
  ComputationInfo info() const {
    eigen_assert(m_isInitialized && "GramSchmidtQR is not initialized.");
    return Success;
  }

  /** \brief Default Constructor. */
  GramSchmidtQR() : m_q(), m_r(), m_isInitialized(false) {}

  /** \brief Default Constructor with memory preallocation
   *
   * Like the default constructor but with preallocation of the internal data
   * according to the specified problem \a size.
   * \sa GramSchmidtQR()
   */
  GramSchmidtQR(Index rows, Index cols) : m_q(rows, rows), m_r(rows, cols), m_isInitialized(false) {}

  /** \brief Constructs a QR factorization from a given matrix.
   *
   * This constructor computes the QR factorization of the matrix \a matrix
   * by calling the compute() method.
   *
   * \sa compute()
   */
  template <typename InputType>
  explicit GramSchmidtQR(const EigenBase<InputType>& matrix)
      : m_q(matrix.rows(), matrix.rows()), m_r(matrix.rows(), matrix.cols()), m_isInitialized(false) {
    compute(matrix.derived());
  }

  /** \brief Computes the QR factorization of the given matrix.
   *
   * \param matrix the matrix of which to compute the QR decomposition.
   * \returns a reference to *this
   *
   * The QR decomposition is computed using the Modified Gram-Schmidt algorithm.
   *
   * \sa class GramSchmidtQR
   */
  template <typename InputType>
  GramSchmidtQR& compute(const EigenBase<InputType>& matrix) {
    const Index rows = matrix.rows();
    const Index cols = matrix.cols();
    const Index size = (std::min)(rows, cols);

    m_q.resize(rows, rows);
    m_r.resize(rows, cols);
    m_r.setZero();

    // Copy input to Q workspace (first cols columns).
    // We work in-place on the first 'cols' columns of m_q.
    MatrixType work = matrix.derived();

    for (Index i = 0; i < size; ++i) {
      // Orthogonalize column i against all previous columns.
      for (Index j = 0; j < i; ++j) {
        Scalar rji = m_q.col(j).dot(work.col(i));
        m_r(j, i) = rji;
        work.col(i) -= rji * m_q.col(j);
      }
      // Normalize.
      RealScalar norm = work.col(i).stableNorm();
      m_r(i, i) = norm;
      if (norm > RealScalar(0)) {
        m_q.col(i) = work.col(i) / norm;
      } else {
        m_q.col(i).setZero();
      }
    }

    // For wide matrices (cols > rows), compute R entries for remaining columns.
    if (cols > rows) {
      m_r.rightCols(cols - rows).noalias() = m_q.adjoint() * work.rightCols(cols - rows);
    }

    // Fill remaining Q columns to complete an orthonormal basis.
    if (size < rows) {
      // Complete Q to a full orthonormal basis using the existing columns.
      // We use a simple approach: start from standard basis vectors and
      // orthogonalize against existing Q columns.
      Index filled = size;
      for (Index k = 0; k < rows && filled < rows; ++k) {
        // Try standard basis vector e_k.
        typename MatrixQType::ColXpr qcol = m_q.col(filled);
        qcol.setUnit(k);
        // Orthogonalize against all existing columns.
        for (Index j = 0; j < filled; ++j) {
          qcol -= m_q.col(j).dot(qcol) * m_q.col(j);
        }
        RealScalar norm = qcol.stableNorm();
        if (norm > RealScalar(1e-10)) {
          qcol /= norm;
          ++filled;
        }
      }
    }

    m_isInitialized = true;
    return *this;
  }

  /** \returns a const reference to the matrix Q */
  const MatrixQType& matrixQ() const {
    eigen_assert(m_isInitialized && "GramSchmidtQR is not initialized.");
    return m_q;
  }

  /** \returns a const reference to the matrix R */
  const MatrixType& matrixR() const {
    eigen_assert(m_isInitialized && "GramSchmidtQR is not initialized.");
    return m_r;
  }

  /** \returns the absolute value of the determinant of the matrix of which
   * *this is the QR decomposition. It has only linear complexity
   * (that is, O(n) where n is the dimension of the square matrix)
   * as the QR decomposition has already been computed.
   *
   * \note This is only for square matrices.
   */
  typename MatrixType::RealScalar absDeterminant() const {
    eigen_assert(m_isInitialized && "GramSchmidtQR is not initialized.");
    eigen_assert(m_r.rows() == m_r.cols() && "GramSchmidtQR::absDeterminant() requires a square matrix.");
    return numext::abs(m_r.diagonal().prod());
  }

  /** \returns the logarithm of the absolute value of the determinant */
  typename MatrixType::RealScalar logAbsDeterminant() const {
    eigen_assert(m_isInitialized && "GramSchmidtQR is not initialized.");
    eigen_assert(m_r.rows() == m_r.cols() && "GramSchmidtQR::logAbsDeterminant() requires a square matrix.");
    return m_r.diagonal().cwiseAbs().array().log().sum();
  }

#ifndef EIGEN_PARSED_BY_DOXYGEN
  template <typename RhsType, typename DstType>
  void _solve_impl(const RhsType& rhs, DstType& dst) const {
    // Solve QR x = rhs => R x = Q^* rhs
    const Index cols = m_r.cols();
    dst.noalias() = m_q.adjoint() * rhs;
    m_r.topRows(cols).template triangularView<Upper>().solveInPlace(dst);
  }

  template <bool Conjugate, typename RhsType, typename DstType>
  void _solve_impl_transposed(const RhsType& rhs, DstType& dst) const {
    // Solve R^T Q^T x = rhs => x = Q R^{-T} rhs
    const Index cols = m_r.cols();
    dst = rhs;
    m_r.topRows(cols).template triangularView<Upper>().transpose().solveInPlace(dst);
    dst = m_q * dst;
  }
#endif

  Index rows() const { return m_q.rows(); }
  Index cols() const { return m_r.cols(); }

 protected:
  MatrixQType m_q;
  MatrixType m_r;
  bool m_isInitialized;
};

/** \returns the Modified Gram-Schmidt QR decomposition of \c *this.
 *
 * \sa class GramSchmidtQR
 */
template <typename Derived>
const GramSchmidtQR<typename MatrixBase<Derived>::PlainObject> MatrixBase<Derived>::gramSchmidtQr() const {
  return GramSchmidtQR<PlainObject>(eval());
}

}  // end namespace Eigen

#endif  // EIGEN_GRAM_SCHMIDT_QR_H
