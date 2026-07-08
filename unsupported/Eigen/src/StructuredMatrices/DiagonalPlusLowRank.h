// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_STRUCTURED_DIAGONAL_PLUS_LOW_RANK_H
#define EIGEN_STRUCTURED_DIAGONAL_PLUS_LOW_RANK_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

template <typename Scalar_, int Size_ = Dynamic, int Rank_ = Dynamic>
class DiagonalPlusLowRank;

namespace internal {

template <typename Scalar_, int Size_, int Rank_>
struct traits<DiagonalPlusLowRank<Scalar_, Size_, Rank_>> {
  using Scalar = Scalar_;
  using StorageKind = Dense;
  using XprKind = MatrixXpr;
  using StorageIndex = int;
  static constexpr int RowsAtCompileTime = Size_;
  static constexpr int ColsAtCompileTime = Size_;
  static constexpr int MaxRowsAtCompileTime = Size_;
  static constexpr int MaxColsAtCompileTime = Size_;
  static constexpr int Flags = NestByRefBit;
};

template <typename Scalar_, int Size_, int Rank_>
struct evaluator_traits<DiagonalPlusLowRank<Scalar_, Size_, Rank_>> {
  using Kind = IndexBased;
  using Shape = StructuredShape;
};

}  // namespace internal

/** \ingroup StructuredMatrices_Module
 * \class DiagonalPlusLowRank
 * \brief An \c n x \c n operator \f$ D + U V^H \f$: a diagonal matrix plus a
 * rank-k correction, stored as its diagonal and the two \c n x \c k factors.
 *
 * Products cost O(nk) instead of O(n^2). Linear systems are solved in O(nk^2)
 * through the Woodbury identity
 * \f$ (D + UV^H)^{-1} = D^{-1} - D^{-1} U (I_k + V^H D^{-1} U)^{-1} V^H D^{-1} \f$,
 * factoring only the \c k x \c k \em capacitance matrix (\ref solve). The
 * determinant follows from the matrix determinant lemma
 * \f$ \det(D)\,\det(I_k + V^H D^{-1} U) \f$ (\ref determinant), and the class is
 * closed under \ref inverse, \ref transpose, \ref conjugate and \ref adjoint.
 *
 * The rank-k correction may have \c k = 0 (a plain diagonal operator). \c k is
 * not required to be small, but every advantage over a dense matrix vanishes as
 * \c k approaches \c n.
 *
 * \tparam Scalar_ the scalar type, real or complex.
 * \tparam Size_ the dimension at compile time, or \c Dynamic (the default).
 * \tparam Rank_ the correction rank at compile time, or \c Dynamic (the default).
 *
 * \sa makeDiagonalPlusLowRank()
 */
template <typename Scalar_, int Size_, int Rank_>
class DiagonalPlusLowRank : public EigenBase<DiagonalPlusLowRank<Scalar_, Size_, Rank_>> {
 public:
  using Scalar = Scalar_;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using StorageIndex = int;
  using DiagonalVector = Matrix<Scalar, Size_, 1>;
  using FactorType = Matrix<Scalar, Size_, Rank_>;
  using CapacitanceType = Matrix<Scalar, Rank_, Rank_>;

  static constexpr int RowsAtCompileTime = Size_;
  static constexpr int ColsAtCompileTime = Size_;
  static constexpr int MaxRowsAtCompileTime = Size_;
  static constexpr int MaxColsAtCompileTime = Size_;
  static constexpr int SizeAtCompileTime = internal::size_at_compile_time(Size_, Size_);
  static constexpr int MaxSizeAtCompileTime = SizeAtCompileTime;
  static constexpr bool IsRowMajor = false;
  // Deliberately no IsVectorAtCompileTime: Ref<const DiagonalPlusLowRank>'s default
  // StrideType argument reads it, so its absence makes internal::is_ref_compatible
  // SFINAE to false and keeps the iterative solvers on their matrix-free path.

  /** Builds the operator \c diag(d) + \c U*V.adjoint(). The factors must have the
   * same number of columns (the correction rank \c k, possibly zero) and as many
   * rows as \a d has entries. */
  template <typename DDerived, typename UDerived, typename VDerived>
  DiagonalPlusLowRank(const MatrixBase<DDerived>& d, const MatrixBase<UDerived>& U, const MatrixBase<VDerived>& V)
      : m_d(d), m_U(U), m_V(V) {
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(DDerived)
    eigen_assert(m_d.size() > 0 && "DiagonalPlusLowRank must be non-empty");
    eigen_assert(m_U.rows() == m_d.size() && m_V.rows() == m_d.size() && m_U.cols() == m_V.cols() &&
                 "factor dimensions do not match");
  }

  EIGEN_DEVICE_FUNC Index rows() const { return m_d.size(); }
  EIGEN_DEVICE_FUNC Index cols() const { return m_d.size(); }
  /** \returns the correction rank \c k. */
  Index correctionRank() const { return m_U.cols(); }

  /** \returns the diagonal vector \c d. */
  const DiagonalVector& diagonal() const { return m_d; }
  /** \returns the left factor \c U. */
  const FactorType& factorU() const { return m_U; }
  /** \returns the right factor \c V. */
  const FactorType& factorV() const { return m_V; }

  /** \returns the coefficient at row \a row and column \a col. */
  Scalar coeff(Index row, Index col) const {
    Scalar r = m_V.row(col).dot(m_U.row(row));  // dot() conjugates its first argument
    if (row == col) r += m_d.coeff(row);
    return r;
  }

  /** \returns the transpose of \c *this, itself a \c DiagonalPlusLowRank
   * operator: \f$ (D + UV^H)^T = D + \bar V \bar U^H \f$. */
  DiagonalPlusLowRank transpose() const { return DiagonalPlusLowRank(m_d, m_V.conjugate(), m_U.conjugate()); }

  /** \returns the complex conjugate of \c *this, itself a \c DiagonalPlusLowRank
   * operator. */
  DiagonalPlusLowRank conjugate() const {
    return DiagonalPlusLowRank(m_d.conjugate(), m_U.conjugate(), m_V.conjugate());
  }

  /** \returns the adjoint of \c *this, itself a \c DiagonalPlusLowRank operator:
   * \f$ (D + UV^H)^H = \bar D + V U^H \f$. */
  DiagonalPlusLowRank adjoint() const { return DiagonalPlusLowRank(m_d.conjugate(), m_V, m_U); }

  /** \returns the capacitance matrix \f$ I_k + V^H D^{-1} U \f$ of the Woodbury
   * identity. \warning The diagonal must have no zero entries. */
  CapacitanceType capacitance() const {
    const Index k = correctionRank();
    CapacitanceType c = CapacitanceType::Identity(k, k);
    c.noalias() += m_V.adjoint() * m_d.cwiseInverse().asDiagonal() * m_U;
    return c;
  }

  /** \returns the solution of \c (*this) * x = b through the Woodbury identity,
   * factoring only the \c k x \c k capacitance matrix: O(nk^2 + k^3) setup and
   * O(nk) per right-hand side. Supports multiple right-hand sides.
   * \warning Requires an invertible diagonal \em and an invertible capacitance
   * matrix (equivalently, an invertible operator); this is not checked beyond
   * the NaN/Inf propagation of the arithmetic itself. */
  template <typename Rhs>
  Matrix<Scalar, Size_, Rhs::ColsAtCompileTime> solve(const MatrixBase<Rhs>& b) const {
    eigen_assert(b.rows() == rows() && "right-hand side has the wrong number of rows");
    const auto dinv = m_d.cwiseInverse();
    Matrix<Scalar, Size_, Rhs::ColsAtCompileTime> x = dinv.asDiagonal() * b;
    if (correctionRank() > 0) {
      PartialPivLU<CapacitanceType> cap(capacitance());
      x.noalias() -= dinv.asDiagonal() * (m_U * cap.solve(m_V.adjoint() * x));
    }
    return x;
  }

  /** \returns the inverse of \c *this, itself a \c DiagonalPlusLowRank operator:
   * by the Woodbury identity, \f$ (D + UV^H)^{-1} = D^{-1} + U' V'^H \f$ with
   * \f$ U' = -D^{-1} U (I_k + V^H D^{-1} U)^{-1} \f$ and \f$ V' = D^{-H} V \f$.
   * \warning Same invertibility requirements as \ref solve. */
  DiagonalPlusLowRank inverse() const {
    const auto dinv = m_d.cwiseInverse();
    FactorType Up(rows(), correctionRank());
    if (correctionRank() > 0) {
      PartialPivLU<CapacitanceType> cap(capacitance());
      Up.noalias() = -(dinv.asDiagonal() * m_U * cap.inverse());
    }
    FactorType Vp = dinv.conjugate().asDiagonal() * m_V;
    return DiagonalPlusLowRank(dinv, Up, Vp);
  }

  /** \returns the determinant through the matrix determinant lemma:
   * \f$ \det(D)\,\det(I_k + V^H D^{-1} U) \f$, in O(nk^2 + k^3) operations.
   * \warning The diagonal must have no zero entries (use the lemma symmetrically
   * or a dense fallback for that case). */
  Scalar determinant() const {
    Scalar det(1);
    for (Index i = 0; i < rows(); ++i) det *= m_d.coeff(i);
    if (correctionRank() > 0) det *= capacitance().determinant();
    return det;
  }

  /** \internal Writes the dense representation into \a dst. Invoked through
   * \c dense = op; */
  template <typename Dest>
  void evalTo(Dest& dst) const {
    dst.noalias() = m_U * m_V.adjoint();
    dst.diagonal() += m_d;
  }

  /** \internal Computes \c dst += (*this), see evalTo(). */
  template <typename Dest>
  void addTo(Dest& dst) const {
    dst.noalias() += m_U * m_V.adjoint();
    dst.diagonal() += m_d;
  }

  /** \internal Computes \c dst -= (*this), see evalTo(). */
  template <typename Dest>
  void subTo(Dest& dst) const {
    dst.noalias() -= m_U * m_V.adjoint();
    dst.diagonal() -= m_d;
  }

  /** \returns the product expression \c (*this) * \a v, evaluated at O(nk)
   * operations without forming the matrix. */
  template <typename Rhs>
  Product<DiagonalPlusLowRank, Rhs, AliasFreeProduct> operator*(const MatrixBase<Rhs>& v) const {
    return Product<DiagonalPlusLowRank, Rhs, AliasFreeProduct>(*this, v.derived());
  }

  /** \internal Computes \c dst += alpha * (*this) * rhs. */
  template <typename Dest, typename Rhs>
  void addProduct(Dest& dst, const Rhs& rhs, const Scalar& alpha) const {
    eigen_assert(rhs.rows() == rows() && "invalid product: dimensions do not match");
    // Evaluate the (possibly expression) right-hand side once.
    const Matrix<Scalar, Size_, Rhs::ColsAtCompileTime> r = rhs;
    dst += alpha * (m_d.asDiagonal() * r);
    if (correctionRank() > 0) {
      const Matrix<Scalar, Rank_, Rhs::ColsAtCompileTime> t = m_V.adjoint() * r;
      dst.noalias() += alpha * (m_U * t);
    }
  }

 private:
  DiagonalVector m_d;
  FactorType m_U;
  FactorType m_V;
};

/** \ingroup StructuredMatrices_Module
 * \returns a \ref DiagonalPlusLowRank operator \c diag(d) + \c U*V.adjoint().
 * The compile-time dimension and rank are deduced from the arguments. */
template <typename DDerived, typename UDerived, typename VDerived>
DiagonalPlusLowRank<typename DDerived::Scalar, DDerived::SizeAtCompileTime, UDerived::ColsAtCompileTime>
makeDiagonalPlusLowRank(const MatrixBase<DDerived>& d, const MatrixBase<UDerived>& U, const MatrixBase<VDerived>& V) {
  return DiagonalPlusLowRank<typename DDerived::Scalar, DDerived::SizeAtCompileTime, UDerived::ColsAtCompileTime>(d, U,
                                                                                                                  V);
}

namespace internal {

// Single product specialization covering every product tag; see the note in
// Circulant.h.
template <typename Scalar_, int Size_, int Rank_, typename Rhs, int ProductTag>
struct generic_product_impl<DiagonalPlusLowRank<Scalar_, Size_, Rank_>, Rhs, StructuredShape, DenseShape, ProductTag>
    : structured_product_impl<DiagonalPlusLowRank<Scalar_, Size_, Rank_>, Rhs> {};

}  // namespace internal

}  // namespace Eigen

#endif  // EIGEN_STRUCTURED_DIAGONAL_PLUS_LOW_RANK_H
