// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_STRUCTURED_VANDERMONDE_H
#define EIGEN_STRUCTURED_VANDERMONDE_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

template <typename Scalar_, int Rows_ = Dynamic, int Cols_ = Dynamic>
class Vandermonde;

template <typename Scalar_>
class BjorckPereyra;

namespace internal {

template <typename Scalar_, int Rows_, int Cols_>
struct traits<Vandermonde<Scalar_, Rows_, Cols_>> {
  using Scalar = Scalar_;
  using StorageKind = Dense;
  using XprKind = MatrixXpr;
  using StorageIndex = int;
  static constexpr int RowsAtCompileTime = Rows_;
  static constexpr int ColsAtCompileTime = Cols_;
  static constexpr int MaxRowsAtCompileTime = Rows_;
  static constexpr int MaxColsAtCompileTime = Cols_;
  static constexpr int Flags = NestByRefBit;
};

template <typename Scalar_, int Rows_, int Cols_>
struct evaluator_traits<Vandermonde<Scalar_, Rows_, Cols_>> {
  using Kind = IndexBased;
  using Shape = StructuredShape;
};

template <typename Scalar_>
struct traits<BjorckPereyra<Scalar_>> : traits<Matrix<Scalar_, Dynamic, Dynamic>> {
  using XprKind = MatrixXpr;
  using StorageKind = SolverStorage;
  using StorageIndex = int;
  using BaseTraits = traits<Matrix<Scalar_, Dynamic, Dynamic>>;
  enum { Flags = BaseTraits::Flags & RowMajorBit, CoeffReadCost = Dynamic };
};

}  // namespace internal

/** \ingroup StructuredMatrices_Module
 * \class Vandermonde
 * \brief An \c m x \c n Vandermonde matrix represented by its node vector.
 *
 * A Vandermonde matrix has entry \c (i,j) equal to \f$ x_i^j \f$, where \c x is
 * the node vector: row \c i holds the powers \c 0..n-1 of node \c x_i, so
 * \c V*a evaluates the polynomial with (ascending) coefficients \c a at every
 * node. The class stores only the \c m nodes; products are evaluated by Horner's
 * rule at O(mn) operations -- the same cost as a dense product, but with O(m)
 * storage and without ever forming the matrix.
 *
 * Square systems are solved in O(n^2) by the Björck-Pereyra algorithm (class
 * \ref BjorckPereyra), whose \c transpose().solve() form covers the dual
 * (moment) system. There is no fast transposed \em product, so the class is not
 * closed under transposition and rectangular least-squares problems are best
 * handled by a dense QR of the materialized matrix.
 *
 * \warning Vandermonde matrices with real nodes are exponentially
 * ill-conditioned: the condition number grows at least like \f$ 2^n \f$ for any
 * real node configuration (Beckermann, 2000). Solves remain surprisingly
 * accurate for monotone node sets and sign-alternating right-hand sides
 * (Björck-Pereyra's celebrated property, see Higham, ASNA ch. 22), but forward
 * errors necessarily scale with the conditioning in general. Complex nodes on
 * the unit circle are the well-conditioned case: for the n-th roots of unity,
 * \f$ V/\sqrt{n} \f$ is unitary.
 *
 * \tparam Scalar_ the scalar type, real or complex.
 * \tparam Rows_ the number of rows (nodes) at compile time, or \c Dynamic.
 * \tparam Cols_ the number of columns (powers) at compile time, or \c Dynamic.
 *
 * \sa class BjorckPereyra, makeVandermonde()
 */
template <typename Scalar_, int Rows_, int Cols_>
class Vandermonde : public EigenBase<Vandermonde<Scalar_, Rows_, Cols_>> {
 public:
  using Scalar = Scalar_;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using StorageIndex = int;
  using NodeVector = Matrix<Scalar, Rows_, 1>;

  static constexpr int RowsAtCompileTime = Rows_;
  static constexpr int ColsAtCompileTime = Cols_;
  static constexpr int MaxRowsAtCompileTime = Rows_;
  static constexpr int MaxColsAtCompileTime = Cols_;
  static constexpr int SizeAtCompileTime = internal::size_at_compile_time(Rows_, Cols_);
  static constexpr int MaxSizeAtCompileTime = SizeAtCompileTime;
  static constexpr bool IsRowMajor = false;
  // Deliberately no IsVectorAtCompileTime: Ref<const Vandermonde>'s default
  // StrideType argument reads it, so its absence makes internal::is_ref_compatible
  // SFINAE to false and keeps the iterative solvers on their matrix-free path.

  /** Builds an \c m x \a cols Vandermonde matrix from the \c m nodes \a nodes. */
  template <typename Derived>
  Vandermonde(const MatrixBase<Derived>& nodes, Index cols) : m_x(nodes), m_cols(cols) {
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
    eigen_assert(m_x.size() > 0 && m_cols > 0 && "Vandermonde must be non-empty");
    eigen_assert((Cols_ == Dynamic || Cols_ == cols) && "cols does not match the compile-time column count");
  }

  /** Builds the square Vandermonde matrix of the nodes \a nodes. */
  template <typename Derived>
  explicit Vandermonde(const MatrixBase<Derived>& nodes) : Vandermonde(nodes, nodes.size()) {}

  EIGEN_DEVICE_FUNC Index rows() const { return m_x.size(); }
  EIGEN_DEVICE_FUNC Index cols() const { return m_cols; }

  /** \returns the node vector. */
  const NodeVector& nodes() const { return m_x; }

  /** \returns the coefficient at row \a row and column \a col, \f$ x_i^j \f$. */
  Scalar coeff(Index row, Index col) const {
    Scalar p(1);
    const Scalar xi = m_x.coeff(row);
    for (Index t = 0; t < col; ++t) p *= xi;
    return p;
  }

  /** \returns the determinant of a \b square Vandermonde matrix through the
   * closed form \f$ \prod_{i<j} (x_j - x_i) \f$, in O(n^2) operations. */
  Scalar determinant() const {
    eigen_assert(rows() == cols() && "Vandermonde::determinant requires a square matrix");
    const Index n = rows();
    Scalar det(1);
    for (Index j = 1; j < n; ++j)
      for (Index i = 0; i < j; ++i) det *= m_x.coeff(j) - m_x.coeff(i);
    return det;
  }

  /** \internal Writes the dense representation into \a dst: column \c j is the
   * elementwise product of column \c j-1 with the nodes, so only cumulative
   * columnwise products are involved. Invoked through \c dense = vandermonde; */
  template <typename Dest>
  void evalTo(Dest& dst) const {
    dst.col(0).setOnes();
    for (Index j = 1; j < m_cols; ++j) dst.col(j) = dst.col(j - 1).cwiseProduct(m_x);
  }

  /** \internal Computes \c dst += (*this), see evalTo(). */
  template <typename Dest>
  void addTo(Dest& dst) const {
    NodeVector p = NodeVector::Ones(rows());
    dst.col(0) += p;
    for (Index j = 1; j < m_cols; ++j) {
      p = p.cwiseProduct(m_x);
      dst.col(j) += p;
    }
  }

  /** \internal Computes \c dst -= (*this), see evalTo(). */
  template <typename Dest>
  void subTo(Dest& dst) const {
    NodeVector p = NodeVector::Ones(rows());
    dst.col(0) -= p;
    for (Index j = 1; j < m_cols; ++j) {
      p = p.cwiseProduct(m_x);
      dst.col(j) -= p;
    }
  }

  /** \returns the product expression \c (*this) * \a a: the polynomial with
   * ascending coefficients \c a (per column) evaluated at every node by Horner's
   * rule, at O(mn) operations and O(1) extra storage. */
  template <typename Rhs>
  Product<Vandermonde, Rhs, AliasFreeProduct> operator*(const MatrixBase<Rhs>& a) const {
    return Product<Vandermonde, Rhs, AliasFreeProduct>(*this, a.derived());
  }

  /** \internal Computes \c dst += alpha * (*this) * rhs by Horner's rule. */
  template <typename Dest, typename Rhs>
  void addProduct(Dest& dst, const Rhs& rhs, const Scalar& alpha) const {
    const Index m = rows(), n = m_cols;
    eigen_assert(rhs.rows() == n && "invalid product: dimensions do not match");
    for (Index k = 0; k < rhs.cols(); ++k)
      for (Index i = 0; i < m; ++i) {
        const Scalar xi = m_x.coeff(i);
        Scalar acc = rhs.coeff(n - 1, k);
        for (Index j = n - 2; j >= 0; --j) acc = acc * xi + rhs.coeff(j, k);
        dst.coeffRef(i, k) += alpha * acc;
      }
  }

 private:
  NodeVector m_x;
  Index m_cols;
};

/** \ingroup StructuredMatrices_Module
 * \returns an \c m x \a cols \ref Vandermonde operator with node vector \a nodes;
 * the compile-time row count is deduced from \a nodes. */
template <typename Derived>
Vandermonde<typename Derived::Scalar, Derived::SizeAtCompileTime, Dynamic> makeVandermonde(
    const MatrixBase<Derived>& nodes, Index cols) {
  return Vandermonde<typename Derived::Scalar, Derived::SizeAtCompileTime, Dynamic>(nodes, cols);
}

/** \ingroup StructuredMatrices_Module
 * \returns the square \ref Vandermonde operator of the nodes \a nodes. */
template <typename Derived>
Vandermonde<typename Derived::Scalar, Derived::SizeAtCompileTime, Derived::SizeAtCompileTime> makeVandermonde(
    const MatrixBase<Derived>& nodes) {
  return Vandermonde<typename Derived::Scalar, Derived::SizeAtCompileTime, Derived::SizeAtCompileTime>(nodes);
}

/** \ingroup StructuredMatrices_Module
 * \class BjorckPereyra
 * \brief Björck-Pereyra O(n^2) solver for square Vandermonde systems.
 *
 * Solves \c V*a = f -- polynomial interpolation: find the coefficients of the
 * polynomial taking values \c f at the nodes -- in O(n^2) operations and O(n)
 * storage, via divided differences in the Newton basis followed by the basis
 * change to monomials (Björck & Pereyra, 1970; Golub & Van Loan, Alg. 4.6.2).
 * The transposed (dual, or moment) system \f$ V^T w = b \f$ is solved by the
 * companion dual recurrences through the standard \c SolverBase idiom:
 * \code
 *   BjorckPereyra<double> bp(V);            // or bp.compute(V);
 *   VectorXd a = bp.solve(f);                // solve V   * a = f
 *   VectorXd w = bp.transpose().solve(b);    // solve V^T * w = b
 *   VectorXd u = bp.adjoint().solve(b);      // solve V^H * u = b
 * \endcode
 *
 * There is no factorization: \c compute() stores the nodes (and flags exactly
 * repeated nodes, which make the matrix singular, through \c info()), and each
 * solve runs the O(n^2) recurrences directly.
 *
 * Despite the exponential conditioning of real-node Vandermonde matrices, the
 * computed solution is often far more accurate than the conditioning suggests:
 * for monotonically ordered nodes and a right-hand side with alternating signs
 * the forward error is governed by a small relative-perturbation bound
 * independent of the condition number (Higham, ASNA ch. 22).
 *
 * \tparam Scalar_ the scalar type, real or complex.
 *
 * \sa class Vandermonde
 */
template <typename Scalar_>
class BjorckPereyra : public SolverBase<BjorckPereyra<Scalar_>> {
 public:
  using Base = SolverBase<BjorckPereyra>;
  friend class SolverBase<BjorckPereyra>;
  EIGEN_GENERIC_PUBLIC_INTERFACE(BjorckPereyra)
  using NodeVector = Matrix<Scalar, Dynamic, 1>;

  /** Default constructor; call \ref compute before \ref solve. */
  BjorckPereyra() : m_isInitialized(false), m_info(InvalidInput) {}

  /** Constructs the solver for the square Vandermonde matrix \a V. */
  template <int Rows_, int Cols_>
  explicit BjorckPereyra(const Vandermonde<Scalar, Rows_, Cols_>& V) : m_isInitialized(false), m_info(InvalidInput) {
    compute(V);
  }

  /** Stores the nodes of the square Vandermonde matrix \a V and scans them for
   * exact duplicates (a repeated node makes the matrix exactly singular, which
   * is reported through \ref info -- solving would divide by zero). */
  template <int Rows_, int Cols_>
  BjorckPereyra& compute(const Vandermonde<Scalar, Rows_, Cols_>& V) {
    eigen_assert(V.rows() == V.cols() && "BjorckPereyra requires a square Vandermonde matrix");
    m_x = V.nodes();
    m_info = Success;
    const Index n = m_x.size();
    for (Index j = 1; j < n && m_info == Success; ++j)
      for (Index i = 0; i < j; ++i)
        if (m_x[i] == m_x[j]) {
          m_info = NumericalIssue;
          break;
        }
    m_isInitialized = true;
    return *this;
  }

  Index rows() const noexcept { return m_x.size(); }
  Index cols() const noexcept { return m_x.size(); }

  /** \returns \c Success, or \c NumericalIssue when the nodes contain an exact
   * duplicate (the matrix is singular). */
  ComputationInfo info() const {
    eigen_assert(m_isInitialized && "BjorckPereyra is not initialized.");
    return m_info;
  }

#ifdef EIGEN_PARSED_BY_DOXYGEN
  /** \returns the solution \c a of \c V*a = \a f, as a lazily evaluated
   * expression. Supports multiple right-hand sides. The transposed and adjoint
   * systems are solved through \c transpose().solve(b) and \c adjoint().solve(b).
   * \pre \ref compute has been called. */
  template <typename Rhs>
  inline const Solve<BjorckPereyra, Rhs> solve(const MatrixBase<Rhs>& f) const;
#endif

#ifndef EIGEN_PARSED_BY_DOXYGEN
  /** \internal Primal solve V*a = rhs: divided differences (Newton coefficients),
   * then the Newton-to-monomial basis change. */
  template <typename RhsType, typename DstType>
  void _solve_impl(const RhsType& rhs, DstType& dst) const {
    const Index n = m_x.size();
    dst = rhs;
    for (Index k = 0; k < rhs.cols(); ++k) {
      auto a = dst.col(k);
      for (Index j = 0; j < n - 1; ++j)
        for (Index i = n - 1; i > j; --i) a[i] = (a[i] - a[i - 1]) / (m_x[i] - m_x[i - j - 1]);
      for (Index j = n - 2; j >= 0; --j)
        for (Index i = j; i < n - 1; ++i) a[i] -= m_x[j] * a[i + 1];
    }
  }

  /** \internal Transposed (dual) solve V^T*w = rhs: the transposes of the primal
   * elementary steps, applied in reverse order; conjugated on the way in and out
   * for the adjoint. */
  template <bool Conjugate, typename RhsType, typename DstType>
  void _solve_impl_transposed(const RhsType& rhs, DstType& dst) const {
    const Index n = m_x.size();
    dst = rhs.template conjugateIf<Conjugate>();
    for (Index k = 0; k < rhs.cols(); ++k) {
      auto w = dst.col(k);
      for (Index j = 0; j < n - 1; ++j)
        for (Index i = n - 1; i > j; --i) w[i] -= m_x[j] * w[i - 1];
      for (Index j = n - 2; j >= 0; --j) {
        for (Index i = j + 1; i < n; ++i) w[i] /= m_x[i] - m_x[i - j - 1];
        for (Index i = j; i < n - 1; ++i) w[i] -= w[i + 1];
      }
    }
    if (Conjugate) dst = dst.conjugate().eval();
  }
#endif

 private:
  NodeVector m_x;
  bool m_isInitialized;
  ComputationInfo m_info;
};

namespace internal {

// Single product specialization covering every product tag; see the note in
// Circulant.h.
template <typename Scalar_, int Rows_, int Cols_, typename Rhs, int ProductTag>
struct generic_product_impl<Vandermonde<Scalar_, Rows_, Cols_>, Rhs, StructuredShape, DenseShape, ProductTag>
    : structured_product_impl<Vandermonde<Scalar_, Rows_, Cols_>, Rhs> {};

}  // namespace internal

}  // namespace Eigen

#endif  // EIGEN_STRUCTURED_VANDERMONDE_H
