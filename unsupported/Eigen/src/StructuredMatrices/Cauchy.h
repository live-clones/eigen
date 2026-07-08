// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_STRUCTURED_CAUCHY_H
#define EIGEN_STRUCTURED_CAUCHY_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

template <typename Scalar_, int Rows_ = Dynamic, int Cols_ = Dynamic>
class Cauchy;

template <typename Scalar_>
class CauchyLU;

namespace internal {

template <typename Scalar_, int Rows_, int Cols_>
struct traits<Cauchy<Scalar_, Rows_, Cols_>> {
  using Scalar = Scalar_;
  using StorageKind = Dense;
  using XprKind = MatrixXpr;
  using StorageIndex = int;
  static constexpr int RowsAtCompileTime = Rows_;
  static constexpr int ColsAtCompileTime = Cols_;
  static constexpr int MaxRowsAtCompileTime = Rows_;
  static constexpr int MaxColsAtCompileTime = Cols_;
  // Deliberately no NestByRefBit: transpose(), conjugate() and adjoint() return
  // owning temporaries, so Product must nest the operator by value for a
  // delayed-evaluated product expression to keep its left factor alive. The copy
  // is O(m+n), negligible against the O(mn) product evaluation.
  static constexpr int Flags = 0;
};

template <typename Scalar_, int Rows_, int Cols_>
struct evaluator_traits<Cauchy<Scalar_, Rows_, Cols_>> {
  using Kind = IndexBased;
  using Shape = StructuredShape;
};

template <typename Scalar_>
struct traits<CauchyLU<Scalar_>> : traits<Matrix<Scalar_, Dynamic, Dynamic>> {
  using XprKind = MatrixXpr;
  using StorageKind = SolverStorage;
  using StorageIndex = int;
  using BaseTraits = traits<Matrix<Scalar_, Dynamic, Dynamic>>;
  enum { Flags = BaseTraits::Flags & RowMajorBit, CoeffReadCost = Dynamic };
};

}  // namespace internal

/** \ingroup StructuredMatrices_Module
 * \class Cauchy
 * \brief An \c m x \c n Cauchy matrix represented by its two node vectors.
 *
 * A Cauchy matrix has entry \c (i,j) equal to \f$ 1/(x_i - y_j) \f$; the class
 * stores only the \c m + \c n nodes. Products are evaluated directly at O(mn)
 * operations -- the same cost as a dense product, but with O(m+n) storage and
 * without ever forming the matrix. (Fast approximate products via multipole
 * expansions are out of scope.)
 *
 * The class is closed under transposition: \f$ C(x,y)^T = C(-y,-x) \f$, and the
 * determinant of a square Cauchy matrix has the classical closed form
 * \f$ \prod_{i<j}(x_j-x_i)(y_i-y_j) \big/ \prod_{i,j}(x_i-y_j) \f$.
 *
 * Because \c operator* returns an Eigen product expression, a \c Cauchy also
 * drops into the matrix-free iterative solvers, and it can be assigned to a
 * dense matrix when an explicit representation is needed. As with any
 * matrix-free operator, the iterative solvers must be instantiated with
 * \c IdentityPreconditioner (e.g.
 * \c GMRES<Cauchy<double>,IdentityPreconditioner>): the default preconditioners
 * read individual coefficients through \c col() or \c InnerIterator, which the
 * structured operators do not expose.
 *
 * Square systems are solved in O(n^2) by \ref CauchyLU, a partially pivoted LU
 * factorization computed through the displacement structure
 * (Gohberg-Kailath-Olshevsky): unlike the Toeplitz/Levinson world, Cauchy
 * structure survives row permutations, which is what makes fast \em pivoted
 * factorization possible. The Hilbert matrix is the Cauchy matrix with
 * \c x_i = i+1, \c y_j = -j.
 *
 * All nodes \c x_i must differ from all nodes \c y_j (entries would be infinite
 * otherwise); this is not checked.
 *
 * \tparam Scalar_ the scalar type, real or complex.
 * \tparam Rows_ the number of rows at compile time, or \c Dynamic (the default).
 * \tparam Cols_ the number of columns at compile time, or \c Dynamic (the default).
 *
 * \sa class CauchyLU, makeCauchy()
 */
template <typename Scalar_, int Rows_, int Cols_>
class Cauchy : public EigenBase<Cauchy<Scalar_, Rows_, Cols_>> {
 public:
  using Scalar = Scalar_;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using StorageIndex = int;
  using RowNodeVector = Matrix<Scalar, Rows_, 1>;
  using ColNodeVector = Matrix<Scalar, Cols_, 1>;

  static constexpr int RowsAtCompileTime = Rows_;
  static constexpr int ColsAtCompileTime = Cols_;
  static constexpr int MaxRowsAtCompileTime = Rows_;
  static constexpr int MaxColsAtCompileTime = Cols_;
  static constexpr int SizeAtCompileTime = internal::size_at_compile_time(Rows_, Cols_);
  static constexpr int MaxSizeAtCompileTime = SizeAtCompileTime;
  static constexpr bool IsRowMajor = false;
  // Deliberately no IsVectorAtCompileTime: Ref<const Cauchy>'s default StrideType
  // argument reads it, so its absence makes internal::is_ref_compatible SFINAE to
  // false and keeps the iterative solvers on their matrix-free path.

  /** Builds a Cauchy matrix from the row nodes \a x and the column nodes \a y:
   * entry \c (i,j) is \c 1/(x[i] - y[j]). */
  template <typename XDerived, typename YDerived>
  Cauchy(const MatrixBase<XDerived>& x, const MatrixBase<YDerived>& y) : m_x(x), m_y(y) {
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(XDerived)
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(YDerived)
    eigen_assert(m_x.size() > 0 && m_y.size() > 0 && "Cauchy node vectors must be non-empty");
  }

  EIGEN_DEVICE_FUNC Index rows() const { return m_x.size(); }
  EIGEN_DEVICE_FUNC Index cols() const { return m_y.size(); }

  /** \returns the row node vector \c x. */
  const RowNodeVector& rowNodes() const { return m_x; }
  /** \returns the column node vector \c y. */
  const ColNodeVector& colNodes() const { return m_y; }

  /** \returns the coefficient at row \a row and column \a col. */
  Scalar coeff(Index row, Index col) const { return Scalar(1) / (m_x.coeff(row) - m_y.coeff(col)); }

  /** \returns the transpose of \c *this, itself a Cauchy operator:
   * \f$ C(x,y)^T = C(-y,-x) \f$. */
  Cauchy<Scalar, Cols_, Rows_> transpose() const { return Cauchy<Scalar, Cols_, Rows_>(-m_y, -m_x); }

  /** \returns the complex conjugate of \c *this, itself a Cauchy operator (the
   * one with conjugated nodes). */
  Cauchy conjugate() const { return Cauchy(m_x.conjugate(), m_y.conjugate()); }

  /** \returns the adjoint of \c *this, itself a Cauchy operator:
   * \f$ C(x,y)^H = C(-\bar y, -\bar x) \f$. */
  Cauchy<Scalar, Cols_, Rows_> adjoint() const {
    return Cauchy<Scalar, Cols_, Rows_>(-m_y.conjugate(), -m_x.conjugate());
  }

  /** \returns the determinant of a \b square Cauchy matrix through the classical
   * closed form \f$ \prod_{i<j}(x_j-x_i)(y_i-y_j) \big/ \prod_{i,j}(x_i-y_j) \f$,
   * in O(n^2) operations. All factors enter a single accumulation kept in the
   * balanced form \c m * 2^e -- every factor and the running value are
   * renormalized to unit magnitude with the power of two tracked separately
   * (exact frexp/ldexp rescaling), numerator factors multiplied in and
   * denominator factors divided out -- so no intermediate can overflow or
   * underflow when the determinant itself is representable. Zero factors
   * (coincident \c x or \c y nodes) and non-finite factors propagate exactly. */
  Scalar determinant() const {
    eigen_assert(rows() == cols() && "Cauchy::determinant requires a square matrix");
    const Index n = rows();
    Scalar det(1);
    Index exponent = 0;
    for (Index j = 1; j < n; ++j)
      for (Index i = 0; i < j; ++i) {
        det = balance(det * balance(m_x.coeff(j) - m_x.coeff(i), exponent), exponent);
        det = balance(det * balance(m_y.coeff(i) - m_y.coeff(j), exponent), exponent);
      }
    for (Index j = 0; j < n; ++j)
      for (Index i = 0; i < n; ++i) {
        Index denomExponent = 0;
        const Scalar d = balance(m_x.coeff(i) - m_y.coeff(j), denomExponent);
        exponent -= denomExponent;
        det = balance(det / d, exponent);
      }
    // ldexp saturates cleanly to zero / infinity once the accumulated exponent
    // leaves the representable range; the clamp only guards the narrowing to int.
    constexpr Index kMaxExponent = Index(1) << 24;
    const int e = static_cast<int>(numext::mini(numext::maxi(exponent, -kMaxExponent), kMaxExponent));
    return applyExponent(det, e);
  }

  /** \internal Writes the dense representation into \a dst, one vectorized
   * column at a time. Invoked through \c dense = cauchy; */
  template <typename Dest>
  void evalTo(Dest& dst) const {
    for (Index j = 0; j < cols(); ++j) dst.col(j) = (m_x.array() - m_y.coeff(j)).inverse().matrix();
  }

  /** \internal Computes \c dst += (*this), see evalTo(). */
  template <typename Dest>
  void addTo(Dest& dst) const {
    for (Index j = 0; j < cols(); ++j) dst.col(j) += (m_x.array() - m_y.coeff(j)).inverse().matrix();
  }

  /** \internal Computes \c dst -= (*this), see evalTo(). */
  template <typename Dest>
  void subTo(Dest& dst) const {
    for (Index j = 0; j < cols(); ++j) dst.col(j) -= (m_x.array() - m_y.coeff(j)).inverse().matrix();
  }

  /** \returns the product expression \c (*this) * \a v, evaluated directly at
   * O(mn) operations and O(1) extra storage. */
  template <typename Rhs>
  Product<Cauchy, Rhs, AliasFreeProduct> operator*(const MatrixBase<Rhs>& v) const {
    return Product<Cauchy, Rhs, AliasFreeProduct>(*this, v.derived());
  }

  /** \internal Computes \c dst += alpha * (*this) * rhs. \c ProductScalar is the
   * promoted scalar of the product (complex when a real operator is applied to a
   * complex right-hand side); the accumulation runs in the promoted type. */
  template <typename Dest, typename Rhs, typename ProductScalar>
  void addProduct(Dest& dst, const Rhs& rhs, const ProductScalar& alpha) const {
    const Index m = rows(), n = cols();
    eigen_assert(rhs.rows() == n && "invalid product: dimensions do not match");
    for (Index k = 0; k < rhs.cols(); ++k)
      for (Index i = 0; i < m; ++i) {
        const Scalar xi = m_x.coeff(i);
        ProductScalar acc(0);
        for (Index j = 0; j < n; ++j) acc += rhs.coeff(j, k) / (xi - m_y.coeff(j));
        dst.coeffRef(i, k) += alpha * acc;
      }
  }

 private:
  /** \internal Applies the exact scaling \c 2^e to a real or complex value
   * componentwise (the overload set covers both possible \c Scalar kinds). */
  static RealScalar applyExponent(const RealScalar& v, int e) { return std::ldexp(v, e); }
  static std::complex<RealScalar> applyExponent(const std::complex<RealScalar>& v, int e) {
    return std::complex<RealScalar>(std::ldexp(v.real(), e), std::ldexp(v.imag(), e));
  }

  /** \internal Rescales \a v by the power of two that brings \c max(|re|,|im|)
   * into [0.5, 1), accumulating the removed exponent into \a exponent. The
   * rescaling is exact, so no roundoff is introduced. Zeros and non-finite
   * values, which must propagate exactly through determinant(), are returned
   * untouched. */
  static Scalar balance(const Scalar& v, Index& exponent) {
    const RealScalar mag = numext::maxi(numext::abs(numext::real(v)), numext::abs(numext::imag(v)));
    if (!(mag > RealScalar(0)) || !(numext::isfinite)(mag)) return v;
    int e;
    std::frexp(mag, &e);
    exponent += e;
    return applyExponent(v, -e);
  }

  RowNodeVector m_x;
  ColNodeVector m_y;
};

/** \ingroup StructuredMatrices_Module
 * \returns a \ref Cauchy operator with row nodes \a x and column nodes \a y. The
 * compile-time dimensions of the operator are deduced from the node vectors. */
template <typename XDerived, typename YDerived>
Cauchy<typename XDerived::Scalar, XDerived::SizeAtCompileTime, YDerived::SizeAtCompileTime> makeCauchy(
    const MatrixBase<XDerived>& x, const MatrixBase<YDerived>& y) {
  return Cauchy<typename XDerived::Scalar, XDerived::SizeAtCompileTime, YDerived::SizeAtCompileTime>(x, y);
}

/** \ingroup StructuredMatrices_Module
 * \class CauchyLU
 * \brief Partially pivoted O(n^2) LU solver for square Cauchy systems
 * (Gohberg-Kailath-Olshevsky).
 *
 * A Cauchy matrix satisfies the displacement equation
 * \f$ D_x C - C D_y = \mathbf{1}\mathbf{1}^T \f$, and -- crucially -- this
 * structure survives row permutations. The GKO algorithm exploits it to compute
 * the row-pivoted factorization \c P*C = L*U in O(n^2) operations: at each step
 * the current column of the Schur complement is generated from O(n) data, the
 * largest entry is chosen as pivot, and the Schur complement stays Cauchy-like
 * with updated generators. This gives Cauchy systems what the Levinson world
 * lacks: genuine partial pivoting at fast-algorithm cost.
 *
 * Usage follows the usual decomposition style, including the transposed and
 * adjoint solves of \c SolverBase (from the same factorization):
 * \code
 *   CauchyLU<double> lu(C);              // or lu.compute(C);
 *   VectorXd u = lu.solve(b);             // solve C   * u = b
 *   VectorXd v = lu.transpose().solve(b); // solve C^T * v = b
 *   VectorXd w = lu.adjoint().solve(b);   // solve C^H * w = b
 * \endcode
 *
 * The factors are stored densely (O(n^2) memory, like the dense LU
 * decompositions).
 *
 * References: I. Gohberg, T. Kailath, V. Olshevsky, "Fast Gaussian elimination
 * with partial pivoting for matrices with displacement structure," Math. Comp.
 * 64 (1995).
 *
 * \tparam Scalar_ the scalar type, real or complex.
 *
 * \sa class Cauchy
 */
template <typename Scalar_>
class CauchyLU : public SolverBase<CauchyLU<Scalar_>> {
 public:
  using Base = SolverBase<CauchyLU>;
  friend class SolverBase<CauchyLU>;
  EIGEN_GENERIC_PUBLIC_INTERFACE(CauchyLU)
  using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic>;
  using DenseVector = Matrix<Scalar, Dynamic, 1>;

  /** Default constructor; call \ref compute before \ref solve. */
  CauchyLU() : m_isInitialized(false), m_info(InvalidInput) {}

  /** Constructs and factorizes from the square Cauchy matrix \a C. */
  template <int Rows_, int Cols_>
  explicit CauchyLU(const Cauchy<Scalar, Rows_, Cols_>& C) : m_isInitialized(false), m_info(InvalidInput) {
    compute(C);
  }

  /** Factorizes the square Cauchy matrix \a C as \c P*C = L*U by the GKO
   * recursion with partial pivoting. \sa solve */
  template <int Rows_, int Cols_>
  CauchyLU& compute(const Cauchy<Scalar, Rows_, Cols_>& C) {
    eigen_assert(C.rows() == C.cols() && "CauchyLU requires a square Cauchy matrix");
    const Index n = C.rows();
    DenseVector x = C.rowNodes();  // row nodes, permuted in place by pivoting
    const DenseVector y = C.colNodes();
    DenseVector a = DenseVector::Ones(n);  // displacement generators of the
    DenseVector b = DenseVector::Ones(n);  // running Schur complement
    m_lu.resize(n, n);
    m_perm.resize(static_cast<std::size_t>(n));
    m_info = Success;

    for (Index k = 0; k < n; ++k) {
      // Column k of the current Schur complement, generated from O(n) data.
      Index piv = k;
      RealScalar best(-1);
      for (Index i = k; i < n; ++i) {
        m_lu(i, k) = a[i] * b[k] / (x[i] - y[k]);
        const RealScalar mag = numext::abs(m_lu(i, k));
        if (!(mag <= best)) {  // negated so a NaN candidate is preferred and propagates
          best = mag;
          piv = i;
        }
      }
      if (piv != k) {
        m_lu.row(piv).head(k + 1).swap(m_lu.row(k).head(k + 1));
        std::swap(x[piv], x[k]);
        std::swap(a[piv], a[k]);
      }
      m_perm[static_cast<std::size_t>(k)] = piv;
      const Scalar pivot = m_lu(k, k);
      if (pivot == Scalar(0)) {
        // Exactly singular: the remaining Schur complement column is zero.
        m_info = NumericalIssue;
        m_lu.row(k).tail(n - k - 1).setZero();
        m_lu.col(k).tail(n - k - 1).setZero();
        continue;
      }
      // L column (unit diagonal implicit), U row, and generator updates that
      // keep the Schur complement Cauchy-like.
      for (Index i = k + 1; i < n; ++i) m_lu(i, k) /= pivot;
      for (Index j = k + 1; j < n; ++j) m_lu(k, j) = a[k] * b[j] / (x[k] - y[j]);
      for (Index i = k + 1; i < n; ++i) a[i] *= (x[i] - x[k]) / (x[i] - y[k]);
      for (Index j = k + 1; j < n; ++j) b[j] *= (y[j] - y[k]) / (y[j] - x[k]);
    }
    m_isInitialized = true;
    return *this;
  }

  Index rows() const noexcept { return m_lu.rows(); }
  Index cols() const noexcept { return m_lu.cols(); }

  /** \returns \c Success, or \c NumericalIssue when the matrix is exactly
   * singular (a zero pivot survived partial pivoting). */
  ComputationInfo info() const {
    eigen_assert(m_isInitialized && "CauchyLU is not initialized.");
    return m_info;
  }

#ifdef EIGEN_PARSED_BY_DOXYGEN
  /** \returns the solution \c u of \c C*u = \a b, as a lazily evaluated
   * expression. Supports multiple right-hand sides. The transposed and adjoint
   * systems reuse the same factorization through \c transpose().solve(b) and
   * \c adjoint().solve(b). \pre \ref compute has been called. */
  template <typename Rhs>
  inline const Solve<CauchyLU, Rhs> solve(const MatrixBase<Rhs>& b) const;
#endif

#ifndef EIGEN_PARSED_BY_DOXYGEN
  /** \internal P*C = L*U, so C*u = b becomes L*U*u = P*b. */
  template <typename RhsType, typename DstType>
  void _solve_impl(const RhsType& rhs, DstType& dst) const {
    dst = rhs;
    for (Index k = 0; k < rows(); ++k) {
      const Index piv = m_perm[static_cast<std::size_t>(k)];
      if (piv != k) dst.row(k).swap(dst.row(piv));
    }
    m_lu.template triangularView<UnitLower>().solveInPlace(dst);
    m_lu.template triangularView<Upper>().solveInPlace(dst);
  }

  /** \internal C^T = U^T L^T P, so C^T*w = b becomes U^T (L^T (P w)) = b: solve
   * the transposed triangles, then undo the transpositions in reverse order;
   * conjugated on the way in and out for the adjoint. */
  template <bool Conjugate, typename RhsType, typename DstType>
  void _solve_impl_transposed(const RhsType& rhs, DstType& dst) const {
    dst = rhs.template conjugateIf<Conjugate>();
    m_lu.template triangularView<Upper>().transpose().solveInPlace(dst);
    m_lu.template triangularView<UnitLower>().transpose().solveInPlace(dst);
    for (Index k = rows() - 1; k >= 0; --k) {
      const Index piv = m_perm[static_cast<std::size_t>(k)];
      if (piv != k) dst.row(k).swap(dst.row(piv));
    }
    if (Conjugate) dst = dst.conjugate().eval();
  }
#endif

 private:
  DenseMatrix m_lu;
  std::vector<Index> m_perm;  // transposition applied at each elimination step
  bool m_isInitialized;
  ComputationInfo m_info;
};

namespace internal {

// Single product specialization covering every product tag; see the note in
// Circulant.h.
template <typename Scalar_, int Rows_, int Cols_, typename Rhs, int ProductTag>
struct generic_product_impl<Cauchy<Scalar_, Rows_, Cols_>, Rhs, StructuredShape, DenseShape, ProductTag>
    : structured_product_impl<Cauchy<Scalar_, Rows_, Cols_>, Rhs> {};

}  // namespace internal

}  // namespace Eigen

#endif  // EIGEN_STRUCTURED_CAUCHY_H
