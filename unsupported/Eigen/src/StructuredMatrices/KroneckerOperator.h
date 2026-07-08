// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// References:
//  [1] C. F. Van Loan, "The ubiquitous Kronecker product", Journal of
//      Computational and Applied Mathematics 123 (2000), 85-100. The vec
//      identity driving every product and solve below -- stated there as
//      Y = C X B^T <=> vec(Y) = (B (x) C) vec(X), i.e. with this file's
//      operand naming (A (x) B) vec(X) = vec(B X A^T) -- together with the
//      factor-wise identities for the inverse, pseudo-inverse,
//      eigendecomposition, SVD and determinant of a Kronecker product.

#ifndef EIGEN_STRUCTURED_KRONECKER_OPERATOR_H
#define EIGEN_STRUCTURED_KRONECKER_OPERATOR_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

template <typename LhsMatrix, typename RhsMatrix>
class KroneckerOperator;

namespace internal {

/** \internal Compile-time product of two dimensions, Dynamic-aware. */
constexpr int kron_dim(int a, int b) { return (a == Dynamic || b == Dynamic) ? Dynamic : a * b; }

template <typename LhsMatrix, typename RhsMatrix>
struct traits<KroneckerOperator<LhsMatrix, RhsMatrix>> {
  using Scalar = typename LhsMatrix::Scalar;
  using StorageKind = Dense;
  using XprKind = MatrixXpr;
  using StorageIndex = int;
  static constexpr int RowsAtCompileTime =
      kron_dim(traits<LhsMatrix>::RowsAtCompileTime, traits<RhsMatrix>::RowsAtCompileTime);
  static constexpr int ColsAtCompileTime =
      kron_dim(traits<LhsMatrix>::ColsAtCompileTime, traits<RhsMatrix>::ColsAtCompileTime);
  static constexpr int MaxRowsAtCompileTime = RowsAtCompileTime;
  static constexpr int MaxColsAtCompileTime = ColsAtCompileTime;
  // Deliberately no NestByRefBit: transpose(), conjugate(), adjoint(), inverse(),
  // eigenvectors(), matrixU() and matrixV() return owning temporaries (the
  // operator stores its factors by value), so Product must nest the operator by
  // value for a delayed-evaluated product expression to keep its left factor
  // alive. The copy is O(m1 n1 + m2 n2), negligible against the product
  // evaluation.
  static constexpr int Flags = 0;
};

template <typename LhsMatrix, typename RhsMatrix>
struct evaluator_traits<KroneckerOperator<LhsMatrix, RhsMatrix>> {
  using Kind = IndexBased;
  using Shape = StructuredShape;
};

}  // namespace internal

/** \ingroup StructuredMatrices_Module
 * \class KroneckerOperator
 * \brief The Kronecker product \f$ A \otimes B \f$ as an implicit operator that is
 * never materialized.
 *
 * For \c A of size \c m1 x \c n1 and \c B of size \c m2 x \c n2, the Kronecker
 * product is the \c m1*m2 x \c n1*n2 block matrix whose block \c (i,j) is
 * \c A(i,j)*B. This class stores only the two factors and evaluates every
 * operation through them:
 *
 * - the matrix-vector product uses the vec identity
 *   \f$ (A \otimes B)\,\mathrm{vec}(X) = \mathrm{vec}(B X A^T) \f$, costing
 *   O(m2 n2 n1 + m2 n1 m1) instead of the O(m1 m2 n1 n2) of a materialized
 *   product;
 * - linear solves and \ref inverse factor through decompositions of \c A and
 *   \c B (\f$ (A \otimes B)^{-1} = A^{-1} \otimes B^{-1} \f$); \ref rank and
 *   minimum-norm least-squares solves (\f$ (A \otimes B)^+ = A^+ \otimes B^+ \f$)
 *   go through the factor SVDs, thresholding the pairwise singular-value
 *   products \f$ \sigma_i(A)\,\sigma_j(B) \f$ -- the singular values of the
 *   Kronecker product -- at the product level;
 * - the eigendecomposition and the (thin) SVD are Kronecker products of the
 *   factor decompositions: the eigenvector and singular-vector matrices are
 *   returned as \c KroneckerOperator objects themselves, never materialized;
 * - \ref determinant uses \f$ \det(A \otimes B) = \det(A)^{n_2}\det(B)^{n_1} \f$,
 *   accumulated in an exponent-balanced form so it neither overflows nor
 *   underflows when the result is representable.
 *
 * The class is closed under \ref transpose, \ref conjugate and \ref adjoint
 * (\f$ (A \otimes B)^T = A^T \otimes B^T \f$). \c operator* returns an Eigen
 * product expression, so the operator plugs into the matrix-free iterative
 * solvers, and it can be assigned to a dense matrix when an explicit
 * representation is needed. As with any matrix-free operator, the iterative
 * solvers must be instantiated with \c IdentityPreconditioner (e.g.
 * \c ConjugateGradient<KroneckerOperator<MatrixXd,MatrixXd>,Lower|Upper,IdentityPreconditioner>):
 * the default preconditioners read individual coefficients through \c col() or
 * \c InnerIterator, which the structured operators do not expose.
 *
 * In contrast to \c kroneckerProduct() (the KroneckerProduct module), which
 * builds an expression meant to be evaluated into a dense matrix, this class is
 * an operator meant to be applied and solved with, without ever forming the
 * product.
 *
 * \tparam LhsMatrix the plain dense matrix type of the left factor \c A.
 * \tparam RhsMatrix the plain dense matrix type of the right factor \c B; its
 *         scalar type must match that of \c LhsMatrix.
 *
 * \sa makeKroneckerOperator(), class Circulant, class Toeplitz
 */
template <typename LhsMatrix, typename RhsMatrix>
class KroneckerOperator : public EigenBase<KroneckerOperator<LhsMatrix, RhsMatrix>> {
 public:
  using Scalar = typename LhsMatrix::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using StorageIndex = int;
  using ComplexScalar = std::complex<RealScalar>;
  // The vec-trick reshapes below identify a vector of length n1*n2 with an
  // n2 x n1 matrix whose columns are stacked, so every workspace and Map taking
  // part in a reshape is pinned to ColMajor explicitly: the semantics must not
  // change under EIGEN_DEFAULT_TO_ROW_MAJOR.
  using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;
  using DenseVector = Matrix<Scalar, Dynamic, 1>;
  using RealVector = Matrix<RealScalar, Dynamic, 1>;
  using ComplexMatrix = Matrix<ComplexScalar, Dynamic, Dynamic, ColMajor>;
  using ComplexVector = Matrix<ComplexScalar, Dynamic, 1>;

  static_assert(std::is_same<Scalar, typename RhsMatrix::Scalar>::value,
                "KroneckerOperator requires both factors to have the same scalar type");

  static constexpr int RowsAtCompileTime =
      internal::kron_dim(LhsMatrix::RowsAtCompileTime, RhsMatrix::RowsAtCompileTime);
  static constexpr int ColsAtCompileTime =
      internal::kron_dim(LhsMatrix::ColsAtCompileTime, RhsMatrix::ColsAtCompileTime);
  static constexpr int MaxRowsAtCompileTime = RowsAtCompileTime;
  static constexpr int MaxColsAtCompileTime = ColsAtCompileTime;
  static constexpr int SizeAtCompileTime = internal::size_at_compile_time(RowsAtCompileTime, ColsAtCompileTime);
  static constexpr int MaxSizeAtCompileTime = SizeAtCompileTime;
  static constexpr bool IsRowMajor = false;
  // Deliberately no IsVectorAtCompileTime: Ref<const KroneckerOperator>'s default
  // StrideType argument reads it, so its absence makes internal::is_ref_compatible
  // SFINAE to false and keeps the iterative solvers on their matrix-free path.

  /** Builds the operator \c A (x) \c B from the two factors, which are copied. */
  template <typename LhsDerived, typename RhsDerived>
  KroneckerOperator(const MatrixBase<LhsDerived>& a, const MatrixBase<RhsDerived>& b) : m_A(a), m_B(b) {
    eigen_assert(m_A.size() > 0 && m_B.size() > 0 && "KroneckerOperator factors must be non-empty");
  }

  EIGEN_DEVICE_FUNC Index rows() const { return m_A.rows() * m_B.rows(); }
  EIGEN_DEVICE_FUNC Index cols() const { return m_A.cols() * m_B.cols(); }

  /** \returns the left factor \c A. */
  const LhsMatrix& lhs() const { return m_A; }
  /** \returns the right factor \c B. */
  const RhsMatrix& rhs() const { return m_B; }

  /** \returns the coefficient at row \a row and column \a col. */
  Scalar coeff(Index row, Index col) const {
    const Index m2 = m_B.rows(), n2 = m_B.cols();
    return m_A.coeff(row / m2, col / n2) * m_B.coeff(row % m2, col % n2);
  }

  /** \returns the transpose \f$ A^T \otimes B^T \f$, itself a Kronecker operator. */
  KroneckerOperator<Matrix<Scalar, LhsMatrix::ColsAtCompileTime, LhsMatrix::RowsAtCompileTime>,
                    Matrix<Scalar, RhsMatrix::ColsAtCompileTime, RhsMatrix::RowsAtCompileTime>>
  transpose() const {
    return {m_A.transpose(), m_B.transpose()};
  }

  /** \returns the conjugate \f$ \bar A \otimes \bar B \f$, itself a Kronecker operator. */
  KroneckerOperator conjugate() const { return {m_A.conjugate(), m_B.conjugate()}; }

  /** \returns the adjoint \f$ A^H \otimes B^H \f$, itself a Kronecker operator. */
  KroneckerOperator<Matrix<Scalar, LhsMatrix::ColsAtCompileTime, LhsMatrix::RowsAtCompileTime>,
                    Matrix<Scalar, RhsMatrix::ColsAtCompileTime, RhsMatrix::RowsAtCompileTime>>
  adjoint() const {
    return {m_A.adjoint(), m_B.adjoint()};
  }

  /** \returns the solution of \c (*this) * x = b for \b square factors, obtained
   * from one LU decomposition per factor: reshaping \c b column-wise as
   * \c mat(b) of size \c n2 x \c n1, the system reads \f$ B X A^T = \mathrm{mat}(b) \f$,
   * so \f$ X = B^{-1} \mathrm{mat}(b) A^{-T} \f$. Supports multiple right-hand
   * sides at O(n1^3 + n2^3 + nrhs (n1 + n2) n1 n2) total cost.
   * \warning Both factors must be invertible, like in \c PartialPivLU; use
   * \ref leastSquaresSolve for rank-deficient or rectangular factors. */
  template <typename Rhs>
  Matrix<Scalar, Dynamic, Rhs::ColsAtCompileTime> solve(const MatrixBase<Rhs>& b) const {
    const Index n1 = m_A.cols(), n2 = m_B.cols();
    eigen_assert(m_A.rows() == n1 && m_B.rows() == n2 && "KroneckerOperator::solve requires square factors");
    eigen_assert(b.rows() == n1 * n2 && "right-hand side has the wrong number of rows");
    PartialPivLU<DenseMatrix> luA(m_A), luB(m_B);
    Matrix<Scalar, Dynamic, Rhs::ColsAtCompileTime> x(n1 * n2, b.cols());
    for (Index k = 0; k < b.cols(); ++k) {
      const DenseVector bc = b.col(k);
      const Map<const DenseMatrix> Bmat(bc.data(), n2, n1);
      const DenseMatrix X = luA.solve(luB.solve(Bmat).transpose()).transpose();
      x.col(k) = Map<const DenseVector>(X.data(), X.size());
    }
    return x;
  }

  /** \returns the minimum-norm least-squares solution of \c (*this) * x = b,
   * through one SVD per factor: with \f$ A = U_A \Sigma_A V_A^H \f$ and
   * \f$ B = U_B \Sigma_B V_B^H \f$, the SVD of the product is
   * \f$ (U_A \otimes U_B)(\Sigma_A \otimes \Sigma_B)(V_A \otimes V_B)^H \f$, and
   * the pseudo-inverse is applied via the vec identity. The singular values of
   * the product are the pairwise products \f$ \sigma_i(A)\,\sigma_j(B) \f$, so
   * the truncation thresholds those products against the product-level
   * threshold of \ref rank -- factor-wise truncation would invert modes that
   * are negligible at the product level. Handles rectangular and rank-deficient
   * factors. Supports multiple right-hand sides. */
  template <typename Rhs>
  Matrix<Scalar, Dynamic, Rhs::ColsAtCompileTime> leastSquaresSolve(const MatrixBase<Rhs>& b) const {
    const Index m1 = m_A.rows(), m2 = m_B.rows();
    eigen_assert(b.rows() == m1 * m2 && "right-hand side has the wrong number of rows");
    JacobiSVD<DenseMatrix> svdA(m_A, ComputeThinU | ComputeThinV), svdB(m_B, ComputeThinU | ComputeThinV);
    const RealVector sa = svdA.singularValues(), sb = svdB.singularValues();
    const RealScalar tol = rankThreshold(sa, sb);
    const Index kA = sa.size(), kB = sb.size();
    Matrix<Scalar, Dynamic, Rhs::ColsAtCompileTime> x(cols(), b.cols());
    for (Index k = 0; k < b.cols(); ++k) {
      const DenseVector bc = b.col(k);
      const Map<const DenseMatrix> Bmat(bc.data(), m2, m1);
      // Project onto the factor singular bases: M = U_B^H mat(b) conj(U_A) is the
      // k_B x k_A matricization of (U_A (x) U_B)^H b, by the vec identity [1].
      DenseMatrix M = svdB.matrixU().adjoint() * Bmat * svdA.matrixU().conjugate();
      // Invert only the pairwise products at/above the product-level threshold.
      // The negated comparison keeps NaN products in the inverted set, so a NaN
      // input propagates to the output instead of being silently zeroed.
      for (Index j = 0; j < kA; ++j)
        for (Index i = 0; i < kB; ++i) M(i, j) = !(sb[i] * sa[j] < tol) ? M(i, j) / (sb[i] * sa[j]) : Scalar(0);
      // Back to the original bases: mat(x) = V_B M V_A^T, again the vec identity.
      const DenseMatrix X = svdB.matrixV() * M * svdA.matrixV().transpose();
      x.col(k) = Map<const DenseVector>(X.data(), X.size());
    }
    return x;
  }

  /** \returns the numerical rank: the number of pairwise singular-value products
   * \f$ \sigma_i(A)\,\sigma_j(B) \f$ -- the singular values of the Kronecker
   * product -- that reach the threshold
   * \c min(rows(),cols()) * epsilon * sigma_max(A) * sigma_max(B), clamped from
   * below at the smallest normal number (the \c SVDBase convention). This is the
   * same threshold \ref leastSquaresSolve uses to decide which modes to invert.
   * Thresholding the products matters: factors that are each full rank against
   * their own threshold can still form pairwise products that are negligible at
   * the product level, so the rank can be smaller than the product of the
   * factor ranks. */
  Index rank() const {
    JacobiSVD<DenseMatrix> svdA(m_A), svdB(m_B);
    const RealVector sa = svdA.singularValues(), sb = svdB.singularValues();
    const RealScalar tol = rankThreshold(sa, sb);
    Index r = 0;
    for (Index i = 0; i < sa.size(); ++i)
      for (Index j = 0; j < sb.size(); ++j)
        if (!(sa[i] * sb[j] < tol)) ++r;  // negated so NaN products count as non-zero
    return r;
  }

  /** \returns the inverse \f$ A^{-1} \otimes B^{-1} \f$, itself a Kronecker
   * operator, for square invertible factors. */
  KroneckerOperator<DenseMatrix, DenseMatrix> inverse() const {
    eigen_assert(m_A.rows() == m_A.cols() && m_B.rows() == m_B.cols() &&
                 "KroneckerOperator::inverse requires square factors");
    return {m_A.inverse().eval(), m_B.inverse().eval()};
  }

  /** \returns the determinant \f$ \det(A)^{n_2} \det(B)^{n_1} \f$ for square
   * factors \c A of size \c n1 and \c B of size \c n2. The product is
   * accumulated from the factor LU diagonals in the balanced form \c m * 2^e --
   * every factor and the running product are renormalized to unit magnitude
   * with the power of two tracked separately -- so the partial products (in
   * particular \c det(A) and \c det(B) themselves, which can overflow or
   * underflow on their own) never leave the representable range when the
   * determinant itself is representable. */
  Scalar determinant() const {
    eigen_assert(m_A.rows() == m_A.cols() && m_B.rows() == m_B.cols() &&
                 "KroneckerOperator::determinant requires square factors");
    Index exponent = 0;
    Scalar mant = balancedDetPow(m_A, m_B.cols(), exponent);
    mant = balance(mant * balancedDetPow(m_B, m_A.cols(), exponent), exponent);
    // ldexp saturates cleanly to zero / infinity once the accumulated exponent
    // leaves the representable range; the clamp only guards the narrowing to int.
    constexpr Index kMaxExponent = Index(1) << 24;
    const int e = static_cast<int>(numext::mini(numext::maxi(exponent, -kMaxExponent), kMaxExponent));
    return ldexpScalar(mant, e);
  }

  /** \returns the eigenvalues for square factors, in Kronecker order: entry
   * \c i*n2 + j is \f$ \lambda_i(A)\,\mu_j(B) \f$, matching column \c i*n2 + j of
   * \ref eigenvectors. The set is not sorted -- there is no canonical eigenvalue
   * order, and sorting would break the Kronecker structure of the eigenvector
   * matrix. */
  ComplexVector eigenvalues() const {
    eigen_assert(m_A.rows() == m_A.cols() && m_B.rows() == m_B.cols() &&
                 "KroneckerOperator::eigenvalues requires square factors");
    ComplexEigenSolver<ComplexMatrix> esA(m_A.template cast<ComplexScalar>(), /*computeEigenvectors=*/false);
    ComplexEigenSolver<ComplexMatrix> esB(m_B.template cast<ComplexScalar>(), /*computeEigenvectors=*/false);
    eigen_assert(esA.info() == Success && esB.info() == Success);
    const Index nA = m_A.cols(), nB = m_B.cols();
    ComplexVector lambda(nA * nB);
    for (Index i = 0; i < nA; ++i)
      for (Index j = 0; j < nB; ++j) lambda[i * nB + j] = esA.eigenvalues()[i] * esB.eigenvalues()[j];
    return lambda;
  }

  /** \returns the matrix of eigenvectors \f$ V_A \otimes V_B \f$ for square
   * factors -- itself a Kronecker operator, never materialized. Column
   * \c i*n2 + j is \f$ v_i(A) \otimes v_j(B) \f$ and matches
   * \c eigenvalues()[i*n2 + j]. Assign it to a dense matrix to materialize. */
  KroneckerOperator<ComplexMatrix, ComplexMatrix> eigenvectors() const {
    eigen_assert(m_A.rows() == m_A.cols() && m_B.rows() == m_B.cols() &&
                 "KroneckerOperator::eigenvectors requires square factors");
    ComplexEigenSolver<ComplexMatrix> esA(m_A.template cast<ComplexScalar>());
    ComplexEigenSolver<ComplexMatrix> esB(m_B.template cast<ComplexScalar>());
    eigen_assert(esA.info() == Success && esB.info() == Success);
    return {esA.eigenvectors(), esB.eigenvectors()};
  }

  /** \returns the singular values of the thin SVD
   * \f$ A \otimes B = U \Sigma V^H \f$ in Kronecker order: entry \c i*k_B + j is
   * \f$ \sigma_i(A)\,\sigma_j(B) \f$ with \c k_A, \c k_B the factor thin ranks
   * \c min(rows, cols), matching the columns of \ref matrixU and \ref matrixV.
   * The values are not sorted (sorting would break the Kronecker structure of
   * \c U and \c V); for rectangular shapes the full SVD pads this set with
   * \c min(rows(),cols()) - k_A*k_B structural zeros. */
  RealVector singularValues() const {
    JacobiSVD<DenseMatrix> svdA(m_A), svdB(m_B);
    const Index kA = svdA.singularValues().size(), kB = svdB.singularValues().size();
    RealVector sv(kA * kB);
    for (Index i = 0; i < kA; ++i)
      for (Index j = 0; j < kB; ++j) sv[i * kB + j] = svdA.singularValues()[i] * svdB.singularValues()[j];
    return sv;
  }

  /** \returns the left singular vectors \f$ U_A \otimes U_B \f$ of the thin SVD,
   * itself a Kronecker operator with orthonormal columns; column \c i*k_B + j
   * matches \c singularValues()[i*k_B + j]. */
  KroneckerOperator<DenseMatrix, DenseMatrix> matrixU() const {
    JacobiSVD<DenseMatrix> svdA(m_A, ComputeThinU), svdB(m_B, ComputeThinU);
    return {svdA.matrixU(), svdB.matrixU()};
  }

  /** \returns the right singular vectors \f$ V_A \otimes V_B \f$ of the thin SVD,
   * itself a Kronecker operator with orthonormal columns; column \c i*k_B + j
   * matches \c singularValues()[i*k_B + j]. */
  KroneckerOperator<DenseMatrix, DenseMatrix> matrixV() const {
    JacobiSVD<DenseMatrix> svdA(m_A, ComputeThinV), svdB(m_B, ComputeThinV);
    return {svdA.matrixV(), svdB.matrixV()};
  }

  /** \internal Writes the dense representation into \a dst, block by block.
   * Invoked through \c dense = kron; */
  template <typename Dest>
  void evalTo(Dest& dst) const {
    const Index m2 = m_B.rows(), n2 = m_B.cols();
    for (Index j = 0; j < m_A.cols(); ++j)
      for (Index i = 0; i < m_A.rows(); ++i) dst.block(i * m2, j * n2, m2, n2) = m_A.coeff(i, j) * m_B;
  }

  /** \internal Computes \c dst += (*this), see evalTo(). */
  template <typename Dest>
  void addTo(Dest& dst) const {
    const Index m2 = m_B.rows(), n2 = m_B.cols();
    for (Index j = 0; j < m_A.cols(); ++j)
      for (Index i = 0; i < m_A.rows(); ++i) dst.block(i * m2, j * n2, m2, n2) += m_A.coeff(i, j) * m_B;
  }

  /** \internal Computes \c dst -= (*this), see evalTo(). */
  template <typename Dest>
  void subTo(Dest& dst) const {
    const Index m2 = m_B.rows(), n2 = m_B.cols();
    for (Index j = 0; j < m_A.cols(); ++j)
      for (Index i = 0; i < m_A.rows(); ++i) dst.block(i * m2, j * n2, m2, n2) -= m_A.coeff(i, j) * m_B;
  }

  /** \returns the product expression \c (*this) * \a x, evaluated through the vec
   * identity without materializing the Kronecker product. */
  template <typename Rhs>
  Product<KroneckerOperator, Rhs, AliasFreeProduct> operator*(const MatrixBase<Rhs>& x) const {
    return Product<KroneckerOperator, Rhs, AliasFreeProduct>(*this, x.derived());
  }

  /** \internal Computes \c dst += alpha * (*this) * rhs through the vec identity
   * \f$ (A \otimes B)\,\mathrm{vec}(X) = \mathrm{vec}(B X A^T) \f$ of [1], column
   * by column. \c ProductScalar is the promoted scalar of the product (complex
   * when a real operator is applied to a complex right-hand side); the
   * workspaces and the accumulation run in the promoted type. */
  template <typename Dest, typename Rhs, typename ProductScalar>
  void addProduct(Dest& dst, const Rhs& rhs, const ProductScalar& alpha) const {
    using ProductVector = Matrix<ProductScalar, Dynamic, 1>;
    using ProductMatrix = Matrix<ProductScalar, Dynamic, Dynamic, ColMajor>;  // ColMajor: see DenseMatrix
    const Index n1 = m_A.cols(), n2 = m_B.cols();
    eigen_assert(rhs.rows() == n1 * n2 && "invalid product: dimensions do not match");
    for (Index k = 0; k < rhs.cols(); ++k) {
      const ProductVector xc = rhs.col(k).template cast<ProductScalar>();
      const Map<const ProductMatrix> X(xc.data(), n2, n1);
      const ProductMatrix Y = m_B * X * m_A.transpose();
      dst.col(k) += alpha * Map<const ProductVector>(Y.data(), Y.size());
    }
  }

 private:
  /** \internal \returns the rank/pseudo-inversion threshold for the pairwise
   * singular-value products of the factor singular values \a sa and \a sb (each
   * sorted decreasingly, so the leading entries are the maxima), in the spirit
   * of the SVD-based pseudo-inverse: \c min(rows,cols) * epsilon * smax with
   * \c smax = sa[0]*sb[0] the largest singular value of the Kronecker product,
   * clamped from below at the smallest normal number so subnormal and zero
   * products -- whose reciprocals overflow -- are treated as exact zeros. */
  RealScalar rankThreshold(const RealVector& sa, const RealVector& sb) const {
    return numext::maxi(RealScalar(numext::mini(rows(), cols())) * NumTraits<RealScalar>::epsilon() * sa[0] * sb[0],
                        (std::numeric_limits<RealScalar>::min)());
  }

  template <typename T>
  static T ldexpImpl(const T& z, int e, std::false_type /*is_complex*/) {
    return std::ldexp(z, e);
  }
  template <typename T>
  static T ldexpImpl(const T& z, int e, std::true_type /*is_complex*/) {
    return T(std::ldexp(numext::real(z), e), std::ldexp(numext::imag(z), e));
  }
  /** \internal Scales \a z by \c 2^e exactly, componentwise for complex scalars. */
  static Scalar ldexpScalar(const Scalar& z, int e) {
    return ldexpImpl(z, e, std::integral_constant<bool, NumTraits<Scalar>::IsComplex != 0>());
  }

  /** \internal Rescales \a z by the power of two that brings \c max(|re|,|im|)
   * into [0.5, 1), accumulating the removed exponent into \a exponent. The
   * rescaling is exact, so no roundoff is introduced. Zeros and non-finite
   * values, which must propagate exactly through determinant(), are returned
   * untouched. */
  static Scalar balance(const Scalar& z, Index& exponent) {
    const RealScalar mag = numext::maxi(numext::abs(numext::real(z)), numext::abs(numext::imag(z)));
    if (!(mag > RealScalar(0)) || !(numext::isfinite)(mag)) return z;
    int e;
    std::frexp(mag, &e);
    exponent += e;
    return ldexpScalar(z, -e);
  }

  /** \internal \returns the mantissa of \c det(M)^power in the balanced form
   * \c m * 2^e, adding \c e into \a exponent. The determinant is accumulated
   * directly from the LU diagonal (times the permutation sign), each entry and
   * the running product renormalized by \ref balance, and the integer power is
   * applied by balanced repeated multiplication -- exact integer-power
   * semantics for negative real and complex determinants (no exp/log
   * branch-cut roundoff), with the bulk of the magnitude carried exactly on the
   * exponent side. */
  static Scalar balancedDetPow(const DenseMatrix& M, Index power, Index& exponent) {
    PartialPivLU<DenseMatrix> lu(M);
    Scalar m = Scalar(RealScalar(lu.permutationP().determinant()));  // +-1
    Index e = 0;
    for (Index i = 0; i < M.rows(); ++i) m = balance(m * balance(lu.matrixLU().coeff(i, i), e), e);
    Scalar r(1);
    Index er = 0;
    for (Index k = 0; k < power; ++k) r = balance(r * m, er);
    exponent += power * e + er;
    return r;
  }

  LhsMatrix m_A;
  RhsMatrix m_B;
};

/** \ingroup StructuredMatrices_Module
 * \returns a \ref KroneckerOperator \c a (x) \c b holding evaluated copies of the
 * factors. The operator type is deduced from the plain types of \a a and \a b. */
template <typename LhsDerived, typename RhsDerived>
KroneckerOperator<typename LhsDerived::PlainObject, typename RhsDerived::PlainObject> makeKroneckerOperator(
    const MatrixBase<LhsDerived>& a, const MatrixBase<RhsDerived>& b) {
  return {a.derived(), b.derived()};
}

namespace internal {

// Single product specialization covering every product tag; see the note in
// Circulant.h.
template <typename LhsMatrix, typename RhsMatrix, typename Rhs, int ProductTag>
struct generic_product_impl<KroneckerOperator<LhsMatrix, RhsMatrix>, Rhs, StructuredShape, DenseShape, ProductTag>
    : structured_product_impl<KroneckerOperator<LhsMatrix, RhsMatrix>, Rhs> {};

}  // namespace internal

}  // namespace Eigen

#endif  // EIGEN_STRUCTURED_KRONECKER_OPERATOR_H
