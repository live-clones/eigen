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
//  [2] N. J. Higham, "Accuracy and Stability of Numerical Algorithms", 2nd ed.,
//      SIAM, 2002, chapter 27. Avoiding spurious overflow by rescaling with
//      powers of two, the technique behind the scaled vec-trick products, the
//      normalized-frame solves, the exponent-split pseudo-inversion and the
//      exponent-balanced determinant.
//  [3] P. H. Sterbenz, "Floating-Point Computation", Prentice-Hall, 1974.
//      Scaling by a power of two is exact, the property every rescaling in this
//      file relies on.

#ifndef EIGEN_STRUCTURED_KRONECKER_OPERATOR_H
#define EIGEN_STRUCTURED_KRONECKER_OPERATOR_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

template <typename LhsMatrix, typename RhsMatrix>
class KroneckerOperator;

namespace internal {

/** \internal Compile-time product of two dimensions: Dynamic when either factor
 * is unknown, and also when the product would not fit in \c int -- no fixed-size
 * dimension that large is usable anyway, and the overflowing multiplication
 * would be ill-formed in a constant expression. */
constexpr int kron_dim(int a, int b) {
  return (a == 0 || b == 0)                            ? 0
         : (a == Dynamic || b == Dynamic)              ? Dynamic
         : (a > (std::numeric_limits<int>::max)() / b) ? Dynamic
                                                       : a * b;
}

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
   *
   * The solves run in a normalized frame [2][3]: the factors and each right-hand
   * side are rescaled to unit magnitude by exact powers of two, and the combined
   * exponent is folded back into the solution entrywise afterwards. In the raw
   * frame the intermediate \f$ B^{-1} \mathrm{mat}(b) \f$ can overflow, or
   * silently underflow to zero, on factor magnitudes alone even when the
   * solution is representable (the subsequent \f$ A^{-T} \f$ rescales it: think
   * \c B huge and \c A tiny); in the normalized frame the intermediates are
   * bounded by the conditioning of the factors, and the final fold saturates to
   * 0/Inf exactly when the true solution leaves the representable range.
   * Partial pivoting is invariant under the uniform power-of-two normalization
   * and every substitution step scales exactly with it, so the result is
   * bit-identical to the unnormalized evaluation whenever that evaluation
   * encounters no intermediate over- or underflow.
   * \warning Both factors must be invertible, like in \c PartialPivLU; use
   * \ref leastSquaresSolve for rank-deficient or rectangular factors. */
  template <typename Rhs>
  Matrix<Scalar, ColsAtCompileTime, Rhs::ColsAtCompileTime> solve(const MatrixBase<Rhs>& b) const {
    EIGEN_STATIC_ASSERT(RowsAtCompileTime == Dynamic || Rhs::RowsAtCompileTime == Dynamic ||
                            int(RowsAtCompileTime) == int(Rhs::RowsAtCompileTime),
                        YOU_MIXED_MATRICES_OF_DIFFERENT_SIZES)
    const Index n1 = m_A.cols(), n2 = m_B.cols();
    eigen_assert(m_A.rows() == n1 && m_B.rows() == n2 && "KroneckerOperator::solve requires square factors");
    eigen_assert(b.rows() == n1 * n2 && "right-hand side has the wrong number of rows");
    // The exponent bounds are 0 for zero or non-finite data, which is then left
    // unnormalized (Inf/NaN propagate through the substitutions as usual).
    const int ea = internal::structured_exponent_bound(m_A);
    const int eb = internal::structured_exponent_bound(m_B);
    PartialPivLU<DenseMatrix> luA(ldexpEntries(m_A, -ea)), luB(ldexpEntries(m_B, -eb));
    Matrix<Scalar, ColsAtCompileTime, Rhs::ColsAtCompileTime> x(n1 * n2, b.cols());
    for (Index k = 0; k < b.cols(); ++k) {
      DenseVector bc = b.col(k);
      const int ec = internal::structured_exponent_bound(bc);
      if (ec != 0) bc = ldexpEntries(bc, -ec);
      const Map<const DenseMatrix> Bmat(bc.data(), n2, n1);
      DenseMatrix X = luA.solve(luB.solve(Bmat).transpose()).transpose();
      // Fold the combined exponent back. The entrywise ldexp saturates exactly
      // where the true solution over- or underflows; a multiplicative fold could
      // not (the combined exponent can exceed the representable range of any
      // fixed number of power-of-two factors).
      const int e = ec - ea - eb;
      if (e != 0) X = ldexpEntries(X, e);
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
   * threshold of \ref rank, in the same overflow-safe ratio form -- factor-wise
   * truncation would invert modes that are negligible at the product level.
   * Handles rectangular and rank-deficient factors. Supports multiple
   * right-hand sides.
   *
   * Unlike \ref solve, no rescaling of the intermediates is needed here: the
   * projection and back-transformation matrices have orthonormal columns, so no
   * intermediate can outgrow the data by more than a dimension factor, exactly
   * as in a dense product. */
  template <typename Rhs>
  Matrix<Scalar, ColsAtCompileTime, Rhs::ColsAtCompileTime> leastSquaresSolve(const MatrixBase<Rhs>& b) const {
    EIGEN_STATIC_ASSERT(RowsAtCompileTime == Dynamic || Rhs::RowsAtCompileTime == Dynamic ||
                            int(RowsAtCompileTime) == int(Rhs::RowsAtCompileTime),
                        YOU_MIXED_MATRICES_OF_DIFFERENT_SIZES)
    const Index m1 = m_A.rows(), m2 = m_B.rows();
    eigen_assert(b.rows() == m1 * m2 && "right-hand side has the wrong number of rows");
    JacobiSVD<DenseMatrix> svdA(m_A, ComputeThinU | ComputeThinV), svdB(m_B, ComputeThinU | ComputeThinV);
    const RealVector sa = svdA.singularValues(), sb = svdB.singularValues();
    const RealScalar tol = relativeRankThreshold();
    const Index kA = sa.size(), kB = sb.size();
    Matrix<Scalar, ColsAtCompileTime, Rhs::ColsAtCompileTime> x(cols(), b.cols());
    if (sa[0] == RealScalar(0) || sb[0] == RealScalar(0)) {
      // An exactly zero factor zeroes the whole operator, whose pseudo-inverse is
      // zero (and the singular-value ratios below would be 0/0).
      x.setZero();
      return x;
    }
    for (Index k = 0; k < b.cols(); ++k) {
      const DenseVector bc = b.col(k);
      const Map<const DenseMatrix> Bmat(bc.data(), m2, m1);
      // Project onto the factor singular bases: M = U_B^H mat(b) conj(U_A) is the
      // k_B x k_A matricization of (U_A (x) U_B)^H b, by the vec identity [1].
      DenseMatrix M = svdB.matrixU().adjoint() * Bmat * svdA.matrixU().conjugate();
      // Invert only the pairwise products at/above the product-level threshold
      // and the smallest normal number, both decided like in rank(). The negated
      // ratio comparison keeps NaN ratios in the inverted set, so a NaN input
      // propagates to the output instead of being silently zeroed. The division
      // splits each singular value into mantissa and exponent [2]: the product
      // sigma_i(B) * sigma_j(A) itself can overflow, or underflow to zero, even
      // when the quotient is representable. Applying the exact power of two
      // first keeps every intermediate within a factor of four of the true
      // quotient.
      for (Index j = 0; j < kA; ++j)
        for (Index i = 0; i < kB; ++i) {
          if (!((sa[j] / sa[0]) * (sb[i] / sb[0]) < tol) && reachesMinNormal(sa[j], sb[i])) {
            int ea, eb;
            const RealScalar ma = std::frexp(sa[j], &ea), mb = std::frexp(sb[i], &eb);
            M(i, j) = ldexpScalar(M(i, j), -(ea + eb)) / (ma * mb);
          } else {
            M(i, j) = Scalar(0);
          }
        }
      // Back to the original bases: mat(x) = V_B M V_A^T, again the vec identity.
      const DenseMatrix X = svdB.matrixV() * M * svdA.matrixV().transpose();
      x.col(k) = Map<const DenseVector>(X.data(), X.size());
    }
    return x;
  }

  /** \returns the numerical rank: the number of pairwise singular-value products
   * \f$ \sigma_i(A)\,\sigma_j(B) \f$ -- the singular values of the Kronecker
   * product -- that reach the threshold
   * \c min(rows(),cols()) * epsilon * sigma_max(A) * sigma_max(B) (the
   * \c SVDBase convention), and that reach the smallest normal number (the
   * \c SVDBase threshold clamp, so subnormal products count as exact zeros).
   * The relative comparison is made in ratio space,
   * \f$ (\sigma_i(A)/\sigma_{max}(A))(\sigma_j(B)/\sigma_{max}(B)) \f$ against
   * \c min(rows(),cols()) * epsilon, and the clamp in exponent space, so that
   * neither the thresholds nor the products can spuriously under- or overflow.
   * This is the same threshold \ref leastSquaresSolve uses to decide which
   * modes to invert. Thresholding the products matters: factors that are each
   * full rank against their own threshold can still form pairwise products
   * that are negligible at the product level, so the rank can be smaller than
   * the product of the factor ranks. */
  Index rank() const {
    JacobiSVD<DenseMatrix> svdA(m_A), svdB(m_B);
    const RealVector sa = svdA.singularValues(), sb = svdB.singularValues();
    // An exactly zero factor zeroes the whole operator (and would make the
    // ratios below 0/0).
    if (sa[0] == RealScalar(0) || sb[0] == RealScalar(0)) return 0;
    const RealScalar tol = relativeRankThreshold();
    Index r = 0;
    for (Index i = 0; i < sa.size(); ++i)
      for (Index j = 0; j < sb.size(); ++j)
        // Negated ratio comparison so NaN ratios count as non-zero.
        if (!((sa[i] / sa[0]) * (sb[j] / sb[0]) < tol) && reachesMinNormal(sa[i], sb[j])) ++r;
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
   * identity without materializing the Kronecker product. The expression carries
   * the default product tag, so assigning it behaves like any dense product: a
   * temporary resolves aliasing between the destination and \a x, and
   * \c .noalias() skips it. */
  template <typename Rhs>
  Product<KroneckerOperator, Rhs> operator*(const MatrixBase<Rhs>& x) const {
    EIGEN_STATIC_ASSERT(ColsAtCompileTime == Dynamic || Rhs::RowsAtCompileTime == Dynamic ||
                            int(ColsAtCompileTime) == int(Rhs::RowsAtCompileTime),
                        INVALID_MATRIX_PRODUCT)
    eigen_assert(x.rows() == cols() && "invalid product: dimensions do not match");
    return Product<KroneckerOperator, Rhs>(*this, x.derived());
  }

  /** \internal Computes \c dst += alpha * (*this) * rhs through the vec identity
   * \f$ (A \otimes B)\,\mathrm{vec}(X) = \mathrm{vec}(B X A^T) \f$ of [1], column
   * by column. \c ProductScalar is the promoted scalar of the product (complex
   * when a real operator is applied to a complex right-hand side); the
   * workspaces and the accumulation run in the promoted type.
   *
   * Evaluated as \c (B X) A^T, the first factor \c B X can overflow even when
   * the product itself is representable (the multiplication by \c A^T rescales
   * it: think \c B huge and \c A tiny). Each column is therefore pre-scaled by
   * an exact power of two [2][3] derived from the component-wise exponent
   * bounds of \c A, \c B and the column -- never from complex moduli, which
   * overflow near the threshold -- so that no intermediate can spuriously
   * overflow, and the
   * exponent is folded back into the result through two representable
   * half-factors, saturating to 0/Inf exactly when the true result leaves the
   * representable range. The scale is one whenever the conservative bound shows
   * no overflow risk, so results of moderate magnitude are bit-identical to the
   * unscaled evaluation. Non-finite data is never scaled (the bounds cannot see
   * past an Inf/NaN, and 0 * Inf would manufacture NaNs); the unscaled GEMMs
   * propagate it entrywise exactly like a dense product. */
  template <typename Dest, typename Rhs, typename ProductScalar>
  void addProduct(Dest& dst, const Rhs& rhs, const ProductScalar& alpha) const {
    using ProductVector = Matrix<ProductScalar, Dynamic, 1>;
    using ProductMatrix = Matrix<ProductScalar, Dynamic, Dynamic, ColMajor>;  // ColMajor: see DenseMatrix
    using ProductReal = typename NumTraits<ProductScalar>::Real;
    const Index n1 = m_A.cols(), n2 = m_B.cols();
    eigen_assert(rhs.rows() == n1 * n2 && "invalid product: dimensions do not match");
    // Exponent budget of the pre-scaling: with max|X| < 2^eX after scaling, the
    // first GEMM is bounded by 2^(eB + eX + bits2) and the full product by
    // 2^(eA + eB + eX + bits1 + bits2), where 2^bits > dimension accounts for
    // the length of the accumulated dot products; both bounds must stay below
    // 2^max_exponent (with two bits of slack).
    const bool factorsFinite = m_A.allFinite() && m_B.allFinite();
    const int expA = internal::structured_exponent_bound(m_A);
    const int expB = internal::structured_exponent_bound(m_B);
    int bits1 = 0, bits2 = 0;
    for (Index t = n1; t > 0; t /= 2) ++bits1;
    for (Index t = n2; t > 0; t /= 2) ++bits2;
    const int budget = std::numeric_limits<ProductReal>::max_exponent - 2;
    for (Index k = 0; k < rhs.cols(); ++k) {
      ProductVector xc = rhs.col(k).template cast<ProductScalar>();
      int e = 0;
      if (factorsFinite && xc.allFinite()) {
        const int expX = internal::structured_exponent_bound(xc);  // 0 for an all-zero column: no scaling
        e = numext::maxi(0, numext::maxi(expB + expX + bits2 - budget, expA + expB + expX + bits1 + bits2 - budget));
        // The cap keeps the half-factors below overflow; it only binds when the
        // true result overflows anyway, so the saturation to Inf is genuine.
        e = numext::mini(e, 2 * budget);
      }
      if (e > 0) {
        // Each power of two is split in halves so that the factors themselves
        // stay representable; scaling by the two exact factors in sequence is
        // still an exact shift wherever the result is representable.
        const ProductReal down1 = ProductReal(std::ldexp(ProductReal(1), -(e / 2)));
        const ProductReal down2 = ProductReal(std::ldexp(ProductReal(1), -(e - e / 2)));
        xc = (xc * down1) * down2;
      }
      const Map<const ProductMatrix> X(xc.data(), n2, n1);
      const ProductMatrix Y = m_B * X * m_A.transpose();
      if (e > 0) {
        const ProductReal up1 = ProductReal(std::ldexp(ProductReal(1), e / 2));
        const ProductReal up2 = ProductReal(std::ldexp(ProductReal(1), e - e / 2));
        dst.col(k) += alpha * ((Map<const ProductVector>(Y.data(), Y.size()) * up1) * up2);
      } else {
        dst.col(k) += alpha * Map<const ProductVector>(Y.data(), Y.size());
      }
    }
  }

 private:
  /** \internal \returns the relative rank/pseudo-inversion threshold for the
   * pairwise singular-value products, in the spirit of the SVD-based
   * pseudo-inverse: a mode \c (i,j) is kept when
   * \c (sa[i]/sa[0]) * (sb[j]/sb[0]) >= min(rows,cols) * epsilon, the \c SVDBase
   * convention for \c sa[i]*sb[j] measured against \c smax = sa[0]*sb[0]. The
   * comparison must happen in ratio space: the absolute form both underflows
   * (\c min(rows,cols) * epsilon * sa[0] can flush to zero before the \c sb[0]
   * multiplication, silently accepting every mode) and overflows
   * (\c sa[0]*sb[0] can exceed the representable range). The ratios never
   * exceed one, and a ratio product that underflows is genuinely below any
   * epsilon-sized threshold. */
  RealScalar relativeRankThreshold() const {
    return RealScalar(numext::mini(rows(), cols())) * NumTraits<RealScalar>::epsilon();
  }

  /** \internal \returns \a M rescaled by \c 2^e entrywise (componentwise for
   * complex scalars): exact where the result is representable [3], with correct
   * saturation to 0/Inf and correctly rounded subnormals beyond -- unlike a
   * multiplication by \c 2^e, whose factor may itself be unrepresentable. */
  template <typename Xpr>
  static typename Xpr::PlainObject ldexpEntries(const Xpr& M, int e) {
    return M.unaryExpr([e](const Scalar& z) { return ldexpScalar(z, e); });
  }

  /** \internal Whether the pairwise singular-value product \c s1*s2 reaches the
   * smallest normal number -- the clamp \c SVDBase places under its
   * pseudo-inversion threshold, so that subnormal singular values (whose
   * reciprocals overflow) are treated as exact zeros. The boundary itself is
   * kept, matching the \c >= convention of \c SVDBase::rank(). The product is
   * compared exponent-safely: the frexp mantissas lie in [0.5, 1), so their
   * product in [0.25, 1) can neither over- nor underflow, and the exponents add
   * as integers. Non-finite singular values return true so they stay in the
   * inverted set and propagate. */
  static bool reachesMinNormal(const RealScalar& s1, const RealScalar& s2) {
    int e1, e2;
    const RealScalar m = std::frexp(s1, &e1) * std::frexp(s2, &e2);
    if (!(numext::isfinite)(m)) return true;  // NaN or Inf singular values must propagate
    if (m == RealScalar(0)) return false;     // an exactly zero product
    // Renormalize the mantissa product into [0.5, 1), so that s1*s2 = m * 2^e
    // reaches the smallest normal number 2^(min_exponent - 1) iff e is at least
    // min_exponent.
    const int e = m < RealScalar(0.5) ? e1 + e2 - 1 : e1 + e2;
    return e >= std::numeric_limits<RealScalar>::min_exponent;
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
