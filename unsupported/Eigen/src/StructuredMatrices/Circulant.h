// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_STRUCTURED_CIRCULANT_H
#define EIGEN_STRUCTURED_CIRCULANT_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

template <typename Scalar_, int Size_ = Dynamic>
class Circulant;

namespace internal {

template <typename Scalar_, int Size_>
struct traits<Circulant<Scalar_, Size_>> {
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

template <typename Scalar_, int Size_>
struct evaluator_traits<Circulant<Scalar_, Size_>> {
  using Kind = IndexBased;
  using Shape = StructuredShape;
};

}  // namespace internal

/** \ingroup StructuredMatrices_Module
 * \class Circulant
 * \brief An \c n x \c n circulant matrix represented by its first column.
 *
 * A circulant matrix is the square matrix whose entry \c (i,j) equals
 * \c c[(i-j) mod n], where \c c is its first column. It is diagonalized by the
 * discrete Fourier transform: its eigenvalues -- the operator's \em symbol,
 * computed once at construction and reused by every product -- are the DFT of
 * \c c. This yields an O(n log n) matrix-vector product (\c operator*), an
 * O(n log n) direct (pseudo-inverse) solve (\ref solve), and closed-form
 * factorizations: the eigendecomposition (\ref eigenvalues, \ref eigenvectors)
 * and the SVD (\ref singularValues, \ref matrixU, \ref matrixV) in the Fourier
 * basis, plus \ref rank, \ref inverse and \ref determinant. The class is closed
 * under \ref transpose, \ref conjugate and \ref adjoint, which reuse the cached
 * symbol instead of recomputing FFTs.
 *
 * The operator stores its own copy of the generating column and derives from
 * \c EigenBase. Because \c operator* returns an Eigen product expression, a
 * \c Circulant also drops into the matrix-free iterative solvers, and it can be
 * assigned to a dense matrix when an explicit representation is needed.
 *
 * \tparam Scalar_ the scalar type, real or complex.
 * \tparam Size_ the dimension at compile time, or \c Dynamic (the default).
 *
 * \sa class Toeplitz, makeCirculant()
 */
template <typename Scalar_, int Size_>
class Circulant : public EigenBase<Circulant<Scalar_, Size_>> {
 public:
  using Scalar = Scalar_;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using StorageIndex = int;
  using Complex = std::complex<RealScalar>;
  using GeneratorType = Matrix<Scalar, Size_, 1>;
  using ComplexVector = Matrix<Complex, Dynamic, 1>;
  using RealVector = Matrix<RealScalar, Size_, 1>;
  using ComplexMatrix = Matrix<Complex, Size_, Size_>;

  static constexpr int RowsAtCompileTime = Size_;
  static constexpr int ColsAtCompileTime = Size_;
  static constexpr int MaxRowsAtCompileTime = Size_;
  static constexpr int MaxColsAtCompileTime = Size_;
  static constexpr int SizeAtCompileTime = internal::size_at_compile_time(Size_, Size_);
  static constexpr int MaxSizeAtCompileTime = SizeAtCompileTime;
  static constexpr bool IsRowMajor = false;
  // Deliberately no IsVectorAtCompileTime: Ref<const Circulant>'s default StrideType
  // argument reads it, so its absence makes internal::is_ref_compatible SFINAE to
  // false and keeps the iterative solvers on their matrix-free path.

  /** Builds a circulant matrix from its first column \a col.
   *
   * When the matrix is large enough for products to take the FFT path, the DFT of
   * \a col -- the eigenvalues of the matrix -- is computed here, once, and reused
   * by every subsequent product and solve. */
  template <typename Derived>
  explicit Circulant(const MatrixBase<Derived>& col) : m_col(col) {
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
    eigen_assert(m_col.size() > 0 && "Circulant generator must be non-empty");
    if (m_col.size() > internal::structured_direct_threshold()) m_symbol = computeSymbol();
  }

  EIGEN_DEVICE_FUNC Index rows() const { return m_col.size(); }
  EIGEN_DEVICE_FUNC Index cols() const { return m_col.size(); }

  /** \returns the generating first column. */
  const GeneratorType& column() const { return m_col; }

  /** \returns the symbol of the operator, i.e. the DFT of the generating column.
   * Its entries are the eigenvalues of the matrix. Cached when the operator is
   * large enough for products to take the FFT path, computed on the fly for small
   * operators. */
  ComplexVector symbol() const { return m_symbol.size() > 0 ? m_symbol : computeSymbol(); }

  /** \returns the coefficient at row \a row and column \a col. */
  Scalar coeff(Index row, Index col) const {
    Index k = row - col;
    if (k < 0) k += rows();
    return m_col.coeff(k);
  }

  /** \returns the transpose of \c *this, itself a \c Circulant operator: the one
   * generated by the index-reversed column. The cached symbol, when present, is
   * reused -- the symbol of the transpose is the index reversal of the symbol --
   * so no FFT is recomputed. */
  Circulant transpose() const {
    const Index n = rows();
    GeneratorType col(n);
    col[0] = m_col[0];
    if (n > 1) col.tail(n - 1) = m_col.tail(n - 1).reverse();
    return Circulant(col, internal::structured_reverse_symbol(m_symbol));
  }

  /** \returns the complex conjugate of \c *this, itself a \c Circulant operator.
   * The cached symbol, when present, is reused: the symbol of the conjugate is the
   * conjugated index reversal of the symbol. */
  Circulant conjugate() const {
    return Circulant(m_col.conjugate(), internal::structured_reverse_symbol(m_symbol).conjugate());
  }

  /** \returns the adjoint of \c *this, itself a \c Circulant operator. The cached
   * symbol, when present, is reused: the symbol of the adjoint is the elementwise
   * conjugate of the symbol (the eigenvalues conjugate while the Fourier
   * eigenbasis stays fixed). */
  Circulant adjoint() const {
    const Index n = rows();
    GeneratorType col(n);
    col[0] = numext::conj(m_col[0]);
    if (n > 1) col.tail(n - 1) = m_col.tail(n - 1).reverse().conjugate();
    return Circulant(col, m_symbol.conjugate());
  }

  /** \internal Writes the dense representation into \a dst; column \c j is the
   * generator rotated downwards by \c j, so only contiguous segment copies are
   * involved. Invoked through \c dense = circulant; */
  template <typename Dest>
  void evalTo(Dest& dst) const {
    const Index n = rows();
    for (Index j = 0; j < n; ++j) {
      dst.col(j).head(j) = m_col.tail(j);
      dst.col(j).tail(n - j) = m_col.head(n - j);
    }
  }

  /** \internal Computes \c dst += (*this), see evalTo(). */
  template <typename Dest>
  void addTo(Dest& dst) const {
    const Index n = rows();
    for (Index j = 0; j < n; ++j) {
      dst.col(j).head(j) += m_col.tail(j);
      dst.col(j).tail(n - j) += m_col.head(n - j);
    }
  }

  /** \internal Computes \c dst -= (*this), see evalTo(). */
  template <typename Dest>
  void subTo(Dest& dst) const {
    const Index n = rows();
    for (Index j = 0; j < n; ++j) {
      dst.col(j).head(j) -= m_col.tail(j);
      dst.col(j).tail(n - j) -= m_col.head(n - j);
    }
  }

  /** \returns the product expression \c (*this) * \a x, evaluated through a fast
   * FFT-based matrix-vector product. */
  template <typename Rhs>
  Product<Circulant, Rhs, AliasFreeProduct> operator*(const MatrixBase<Rhs>& x) const {
    return Product<Circulant, Rhs, AliasFreeProduct>(*this, x.derived());
  }

  /** \returns the minimum-norm least-squares solution of \c (*this) * x = b,
   * computed directly in the Fourier domain. Symbol entries whose modulus exceeds
   * the rank threshold (see \ref rank) are inverted; the remaining ones are
   * treated as exact zeros, so the result is the pseudo-inverse applied to \a b.
   * For a non-singular operator this is the exact solution
   * \c x = ifft( fft(b) ./ fft(c) ). Supports multiple right-hand sides. */
  template <typename Rhs>
  Matrix<Scalar, Size_, Rhs::ColsAtCompileTime> solve(const MatrixBase<Rhs>& b) const {
    const Index n = rows();
    eigen_assert(b.rows() == n && "right-hand side has the wrong number of rows");
    const ComplexVector s = symbol();
    const RealScalar tol = rankThreshold(s);
    ComplexVector sinv(n);
    // The negated comparison keeps NaN symbol entries in the inverted set, so a
    // NaN input propagates to the output instead of being silently zeroed.
    for (Index k = 0; k < n; ++k) sinv[k] = !(numext::abs(s[k]) <= tol) ? Complex(1) / s[k] : Complex(0);
    Matrix<Scalar, Size_, Rhs::ColsAtCompileTime> x(n, b.cols());
    x.setZero();
    // Applying the circulant operator whose symbol is the (pseudo-)inverse of ours
    // is exactly the (pseudo-)inverse operator.
    internal::structured_fft_apply(x, sinv, n, b.derived(), Scalar(1));
    return x;
  }

  /** \returns the numerical rank: the number of symbol entries whose modulus
   * exceeds the threshold \c n * epsilon * max_k|symbol[k]|. This is the same
   * threshold \ref solve uses to decide which Fourier components to invert. */
  Index rank() const {
    const ComplexVector s = symbol();
    const RealScalar tol = rankThreshold(s);
    Index r = 0;
    for (Index k = 0; k < s.size(); ++k)
      if (!(numext::abs(s[k]) <= tol)) ++r;  // negated so NaN entries count as non-zero
    return r;
  }

  /** \returns the inverse of \c *this, itself a \c Circulant operator: the one
   * generated by \c ifft(1 ./ symbol), the first column of the inverse matrix.
   * \warning The operator must be non-singular; use \ref solve for a
   * pseudo-inverse solve of a rank-deficient operator. */
  Circulant inverse() const {
    const Index n = rows();
    const ComplexVector sinv = symbol().cwiseInverse();
    GeneratorType col(n);
    if (n == 1) {
      col = internal::structured_scalar_part_impl<Scalar>::run(sinv);
    } else {
      FFT<RealScalar> fft;
      ComplexVector ct(n);
      fft.inv(ct, sinv, n);
      col = internal::structured_scalar_part_impl<Scalar>::run(ct);
    }
    return Circulant(col, n > internal::structured_direct_threshold() ? sinv : ComplexVector());
  }

  /** \returns the determinant, i.e. the product of the eigenvalues (the symbol
   * entries). For a real operator the product is real up to roundoff, and its
   * real part is returned. */
  Scalar determinant() const {
    const ComplexVector s = symbol();
    Complex det(1);
    for (Index k = 0; k < s.size(); ++k) det *= s[k];
    return internal::structured_scalar_part_impl<Scalar>::run_scalar(det);
  }

  /** \returns the eigenvalues: eigenvalue \c k is \c symbol()[k], and its
   * (unit-norm) eigenvector is the Fourier vector \c f_k with
   * \f$ (f_k)_j = e^{2\pi i j k / n} / \sqrt{n} \f$, see \ref eigenvectors.
   * Every circulant matrix is diagonalized by this same Fourier basis. */
  ComplexVector eigenvalues() const { return symbol(); }

  /** \returns the unitary matrix of eigenvectors: column \c k is the Fourier
   * vector \c f_k matching \c eigenvalues()[k].
   * \note The eigenvector matrix is materialized as a dense \c n x \c n matrix;
   * unlike the other methods of this class this costs O(n^2) storage. */
  ComplexMatrix eigenvectors() const {
    const Index n = rows();
    ComplexMatrix F(n, n);
    for (Index k = 0; k < n; ++k) fourierColumn(F, k, k);
    return F;
  }

  /** \returns the singular values, sorted in decreasing order: the moduli of the
   * symbol entries. The ordering is shared with \ref matrixU and \ref matrixV, so
   * together they form the SVD \c *this = U * singularValues().asDiagonal() * V^H. */
  RealVector singularValues() const {
    const ComplexVector s = symbol();
    const std::vector<Index> perm = svdPermutation(s);
    RealVector sv(s.size());
    for (Index t = 0; t < s.size(); ++t) sv[t] = numext::abs(s[perm[t]]);
    return sv;
  }

  /** \returns the matrix of left singular vectors \c U: column \c t is the Fourier
   * vector of the t-th largest symbol entry, scaled by its phase (phase 1 for a
   * zero entry). Dense \c n x \c n, see the note in \ref eigenvectors. */
  ComplexMatrix matrixU() const {
    const Index n = rows();
    const ComplexVector s = symbol();
    const std::vector<Index> perm = svdPermutation(s);
    ComplexMatrix U(n, n);
    for (Index t = 0; t < n; ++t) {
      fourierColumn(U, perm[t], t);
      const RealScalar a = numext::abs(s[perm[t]]);
      if (a > RealScalar(0)) U.col(t) *= s[perm[t]] / a;
    }
    return U;
  }

  /** \returns the matrix of right singular vectors \c V: column \c t is the
   * Fourier vector of the t-th largest symbol entry. Dense \c n x \c n, see the
   * note in \ref eigenvectors. */
  ComplexMatrix matrixV() const {
    const Index n = rows();
    const ComplexVector s = symbol();
    const std::vector<Index> perm = svdPermutation(s);
    ComplexMatrix V(n, n);
    for (Index t = 0; t < n; ++t) fourierColumn(V, perm[t], t);
    return V;
  }

  /** \internal Computes \c dst += alpha * (*this) * rhs. */
  template <typename Dest, typename Rhs>
  void addProduct(Dest& dst, const Rhs& rhs, const Scalar& alpha) const {
    const Index n = rows();
    eigen_assert(rhs.rows() == n && "invalid product: dimensions do not match");

    if (n <= internal::structured_scalar_threshold()) {
      // Tiny sizes: a plain scalar loop beats the segment-based path below, whose
      // per-segment setup dominates when segments hold only a few entries.
      for (Index k = 0; k < rhs.cols(); ++k)
        for (Index i = 0; i < n; ++i) {
          Scalar acc(0);
          for (Index j = 0; j < n; ++j) acc += coeff(i, j) * rhs.coeff(j, k);
          dst.coeffRef(i, k) += alpha * acc;
        }
      return;
    }

    if (n <= internal::structured_direct_threshold()) {
      // Direct path: accumulate x_j times the j-th column of the operator, which
      // is the generator rotated downwards by j. Only contiguous, forward segment
      // operations are involved, so everything vectorizes.
      for (Index k = 0; k < rhs.cols(); ++k) {
        auto dstCol = dst.col(k);
        for (Index j = 0; j < n; ++j) {
          const Scalar xj = alpha * rhs.coeff(j, k);
          dstCol.head(j) += xj * m_col.tail(j);
          dstCol.tail(n - j) += xj * m_col.head(n - j);
        }
      }
      return;
    }

    internal::structured_fft_apply(dst, m_symbol, n, rhs, alpha);
  }

 private:
  /** \internal Builds an operator from a generator and an already-known symbol
   * (empty for small operators), skipping the FFT of the public constructor. Used
   * by transpose(), conjugate(), adjoint() and inverse(), whose symbols are cheap
   * transformations of the existing one. */
  Circulant(const GeneratorType& col, const ComplexVector& symbol) : m_col(col), m_symbol(symbol) {}

  /** \internal \returns the rank/pseudo-inversion threshold for \a s, in the
   * spirit of the SVD-based pseudo-inverse: n * epsilon * max_k|s[k]|, clamped
   * from below like SVDBase::rank() so an all-tiny (or all-zero) symbol yields
   * rank zero instead of overflowing 1/s in solve(). */
  static RealScalar rankThreshold(const ComplexVector& s) {
    return numext::maxi(RealScalar(s.size()) * NumTraits<RealScalar>::epsilon() * s.cwiseAbs().maxCoeff(),
                        (std::numeric_limits<RealScalar>::min)());
  }

  /** \internal Writes the unit-norm Fourier eigenvector \c f_k into column
   * \a dstCol of \a F: (f_k)_j = exp(2 pi i j k / n) / sqrt(n). The index product
   * j*k is accumulated incrementally modulo n, so the argument passed to polar()
   * stays O(2 pi) -- keeping full accuracy for any n -- and no Index overflow can
   * occur. */
  void fourierColumn(ComplexMatrix& F, Index k, Index dstCol) const {
    const Index n = rows();
    const RealScalar scale = RealScalar(1) / numext::sqrt(RealScalar(n));
    Index jk = 0;  // j * k mod n
    for (Index j = 0; j < n; ++j) {
      F(j, dstCol) = std::polar(scale, RealScalar(2 * EIGEN_PI) * RealScalar(jk) / RealScalar(n));
      jk += k;
      if (jk >= n) jk -= n;
    }
  }

  /** \internal \returns the indices of \a s sorted by decreasing modulus; the
   * shared ordering of singularValues(), matrixU() and matrixV(). The sort is
   * stable so repeated calls agree even in the presence of ties, and NaN moduli
   * order last (comparing through NaN directly would break the strict weak
   * ordering std::stable_sort requires). */
  static std::vector<Index> svdPermutation(const ComplexVector& s) {
    std::vector<Index> perm(static_cast<std::size_t>(s.size()));
    for (Index k = 0; k < s.size(); ++k) perm[static_cast<std::size_t>(k)] = k;
    std::stable_sort(perm.begin(), perm.end(), [&s](Index a, Index b) {
      const RealScalar ka = numext::abs(s[a]), kb = numext::abs(s[b]);
      const bool nanA = (ka != ka), nanB = (kb != kb);
      if (nanA || nanB) return nanB && !nanA;
      return ka > kb;
    });
    return perm;
  }

  /** \internal \returns the DFT of the generating column. */
  ComplexVector computeSymbol() const {
    const Index n = m_col.size();
    const ComplexVector cc = m_col.template cast<Complex>();
    if (n == 1) return cc;  // the DFT of a single sample is the identity
    ComplexVector symbol(n);
    FFT<RealScalar> fft;
    fft.fwd(symbol, cc, n);
    return symbol;
  }

  GeneratorType m_col;
  ComplexVector m_symbol;
};

/** \ingroup StructuredMatrices_Module
 * \returns a \ref Circulant operator with first column \a col. The compile-time
 * size of the operator is deduced from \a col. */
template <typename Derived>
Circulant<typename Derived::Scalar, Derived::SizeAtCompileTime> makeCirculant(const MatrixBase<Derived>& col) {
  return Circulant<typename Derived::Scalar, Derived::SizeAtCompileTime>(col);
}

namespace internal {

// The StructuredShape key makes this single partial specialization cover every
// product tag (GEMV, GEMM, coeff-based, ...) without ambiguity against the stock
// dense specializations, so products of any size and fixedness reach the fast path.
template <typename Scalar_, int Size_, typename Rhs, int ProductTag>
struct generic_product_impl<Circulant<Scalar_, Size_>, Rhs, StructuredShape, DenseShape, ProductTag>
    : structured_product_impl<Circulant<Scalar_, Size_>, Rhs> {};

}  // namespace internal

}  // namespace Eigen

#endif  // EIGEN_STRUCTURED_CIRCULANT_H
