// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// References:
//  [1] P. J. Davis, "Circulant Matrices", Wiley, 1979. Diagonalization of
//      circulant and block circulant matrices by the DFT; the closed-form
//      eigenstructure used by eigenvalues()/eigenvectors(), and the SVD and
//      pseudo-inverse below, follow from it by taking moduli/phases of the
//      eigenvalues.
//  [2] R. H. Chan and X.-Q. Jin, "An Introduction to Iterative Toeplitz
//      Solvers", SIAM, 2007. BCCB matrices are diagonalized by the 2-D DFT
//      F_{n1} (x) F_{n2}; the FFT-based products and solves below, and the use
//      of BCCB operators as preconditioners for two-level Toeplitz (BTTB)
//      systems, follow this reference.
//  [3] G. H. Golub and C. F. Van Loan, "Matrix Computations", 4th ed., Johns
//      Hopkins University Press, 2013, chapter 5.4 (numerical rank conventions).

#ifndef EIGEN_STRUCTURED_BCCB_H
#define EIGEN_STRUCTURED_BCCB_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

template <typename Scalar_, int BlockSize_ = Dynamic, int NumBlocks_ = Dynamic>
class Bccb;

namespace internal {

/** \internal Compile-time product of the two circulant levels, Dynamic-aware. */
constexpr int bccb_dim(int a, int b) { return (a == Dynamic || b == Dynamic) ? Dynamic : a * b; }

template <typename Scalar_, int BlockSize_, int NumBlocks_>
struct traits<Bccb<Scalar_, BlockSize_, NumBlocks_>> {
  using Scalar = Scalar_;
  using StorageKind = Dense;
  using XprKind = MatrixXpr;
  using StorageIndex = int;
  static constexpr int RowsAtCompileTime = bccb_dim(BlockSize_, NumBlocks_);
  static constexpr int ColsAtCompileTime = RowsAtCompileTime;
  static constexpr int MaxRowsAtCompileTime = RowsAtCompileTime;
  static constexpr int MaxColsAtCompileTime = RowsAtCompileTime;
  // Deliberately no NestByRefBit: transpose(), conjugate() and adjoint() return
  // owning temporaries, so Product must nest the operator by value for a
  // delayed-evaluated product expression to keep its left factor alive. The copy
  // is O(N), negligible against the O(N log N) product evaluation.
  static constexpr int Flags = 0;
};

template <typename Scalar_, int BlockSize_, int NumBlocks_>
struct evaluator_traits<Bccb<Scalar_, BlockSize_, NumBlocks_>> {
  using Kind = IndexBased;
  using Shape = StructuredShape;
};

}  // namespace internal

/** \ingroup StructuredMatrices_Module
 * \class Bccb
 * \brief A block circulant matrix with circulant blocks (BCCB), the matrix of a
 * two-dimensional circular convolution, represented by its n2 x n1 generating
 * array.
 *
 * A BCCB matrix is the \c N x \c N matrix, \c N = n1*n2, that is circulant at two
 * levels: it is an n1 x n1 block circulant whose n2 x n2 blocks are themselves
 * circulant. With the generating array \c G (column \c k holds the first column
 * of the k-th block), entry \c (i,j) with \c i = b1*n2 + i2, \c j = c1*n2 + j2
 * equals \c G((i2-j2) mod n2, (b1-c1) mod n1), and the operator acts on a
 * column-major reshaped vector as the 2-D circular convolution
 * \c C*vec(X) = vec(G (*) X).
 *
 * BCCB matrices are diagonalized by the 2-D discrete Fourier transform
 * \f$ F_{n_1} \otimes F_{n_2} \f$ ([1], [2]): the operator's \em symbol -- the 2-D DFT of
 * \c G, computed once at construction and reused by every product -- holds the
 * eigenvalues. This yields O(N log N) products (\c operator*), an O(N log N)
 * direct (pseudo-inverse) solve (\ref solve), and closed-form factorizations:
 * the eigendecomposition (\ref eigenvalues, \ref eigenvectors) and the SVD
 * (\ref singularValues, \ref matrixU, \ref matrixV) in the 2-D Fourier basis,
 * plus \ref rank, \ref inverse and \ref determinant. The class is closed under
 * \ref transpose, \ref conjugate and \ref adjoint, which reuse the cached symbol.
 * BCCB operators are the workhorse of image deblurring with periodic boundary
 * conditions, and the natural preconditioners for two-level Toeplitz (BTTB)
 * systems [2].
 *
 * The operator stores its own copy of the generating array and derives from
 * \c EigenBase. Because \c operator* returns an Eigen product expression, a
 * \c Bccb also drops into the matrix-free iterative solvers, and it can be
 * assigned to a dense matrix when an explicit representation is needed. As with
 * any matrix-free operator, the iterative solvers must be instantiated with
 * \c IdentityPreconditioner (e.g.
 * \c ConjugateGradient<Bccb<double>,Lower|Upper,IdentityPreconditioner>):
 * the default preconditioners read individual coefficients through \c col() or
 * \c InnerIterator, which the structured operators do not expose.
 *
 * The FFTs run at the exact sizes \c n1 and \c n2 (circular convolution allows no
 * padding), so with the default kissfft backend prefer dimensions without large
 * prime factors.
 *
 * \tparam Scalar_ the scalar type, real or complex.
 * \tparam BlockSize_ the circulant block dimension \c n2 at compile time, or
 *         \c Dynamic (the default).
 * \tparam NumBlocks_ the number of blocks \c n1 at compile time, or \c Dynamic
 *         (the default).
 *
 * \sa class Circulant, makeBccb()
 */
template <typename Scalar_, int BlockSize_, int NumBlocks_>
class Bccb : public EigenBase<Bccb<Scalar_, BlockSize_, NumBlocks_>> {
 public:
  using Scalar = Scalar_;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using StorageIndex = int;
  using Complex = std::complex<RealScalar>;
  // The two-level structure keys on the *column-major* flattening of the n2 x n1
  // generating array and symbol (entry (k2, k1) is flat index k1*n2 + k2), so
  // every internal 2-D type is pinned to ColMajor explicitly: the semantics must
  // not change under EIGEN_DEFAULT_TO_ROW_MAJOR. The single-row special case
  // only satisfies Eigen's rule that 1 x n matrices be row-major; with one row
  // the two orders coincide.
  using GeneratorType =
      Matrix<Scalar, BlockSize_, NumBlocks_, (BlockSize_ == 1 && NumBlocks_ != 1) ? int(RowMajor) : int(ColMajor)>;
  using ComplexArray = Matrix<Complex, Dynamic, Dynamic, ColMajor>;
  using ComplexVector = Matrix<Complex, Dynamic, 1>;
  using RealVector = Matrix<RealScalar, Dynamic, 1>;
  using ComplexMatrix = Matrix<Complex, Dynamic, Dynamic, ColMajor>;

  static constexpr int RowsAtCompileTime = internal::bccb_dim(BlockSize_, NumBlocks_);
  static constexpr int ColsAtCompileTime = RowsAtCompileTime;
  static constexpr int MaxRowsAtCompileTime = RowsAtCompileTime;
  static constexpr int MaxColsAtCompileTime = RowsAtCompileTime;
  static constexpr int SizeAtCompileTime = internal::size_at_compile_time(RowsAtCompileTime, ColsAtCompileTime);
  static constexpr int MaxSizeAtCompileTime = SizeAtCompileTime;
  static constexpr bool IsRowMajor = false;
  // Deliberately no IsVectorAtCompileTime: Ref<const Bccb>'s default StrideType
  // argument reads it, so its absence makes internal::is_ref_compatible SFINAE to
  // false and keeps the iterative solvers on their matrix-free path.

  /** Builds a BCCB matrix from its generating array \a generator: column \c k is
   * the first column of the k-th circulant block.
   *
   * When the matrix is large enough for products to take the FFT path, the 2-D
   * DFT of the array -- the eigenvalues of the matrix -- is computed here, once,
   * and reused by every subsequent product and solve. */
  template <typename Derived>
  explicit Bccb(const MatrixBase<Derived>& generator) : m_g(generator) {
    eigen_assert(m_g.rows() > 0 && m_g.cols() > 0 && "Bccb generator must be non-empty");
    if (m_g.size() > internal::structured_direct_threshold()) m_symbol = computeSymbol();
    m_fftUsable = computeFftUsable();
  }

  EIGEN_DEVICE_FUNC Index rows() const { return m_g.size(); }
  EIGEN_DEVICE_FUNC Index cols() const { return m_g.size(); }

  /** \returns the circulant block dimension \c n2. */
  Index blockSize() const { return m_g.rows(); }
  /** \returns the number of blocks \c n1 in each block row. */
  Index numBlocks() const { return m_g.cols(); }
  /** \returns the generating array. */
  const GeneratorType& generator() const { return m_g; }

  /** \returns the symbol of the operator: the 2-D DFT of the generating array,
   * an n2 x n1 complex array whose entries are the eigenvalues of the matrix
   * (see \ref eigenvalues for the ordering). Cached when the operator is large
   * enough for products to take the FFT path, computed on the fly for small
   * operators. */
  ComplexArray symbol() const { return m_symbol.size() > 0 ? m_symbol : computeSymbol(); }

  /** \returns the coefficient at row \a row and column \a col. */
  Scalar coeff(Index row, Index col) const {
    const Index n2 = blockSize();
    Index k2 = row % n2 - col % n2;
    if (k2 < 0) k2 += n2;
    Index k1 = row / n2 - col / n2;
    if (k1 < 0) k1 += numBlocks();
    return m_g.coeff(k2, k1);
  }

  /** \returns the transpose of \c *this, itself a \c Bccb operator: the one
   * generated by the array index-reversed in both dimensions. The cached symbol,
   * when present, is reused -- the symbol of the transpose is its two-dimensional
   * index reversal -- so no FFT is recomputed. */
  Bccb transpose() const { return Bccb(reverse2(m_g), reverse2(m_symbol)); }

  /** \returns the complex conjugate of \c *this, itself a \c Bccb operator. The
   * cached symbol, when present, is reused: the symbol of the conjugate is the
   * conjugated two-dimensional index reversal of the symbol. */
  Bccb conjugate() const { return Bccb(m_g.conjugate(), reverse2(m_symbol).conjugate()); }

  /** \returns the adjoint of \c *this, itself a \c Bccb operator. The cached
   * symbol, when present, is reused: the symbol of the adjoint is the elementwise
   * conjugate of the symbol (the eigenvalues conjugate while the 2-D Fourier
   * eigenbasis stays fixed). */
  Bccb adjoint() const { return Bccb(reverse2(m_g).conjugate(), m_symbol.conjugate()); }

  /** \returns the minimum-norm least-squares solution of \c (*this) * x = b,
   * computed directly in the 2-D Fourier domain. Symbol entries whose modulus
   * reaches the rank threshold (see \ref rank) are inverted; the remaining ones
   * are treated as exact zeros, so the result is the pseudo-inverse applied to
   * \a b. For a non-singular operator this is the exact solution. Supports
   * multiple right-hand sides. */
  template <typename Rhs>
  Matrix<Scalar, RowsAtCompileTime, Rhs::ColsAtCompileTime> solve(const MatrixBase<Rhs>& b) const {
    EIGEN_STATIC_ASSERT(RowsAtCompileTime == Dynamic || Rhs::RowsAtCompileTime == Dynamic ||
                            int(RowsAtCompileTime) == int(Rhs::RowsAtCompileTime),
                        YOU_MIXED_MATRICES_OF_DIFFERENT_SIZES)
    const Index N = rows();
    eigen_assert(b.rows() == N && "right-hand side has the wrong number of rows");
    const ComplexArray s = symbol();
    const RealScalar tol = rankThreshold(s);
    ComplexArray sinv(s.rows(), s.cols());
    // Strictly-below-threshold entries are zeroed, matching SVDBase::rank(), so a
    // smallest-normal 1x1 operator stays invertible. The negated comparison keeps
    // NaN symbol entries in the inverted set, so a NaN input propagates to the
    // output instead of being silently zeroed.
    for (Index k1 = 0; k1 < s.cols(); ++k1)
      for (Index k2 = 0; k2 < s.rows(); ++k2)
        sinv(k2, k1) = !(numext::abs(s(k2, k1)) < tol) ? Complex(1) / s(k2, k1) : Complex(0);
    Matrix<Scalar, RowsAtCompileTime, Rhs::ColsAtCompileTime> x(N, b.cols());
    if (!b.allFinite()) {
      // A non-finite right-hand side cannot go through the transforms (see
      // addProduct): apply the pseudo-inverse -- itself a BCCB operator,
      // generated by the 2-D inverse DFT of the thresholded reciprocal symbol --
      // through the direct kernel so Inf/NaN propagate entrywise.
      ComplexArray g = sinv;
      ifft2(g);
      const GeneratorType pg = internal::structured_scalar_part_impl<Scalar>::run(g);
      x.setZero();
      Bccb(pg, ComplexArray()).directProduct(x, b.derived(), Scalar(1));
      return x;
    }
    applySymbol(x, sinv, b.derived(), Scalar(1), /*accumulate=*/false);
    return x;
  }

  /** \returns the numerical rank: the number of symbol entries whose modulus is
   * no smaller than the threshold \c N * epsilon * max|symbol|, clamped from
   * below like \c SVDBase::rank(). This is the same threshold \ref solve uses to
   * decide which Fourier components to invert, and the comparison is strict like
   * SVDBase's, so an entry sitting exactly on the threshold still counts as
   * non-zero. */
  Index rank() const {
    const ComplexArray s = symbol();
    const RealScalar tol = rankThreshold(s);
    Index r = 0;
    for (Index k1 = 0; k1 < s.cols(); ++k1)
      for (Index k2 = 0; k2 < s.rows(); ++k2)
        if (!(numext::abs(s(k2, k1)) < tol)) ++r;  // negated so NaN entries count as non-zero
    return r;
  }

  /** \returns the inverse of \c *this, itself a \c Bccb operator: the one
   * generated by the 2-D inverse DFT of the entrywise-inverted symbol.
   * \warning The operator must be non-singular; use \ref solve for a
   * pseudo-inverse solve of a rank-deficient operator. */
  Bccb inverse() const {
    ComplexArray sinv = symbol().cwiseInverse();
    ComplexArray g = sinv;
    ifft2(g);
    GeneratorType ginv = internal::structured_scalar_part_impl<Scalar>::run(g);
    return Bccb(ginv, rows() > internal::structured_direct_threshold() ? sinv : ComplexArray());
  }

  /** \returns the determinant, i.e. the product of the eigenvalues (the symbol
   * entries). The product is accumulated in the balanced form \c m * 2^e -- every
   * factor and the running product are renormalized to unit magnitude with the
   * power of two tracked separately -- so the partial products can neither
   * overflow nor underflow when the determinant itself is representable, whatever
   * the ordering of large and small eigenvalues. For a real operator the product
   * is real up to roundoff, and its real part is returned. */
  Scalar determinant() const {
    const ComplexArray s = symbol();
    Complex det(1);
    Index exponent = 0;
    for (Index k1 = 0; k1 < s.cols(); ++k1)
      for (Index k2 = 0; k2 < s.rows(); ++k2) det = balance(det * balance(s(k2, k1), exponent), exponent);
    // ldexp saturates cleanly to zero / infinity once the accumulated exponent
    // leaves the representable range; the clamp only guards the narrowing to int.
    constexpr Index kMaxExponent = Index(1) << 24;
    const int e = static_cast<int>(numext::mini(numext::maxi(exponent, -kMaxExponent), kMaxExponent));
    return toScalar(Complex(std::ldexp(numext::real(det), e), std::ldexp(numext::imag(det), e)),
                    std::is_same<RealScalar, Scalar>());
  }

  /** \returns the eigenvalues as the column-major flattening of the symbol:
   * eigenvalue \c f1*n2 + f2 is \c symbol()(f2, f1), and its (unit-norm)
   * eigenvector is the Kronecker product of the 1-D Fourier vectors of
   * frequencies \c f1 and \c f2, i.e. column \c f1*n2 + f2 of \ref eigenvectors.
   * Every BCCB matrix is diagonalized by this same 2-D Fourier basis. */
  ComplexVector eigenvalues() const {
    const ComplexArray s = symbol();
    return Map<const ComplexVector>(s.data(), s.size());
  }

  /** \returns the unitary matrix of eigenvectors: column \c f1*n2 + f2 is the
   * 2-D Fourier vector matching \c eigenvalues()[f1*n2 + f2].
   * \note The eigenvector matrix is materialized as a dense \c N x \c N matrix;
   * unlike the other methods of this class this costs O(N^2) storage. */
  ComplexMatrix eigenvectors() const {
    const Index n2 = blockSize(), n1 = numBlocks(), N = rows();
    ComplexMatrix F(N, N);
    for (Index f1 = 0; f1 < n1; ++f1)
      for (Index f2 = 0; f2 < n2; ++f2) fourierColumn(F, f1, f2, f1 * n2 + f2);
    return F;
  }

  /** \returns the singular values, sorted in decreasing order: the moduli of the
   * symbol entries. The ordering is shared with \ref matrixU and \ref matrixV,
   * so together they form the SVD \c *this = U * singularValues().asDiagonal() * V^H. */
  RealVector singularValues() const {
    const ComplexVector s = eigenvalues();
    const std::vector<Index> perm = svdPermutation(s);
    RealVector sv(s.size());
    for (Index t = 0; t < s.size(); ++t) sv[t] = numext::abs(s[perm[t]]);
    return sv;
  }

  /** \returns the matrix of left singular vectors \c U: column \c t is the 2-D
   * Fourier vector of the t-th largest symbol entry, scaled by its phase (phase 1
   * for a zero entry). Dense \c N x \c N, see the note in \ref eigenvectors. */
  ComplexMatrix matrixU() const {
    const Index n2 = blockSize(), N = rows();
    const ComplexVector s = eigenvalues();
    const std::vector<Index> perm = svdPermutation(s);
    ComplexMatrix U(N, N);
    for (Index t = 0; t < N; ++t) {
      fourierColumn(U, perm[t] / n2, perm[t] % n2, t);
      const RealScalar a = numext::abs(s[perm[t]]);
      if (a > RealScalar(0)) U.col(t) *= s[perm[t]] / a;
    }
    return U;
  }

  /** \returns the matrix of right singular vectors \c V: column \c t is the 2-D
   * Fourier vector of the t-th largest symbol entry. Dense \c N x \c N, see the
   * note in \ref eigenvectors. */
  ComplexMatrix matrixV() const {
    const Index n2 = blockSize(), N = rows();
    const ComplexVector s = eigenvalues();
    const std::vector<Index> perm = svdPermutation(s);
    ComplexMatrix V(N, N);
    for (Index t = 0; t < N; ++t) fourierColumn(V, perm[t] / n2, perm[t] % n2, t);
    return V;
  }

  /** \internal Writes the dense representation into \a dst: the (b1,c1) block is
   * the generator column (b1-c1) mod n1 rotated downwards by the within-block
   * column index, so only contiguous segment copies are involved. Invoked
   * through \c dense = bccb; */
  template <typename Dest>
  void evalTo(Dest& dst) const {
    const Index n2 = blockSize(), n1 = numBlocks();
    for (Index c1 = 0; c1 < n1; ++c1)
      for (Index j2 = 0; j2 < n2; ++j2) {
        auto col = dst.col(c1 * n2 + j2);
        for (Index b1 = 0; b1 < n1; ++b1) {
          Index k1 = b1 - c1;
          if (k1 < 0) k1 += n1;
          col.segment(b1 * n2, j2) = m_g.col(k1).tail(j2);
          col.segment(b1 * n2 + j2, n2 - j2) = m_g.col(k1).head(n2 - j2);
        }
      }
  }

  /** \internal Computes \c dst += (*this), see evalTo(). */
  template <typename Dest>
  void addTo(Dest& dst) const {
    const Index n2 = blockSize(), n1 = numBlocks();
    for (Index c1 = 0; c1 < n1; ++c1)
      for (Index j2 = 0; j2 < n2; ++j2) {
        auto col = dst.col(c1 * n2 + j2);
        for (Index b1 = 0; b1 < n1; ++b1) {
          Index k1 = b1 - c1;
          if (k1 < 0) k1 += n1;
          col.segment(b1 * n2, j2) += m_g.col(k1).tail(j2);
          col.segment(b1 * n2 + j2, n2 - j2) += m_g.col(k1).head(n2 - j2);
        }
      }
  }

  /** \internal Computes \c dst -= (*this), see evalTo(). */
  template <typename Dest>
  void subTo(Dest& dst) const {
    const Index n2 = blockSize(), n1 = numBlocks();
    for (Index c1 = 0; c1 < n1; ++c1)
      for (Index j2 = 0; j2 < n2; ++j2) {
        auto col = dst.col(c1 * n2 + j2);
        for (Index b1 = 0; b1 < n1; ++b1) {
          Index k1 = b1 - c1;
          if (k1 < 0) k1 += n1;
          col.segment(b1 * n2, j2) -= m_g.col(k1).tail(j2);
          col.segment(b1 * n2 + j2, n2 - j2) -= m_g.col(k1).head(n2 - j2);
        }
      }
  }

  /** \returns the product expression \c (*this) * \a x, evaluated through a fast
   * 2-D-FFT-based matrix-vector product. The expression carries the default
   * product tag, so assigning it behaves like any dense product: a temporary
   * resolves aliasing between the destination and \a x, and \c .noalias() skips
   * it. */
  template <typename Rhs>
  Product<Bccb, Rhs> operator*(const MatrixBase<Rhs>& x) const {
    EIGEN_STATIC_ASSERT(ColsAtCompileTime == Dynamic || Rhs::RowsAtCompileTime == Dynamic ||
                            int(ColsAtCompileTime) == int(Rhs::RowsAtCompileTime),
                        INVALID_MATRIX_PRODUCT)
    eigen_assert(x.rows() == cols() && "invalid product: dimensions do not match");
    return Product<Bccb, Rhs>(*this, x.derived());
  }

  /** \internal Computes \c dst += alpha * (*this) * rhs. \c ProductScalar is the
   * promoted scalar of the product (complex when a real operator is applied to a
   * complex right-hand side); the accumulation runs in the promoted type.
   *
   * Non-finite data -- in the generating array, the cached symbol (which can
   * overflow even for a finite generator: the 2-D DFT accumulates up to N
   * addends) or the right-hand side -- takes the direct O(N^2) kernel: the
   * transforms would smear a single Inf/NaN into NaNs across the whole output,
   * where the dense product only propagates it through the dot products that
   * touch it. */
  template <typename Dest, typename Rhs, typename ProductScalar>
  void addProduct(Dest& dst, const Rhs& rhs, const ProductScalar& alpha) const {
    const Index N = rows();
    eigen_assert(rhs.rows() == N && "invalid product: dimensions do not match");
    if (N <= internal::structured_direct_threshold() || !m_fftUsable || !rhs.allFinite())
      directProduct(dst, rhs, alpha);
    else
      applySymbol(dst, m_symbol, rhs, alpha, /*accumulate=*/true);
  }

 private:
  /** \internal Builds an operator from a generating array and an already-known
   * symbol (empty for small operators), skipping the FFT of the public
   * constructor. Used by transpose(), conjugate(), adjoint() and inverse(),
   * whose symbols are cheap transformations of the existing one. */
  Bccb(const GeneratorType& g, const ComplexArray& symbol) : m_g(g), m_symbol(symbol) {
    m_fftUsable = computeFftUsable();
  }

  /** \internal Whether products may take the FFT path: the generating array and
   * the cached symbol must be finite. The symbol accumulates up to N addends, so
   * it can overflow to Inf even for a finite generator; such operators fall back
   * to the direct kernel, which stays exact. */
  bool computeFftUsable() const { return m_g.allFinite() && (m_symbol.size() == 0 || m_symbol.allFinite()); }

  /** \internal Direct O(N^2) product kernel: computes \c dst += alpha * (*this)
   * * rhs without transforms, as a plain scalar loop. Serves operators below the
   * FFT threshold (where the two-level segment-based middle tier of the 1-D
   * operators does not pay off: the within-block segments are too short) and any
   * product involving non-finite data, whose entrywise IEEE semantics the
   * transforms cannot preserve. */
  template <typename Dest, typename Rhs, typename ProductScalar>
  void directProduct(Dest& dst, const Rhs& rhs, const ProductScalar& alpha) const {
    const Index N = rows();
    // A unit alpha must not multiply: even the identity complex scalar (1,0)
    // pollutes an (Inf,0) value with NaN through the 0*Inf cross term.
    const bool unitAlpha = alpha == ProductScalar(1);
    for (Index k = 0; k < rhs.cols(); ++k)
      for (Index i = 0; i < N; ++i) {
        ProductScalar acc(0);
        for (Index j = 0; j < N; ++j) acc += coeff(i, j) * rhs.coeff(j, k);
        dst.coeffRef(i, k) += unitAlpha ? acc : ProductScalar(alpha * acc);
      }
  }

  /** \internal Applies the operator whose 2-D symbol is \a s to every column of
   * \a rhs: reshape to n2 x n1 (column-major), 2-D FFT, multiply by \a s,
   * back-transform, take the \c ProductScalar part. Adds into \a dst when
   * \a accumulate, overwrites otherwise.
   *
   * The transforms sum up to \c N inputs, so intermediates can overflow even
   * when every entry of the true result is representable. To keep the whole
   * pipeline overflow-free for finite inputs anywhere in the floating-point
   * range, the symbol and each right-hand-side column are rescaled by an exact
   * power of two that brings their maximum modulus below one -- huge generators
   * make huge symbols, hence both sides -- and the removed exponent is folded
   * back into the output through per-entry ldexp, which saturates cleanly to
   * zero or infinity if the true result itself leaves the representable range.
   * The exponents come from the component-wise magnitudes (see
   * internal::structured_exponent_bound()): the modulus of a finite complex
   * value near the overflow threshold is not representable, which would
   * silently disable the scaling exactly where it is needed. Zero columns
   * short-circuit to zero, but only under a finite symbol: Inf*0 and NaN*0 are
   * NaN entrywise. Non-finite data must otherwise not reach this function; the
   * products route it to the direct kernel (see addProduct), and solve() takes
   * its direct fallback for non-finite right-hand sides. */
  template <typename Dest, typename Rhs, typename ProductScalar>
  void applySymbol(Dest& dst, const ComplexArray& s, const Rhs& rhs, const ProductScalar& alpha,
                   bool accumulate) const {
    const Index n2 = blockSize(), n1 = numBlocks(), N = rows();
    const bool sFinite = s.allFinite();
    const int es = internal::structured_exponent_bound(s);  // max|s| < 2^es
    ComplexArray sScaled;
    if (es != 0) {
      sScaled = s;
      ldexpInPlace(sScaled, -es);
    }
    const ComplexArray& sUse = es != 0 ? sScaled : s;
    Matrix<ProductScalar, Dynamic, 1> xc(N);
    ComplexArray X(n2, n1);
    for (Index k = 0; k < rhs.cols(); ++k) {
      xc = rhs.col(k).template cast<ProductScalar>();
      if (sFinite && (xc.array() == ProductScalar(0)).all()) {
        // An exactly zero column maps to an exactly zero column -- unless the
        // symbol holds Inf or NaN, whose products with zero are NaN.
        if (!accumulate) dst.col(k).setZero();
        continue;
      }
      const int ex = internal::structured_exponent_bound(xc);  // 0 for an all-zero column: no scaling
      X = Map<const Matrix<ProductScalar, Dynamic, Dynamic, ColMajor>>(xc.data(), n2, n1).template cast<Complex>();
      ldexpInPlace(X, -ex);
      fft2(X);
      X.array() *= sUse.array();
      ifft2(X);
      ldexpInPlace(X, ex + es);
      const Map<const ComplexVector> xv(X.data(), N);
      if (accumulate)
        dst.col(k) += alpha * internal::structured_scalar_part_impl<ProductScalar>::run(xv);
      else
        dst.col(k) = alpha * internal::structured_scalar_part_impl<ProductScalar>::run(xv);
    }
  }

  /** \internal Multiplies every entry of \a X by 2^e, exactly. The per-entry
   * ldexp saturates to zero / infinity component-wise without ever forming the
   * (possibly unrepresentable) scale factor 2^e itself. */
  static void ldexpInPlace(ComplexArray& X, int e) {
    if (e == 0) return;
    for (Index k1 = 0; k1 < X.cols(); ++k1)
      for (Index k2 = 0; k2 < X.rows(); ++k2) {
        Complex& z = X.coeffRef(k2, k1);
        z = Complex(std::ldexp(numext::real(z), e), std::ldexp(numext::imag(z), e));
      }
  }

  /** \internal In-place 2-D FFT by the row-column algorithm: transform every
   * column (length n2), then every row (length n1). Length-1 transforms are the
   * identity (and unsupported by the kissfft backend), hence the guards. */
  void fft2(ComplexArray& X) const {
    FFT<RealScalar> fft;
    const Index n2 = X.rows(), n1 = X.cols();
    if (n2 > 1) {
      ComplexVector tmp(n2);
      for (Index k1 = 0; k1 < n1; ++k1) {
        ComplexVector colv = X.col(k1);
        fft.fwd(tmp, colv, n2);
        X.col(k1) = tmp;
      }
    }
    if (n1 > 1) {
      ComplexVector tmp(n1), rowv(n1);
      for (Index k2 = 0; k2 < n2; ++k2) {
        rowv = X.row(k2).transpose();
        fft.fwd(tmp, rowv, n1);
        X.row(k2) = tmp.transpose();
      }
    }
  }

  /** \internal In-place 2-D inverse FFT, see fft2(). */
  void ifft2(ComplexArray& X) const {
    FFT<RealScalar> fft;
    const Index n2 = X.rows(), n1 = X.cols();
    if (n2 > 1) {
      ComplexVector tmp(n2);
      for (Index k1 = 0; k1 < n1; ++k1) {
        ComplexVector colv = X.col(k1);
        fft.inv(tmp, colv, n2);
        X.col(k1) = tmp;
      }
    }
    if (n1 > 1) {
      ComplexVector tmp(n1), rowv(n1);
      for (Index k2 = 0; k2 < n2; ++k2) {
        rowv = X.row(k2).transpose();
        fft.inv(tmp, rowv, n1);
        X.row(k2) = tmp.transpose();
      }
    }
  }

  /** \internal \returns the 2-D DFT of the generating array. */
  ComplexArray computeSymbol() const {
    ComplexArray s = m_g.template cast<Complex>();
    fft2(s);
    return s;
  }

  /** \internal \returns \a M index-reversed in both dimensions:
   * result(k2, k1) = M((-k2) mod n2, (-k1) mod n1). Row 0 and column 0 stay in
   * place; the rest is a two-dimensional reversal. Empty input stays empty. */
  template <typename MatType>
  static MatType reverse2(const MatType& M) {
    const Index r = M.rows(), c = M.cols();
    MatType R(r, c);
    if (M.size() == 0) return R;
    R(0, 0) = M(0, 0);
    if (c > 1) R.row(0).tail(c - 1) = M.row(0).tail(c - 1).reverse();
    if (r > 1) R.col(0).tail(r - 1) = M.col(0).tail(r - 1).reverse();
    if (r > 1 && c > 1) R.bottomRightCorner(r - 1, c - 1) = M.bottomRightCorner(r - 1, c - 1).reverse();
    return R;
  }

  /** \internal \returns the rank/pseudo-inversion threshold for \a s, in the
   * spirit of the SVD-based pseudo-inverse (the N * epsilon * max|s| convention
   * of [3], chapter 5.4): clamped from below at the smallest normal number so
   * subnormal and zero symbol entries -- whose reciprocals overflow -- are
   * treated as exact zeros. Entries at or above the threshold, in particular a
   * smallest-normal entry, are inverted (their reciprocals are finite). */
  static RealScalar rankThreshold(const ComplexArray& s) {
    return numext::maxi(RealScalar(s.size()) * NumTraits<RealScalar>::epsilon() * s.cwiseAbs().maxCoeff(),
                        (std::numeric_limits<RealScalar>::min)());
  }

  /** \internal Rescales \a z by the power of two that brings \c max(|re|,|im|)
   * into [0.5, 1), accumulating the removed exponent into \a exponent. The
   * rescaling is exact, so no roundoff is introduced. Zeros and non-finite
   * values, which must propagate exactly through determinant(), are returned
   * untouched. */
  static Complex balance(const Complex& z, Index& exponent) {
    const RealScalar mag = numext::maxi(numext::abs(numext::real(z)), numext::abs(numext::imag(z)));
    if (!(mag > RealScalar(0)) || !(numext::isfinite)(mag)) return z;
    int e;
    std::frexp(mag, &e);
    exponent += e;
    return Complex(std::ldexp(numext::real(z), -e), std::ldexp(numext::imag(z), -e));
  }

  /** \internal Writes the unit-norm 2-D Fourier eigenvector of frequencies
   * \c (f1, f2) into column \a dstCol of \a F: entry \c b1*n2 + i2 is
   * exp(2 pi i (b1 f1 / n1 + i2 f2 / n2)) / sqrt(N). All frequency products are
   * accumulated incrementally modulo their length, so the angles stay O(2 pi) at
   * full accuracy and no Index overflow can occur. */
  void fourierColumn(ComplexMatrix& F, Index f1, Index f2, Index dstCol) const {
    const Index n2 = blockSize(), n1 = numBlocks();
    const RealScalar scale = RealScalar(1) / numext::sqrt(RealScalar(rows()));
    ComplexVector w2(n2);
    Index jf = 0;  // i2 * f2 mod n2
    for (Index i2 = 0; i2 < n2; ++i2) {
      w2[i2] = std::polar(scale, RealScalar(2 * EIGEN_PI) * RealScalar(jf) / RealScalar(n2));
      jf += f2;
      if (jf >= n2) jf -= n2;
    }
    Index bf = 0;  // b1 * f1 mod n1
    for (Index b1 = 0; b1 < n1; ++b1) {
      const Complex w1 = std::polar(RealScalar(1), RealScalar(2 * EIGEN_PI) * RealScalar(bf) / RealScalar(n1));
      F.col(dstCol).segment(b1 * n2, n2) = w1 * w2;
      bf += f1;
      if (bf >= n1) bf -= n1;
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

  /** \internal Projects the complex determinant onto \c Scalar. */
  static Scalar toScalar(const Complex& z, std::true_type /*scalar_is_real*/) { return numext::real(z); }
  static Scalar toScalar(const Complex& z, std::false_type /*scalar_is_real*/) { return z; }

  GeneratorType m_g;
  ComplexArray m_symbol;
  bool m_fftUsable;
};

/** \ingroup StructuredMatrices_Module
 * \returns a \ref Bccb operator with generating array \a generator. The
 * compile-time dimensions of the operator are deduced from the array. */
template <typename Derived>
Bccb<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> makeBccb(
    const MatrixBase<Derived>& generator) {
  return Bccb<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>(generator);
}

namespace internal {

// Single product specialization covering every product tag; see the note in
// Circulant.h.
template <typename Scalar_, int BlockSize_, int NumBlocks_, typename Rhs, int ProductTag>
struct generic_product_impl<Bccb<Scalar_, BlockSize_, NumBlocks_>, Rhs, StructuredShape, DenseShape, ProductTag>
    : structured_product_impl<Bccb<Scalar_, BlockSize_, NumBlocks_>, Rhs> {};

}  // namespace internal

}  // namespace Eigen

#endif  // EIGEN_STRUCTURED_BCCB_H
