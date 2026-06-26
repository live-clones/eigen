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

template <typename Scalar_>
class Circulant;

namespace internal {

template <typename Scalar_>
struct traits<Circulant<Scalar_>> {
  typedef Scalar_ Scalar;
  typedef Eigen::Dense StorageKind;
  typedef Eigen::MatrixXpr XprKind;
  typedef int StorageIndex;
  enum {
    RowsAtCompileTime = Dynamic,
    ColsAtCompileTime = Dynamic,
    MaxRowsAtCompileTime = Dynamic,
    MaxColsAtCompileTime = Dynamic,
    Flags = NestByRefBit
  };
};

}  // namespace internal

/** \ingroup StructuredMatrices_Module
 * \class Circulant
 * \brief An \c n x \c n circulant matrix represented by its first column.
 *
 * A circulant matrix is the square matrix whose entry \c (i,j) equals
 * \c c[(i-j) mod n], where \c c is its first column. It is diagonalized by the
 * discrete Fourier transform, which yields an O(n log n) matrix-vector product
 * (\c operator*) and an O(n log n) direct solve (\ref solve).
 *
 * The operator stores its own copy of the generating column and derives from
 * \c EigenBase. Because \c operator* returns an Eigen product expression, a
 * \c Circulant also drops into the matrix-free iterative solvers.
 *
 * \tparam Scalar_ the scalar type, real or complex.
 *
 * \sa class Toeplitz, makeCirculant()
 */
template <typename Scalar_>
class Circulant : public EigenBase<Circulant<Scalar_>> {
 public:
  typedef Scalar_ Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef int StorageIndex;
  typedef std::complex<RealScalar> Complex;
  typedef Matrix<Scalar, Dynamic, 1> DenseVector;
  typedef Matrix<Scalar, Dynamic, Dynamic> DenseMatrix;
  typedef Matrix<Complex, Dynamic, 1> ComplexVector;

  enum {
    RowsAtCompileTime = Dynamic,
    ColsAtCompileTime = Dynamic,
    MaxRowsAtCompileTime = Dynamic,
    MaxColsAtCompileTime = Dynamic,
    IsRowMajor = false
  };

  /** Builds a circulant matrix from its first column \a col. */
  template <typename Derived>
  explicit Circulant(const MatrixBase<Derived>& col) : m_col(col) {
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
    eigen_assert(m_col.size() > 0 && "Circulant generator must be non-empty");
  }

  EIGEN_DEVICE_FUNC Index rows() const { return m_col.size(); }
  EIGEN_DEVICE_FUNC Index cols() const { return m_col.size(); }

  /** \returns the generating first column. */
  const DenseVector& column() const { return m_col; }

  /** \returns the coefficient at row \a row and column \a col. */
  Scalar coeff(Index row, Index col) const {
    Index k = row - col;
    if (k < 0) k += rows();
    return m_col.coeff(k);
  }

  /** \returns the equivalent dense matrix (mainly for testing and debugging). */
  DenseMatrix toDense() const {
    const Index n = rows();
    DenseMatrix dense(n, n);
    for (Index j = 0; j < n; ++j)
      for (Index i = 0; i < n; ++i) dense(i, j) = coeff(i, j);
    return dense;
  }

  /** \returns the product expression \c (*this) * \a x, evaluated through a fast
   * FFT-based matrix-vector product. */
  template <typename Rhs>
  Product<Circulant, Rhs, AliasFreeProduct> operator*(const MatrixBase<Rhs>& x) const {
    return Product<Circulant, Rhs, AliasFreeProduct>(*this, x.derived());
  }

  /** \returns the solution \c x of \c (*this) * x = b, computed directly in the
   * Fourier domain: \c x = ifft( fft(b) ./ fft(c) ).
   * \warning The circulant matrix is assumed to be non-singular. */
  template <typename Rhs>
  Matrix<Scalar, Dynamic, Rhs::ColsAtCompileTime> solve(const MatrixBase<Rhs>& b) const {
    const Index n = rows();
    eigen_assert(b.rows() == n && "right-hand side has the wrong number of rows");
    Matrix<Scalar, Dynamic, Rhs::ColsAtCompileTime> x(n, b.cols());

    FFT<RealScalar> fft;
    ComplexVector cc = m_col.template cast<Complex>();
    ComplexVector cf(n);
    fft.fwd(cf, cc, n);

    ComplexVector bf(n), xt(n);
    for (Index k = 0; k < b.cols(); ++k) {
      ComplexVector bc = b.col(k).template cast<Complex>();
      fft.fwd(bf, bc, n);
      bf = bf.cwiseQuotient(cf);
      fft.inv(xt, bf, n);
      x.col(k) = internal::structured_from_complex<Scalar>(xt);
    }
    return x;
  }

  /** \internal Computes \c dst += alpha * (*this) * rhs. */
  template <typename Dest, typename Rhs>
  void addProduct(Dest& dst, const Rhs& rhs, const Scalar& alpha) const {
    const Index n = rows();
    eigen_assert(rhs.rows() == n && "invalid product: dimensions do not match");

    if (n <= internal::structured_direct_threshold()) {
      for (Index k = 0; k < rhs.cols(); ++k)
        for (Index i = 0; i < n; ++i) {
          Scalar acc(0);
          for (Index j = 0; j < n; ++j) acc += coeff(i, j) * rhs.coeff(j, k);
          dst.coeffRef(i, k) += alpha * acc;
        }
      return;
    }

    FFT<RealScalar> fft;
    ComplexVector cc = m_col.template cast<Complex>();
    ComplexVector symbol(n);
    fft.fwd(symbol, cc, n);
    internal::structured_fft_apply<Scalar>(dst, symbol, n, n, rhs, alpha);
  }

 private:
  DenseVector m_col;
};

/** \ingroup StructuredMatrices_Module
 * \returns a \ref Circulant operator with first column \a col. */
template <typename Derived>
Circulant<typename Derived::Scalar> makeCirculant(const MatrixBase<Derived>& col) {
  return Circulant<typename Derived::Scalar>(col);
}

namespace internal {

template <typename Scalar_, typename Rhs>
struct generic_product_impl<Circulant<Scalar_>, Rhs, DenseShape, DenseShape, GemvProduct>
    : structured_product_impl<Circulant<Scalar_>, Rhs> {};

template <typename Scalar_, typename Rhs>
struct generic_product_impl<Circulant<Scalar_>, Rhs, DenseShape, DenseShape, GemmProduct>
    : structured_product_impl<Circulant<Scalar_>, Rhs> {};

}  // namespace internal

}  // namespace Eigen

#endif  // EIGEN_STRUCTURED_CIRCULANT_H
