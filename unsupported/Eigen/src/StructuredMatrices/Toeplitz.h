// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_STRUCTURED_TOEPLITZ_H
#define EIGEN_STRUCTURED_TOEPLITZ_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

template <typename Scalar_>
class Toeplitz;

namespace internal {

template <typename Scalar_>
struct traits<Toeplitz<Scalar_>> {
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
 * \class Toeplitz
 * \brief An \c m x \c n Toeplitz matrix represented by its first column and row.
 *
 * A Toeplitz matrix has constant diagonals: entry \c (i,j) equals \c c[i-j] when
 * \c i>=j and \c r[j-i] otherwise, where \c c is its first column and \c r its
 * first row. The diagonal entry is taken from \c c, so \c r[0] is ignored.
 *
 * The matrix-vector product (\c operator*) is evaluated in O(n log n) by
 * embedding the Toeplitz matrix in a larger circulant matrix and applying the
 * FFT. As with \ref Circulant, \c operator* returns an Eigen product expression,
 * so a \c Toeplitz also plugs into the matrix-free iterative solvers.
 *
 * \tparam Scalar_ the scalar type, real or complex.
 *
 * \sa class Circulant, makeToeplitz()
 */
template <typename Scalar_>
class Toeplitz : public EigenBase<Toeplitz<Scalar_>> {
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

  /** Builds a Toeplitz matrix from its first column \a col and first row \a row.
   * The diagonal is taken from \a col, hence \c row[0] is ignored. */
  template <typename ColDerived, typename RowDerived>
  Toeplitz(const MatrixBase<ColDerived>& col, const MatrixBase<RowDerived>& row) : m_col(col), m_row(row) {
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(ColDerived)
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(RowDerived)
    eigen_assert(m_col.size() > 0 && m_row.size() > 0 && "Toeplitz generators must be non-empty");
  }

  EIGEN_DEVICE_FUNC Index rows() const { return m_col.size(); }
  EIGEN_DEVICE_FUNC Index cols() const { return m_row.size(); }

  /** \returns the generating first column. */
  const DenseVector& column() const { return m_col; }
  /** \returns the generating first row. */
  const DenseVector& row() const { return m_row; }

  /** \returns the coefficient at row \a row and column \a col. */
  Scalar coeff(Index row, Index col) const {
    Index k = row - col;
    return k >= 0 ? m_col.coeff(k) : m_row.coeff(-k);
  }

  /** \returns the equivalent dense matrix (mainly for testing and debugging). */
  DenseMatrix toDense() const {
    const Index m = rows(), n = cols();
    DenseMatrix dense(m, n);
    for (Index j = 0; j < n; ++j)
      for (Index i = 0; i < m; ++i) dense(i, j) = coeff(i, j);
    return dense;
  }

  /** \returns the product expression \c (*this) * \a x, evaluated through a fast
   * FFT-based matrix-vector product (circulant embedding). */
  template <typename Rhs>
  Product<Toeplitz, Rhs, AliasFreeProduct> operator*(const MatrixBase<Rhs>& x) const {
    return Product<Toeplitz, Rhs, AliasFreeProduct>(*this, x.derived());
  }

  /** \internal Computes \c dst += alpha * (*this) * rhs. */
  template <typename Dest, typename Rhs>
  void addProduct(Dest& dst, const Rhs& rhs, const Scalar& alpha) const {
    const Index m = rows(), n = cols();
    eigen_assert(rhs.rows() == n && "invalid product: dimensions do not match");

    if (m <= internal::structured_direct_threshold() && n <= internal::structured_direct_threshold()) {
      for (Index k = 0; k < rhs.cols(); ++k)
        for (Index i = 0; i < m; ++i) {
          Scalar acc(0);
          for (Index j = 0; j < n; ++j) acc += coeff(i, j) * rhs.coeff(j, k);
          dst.coeffRef(i, k) += alpha * acc;
        }
      return;
    }

    // Embed the Toeplitz matrix into a circulant matrix of size p >= m+n-1 whose
    // first column is [c; 0...0; r[n-1], ..., r[1]]. Multiplying that circulant
    // by [x; 0] reproduces the Toeplitz product in its leading m entries.
    const Index p = internal::fft_next_good_size(m + n - 1);
    ComplexVector emb = ComplexVector::Zero(p);
    for (Index i = 0; i < m; ++i) emb[i] = Complex(m_col.coeff(i));
    for (Index k = 1; k < n; ++k) emb[p - k] = Complex(m_row.coeff(k));

    FFT<RealScalar> fft;
    ComplexVector symbol(p);
    fft.fwd(symbol, emb, p);
    internal::structured_fft_apply<Scalar>(dst, symbol, p, m, rhs, alpha);
  }

 private:
  DenseVector m_col;
  DenseVector m_row;
};

/** \ingroup StructuredMatrices_Module
 * \returns a \ref Toeplitz operator with first column \a col and first row \a row. */
template <typename ColDerived, typename RowDerived>
Toeplitz<typename ColDerived::Scalar> makeToeplitz(const MatrixBase<ColDerived>& col,
                                                   const MatrixBase<RowDerived>& row) {
  return Toeplitz<typename ColDerived::Scalar>(col, row);
}

namespace internal {

template <typename Scalar_, typename Rhs>
struct generic_product_impl<Toeplitz<Scalar_>, Rhs, DenseShape, DenseShape, GemvProduct>
    : structured_product_impl<Toeplitz<Scalar_>, Rhs> {};

template <typename Scalar_, typename Rhs>
struct generic_product_impl<Toeplitz<Scalar_>, Rhs, DenseShape, DenseShape, GemmProduct>
    : structured_product_impl<Toeplitz<Scalar_>, Rhs> {};

}  // namespace internal

}  // namespace Eigen

#endif  // EIGEN_STRUCTURED_TOEPLITZ_H
