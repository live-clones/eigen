// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_STRUCTURED_MATRIX_UTILS_H
#define EIGEN_STRUCTURED_MATRIX_UTILS_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

// Below this dimension the FFT setup costs more than a plain O(n^2) evaluation,
// so the structured operators fall back to a direct coefficient-based product.
constexpr Index structured_direct_threshold() { return 32; }

/** \internal
 * \returns the smallest integer >= \a n whose only prime factors are 2, 3 and 5.
 *
 * Such "5-smooth" sizes keep the FFT fast and sidestep the default kissfft
 * backend's poor handling of sizes with large prime factors. Used to pad the
 * circulant embedding of a Toeplitz matrix. The {2,3,5}-smooth numbers are dense
 * enough that the linear search returns after only a handful of steps.
 */
inline Index fft_next_good_size(Index n) {
  if (n < 1) return 1;
  for (Index m = n;; ++m) {
    Index r = m;
    while (r % 2 == 0) r /= 2;
    while (r % 3 == 0) r /= 3;
    while (r % 5 == 0) r /= 5;
    if (r == 1) return m;
  }
}

/** \internal Accumulate \c dst.col(k) += alpha * seg, taking the real part for a
 * real scalar type and the value itself for a complex scalar type. */
template <typename Scalar, typename Dest, typename ComplexSeg, std::enable_if_t<NumTraits<Scalar>::IsComplex, int> = 0>
inline void structured_accumulate_col(Dest& dst, Index k, const ComplexSeg& seg, const Scalar& alpha) {
  dst.col(k) += alpha * seg;
}

template <typename Scalar, typename Dest, typename ComplexSeg, std::enable_if_t<!NumTraits<Scalar>::IsComplex, int> = 0>
inline void structured_accumulate_col(Dest& dst, Index k, const ComplexSeg& seg, const Scalar& alpha) {
  dst.col(k) += alpha * seg.real();
}

/** \internal \returns a dense \a Scalar vector from the complex segment \a seg,
 * dropping the (numerically negligible) imaginary part for a real scalar type. */
template <typename Scalar, typename ComplexSeg, std::enable_if_t<NumTraits<Scalar>::IsComplex, int> = 0>
inline Matrix<Scalar, Dynamic, 1> structured_from_complex(const ComplexSeg& seg) {
  return seg;
}

template <typename Scalar, typename ComplexSeg, std::enable_if_t<!NumTraits<Scalar>::IsComplex, int> = 0>
inline Matrix<Scalar, Dynamic, 1> structured_from_complex(const ComplexSeg& seg) {
  return seg.real();
}

/** \internal
 * Computes \c dst += alpha * (op * rhs) for the structured operator \a op, given
 * the precomputed length-\a p DFT \a symbol of its first column (circulant) or of
 * its circulant embedding (Toeplitz). Each right-hand-side column is transformed,
 * multiplied pointwise by \a symbol, and transformed back; the leading \a outSize
 * entries form the corresponding output column.
 */
template <typename Scalar, typename Dest, typename Rhs>
void structured_fft_apply(Dest& dst, const Matrix<std::complex<typename NumTraits<Scalar>::Real>, Dynamic, 1>& symbol,
                          Index p, Index outSize, const Rhs& rhs, const Scalar& alpha) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef std::complex<RealScalar> Complex;
  typedef Matrix<Complex, Dynamic, 1> ComplexVector;

  FFT<RealScalar> fft;
  ComplexVector xf(p), yt(p);
  for (Index k = 0; k < rhs.cols(); ++k) {
    ComplexVector xc = rhs.col(k).template cast<Complex>();
    fft.fwd(xf, xc, p);  // zero-pads xc up to length p
    xf = symbol.cwiseProduct(xf);
    fft.inv(yt, xf, p);
    structured_accumulate_col<Scalar>(dst, k, yt.head(outSize), alpha);
  }
}

/** \internal Shared product implementation for the structured operator types.
 * Forwards to the operator's \c addProduct member, which performs the fast
 * matrix-vector product. The same body serves both the matrix-vector
 * (\c GemvProduct) and matrix-matrix (\c GemmProduct) dispatch tags. */
template <typename Op, typename Rhs>
struct structured_product_impl : generic_product_impl_base<Op, Rhs, structured_product_impl<Op, Rhs>> {
  typedef typename Product<Op, Rhs>::Scalar Scalar;
  template <typename Dest>
  static void scaleAndAddTo(Dest& dst, const Op& lhs, const Rhs& rhs, const Scalar& alpha) {
    lhs.addProduct(dst, rhs, alpha);
  }
};

}  // namespace internal

}  // namespace Eigen

#endif  // EIGEN_STRUCTURED_MATRIX_UTILS_H
