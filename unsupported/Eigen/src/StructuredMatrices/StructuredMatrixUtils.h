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

// Evaluator shape of the structured operator types. Keying the operators on their
// own shape (instead of DenseShape) routes dense assignment through the
// EigenBase2EigenBase path (i.e. evalTo/addTo/subTo) and products through a single
// generic_product_impl partial specialization covering every product tag.
struct StructuredShape {};

// Below this dimension the FFT setup costs more than a plain O(n^2) evaluation,
// so the structured operators fall back to a direct segment-based product.
constexpr Index structured_direct_threshold() { return 32; }

// Below this dimension even the segment-based direct product loses to a plain
// scalar loop: the per-segment setup dominates when the average segment holds
// fewer than a couple of packets (measured crossover on AVX2 hardware).
constexpr Index structured_scalar_threshold() { return 16; }

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

/** \internal View a complex-valued expression as \a Scalar: the expression itself
 * when \a Scalar is complex, its real part when \a Scalar is real (the imaginary
 * part then only holds numerically negligible roundoff). \c run_scalar is the
 * single-coefficient analogue. A single dispatch struct keeps the number of
 * instantiated helpers down to one per scalar type. */
template <typename Scalar, bool IsComplex = NumTraits<Scalar>::IsComplex>
struct structured_scalar_part_impl {
  template <typename Xpr>
  static const Xpr& run(const Xpr& xpr) {
    return xpr;
  }
  static const Scalar& run_scalar(const Scalar& x) { return x; }
};

template <typename Scalar>
struct structured_scalar_part_impl<Scalar, false> {
  template <typename Xpr>
  static typename Xpr::RealReturnType run(const Xpr& xpr) {
    return xpr.real();
  }
  static Scalar run_scalar(const std::complex<Scalar>& x) { return numext::real(x); }
};

/** \internal \returns an exponent bound \c e with \c max_k|x[k]| < 2^e, or 0 when
 * \a x is zero or holds non-finite values. The bound is derived from the
 * component-wise magnitudes, never from the modulus: a finite complex value near
 * the overflow threshold has a non-representable modulus, which would silently
 * disable the overflow-protection scaling exactly where it is needed. Bounding
 * the modulus by twice the largest component costs at most one extra bit. */
template <typename Xpr>
int structured_exponent_bound(const Xpr& x) {
  using RealScalar = typename NumTraits<typename Xpr::Scalar>::Real;
  RealScalar m;
  if (NumTraits<typename Xpr::Scalar>::IsComplex)
    m = numext::maxi(x.real().cwiseAbs().maxCoeff(), x.imag().cwiseAbs().maxCoeff());
  else
    m = x.cwiseAbs().maxCoeff();
  int e = 0;
  if (m > RealScalar(0) && (numext::isfinite)(m)) {
    std::frexp(m, &e);
    if (NumTraits<typename Xpr::Scalar>::IsComplex) ++e;
  }
  return e;
}

/** \internal
 * Computes \c dst.col(k) += alpha * ifft( symbol .* fft(rhs.col(k)) ) for every
 * column of \a rhs, i.e. applies the circulant operator whose eigenvalues are
 * \a symbol. The leading \a outSize entries of each back-transform form the
 * corresponding output column. All workspace is allocated once outside the
 * per-column loop; right-hand sides shorter than the transform length are
 * zero-padded into the preallocated buffer so the FFT never re-allocates.
 *
 * The transforms accumulate up to \c p addends, so a finite column near the
 * overflow threshold (or one applied through a symbol of huge magnitude) would
 * overflow inside the FFT and turn a representable result into Inf/NaN. Each
 * column is therefore scaled down by a power of two -- an exact shift, no
 * roundoff -- derived from the column's and the symbol's largest component-wise
 * exponents (see structured_exponent_bound()), and the exponent is folded back
 * into the output after the back-transform. The scale is one whenever the
 * conservative intermediate bound cannot overflow, so results are bit-identical
 * for inputs of moderate magnitude; zero columns are never scaled.
 *
 * Genuinely non-finite data (NaN or Inf in the symbol or the column) must not
 * reach this function: the transforms mix every input entry into every output
 * entry, so a single special value would contaminate the whole column with NaN
 * where the dense product only propagates it through the dot products that
 * touch it. The operators route such data to their direct kernels instead.
 */
template <typename Scalar, typename Dest, typename Rhs>
void structured_fft_apply(Dest& dst, const Matrix<std::complex<typename NumTraits<Scalar>::Real>, Dynamic, 1>& symbol,
                          Index outSize, const Rhs& rhs, const Scalar& alpha) {
  using RealScalar = typename NumTraits<Scalar>::Real;
  using Complex = std::complex<RealScalar>;
  using ComplexVector = Matrix<Complex, Dynamic, 1>;

  const Index p = symbol.size();
  eigen_assert(rhs.rows() <= p && outSize <= p);

  if (p == 1) {
    // Degenerate 1x1 operator: the length-1 transform is the identity (and is not
    // supported by the kissfft backend anyway).
    dst.row(0) +=
        alpha * structured_scalar_part_impl<Scalar>::run(symbol.coeff(0) * rhs.row(0).template cast<Complex>());
    return;
  }

  // Exponent budget of the power-of-two scaling: with the column scaled so that
  // its magnitudes stay below 2^c, the intermediates are bounded by
  // 2^(c + s + 2*ceil(log2(p)) + 1) -- each transform accumulates up to p
  // addends and the complex multiplication by the symbol (magnitudes below 2^s)
  // contributes one more bit -- which must stay below 2^max_exponent.
  int log2p = 0;
  for (Index t = p; t > 0; t /= 2) ++log2p;
  const int budget = std::numeric_limits<RealScalar>::max_exponent - 2 * log2p - 2;
  const int symbolExp = structured_exponent_bound(symbol);  // max|symbol| < 2^symbolExp

  FFT<RealScalar> fft;
  ComplexVector xt = ComplexVector::Zero(p);  // the zero padding beyond rhs.rows() is never overwritten
  ComplexVector xf(p), yt(p);
  for (Index k = 0; k < rhs.cols(); ++k) {
    const int colExp = structured_exponent_bound(rhs.col(k));  // 0 for an all-zero column: no scaling
    const int e = numext::maxi(colExp + symbolExp - budget, 0);
    // Each power of two is split in halves so that the factors themselves stay
    // inside the exponent range even when e exceeds it (a huge column applied
    // through a huge symbol); scaling by the two exact factors in sequence is
    // still an exact shift wherever the result is representable.
    const RealScalar down1 = std::ldexp(RealScalar(1), -(e / 2)), down2 = std::ldexp(RealScalar(1), -(e - e / 2));
    const RealScalar up1 = std::ldexp(RealScalar(1), e / 2), up2 = std::ldexp(RealScalar(1), e - e / 2);
    xt.head(rhs.rows()) = ((rhs.col(k) * down1) * down2).template cast<Complex>();
    fft.fwd(xf, xt, p);
    xf.array() *= symbol.array();
    fft.inv(yt, xf, p);
    dst.col(k) += alpha * structured_scalar_part_impl<Scalar>::run((yt.head(outSize) * up1) * up2);
  }
}

/** \internal Shared product implementation for the structured operator types.
 * Forwards to the operator's \c addProduct member, which performs the fast
 * matrix-vector product. The same body serves every dense product dispatch tag.
 *
 * The structured products carry the default product tag, so assignment has the
 * ordinary dense-product semantics: \c x = op * expr first materializes the
 * product into a temporary, which resolves every form of aliasing between the
 * destination and the right-hand side (same object, overlapping views,
 * expressions referencing the destination, destinations resized by the
 * assignment), and \c .noalias() skips the temporary under the usual caller
 * promise that no aliasing exists. */
template <typename Op, typename Rhs>
struct structured_product_impl : generic_product_impl_base<Op, Rhs, structured_product_impl<Op, Rhs>> {
  using Scalar = typename Product<Op, Rhs>::Scalar;

  template <typename Dest>
  static void evalTo(Dest& dst, const Op& lhs, const Rhs& rhs) {
    dst.setZero();
    lhs.addProduct(dst, rhs, Scalar(1));
  }

  template <typename Dest>
  static void scaleAndAddTo(Dest& dst, const Op& lhs, const Rhs& rhs, const Scalar& alpha) {
    lhs.addProduct(dst, rhs, alpha);
  }
};

}  // namespace internal

}  // namespace Eigen

#endif  // EIGEN_STRUCTURED_MATRIX_UTILS_H
