// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_APPROX_COMPARISONS_H
#define EIGEN_APPROX_COMPARISONS_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

// Keep custom scalars on the algebraic path: exponent scaling requires binary floating-point arithmetic.
template <typename Scalar>
struct use_scaled_comparison
    : bool_constant<(std::is_floating_point<Scalar>::value && std::numeric_limits<Scalar>::radix == 2) ||
                    std::is_same<Scalar, half>::value || std::is_same<Scalar, bfloat16>::value> {};

template <typename RealScalar>
struct use_scaled_comparison<std::complex<RealScalar>> : use_scaled_comparison<RealScalar> {};

// A nonnegative magnitude fraction * 2^exponent, including norms larger than the scalar range. Positive finite
// magnitudes order lexicographically by (exponent, fraction), the member order.
template <typename RealScalar>
struct comparison_magnitude {
  int exponent = 0;
  RealScalar fraction;

  EIGEN_DEVICE_FUNC explicit comparison_magnitude(const RealScalar& value) : fraction(value) {
    if (value > RealScalar(0) && value <= NumTraits<RealScalar>::highest()) {
      EIGEN_USING_STD(frexp);
      fraction = frexp(value, &exponent);
    }
  }

  template <typename OtherRealScalar>
  EIGEN_DEVICE_FUNC explicit comparison_magnitude(const comparison_magnitude<OtherRealScalar>& value)
      : exponent(value.exponent), fraction(RealScalar(value.fraction)) {}

  EIGEN_DEVICE_FUNC void multiply(const RealScalar& value) {
    const comparison_magnitude factor(value);
    const comparison_magnitude product(fraction * factor.fraction);
    fraction = product.fraction;
    exponent += factor.exponent + product.exponent;
  }

  EIGEN_DEVICE_FUNC bool isFinite() const { return fraction <= RealScalar(1); }
};

template <typename RealScalar>
EIGEN_DEVICE_FUNC bool operator<=(const comparison_magnitude<RealScalar>& x,
                                  const comparison_magnitude<RealScalar>& y) {
  if (!x.isFinite()) return false;
  // Zero and non-finite magnitudes carry no exponent.
  if (!y.isFinite() || x.fraction == RealScalar(0) || y.fraction == RealScalar(0)) return x.fraction <= y.fraction;
  return x.exponent < y.exponent || (x.exponent == y.exponent && x.fraction <= y.fraction);
}

template <typename RealScalar, typename OtherRealScalar>
EIGEN_DEVICE_FUNC bool operator<=(const comparison_magnitude<RealScalar>& x,
                                  const comparison_magnitude<OtherRealScalar>& y) {
  using Common = std::common_type_t<RealScalar, OtherRealScalar>;
  return comparison_magnitude<Common>(x) <= comparison_magnitude<Common>(y);
}

// The scaled path rescales its operands with operator*(Scalar). Dense-shaped expressions stay lazy, so nothing is
// allocated; another shape may hide that operator (Homogeneous does) and is evaluated first.
template <typename X>
using scaled_comparison_operand_t =
    std::conditional_t<std::is_same<typename evaluator_traits<X>::Shape, DenseShape>::value, const X&,
                       typename plain_object_eval<X>::type>;

template <typename Components>
EIGEN_DEVICE_FUNC typename Components::Scalar scaled_comparison_max_coeff(const Components& components) {
  using RealScalar = typename Components::Scalar;
  if (components.size() == 0) return RealScalar(0);
  return safe_scaling<RealScalar>::recover_flushed_max_coeff(components,
                                                             components.cwiseAbs().template maxCoeff<PropagateNaN>());
}

template <typename Derived>
EIGEN_DEVICE_FUNC comparison_magnitude<typename stable_norm_accumulator<typename Derived::RealScalar>::type>
scaled_comparison_norm(const Derived& xExpr) {
  using RealScalar = typename stable_norm_accumulator<typename Derived::RealScalar>::type;
  scaled_comparison_operand_t<Derived> x(xExpr);
  const auto& matrix = x.matrix();
  const auto& realComponents = matrix.realView();
  const auto& components = realComponents.template cast<RealScalar>();
  const RealScalar scale = scaled_comparison_max_coeff(components);
  // Classify first so NaNs do not reach the ordered comparison.
  if (!(numext::isfinite)(scale) || !(scale > RealScalar(0))) return comparison_magnitude<RealScalar>(scale);
  RealScalar squaredNorm = RealScalar(0);
  const auto factors = safe_scaling<RealScalar>::with_scaled(
      components, scale, [&](const auto& scaled) { squaredNorm = scaled.squaredNorm(); });
  comparison_magnitude<RealScalar> result(numext::sqrt(squaredNorm));
  result.multiply(factors.scale);
  return result;
}

template <typename X, typename Y>
EIGEN_DEVICE_FUNC comparison_magnitude<typename stable_norm_accumulator<typename X::RealScalar>::type>
scaled_comparison_distance(const X& xExpr, const Y& yExpr) {
  using Accumulator = typename stable_norm_accumulator<typename X::RealScalar>::type;
  using WideScalar =
      std::conditional_t<NumTraits<typename X::Scalar>::IsComplex || NumTraits<typename Y::Scalar>::IsComplex,
                         std::complex<Accumulator>, Accumulator>;
  scaled_comparison_operand_t<X> x(xExpr);
  scaled_comparison_operand_t<Y> y(yExpr);
  const auto& matrixX = x.matrix();
  const auto& matrixY = y.matrix();
  const auto& wideX = matrixX.template cast<WideScalar>();
  const auto& wideY = matrixY.template cast<WideScalar>();
  // FTZ flushes a subnormal difference of normal operands. Scaling a maximum M < 1 up by a power of two first loses
  // only differences below M * min; scaling larger operands down could underflow their small components.
  const Accumulator maxCoeff = numext::mini(
      Accumulator(1),
      numext::maxi(scaled_comparison_max_coeff(wideX.realView()), scaled_comparison_max_coeff(wideY.realView())));
  const safe_scaling_factors<Accumulator> factors = supports_power_of_two_scaling<Accumulator>::value
                                                        ? safe_scaling<Accumulator>::compute_floor_factors(maxCoeff)
                                                        : safe_scaling_factors<Accumulator>();
  // Both calls share one expression type, so scaled_comparison_norm is instantiated once.
  auto difference = scaled_comparison_norm(wideX * factors.invScale - wideY * factors.invScale);
  if (!difference.isFinite()) {
    // Finite operands can overflow on subtraction; halving first keeps every component representable.
    const Accumulator halfInvScale = factors.invScale * Accumulator(0.5);
    difference = scaled_comparison_norm(wideX * halfInvScale - wideY * halfInvScale);
    difference.multiply(Accumulator(2));
  }
  difference.multiply(factors.scale);
  return difference;
}

// Coefficients widened to the stable-norm accumulator, in which ordinary comparisons square and sum. The cast is the
// identity for float and double; half and bfloat16 widen exactly to float, where no half square underflows.
template <typename X>
using approx_comparison_wide_t =
    std::conditional_t<NumTraits<typename X::Scalar>::IsComplex,
                       std::complex<typename stable_norm_accumulator<typename X::RealScalar>::type>,
                       typename stable_norm_accumulator<typename X::RealScalar>::type>;

template <typename Scalar, bool = use_scaled_comparison<Scalar>::value>
struct approx_comparison_impl {
  using RealScalar = typename NumTraits<Scalar>::Real;

  template <typename X, typename Y>
  EIGEN_DEVICE_FUNC static bool isApprox(const X& x, const Y& y, const RealScalar& prec) {
    return (x.matrix() - y.matrix()).cwiseAbs2().sum() <=
           prec * prec * numext::mini(x.cwiseAbs2().sum(), y.cwiseAbs2().sum());
  }

  template <typename X, typename Y>
  EIGEN_DEVICE_FUNC static bool isMuchSmallerThan(const X& x, const Y& y, const RealScalar& prec) {
    return x.cwiseAbs2().sum() <= numext::abs2(prec) * y.cwiseAbs2().sum();
  }

  template <typename X>
  EIGEN_DEVICE_FUNC static bool isMuchSmallerThan(const X& x, const RealScalar& y, const RealScalar& prec) {
    return x.cwiseAbs2().sum() <= numext::abs2(prec * y);
  }
};

template <typename Scalar>
struct approx_comparison_impl<Scalar, true> {
  using RealScalar = typename NumTraits<Scalar>::Real;
  using Accumulator = typename stable_norm_accumulator<RealScalar>::type;
  template <typename Y>
  using CommonAccumulator =
      std::common_type_t<Accumulator, typename stable_norm_accumulator<typename Y::RealScalar>::type>;

  template <typename ValueScalar>
  EIGEN_DEVICE_FUNC static typename stable_norm_accumulator<ValueScalar>::type squared_norm_lower_bound(Index size) {
    using ValueAccumulator = typename stable_norm_accumulator<ValueScalar>::type;
    // Squares accumulate in ValueAccumulator; below n * min / epsilon, flushed squares can affect the comparison.
    return stable_normalization_normal_min<ValueAccumulator, ValueAccumulator>::run() /
           NumTraits<ValueAccumulator>::epsilon() * ValueAccumulator(size);
  }

  template <typename BoundScalar>
  EIGEN_DEVICE_FUNC static bool safe_squared_norm(const BoundScalar& value, Index size) {
    using Common = std::common_type_t<Accumulator, BoundScalar>;
    return Common(value) >= Common(squared_norm_lower_bound<RealScalar>(size)) &&
           Common(value) <= Common(NumTraits<Accumulator>::highest());
  }

  template <typename X, typename Y>
  EIGEN_DEVICE_FUNC static bool isApprox(const X& x, const Y& y, const RealScalar& prec) {
    // Widening must not admit operands that cannot be subtracted, such as half and float.
    EIGEN_CHECK_BINARY_COMPATIBILITY(scalar_difference_op<typename X::Scalar EIGEN_COMMA typename Y::Scalar>,
                                     typename X::Scalar, typename Y::Scalar)
    const auto& wideX = x.template cast<approx_comparison_wide_t<X>>();
    const auto& wideY = y.template cast<approx_comparison_wide_t<Y>>();
    const Accumulator x2 = wideX.cwiseAbs2().sum();
    const Accumulator y2 = wideY.cwiseAbs2().sum();
    const Accumulator minimum = numext::mini(x2, y2);
    const Accumulator precision2 = Accumulator(prec) * Accumulator(prec);
    const Accumulator bound = precision2 * minimum;
    // Only the smaller norm enters the bound; overflow of the larger norm is harmless.
    if (safe_squared_norm(bound, x.size()) && minimum >= squared_norm_lower_bound<RealScalar>(x.size()) &&
        precision2 >= squared_norm_lower_bound<RealScalar>(1))
      return (wideX.matrix() - wideY.matrix()).cwiseAbs2().sum() <= bound;

    return isApprox_scaled(x, y, prec);
  }

  template <typename X, typename Y>
  EIGEN_DEVICE_FUNC static bool isMuchSmallerThan(const X& x, const Y& y, const RealScalar& prec) {
    using Common = CommonAccumulator<Y>;
    typename nested_eval<X, 2>::type nested(x);
    typename nested_eval<Y, 2>::type otherNested(y);
    const Accumulator x2 = nested.template cast<approx_comparison_wide_t<X>>().cwiseAbs2().sum();
    const auto y2 = otherNested.template cast<approx_comparison_wide_t<Y>>().cwiseAbs2().sum();
    const Accumulator precision2 = numext::abs2(Accumulator(prec));
    const Common bound = Common(precision2) * Common(y2);
    // A finite bound above the flushing error makes overflow/underflow of x2 harmless.
    if (safe_squared_norm(bound, x.size()) &&
        Common(y2) >= Common(squared_norm_lower_bound<typename Y::RealScalar>(y.size())) &&
        precision2 >= squared_norm_lower_bound<RealScalar>(1))
      return Common(x2) <= bound;
    return isMuchSmallerThan_scaled(nested, otherNested, prec);
  }

  template <typename X>
  EIGEN_DEVICE_FUNC static bool isMuchSmallerThan(const X& x, const RealScalar& y, const RealScalar& prec) {
    typename nested_eval<X, 2>::type nested(x);
    const Accumulator x2 = nested.template cast<approx_comparison_wide_t<X>>().cwiseAbs2().sum();
    const Accumulator bound = numext::abs2(Accumulator(prec) * Accumulator(y));
    if (safe_squared_norm(bound, x.size())) return x2 <= bound;
    return isMuchSmallerThan_scaled(nested, y, prec);
  }

 private:
  // Keep exponent scaling from inhibiting inlining of ordinary comparisons.
  template <typename X, typename Y>
  EIGEN_DEVICE_FUNC static EIGEN_DONT_INLINE bool isApprox_scaled(const X& xExpr, const Y& yExpr,
                                                                  const RealScalar& prec) {
    scaled_comparison_operand_t<X> x(xExpr);
    scaled_comparison_operand_t<Y> y(yExpr);
    const auto nx = scaled_comparison_norm(x);
    const auto ny = scaled_comparison_norm(y);
    if (!nx.isFinite() || !ny.isFinite()) return false;
    auto tolerance = nx <= ny ? nx : ny;
    tolerance.multiply(numext::abs(Accumulator(prec)));
    return scaled_comparison_distance(x, y) <= tolerance;
  }

  template <typename X, typename Y>
  EIGEN_DEVICE_FUNC static EIGEN_DONT_INLINE bool isMuchSmallerThan_scaled(const X& x, const Y& y,
                                                                           const RealScalar& prec) {
    using Common = CommonAccumulator<Y>;
    comparison_magnitude<Common> tolerance(scaled_comparison_norm(y));
    tolerance.multiply(numext::abs(Common(prec)));
    return scaled_comparison_norm(x) <= tolerance;
  }

  template <typename X>
  EIGEN_DEVICE_FUNC static EIGEN_DONT_INLINE bool isMuchSmallerThan_scaled(const X& x, const RealScalar& y,
                                                                           const RealScalar& prec) {
    comparison_magnitude<Accumulator> tolerance(numext::abs(Accumulator(y)));
    tolerance.multiply(numext::abs(Accumulator(prec)));
    return scaled_comparison_norm(x) <= tolerance;
  }
};

// Exponent scaling needs binary floating-point on both sides.
template <typename Derived, typename OtherDerived>
using approx_comparison_impl_t =
    approx_comparison_impl<typename Derived::Scalar, use_scaled_comparison<typename Derived::Scalar>::value &&
                                                         use_scaled_comparison<typename OtherDerived::Scalar>::value>;

template <typename Derived, typename OtherDerived, bool is_integer = NumTraits<typename Derived::Scalar>::IsInteger>
struct isApprox_selector {
  EIGEN_DEVICE_FUNC static bool run(const Derived& x, const OtherDerived& y, const typename Derived::RealScalar& prec) {
    typename internal::nested_eval<Derived, 2>::type nested(x);
    typename internal::nested_eval<OtherDerived, 2>::type otherNested(y);
    return approx_comparison_impl_t<Derived, OtherDerived>::isApprox(nested, otherNested, prec);
  }
};

template <typename Derived, typename OtherDerived>
struct isApprox_selector<Derived, OtherDerived, true> {
  EIGEN_DEVICE_FUNC static bool run(const Derived& x, const OtherDerived& y, const typename Derived::RealScalar&) {
    return x.matrix() == y.matrix();
  }
};

template <typename Derived, typename OtherDerived, bool is_integer = NumTraits<typename Derived::Scalar>::IsInteger>
struct isMuchSmallerThan_object_selector {
  EIGEN_DEVICE_FUNC static bool run(const Derived& x, const OtherDerived& y, const typename Derived::RealScalar& prec) {
    return approx_comparison_impl_t<Derived, OtherDerived>::isMuchSmallerThan(x, y, prec);
  }
};

template <typename Derived, typename OtherDerived>
struct isMuchSmallerThan_object_selector<Derived, OtherDerived, true> {
  EIGEN_DEVICE_FUNC static bool run(const Derived& x, const OtherDerived&, const typename Derived::RealScalar&) {
    return x.matrix() == Derived::Zero(x.rows(), x.cols()).matrix();
  }
};

template <typename Derived, bool is_integer = NumTraits<typename Derived::Scalar>::IsInteger>
struct isMuchSmallerThan_scalar_selector {
  EIGEN_DEVICE_FUNC static bool run(const Derived& x, const typename Derived::RealScalar& y,
                                    const typename Derived::RealScalar& prec) {
    return approx_comparison_impl<typename Derived::Scalar>::isMuchSmallerThan(x, y, prec);
  }
};

template <typename Derived>
struct isMuchSmallerThan_scalar_selector<Derived, true>
    : isMuchSmallerThan_object_selector<Derived, typename Derived::RealScalar, true> {};

}  // end namespace internal

/** \returns \c true if \c *this is approximately equal to \a other, within the precision
 * determined by \a prec.
 *
 * \note This is a relative norm comparison. Two vectors \f$ v \f$ and \f$ w \f$
 * are considered to be approximately equal within precision \f$ p \f$ if
 * \f[ \Vert v - w \Vert \leqslant p\,\min(\Vert v\Vert, \Vert w\Vert). \f]
 * For matrices, the comparison is done using the Hilbert-Schmidt norm (aka Frobenius norm
 * L2 norm).
 *
 * \a prec is a relative tolerance chosen by the caller, not a bound on floating-point rounding error.
 * The default, NumTraits<Scalar>::dummy_precision(), does not account for the operation, dimensions,
 * or conditioning of the problem. Choose an explicit tolerance when checking numerical accuracy.
 * A norm comparison also does not bound the relative error of each coefficient.
 *
 * \note Because of the multiplicativeness of this comparison, one can't use this function
 * to check whether \c *this is approximately equal to the zero matrix or vector.
 * Indeed, \c isApprox(zero) returns false unless \c *this itself is exactly the zero matrix
 * or vector. If you want to test whether \c *this is small relative to a reference norm,
 * use isMuchSmallerThan() instead.
 *
 * \sa isMuchSmallerThan(), isZero()
 */
template <typename Derived>
template <typename OtherDerived>
EIGEN_DEVICE_FUNC constexpr bool DenseBase<Derived>::isApprox(const DenseBase<OtherDerived>& other,
                                                              const RealScalar& prec) const {
  return internal::isApprox_selector<Derived, OtherDerived>::run(derived(), other.derived(), prec);
}

/** \returns \c true if the norm of \c *this is much smaller than \a other,
 * within the precision determined by \a prec.
 *
 * \note This is a relative norm comparison. A vector \f$ v \f$ is
 * considered to be much smaller than \f$ x \f$ within precision \f$ p \f$ if
 * \f[ \Vert v \Vert \leqslant p\,\vert x\vert. \f]
 *
 * For matrices, the comparison is done using the Hilbert-Schmidt norm. For this reason,
 * the value of the reference scalar \a other should come from the Hilbert-Schmidt norm
 * of a reference matrix of same dimensions.
 *
 * \sa isApprox(), isMuchSmallerThan(const DenseBase<OtherDerived>&, RealScalar) const
 */
template <typename Derived>
EIGEN_DEVICE_FUNC constexpr bool DenseBase<Derived>::isMuchSmallerThan(const typename NumTraits<Scalar>::Real& other,
                                                                       const RealScalar& prec) const {
  return internal::isMuchSmallerThan_scalar_selector<Derived>::run(derived(), other, prec);
}

/** \returns \c true if the norm of \c *this is much smaller than the norm of \a other,
 * within the precision determined by \a prec.
 *
 * \note This is a relative norm comparison. A vector \f$ v \f$ is
 * considered to be much smaller than a vector \f$ w \f$ within precision \f$ p \f$ if
 * \f[ \Vert v \Vert \leqslant p\,\Vert w\Vert. \f]
 * For matrices, the comparison is done using the Hilbert-Schmidt norm.
 *
 * \sa isApprox(), isMuchSmallerThan(const RealScalar&, RealScalar) const
 */
template <typename Derived>
template <typename OtherDerived>
EIGEN_DEVICE_FUNC constexpr bool DenseBase<Derived>::isMuchSmallerThan(const DenseBase<OtherDerived>& other,
                                                                       const RealScalar& prec) const {
  return internal::isMuchSmallerThan_object_selector<Derived, OtherDerived>::run(derived(), other.derived(), prec);
}

}  // end namespace Eigen

#endif  // EIGEN_APPROX_COMPARISONS_H
