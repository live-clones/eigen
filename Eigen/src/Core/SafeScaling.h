// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_SAFE_SCALING_H
#define EIGEN_SAFE_SCALING_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {
namespace internal {

// Select factors for a positive finite scale. Supported binary floating-point scalars use normal powers of two;
// other scalar types use division for matrix scaling and can request a clamped reciprocal for stableNorm(). General
// scaling rounds down so it does not discard a representable tail that division by the original value would preserve.
// Stable reductions separately round up to keep scaled magnitudes at most one. At the upper exponent boundary, the
// scale is clamped to keep its reciprocal normal; for the standard binary formats this leaves magnitudes below four.
template <typename Scalar>
struct supports_power_of_two_scaling
    : bool_constant<(std::is_same<Scalar, float>::value || std::is_same<Scalar, double>::value) &&
                    std::numeric_limits<Scalar>::is_iec559 && std::numeric_limits<Scalar>::radix == 2 &&
                    (sizeof(Scalar) == sizeof(numext::uint32_t) || sizeof(Scalar) == sizeof(numext::uint64_t))> {};

#if !defined(EIGEN_GPU_COMPILE_PHASE)
template <>
struct supports_power_of_two_scaling<long double> : bool_constant<std::numeric_limits<long double>::radix == 2> {};
#endif

template <>
struct supports_power_of_two_scaling<half> : true_type {};

template <>
struct supports_power_of_two_scaling<bfloat16> : true_type {};

template <typename Scalar, bool = supports_power_of_two_scaling<Scalar>::value>
struct safe_scaling;

template <typename Scalar>
struct safe_scaling_factors {
  Scalar scale = Scalar(1);
  // Division-only arithmetic scaling leaves this unused reciprocal at one.
  Scalar invScale = Scalar(1);
};

template <typename Scalar>
EIGEN_DEVICE_FUNC EIGEN_DONT_INLINE typename binary_floating_point_traits<Scalar>::Bits
scale_binary_bits_by_power_of_two(const typename binary_floating_point_traits<Scalar>::Bits valueBits,
                                  const typename binary_floating_point_traits<Scalar>::Bits factorExponentBits) {
  using Binary = binary_floating_point_traits<Scalar>;
  using Bits = typename Binary::Bits;

  const Bits valueExponentBits = valueBits & Binary::kExponentMask;
  const Bits fraction = valueBits & Binary::kFractionMask;
  if (valueExponentBits == Binary::kExponentMask || (valueExponentBits == 0 && fraction == 0)) return valueBits;

  Bits significand = fraction;
  int exponent = int(valueExponentBits >> Binary::kFractionBits);
  if (exponent == 0) {
    exponent = 1;
    while (significand < Binary::kExponentUnit) {
      significand <<= 1;
      --exponent;
    }
  }
  exponent += int(factorExponentBits >> Binary::kFractionBits) - Binary::kExponentBias;
  eigen_internal_assert(exponent > 0 && exponent < int(Binary::kExponentMask >> Binary::kFractionBits));
  return (valueBits & Binary::kSignBit) | (Bits(exponent) << Binary::kFractionBits) |
         (significand & Binary::kFractionMask);
}

// Recovery-only multiplication: factor is a positive normal power of two and every nonzero finite result must be
// normal and finite. Our tiny-input scaling factors guarantee this; no output rounding is needed. Keep the integer-only
// ABI so compilers cannot replace reconstruction with FTZ/DAZ-sensitive floating-point arithmetic.
template <typename Scalar>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar scale_binary_by_power_of_two(const Scalar& value, const Scalar& factor) {
  using Binary = binary_floating_point_traits<Scalar>;
  using Bits = typename Binary::Bits;
  const Bits factorExponentBits = Binary::bits(factor);
  eigen_internal_assert(factorExponentBits != 0 && factorExponentBits < Binary::kExponentMask &&
                        (factorExponentBits & ~Binary::kExponentMask) == 0);
  return numext::bit_cast<Scalar>(scale_binary_bits_by_power_of_two<Scalar>(Binary::bits(value), factorExponentBits));
}

template <typename Scalar>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE std::complex<Scalar> scale_binary_by_power_of_two(
    const std::complex<Scalar>& value, const Scalar& factor) {
  return std::complex<Scalar>(scale_binary_by_power_of_two(value.real(), factor),
                              scale_binary_by_power_of_two(value.imag(), factor));
}

template <typename FactorScalar>
struct scale_by_power_of_two_op {
  EIGEN_DEVICE_FUNC explicit scale_by_power_of_two_op(const FactorScalar& factor) : m_factor(factor) {}

  template <typename CoeffScalar>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffScalar operator()(const CoeffScalar& value) const {
    return scale_binary_by_power_of_two(value, m_factor);
  }

  FactorScalar m_factor;
};

template <typename FactorScalar>
struct functor_traits<scale_by_power_of_two_op<FactorScalar>> {
  // Integer significand recovery is scalar and calls an out-of-line helper.
  static constexpr int Cost = 10 * NumTraits<FactorScalar>::MulCost;
  static constexpr bool PacketAccess = false;
  static constexpr bool IsRepeatable = true;
};

template <typename FactorScalar, typename CoeffScalar>
struct use_subnormal_preserving_scaling
    : bool_constant<
          (std::is_same<FactorScalar, float>::value &&
           (std::is_same<CoeffScalar, float>::value || std::is_same<CoeffScalar, std::complex<float>>::value)) ||
          (std::is_same<FactorScalar, double>::value &&
           (std::is_same<CoeffScalar, double>::value || std::is_same<CoeffScalar, std::complex<double>>::value))> {};

template <typename Scalar, bool IsPowerOfTwo_>
struct safe_scaling_operations {
 private:
  using Factors = safe_scaling_factors<Scalar>;

  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE bool needs_subnormal_recovery(const Scalar& value) {
    using Binary = binary_floating_point_traits<Scalar>;
    using Bits = typename Binary::Bits;
    constexpr int kDirectScaleExponent = std::numeric_limits<Scalar>::min_exponent + Binary::kFractionBits - 1;
    constexpr Bits kThreshold = Bits(kDirectScaleExponent + Binary::kExponentBias) << Binary::kFractionBits;
    return Binary::magnitude(value) - Bits(1) < kThreshold - Bits(1);
  }

  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE bool is_identity(const Factors& factors) {
    return factors.scale == Scalar(1) && factors.invScale == Scalar(1);
  }

  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE bool has_normal_reciprocal(const Scalar&, const Scalar&, false_type) {
    return false;
  }

  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE bool has_normal_reciprocal(const Scalar& value, const Scalar& normalMin,
                                                                          true_type) {
    return value >= normalMin && value <= Scalar(1) / normalMin;
  }

  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE bool is_positive_finite(const Scalar& value, false_type) {
    return value > Scalar(0) && value <= NumTraits<Scalar>::highest();
  }

  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE bool is_positive_finite(const Scalar& value, true_type) {
    using Binary = binary_floating_point_traits<Scalar>;
    const typename Binary::Bits bits = Binary::bits(value);
    return bits > 0 && bits < Binary::kExponentMask;
  }

  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Factors select_factors(const Scalar& maxCoeff) {
    using IsIeeeBinary = bool_constant<std::is_same<Scalar, float>::value || std::is_same<Scalar, double>::value>;
    if (!is_positive_finite(maxCoeff, IsIeeeBinary())) return Factors{};
    return safe_scaling<Scalar, IsPowerOfTwo_>::compute_floor_factors(maxCoeff);
  }

  template <typename Src, typename Func>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void with_scaled_impl(const Src& src, const Scalar&,
                                                                     const Factors& factors, const Func& func,
                                                                     false_type) {
    if (is_identity(factors))
      func(src);
    else EIGEN_IF_CONSTEXPR (IsPowerOfTwo_)
      func(src * factors.invScale);
    else
      func(src / factors.scale);
  }

  template <typename Src, typename Func>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void with_scaled_impl(const Src& src, const Scalar& maxCoeff,
                                                                     const Factors& factors, const Func& func,
                                                                     true_type) {
    if (!is_identity(factors) && needs_subnormal_recovery(maxCoeff))
      func(src.unaryExpr(scale_by_power_of_two_op<Scalar>(factors.invScale)));
    else
      with_scaled_impl(src, maxCoeff, factors, func, false_type());
  }

 public:
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Factors
  compute_ceiling_factors_with_normal_reciprocal(const Scalar& value) {
    Factors factors = safe_scaling<Scalar, IsPowerOfTwo_>::compute_ceiling_factors(value);
    EIGEN_IF_CONSTEXPR (!IsPowerOfTwo_) {
      if (factors.invScale > NumTraits<Scalar>::highest()) {
        factors.invScale = NumTraits<Scalar>::highest();
        factors.scale = Scalar(1) / factors.invScale;
      }
    }
    return factors;
  }

  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE bool try_compute_ceiling_factors_with_normal_reciprocal(
      const Scalar& value, const Scalar& normalMin, Factors& factors) {
    if (!IsPowerOfTwo_ &&
        !has_normal_reciprocal(value, normalMin, bool_constant<std::is_floating_point<Scalar>::value>()))
      return false;
    factors = compute_ceiling_factors_with_normal_reciprocal(value);
    return true;
  }

  // Dispatch once per expression; ordinary scaling retains packet access. The callback consumes the lazy
  // expression before its operands expire, without requiring a common C++ type for the different paths.
  template <typename Src, typename Func>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Factors with_scaled(const Src& src, const Scalar& maxCoeff,
                                                                   const Func& func) {
    const Factors factors = select_factors(maxCoeff);
    constexpr bool kPreserveSubnormalInputs =
        IsPowerOfTwo_ && use_subnormal_preserving_scaling<Scalar, typename Src::Scalar>::value;
    with_scaled_impl(src, maxCoeff, factors, func, bool_constant<kPreserveSubnormalInputs>());
    return factors;
  }

  template <typename Dest, typename Src>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void unscale_to(Dest&& dest, const Src& src, const Factors& factors) {
    if (factors.scale == Scalar(1)) {
      dest = src;
      return;
    }
    dest = src * factors.scale;
  }

  template <typename ValueType>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void unscale_in_place(ValueType& value, const Factors& factors) {
    if (factors.scale == Scalar(1)) return;
    value *= factors.scale;
  }

  template <typename MatrixType>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void scale_in_place(MatrixType& matrix, const Scalar& maxCoeff,
                                                                   const Factors& factors) {
    if (!is_identity(factors)) scale_to(matrix, matrix, maxCoeff, factors);
  }

  template <typename Dest, typename Src>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void scale_to(Dest& dest, const Src& src, const Scalar& maxCoeff,
                                                             const Factors& factors) {
    constexpr bool kPreserveSubnormalInputs =
        IsPowerOfTwo_ && use_subnormal_preserving_scaling<Scalar, typename Src::Scalar>::value;
    with_scaled_impl(
        src, maxCoeff, factors, [&](const auto& scaled) { dest = scaled; }, bool_constant<kPreserveSubnormalInputs>());
  }

  template <typename Dest, typename Src>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Factors scale_to(Dest& dest, const Src& src, const Scalar& maxCoeff) {
    const Factors factors = select_factors(maxCoeff);
    scale_to(dest, src, maxCoeff, factors);
    return factors;
  }
};

template <typename Scalar, bool>
struct safe_scaling : safe_scaling_operations<Scalar, false> {
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE safe_scaling_factors<Scalar> compute_floor_factors(const Scalar& value) {
    return {value, Scalar(1)};
  }

  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE safe_scaling_factors<Scalar> compute_ceiling_factors(
      const Scalar& value) {
    return {value, Scalar(1) / value};
  }
};

template <typename Scalar>
struct safe_scaling<Scalar, true> : safe_scaling_operations<Scalar, true> {
 private:
  using Binary = binary_floating_point_traits<Scalar>;
  using Bits = typename Binary::Bits;

  template <bool RoundUp>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE safe_scaling_factors<Scalar> compute_power_of_two_factors(
      const Scalar& value) {
    constexpr Bits kMaxScale = Binary::kMaxFiniteExponentBits - Binary::kExponentUnit;
    Bits scaleBits = RoundUp ? numext::bit_cast<Bits>(numext::ceil_power_of_two(value))
                             : Binary::bits(value) & Binary::kExponentMask;
    if (scaleBits < Binary::kExponentUnit) scaleBits = Binary::kExponentUnit;
    if (scaleBits > kMaxScale) scaleBits = kMaxScale;
    const Bits invScaleBits = Bits(Binary::kMaxFiniteExponentBits - scaleBits);
    return {numext::bit_cast<Scalar>(scaleBits), numext::bit_cast<Scalar>(invScaleBits)};
  }

 public:
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE safe_scaling_factors<Scalar> compute_floor_factors(const Scalar& value) {
    return compute_power_of_two_factors<false>(value);
  }

  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE safe_scaling_factors<Scalar> compute_ceiling_factors(
      const Scalar& value) {
    return compute_power_of_two_factors<true>(value);
  }
};

#if !defined(EIGEN_GPU_COMPILE_PHASE)
template <>
struct safe_scaling<long double, true> : safe_scaling_operations<long double, true> {
  static EIGEN_STRONG_INLINE safe_scaling_factors<long double> compute_floor_factors(const long double& value) {
    EIGEN_USING_STD(frexp);
    int exponent = 0;
    frexp(value, &exponent);
    return compute_factors_from_exponent(exponent - 1);
  }

  static EIGEN_STRONG_INLINE safe_scaling_factors<long double> compute_ceiling_factors(const long double& value) {
    EIGEN_USING_STD(frexp);
    int exponent = 0;
    const long double fraction = frexp(value, &exponent);
    if (fraction == 0.5L) --exponent;
    return compute_factors_from_exponent(exponent);
  }

 private:
  static EIGEN_STRONG_INLINE safe_scaling_factors<long double> compute_factors_from_exponent(int exponent) {
    constexpr int kMinNormalExponent = std::numeric_limits<long double>::min_exponent - 1;
    constexpr int kMaxNormalExponent = std::numeric_limits<long double>::max_exponent - 1;
    constexpr int kMinScaleExponent =
        kMinNormalExponent > -kMaxNormalExponent ? kMinNormalExponent : -kMaxNormalExponent;
    constexpr int kMaxScaleExponent =
        kMaxNormalExponent < -kMinNormalExponent ? kMaxNormalExponent : -kMinNormalExponent;
    if (exponent < kMinScaleExponent) exponent = kMinScaleExponent;
    if (exponent > kMaxScaleExponent) exponent = kMaxScaleExponent;

    EIGEN_USING_STD(ldexp);
    const long double scale = ldexp(1.0L, exponent);
    return {scale, 1.0L / scale};
  }
};
#endif

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_SAFE_SCALING_H
