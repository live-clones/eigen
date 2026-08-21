// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_SAFE_SCALING_H
#define EIGEN_SAFE_SCALING_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {
namespace internal {

// Select reciprocal factors for a positive finite scale. Supported binary floating-point scalars use normal powers of
// two; other scalar types retain arithmetic scaling and can request a clamped reciprocal for stableNorm(). General
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
  Scalar invScale = Scalar(1);

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool isIdentity() const { return scale == Scalar(1) && invScale == Scalar(1); }
};

template <typename Derived, typename FactorScalar>
class safe_scaled_expression;

template <typename Scalar, bool = std::is_floating_point<Scalar>::value>
struct safe_scaling_has_normal_reciprocal {
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE bool run(const Scalar&, const Scalar&) { return false; }
};

template <typename Scalar>
struct safe_scaling_has_normal_reciprocal<Scalar, true> {
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE bool run(const Scalar& value, const Scalar& normalMin) {
    return value >= normalMin && value <= Scalar(1) / normalMin;
  }
};

template <typename Scalar>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar scale_binary_preserving_subnormal_input(const Scalar& value,
                                                                                     const Scalar& factor) {
  using Bits = typename numext::get_integer_by_size<sizeof(Scalar)>::unsigned_type;
  constexpr int kFractionBits = std::numeric_limits<Scalar>::digits - 1;
  constexpr Bits kSignBit = Bits(1) << (sizeof(Scalar) * CHAR_BIT - 1);
  constexpr Bits kExponentMask = ((Bits(1) << (sizeof(Scalar) * CHAR_BIT - std::numeric_limits<Scalar>::digits)) - 1)
                                 << kFractionBits;
  const Bits bits = numext::bit_cast<Bits>(value);
  const Bits significand = bits & ~kSignBit;
  if ((bits & kExponentMask) != 0 || significand == 0) return value * factor;

  const Bits factorBits = numext::bit_cast<Bits>(factor);
  const int factorExponent =
      int((factorBits & kExponentMask) >> kFractionBits) - (std::numeric_limits<Scalar>::max_exponent - 1);
  EIGEN_USING_STD(ldexp);
  Scalar result = ldexp(Scalar(significand), factorExponent + std::numeric_limits<Scalar>::min_exponent -
                                                 std::numeric_limits<Scalar>::digits);
  if ((bits & kSignBit) != 0) result = -result;
  return result;
}

template <typename Scalar>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE std::complex<Scalar> scale_binary_preserving_subnormal_input(
    const std::complex<Scalar>& value, const Scalar& factor) {
  return std::complex<Scalar>(scale_binary_preserving_subnormal_input(value.real(), factor),
                              scale_binary_preserving_subnormal_input(value.imag(), factor));
}

template <typename FactorScalar, typename CoeffScalar>
struct use_subnormal_preserving_scaling
    : bool_constant<
          (std::is_same<FactorScalar, float>::value &&
           (std::is_same<CoeffScalar, float>::value || std::is_same<CoeffScalar, std::complex<float>>::value)) ||
          (std::is_same<FactorScalar, double>::value &&
           (std::is_same<CoeffScalar, double>::value || std::is_same<CoeffScalar, std::complex<double>>::value))> {};

template <typename Dest, typename Src, typename = void>
struct safe_scaling_destination_traits {
  using Scalar = remove_all_t<Src>;
  static constexpr bool IsExpression = false;
};

template <typename Dest, typename Src>
struct safe_scaling_destination_traits<
    Dest, Src,
    void_t<typename std::remove_reference_t<Dest>::Scalar, decltype(std::declval<Dest&>().rows()),
           decltype(std::declval<Dest&>().cols()), decltype(std::declval<Dest&>().coeffRef(Index(0), Index(0)))>> {
  using Scalar = typename std::remove_reference_t<Dest>::Scalar;
  static constexpr bool IsExpression = true;
};

template <typename Scalar, bool IsPowerOfTwo_>
struct safe_scaling_operations {
  using Factors = safe_scaling_factors<Scalar>;

 private:
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE bool can_scale_directly(const Scalar& maxCoeff) {
    using Bits = typename numext::get_integer_by_size<sizeof(Scalar)>::unsigned_type;
    constexpr int kFractionBits = std::numeric_limits<Scalar>::digits - 1;
    constexpr int kExponentBias = std::numeric_limits<Scalar>::max_exponent - 1;
    constexpr int kDirectScaleExponent = std::numeric_limits<Scalar>::min_exponent + kFractionBits - 1;
    constexpr Bits kDirectScaleThreshold = Bits(kDirectScaleExponent + kExponentBias) << kFractionBits;
    return maxCoeff >= numext::bit_cast<Scalar>(kDirectScaleThreshold);
  }

  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Factors select_factors(const Scalar& maxCoeff) {
    if (!(numext::isfinite)(maxCoeff) || numext::is_exactly_zero(maxCoeff)) return Factors{};
    return safe_scaling<Scalar, IsPowerOfTwo_>::compute_factors(maxCoeff);
  }

  template <typename MatrixType>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void scale_matrix_column(MatrixType& matrix, const Index col,
                                                                        const Index firstRow, const Index endRow,
                                                                        const Scalar& factor) {
    for (Index row = firstRow; row < endRow; ++row) {
      matrix.coeffRef(row, col) = scale_binary_preserving_subnormal_input(matrix.coeff(row, col), factor);
    }
  }

  template <typename MatrixType>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void scale_matrix_in_place_impl(MatrixType& matrix, const Scalar&,
                                                                               const Scalar& scale,
                                                                               const Scalar& invScale,
                                                                               bool_constant<false>) {
    EIGEN_IF_CONSTEXPR (IsPowerOfTwo_)
      matrix *= invScale;
    else
      matrix /= scale;
  }

  template <typename MatrixType>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void scale_matrix_in_place_impl(MatrixType& matrix,
                                                                               const Scalar& maxCoeff, const Scalar&,
                                                                               const Scalar& invScale,
                                                                               bool_constant<true>) {
    // Recover subnormal inputs through their integer significands before FTZ/DAZ modes can flush them.
    if (can_scale_directly(maxCoeff)) {
      matrix *= invScale;
      return;
    }
    for (Index col = 0; col < matrix.cols(); ++col) {
      scale_matrix_column(matrix, col, 0, matrix.rows(), invScale);
    }
  }

  template <typename MatrixType, unsigned int Mode>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void scale_matrix_in_place_impl(TriangularView<MatrixType, Mode>& matrix,
                                                                               const Scalar& maxCoeff, const Scalar&,
                                                                               const Scalar& invScale,
                                                                               bool_constant<true>) {
    if (can_scale_directly(maxCoeff)) {
      matrix *= invScale;
      return;
    }
    constexpr bool kLower = (Mode & Lower) == Lower;
    constexpr bool kExplicitDiagonal = (Mode & (UnitDiag | ZeroDiag)) == 0;
    for (Index col = 0; col < matrix.cols(); ++col) {
      const Index firstRow = kLower ? col + Index(!kExplicitDiagonal) : 0;
      const Index endRow = kLower ? matrix.rows() : numext::mini(matrix.rows(), col + Index(kExplicitDiagonal));
      scale_matrix_column(matrix, col, firstRow, endRow, invScale);
    }
  }

  template <typename Dest, typename Src, bool IsPowerOfTwo, typename IsExpression>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void scale_to_impl(Dest& dest, const Src& src, const Scalar&,
                                                                  const Scalar& scale, const Scalar& invScale,
                                                                  bool_constant<false>, bool_constant<IsPowerOfTwo>,
                                                                  IsExpression) {
    EIGEN_IF_CONSTEXPR (IsPowerOfTwo)
      dest = src * invScale;
    else
      dest = src / scale;
  }

  template <typename Dest, typename Src>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void scale_to_impl(Dest& dest, const Src& src, const Scalar& maxCoeff,
                                                                  const Scalar& scale, const Scalar& invScale,
                                                                  bool_constant<true>, bool_constant<true>,
                                                                  bool_constant<true>) {
    if (can_scale_directly(maxCoeff)) {
      dest = src * invScale;
      return;
    }
    dest = src;
    scale_matrix_in_place_impl(dest, maxCoeff, scale, invScale, bool_constant<true>());
  }

  template <typename Dest, typename Src>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void scale_to_impl(Dest& dest, const Src& src, const Scalar& maxCoeff,
                                                                  const Scalar&, const Scalar& invScale,
                                                                  bool_constant<true>, bool_constant<true>,
                                                                  bool_constant<false>) {
    if (can_scale_directly(maxCoeff)) {
      dest = src * invScale;
    } else {
      dest = scale_binary_preserving_subnormal_input(src, invScale);
    }
  }

  template <typename MatrixType>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void scale_in_place_with_factors(MatrixType& matrix,
                                                                                const Scalar& maxCoeff,
                                                                                const Factors& factors) {
    if (factors.isIdentity()) return;
    using CoeffScalar = typename MatrixType::Scalar;
    constexpr bool kPreserveSubnormalInputs =
        IsPowerOfTwo_ && use_subnormal_preserving_scaling<Scalar, CoeffScalar>::value;
    scale_matrix_in_place_impl(matrix, maxCoeff, factors.scale, factors.invScale,
                               bool_constant<kPreserveSubnormalInputs>());
  }

 public:
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Factors
  compute_ceiling_factors_with_normal_reciprocal(const Scalar& value);

  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE bool try_compute_ceiling_factors_with_normal_reciprocal(
      const Scalar& value, const Scalar& normalMin, Factors& factors) {
    if (!IsPowerOfTwo_ && !safe_scaling_has_normal_reciprocal<Scalar>::run(value, normalMin)) return false;
    factors = safe_scaling<Scalar, IsPowerOfTwo_>::compute_ceiling_factors(value);
    return true;
  }

  template <typename Derived>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE safe_scaled_expression<Derived, Scalar> scaled_expression(
      const Derived& value, const Scalar& maxCoeff, Factors& factors);

  template <typename Dest, typename Src>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void unscale_to(Dest&& dest, const Src& src, const Factors& factors) {
    if (factors.isIdentity()) {
      dest = src;
      return;
    }
    dest = src * factors.scale;
  }

  template <typename ValueType>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void unscale_in_place(ValueType& value, const Factors& factors) {
    if (factors.isIdentity()) return;
    value *= factors.scale;
  }

  template <typename MatrixType>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Factors scale_in_place(MatrixType& matrix, const Scalar& maxCoeff) {
    const Factors factors = select_factors(maxCoeff);
    scale_in_place_with_factors(matrix, maxCoeff, factors);
    return factors;
  }

  template <typename MatrixType>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void scale_in_place(MatrixType& matrix, const Scalar& maxCoeff,
                                                                   const Factors& factors) {
    scale_in_place_with_factors(matrix, maxCoeff, factors);
  }

  template <typename Dest, typename Src>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void scale_to(Dest& dest, const Src& src, const Scalar& maxCoeff,
                                                             const Factors& factors) {
    if (factors.isIdentity()) {
      dest = src;
      return;
    }
    using DestinationTraits = safe_scaling_destination_traits<Dest, Src>;
    using CoeffScalar = typename DestinationTraits::Scalar;
    constexpr bool kPreserveSubnormalInputs =
        IsPowerOfTwo_ && use_subnormal_preserving_scaling<Scalar, CoeffScalar>::value;
    scale_to_impl(dest, src, maxCoeff, factors.scale, factors.invScale, bool_constant<kPreserveSubnormalInputs>(),
                  bool_constant<IsPowerOfTwo_>(), bool_constant<DestinationTraits::IsExpression>());
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
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE safe_scaling_factors<Scalar> compute_factors(const Scalar& value) {
    return {value, Scalar(1) / value};
  }

  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE safe_scaling_factors<Scalar> compute_ceiling_factors(
      const Scalar& value) {
    return compute_factors(value);
  }
};

template <typename Scalar>
struct safe_scaling<Scalar, true> : safe_scaling_operations<Scalar, true> {
 private:
  using Bits = typename numext::get_integer_by_size<sizeof(Scalar)>::unsigned_type;

  template <bool RoundUp>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE safe_scaling_factors<Scalar> compute_power_of_two_factors(
      const Scalar& value) {
    constexpr int kFractionBits = std::numeric_limits<Scalar>::digits - 1;
    constexpr int kExponentBits = int(sizeof(Scalar) * CHAR_BIT) - std::numeric_limits<Scalar>::digits;
    constexpr Bits kExponentUnit = Bits(1) << kFractionBits;
    constexpr Bits kExponentMask = ((Bits(1) << kExponentBits) - 1) << kFractionBits;
    constexpr Bits kMaxFiniteExponent = kExponentMask - kExponentUnit;
    constexpr Bits kMaxScale = kMaxFiniteExponent - kExponentUnit;
    Bits scaleBits = RoundUp ? numext::bit_cast<Bits>(numext::ceil_power_of_two(value))
                             : numext::bit_cast<Bits>(value) & kExponentMask;
    if (scaleBits < kExponentUnit) scaleBits = kExponentUnit;
    if (scaleBits > kMaxScale) scaleBits = kMaxScale;
    const Bits invScaleBits = Bits(kMaxFiniteExponent - scaleBits);
    return {numext::bit_cast<Scalar>(scaleBits), numext::bit_cast<Scalar>(invScaleBits)};
  }

 public:
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE safe_scaling_factors<Scalar> compute_factors(const Scalar& value) {
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
  static EIGEN_STRONG_INLINE safe_scaling_factors<long double> compute_factors(const long double& value) {
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

template <typename Scalar, bool IsPowerOfTwo_>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE safe_scaling_factors<Scalar>
safe_scaling_operations<Scalar, IsPowerOfTwo_>::compute_ceiling_factors_with_normal_reciprocal(const Scalar& value) {
  safe_scaling_factors<Scalar> factors = safe_scaling<Scalar, IsPowerOfTwo_>::compute_ceiling_factors(value);
  EIGEN_IF_CONSTEXPR (!IsPowerOfTwo_) {
    if (factors.invScale > NumTraits<Scalar>::highest()) {
      factors.invScale = NumTraits<Scalar>::highest();
      factors.scale = Scalar(1) / factors.invScale;
    }
  }
  return factors;
}

template <typename Derived, typename FactorScalar>
struct traits<safe_scaled_expression<Derived, FactorScalar>> {
  using ReturnType = typename plain_matrix_type<Derived>::type;
};

template <typename Derived, typename FactorScalar>
class safe_scaled_expression : public ReturnByValue<safe_scaled_expression<Derived, FactorScalar>> {
 public:
  using Nested = typename ref_selector<Derived>::type;
  using ScaledAdjointReturnType = typename Derived::AdjointReturnType;

  EIGEN_DEVICE_FUNC safe_scaled_expression(const Derived& value, const FactorScalar& maxCoeff,
                                           const safe_scaling_factors<FactorScalar>& factors)
      : m_value(value), m_maxCoeff(maxCoeff), m_factors(factors) {}

  EIGEN_DEVICE_FUNC constexpr Index rows() const noexcept { return m_value.rows(); }
  EIGEN_DEVICE_FUNC constexpr Index cols() const noexcept { return m_value.cols(); }

  template <typename Dest>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void evalTo(Dest& dest) const {
    safe_scaling<FactorScalar>::scale_to(dest, m_value, m_maxCoeff, m_factors);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE safe_scaled_expression<ScaledAdjointReturnType, FactorScalar> adjoint() const {
    return safe_scaled_expression<ScaledAdjointReturnType, FactorScalar>(m_value.adjoint(), m_maxCoeff, m_factors);
  }

 private:
  Nested m_value;
  FactorScalar m_maxCoeff;
  safe_scaling_factors<FactorScalar> m_factors;
};

template <typename Scalar, bool IsPowerOfTwo_>
template <typename Derived>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE safe_scaled_expression<Derived, Scalar>
safe_scaling_operations<Scalar, IsPowerOfTwo_>::scaled_expression(const Derived& value, const Scalar& maxCoeff,
                                                                  Factors& factors) {
  factors = select_factors(maxCoeff);
  return safe_scaled_expression<Derived, Scalar>(value, maxCoeff, factors);
}

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_SAFE_SCALING_H
