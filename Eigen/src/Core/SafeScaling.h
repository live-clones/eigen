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
};

template <typename Derived, typename FactorScalar>
class safe_scaled_expression;

// Multiply by a positive normal power of two through the IEEE representation so FTZ/DAZ cannot consume a subnormal
// input or result.
template <typename Scalar>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar scale_binary_by_power_of_two(const Scalar& value, const Scalar& factor) {
  using Binary = binary_floating_point_traits<Scalar>;
  using Bits = typename Binary::Bits;

  const Bits valueBits = Binary::bits(value);
  const Bits valueExponentBits = valueBits & Binary::kExponentMask;
  const Bits fraction = valueBits & Binary::kFractionMask;
  if (valueExponentBits == Binary::kExponentMask || (valueExponentBits == 0 && fraction == 0)) return value;

  const Bits factorExponentBits = Binary::bits(factor) & Binary::kExponentMask;
  if (factorExponentBits == 0 || factorExponentBits == Binary::kExponentMask) return value * factor;

  Bits significand = fraction;
  int exponent = std::numeric_limits<Scalar>::min_exponent - std::numeric_limits<Scalar>::digits;
  if (valueExponentBits != 0) {
    significand |= Binary::kExponentUnit;
    exponent = int(valueExponentBits >> Binary::kFractionBits) - Binary::kExponentBias - Binary::kFractionBits;
  }
  exponent += int(factorExponentBits >> Binary::kFractionBits) - Binary::kExponentBias;

  int highestBit = 0;
  for (Bits bits = significand; bits > Bits(1); bits >>= 1) ++highestBit;
  const int resultExponent = exponent + highestBit;
  const Bits sign = valueBits & Binary::kSignBit;
  if (resultExponent > Binary::kMaxExponent) return numext::bit_cast<Scalar>(sign | Binary::kExponentMask);
  if (resultExponent >= Binary::kMinExponent) {
    const Bits normalized = significand << (Binary::kFractionBits - highestBit);
    const Bits resultExponentBits = Bits(resultExponent + Binary::kExponentBias) << Binary::kFractionBits;
    return numext::bit_cast<Scalar>(sign | resultExponentBits | (normalized & Binary::kFractionMask));
  }

  const int subnormalExponent = std::numeric_limits<Scalar>::min_exponent - std::numeric_limits<Scalar>::digits;
  const int leftShift = exponent - subnormalExponent;
  Bits rounded = 0;
  if (leftShift >= 0) {
    rounded = significand << leftShift;
  } else {
    const int rightShift = -leftShift;
    if (rightShift <= std::numeric_limits<Scalar>::digits) {
      rounded = significand >> rightShift;
      const Bits remainderMask = (Bits(1) << rightShift) - 1;
      const Bits remainder = significand & remainderMask;
      const Bits halfway = Bits(1) << (rightShift - 1);
      if (remainder > halfway || (remainder == halfway && (rounded & Bits(1)) != 0)) ++rounded;
    }
  }
  return numext::bit_cast<Scalar>(sign | rounded);
}

template <typename Scalar>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE std::complex<Scalar> scale_binary_by_power_of_two(
    const std::complex<Scalar>& value, const Scalar& factor) {
  return std::complex<Scalar>(scale_binary_by_power_of_two(value.real(), factor),
                              scale_binary_by_power_of_two(value.imag(), factor));
}

template <typename FactorScalar, typename CoeffScalar>
struct use_subnormal_preserving_scaling
    : bool_constant<
          (std::is_same<FactorScalar, float>::value &&
           (std::is_same<CoeffScalar, float>::value || std::is_same<CoeffScalar, std::complex<float>>::value)) ||
          (std::is_same<FactorScalar, double>::value &&
           (std::is_same<CoeffScalar, double>::value || std::is_same<CoeffScalar, std::complex<double>>::value))> {};

template <typename Scalar, bool = std::is_same<Scalar, float>::value || std::is_same<Scalar, double>::value>
struct safe_scaling_needs_subnormal_recovery {
 private:
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Scalar threshold() {
    return (std::numeric_limits<Scalar>::min)() / NumTraits<Scalar>::epsilon();
  }

 public:
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE bool run(const Scalar& value) {
    const Scalar magnitude = numext::abs(value);
    return magnitude > Scalar(0) && magnitude < threshold();
  }

  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE bool run_or_zero(const Scalar& value) {
    return numext::abs(value) < threshold();
  }
};

template <typename Scalar>
struct safe_scaling_needs_subnormal_recovery<Scalar, true> {
 private:
  using Binary = binary_floating_point_traits<Scalar>;
  using Bits = typename Binary::Bits;

  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Bits threshold() {
    constexpr int kDirectScaleExponent = std::numeric_limits<Scalar>::min_exponent + Binary::kFractionBits - 1;
    return Bits(kDirectScaleExponent + Binary::kExponentBias) << Binary::kFractionBits;
  }

 public:
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE bool run(const Scalar& value) {
    return Binary::magnitude(value) - Bits(1) < threshold() - Bits(1);
  }

  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE bool run_or_zero(const Scalar& value) {
    return Binary::magnitude(value) < threshold();
  }
};

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
 private:
  using Factors = safe_scaling_factors<Scalar>;

  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE bool needs_subnormal_recovery(const Scalar& value) {
    return safe_scaling_needs_subnormal_recovery<Scalar>::run(value);
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

  template <typename MatrixType>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void scale_matrix_column(MatrixType& matrix, const Index col,
                                                                        const Index firstRow, const Index endRow,
                                                                        const Scalar& factor) {
    for (Index row = firstRow; row < endRow; ++row) {
      matrix.coeffRef(row, col) = scale_binary_by_power_of_two(matrix.coeff(row, col), factor);
    }
  }

  template <typename MatrixType>
  EIGEN_DEVICE_FUNC EIGEN_DONT_INLINE static void scale_matrix_in_place_preserving_subnormal_inputs(
      MatrixType& matrix, const Scalar& factor) {
    for (Index col = 0; col < matrix.cols(); ++col) {
      scale_matrix_column(matrix, col, 0, matrix.rows(), factor);
    }
  }

  template <typename MatrixType, unsigned int Mode>
  EIGEN_DEVICE_FUNC EIGEN_DONT_INLINE static void scale_matrix_in_place_preserving_subnormal_inputs(
      TriangularView<MatrixType, Mode>& matrix, const Scalar& factor) {
    constexpr bool kLower = (Mode & Lower) == Lower;
    constexpr bool kExplicitDiagonal = (Mode & (UnitDiag | ZeroDiag)) == 0;
    for (Index col = 0; col < matrix.cols(); ++col) {
      const Index firstRow = kLower ? col + Index(!kExplicitDiagonal) : 0;
      const Index endRow = kLower ? matrix.rows() : numext::mini(matrix.rows(), col + Index(kExplicitDiagonal));
      scale_matrix_column(matrix, col, firstRow, endRow, factor);
    }
  }

  template <typename Dest, typename Src>
  EIGEN_DEVICE_FUNC EIGEN_DONT_INLINE static void scale_scalar_to_preserving_subnormal_input(Dest& dest, const Src& src,
                                                                                             const Scalar& factor) {
    dest = scale_binary_by_power_of_two(src, factor);
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
    if (!needs_subnormal_recovery(maxCoeff)) {
      matrix *= invScale;
      return;
    }
    scale_matrix_in_place_preserving_subnormal_inputs(matrix, invScale);
  }

  template <typename MatrixType, unsigned int Mode>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void scale_matrix_in_place_impl(TriangularView<MatrixType, Mode>& matrix,
                                                                               const Scalar& maxCoeff, const Scalar&,
                                                                               const Scalar& invScale,
                                                                               bool_constant<true>) {
    if (!needs_subnormal_recovery(maxCoeff)) {
      matrix *= invScale;
      return;
    }
    scale_matrix_in_place_preserving_subnormal_inputs(matrix, invScale);
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
                                                                  const Scalar&, const Scalar& invScale,
                                                                  bool_constant<true>, bool_constant<true>,
                                                                  bool_constant<true>) {
    if (!needs_subnormal_recovery(maxCoeff)) {
      dest = src * invScale;
      return;
    }
    dest = src;
    scale_matrix_in_place_preserving_subnormal_inputs(dest, invScale);
  }

  template <typename Dest, typename Src>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void scale_to_impl(Dest& dest, const Src& src, const Scalar& maxCoeff,
                                                                  const Scalar&, const Scalar& invScale,
                                                                  bool_constant<true>, bool_constant<true>,
                                                                  bool_constant<false>) {
    if (!needs_subnormal_recovery(maxCoeff)) {
      dest = src * invScale;
    } else {
      scale_scalar_to_preserving_subnormal_input(dest, src, invScale);
    }
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

  template <typename Derived>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE safe_scaled_expression<Derived, Scalar> scaled_expression(
      const Derived& value, const Scalar& maxCoeff, Factors& factors);

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
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Factors scale_in_place(MatrixType& matrix, const Scalar& maxCoeff) {
    const Factors factors = select_factors(maxCoeff);
    scale_in_place(matrix, maxCoeff, factors);
    return factors;
  }

  template <typename MatrixType>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void scale_in_place(MatrixType& matrix, const Scalar& maxCoeff,
                                                                   const Factors& factors) {
    if (is_identity(factors)) return;
    using CoeffScalar = typename MatrixType::Scalar;
    constexpr bool kPreserveSubnormalInputs =
        IsPowerOfTwo_ && use_subnormal_preserving_scaling<Scalar, CoeffScalar>::value;
    scale_matrix_in_place_impl(matrix, maxCoeff, factors.scale, factors.invScale,
                               bool_constant<kPreserveSubnormalInputs>());
  }

  template <typename Dest, typename Src>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void scale_to(Dest& dest, const Src& src, const Scalar& maxCoeff,
                                                             const Factors& factors) {
    if (is_identity(factors)) {
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
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE safe_scaling_factors<Scalar> compute_floor_factors(const Scalar& value) {
    return {value, Scalar(1) / value};
  }

  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE safe_scaling_factors<Scalar> compute_ceiling_factors(
      const Scalar& value) {
    return compute_floor_factors(value);
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

template <typename Derived, typename FactorScalar>
struct traits<safe_scaled_expression<Derived, FactorScalar>> {
  using ReturnType = typename plain_matrix_type<Derived>::type;
};

template <typename Derived, typename FactorScalar>
class safe_scaled_expression : public ReturnByValue<safe_scaled_expression<Derived, FactorScalar>> {
 public:
  EIGEN_DEVICE_FUNC safe_scaled_expression(const Derived& value, const FactorScalar& maxCoeff,
                                           const safe_scaling_factors<FactorScalar>& factors)
      : m_value(value), m_maxCoeff(maxCoeff), m_factors(factors) {}

  EIGEN_DEVICE_FUNC constexpr Index rows() const noexcept { return m_value.rows(); }
  EIGEN_DEVICE_FUNC constexpr Index cols() const noexcept { return m_value.cols(); }

  template <typename Dest>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void evalTo(Dest& dest) const {
    safe_scaling<FactorScalar>::scale_to(dest, m_value, m_maxCoeff, m_factors);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE safe_scaled_expression<typename Derived::AdjointReturnType, FactorScalar>
  adjoint() const {
    return safe_scaled_expression<typename Derived::AdjointReturnType, FactorScalar>(m_value.adjoint(), m_maxCoeff,
                                                                                     m_factors);
  }

 private:
  using Nested = typename ref_selector<Derived>::type;
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
