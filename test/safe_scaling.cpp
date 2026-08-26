// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include "fp_control.h"

template <typename T>
void check_power_of_two_scaling_factor() {
  static_assert(internal::supports_power_of_two_scaling<T>::value, "power-of-two scaling must be enabled");
  using Factors = internal::safe_scaling_factors<T>;

  const auto check_factor = [](const T& value) {
    const auto factors = internal::safe_scaling<T>::compute_floor_factors(value);
    VERIFY(factors.scale >= (std::numeric_limits<T>::min)());
    VERIFY(factors.invScale >= (std::numeric_limits<T>::min)());
    VERIFY_IS_EQUAL(factors.scale * factors.invScale, T(1));
    return factors;
  };

  const auto check_round_trip = [&](const T& value) {
    const auto factors = check_factor(value);
    T scaled;
    const auto scaleToFactors = internal::safe_scaling<T>::scale_to(scaled, value, value);
    VERIFY_IS_EQUAL(scaleToFactors.scale, factors.scale);
    VERIFY_IS_EQUAL(scaleToFactors.invScale, factors.invScale);
    Matrix<T, 1, 1> scaledInPlace;
    scaledInPlace(0) = value;
    const auto selectedFactors = internal::safe_scaling<T>::scale_in_place(scaledInPlace, value);
    VERIFY_IS_EQUAL(selectedFactors.scale, factors.scale);
    VERIFY_IS_EQUAL(selectedFactors.invScale, factors.invScale);
    VERIFY_IS_EQUAL(scaledInPlace(0), scaled);

    T restored;
    internal::safe_scaling<T>::unscale_to(restored, scaled, factors);
    VERIFY_IS_EQUAL(restored, value);
    internal::safe_scaling<T>::unscale_in_place(scaled, factors);
    VERIFY_IS_EQUAL(scaled, value);
  };

  const auto denormFactors = check_factor(std::numeric_limits<T>::denorm_min());
  VERIFY_IS_EQUAL(denormFactors.scale, (std::numeric_limits<T>::min)());
  Factors normalReciprocalFactors;
  VERIFY(internal::safe_scaling<T>::try_compute_ceiling_factors_with_normal_reciprocal(
      std::numeric_limits<T>::denorm_min(), (std::numeric_limits<T>::min)(), normalReciprocalFactors));
  VERIFY_IS_EQUAL(normalReciprocalFactors.scale, denormFactors.scale);
  VERIFY_IS_EQUAL(normalReciprocalFactors.invScale, denormFactors.invScale);

  const T denormMin = std::numeric_limits<T>::denorm_min();
  T scaledDenorm;
  internal::safe_scaling<T>::scale_to(scaledDenorm, denormMin, denormMin);
  Matrix<T, 1, 1> scaledDenormInPlace;
  scaledDenormInPlace(0) = denormMin;
  internal::safe_scaling<T>::scale_in_place(scaledDenormInPlace, denormMin);
  VERIFY_IS_EQUAL(scaledDenormInPlace(0), scaledDenorm);
  VERIFY(scaledDenorm > T(0));

  check_round_trip((std::numeric_limits<T>::min)());
  check_round_trip(T(0.75));
  check_round_trip(T(3));
  check_round_trip((std::numeric_limits<T>::max)());

  const auto floorFactors = internal::safe_scaling<T>::compute_floor_factors(T(1.5));
  const auto ceilingFactors = internal::safe_scaling<T>::compute_ceiling_factors_with_normal_reciprocal(T(1.5));
  VERIFY_IS_EQUAL(floorFactors.scale, T(1));
  VERIFY_IS_EQUAL(ceilingFactors.scale, T(2));

  const T highest = (std::numeric_limits<T>::max)();
  const auto highestCeilingFactors = internal::safe_scaling<T>::compute_ceiling_factors_with_normal_reciprocal(highest);
  VERIFY_IS_EQUAL(highestCeilingFactors.scale, check_factor(highest).scale);
  VERIFY_IS_EQUAL(highestCeilingFactors.scale * highestCeilingFactors.invScale, T(1));
  const T scaledHighest = highest * highestCeilingFactors.invScale;
  VERIFY(scaledHighest >= T(1));
  if (std::numeric_limits<T>::max_exponent + std::numeric_limits<T>::min_exponent <= 3) {
    VERIFY(scaledHighest < T(4));
  }
}

template <typename T>
void check_safe_scaling_special_values() {
  using Scaling = internal::safe_scaling<T>;
  using Factors = internal::safe_scaling_factors<T>;

  const Factors identity;
  VERIFY_IS_EQUAL(identity.scale, T(1));
  VERIFY_IS_EQUAL(identity.invScale, T(1));

  Matrix<T, 1, 1> zero = Matrix<T, 1, 1>::Zero();
  const auto zeroFactors = Scaling::scale_in_place(zero, T(0));
  VERIFY_IS_EQUAL(zeroFactors.scale, T(1));
  VERIFY_IS_EQUAL(zeroFactors.invScale, T(1));
  VERIFY_IS_EQUAL(zero(0), T(0));

  for (const T special : {std::numeric_limits<T>::infinity(), std::numeric_limits<T>::quiet_NaN()}) {
    Matrix<T, 2, 1> input;
    input << special, T(2);
    Matrix<T, 2, 1> scaledInPlace = input;
    const auto inPlaceFactors = Scaling::scale_in_place(scaledInPlace, special);
    VERIFY_IS_EQUAL(inPlaceFactors.scale, T(1));
    VERIFY_IS_EQUAL(inPlaceFactors.invScale, T(1));
    VERIFY_IS_EQUAL(scaledInPlace(1), T(2));
    if ((numext::isnan)(special)) {
      VERIFY((numext::isnan)(scaledInPlace(0)));
    } else {
      VERIFY((numext::isinf)(scaledInPlace(0)));
    }
  }
}

void check_safe_scaling_special_value_frontends() {
  using Scaling = internal::safe_scaling<float>;
  using Factors = internal::safe_scaling_factors<float>;

  for (const float special : {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
    Vector2f input(special, 2.0f);
    Vector2f scaledTo;
    const auto scaleToFactors = Scaling::scale_to(scaledTo, input, special);
    VERIFY_IS_EQUAL(scaleToFactors.scale, 1.0f);
    VERIFY_IS_EQUAL(scaleToFactors.invScale, 1.0f);

    Factors expressionFactors;
    const Vector2f scaledExpression = Scaling::scaled_expression(input, special, expressionFactors);
    VERIFY_IS_EQUAL(expressionFactors.scale, 1.0f);
    VERIFY_IS_EQUAL(expressionFactors.invScale, 1.0f);
    VERIFY_IS_EQUAL(scaledTo(1), 2.0f);
    VERIFY_IS_EQUAL(scaledExpression(1), 2.0f);
    if ((numext::isnan)(special)) {
      VERIFY((numext::isnan)(scaledTo(0)));
      VERIFY((numext::isnan)(scaledExpression(0)));
    } else {
      VERIFY((numext::isinf)(scaledTo(0)));
      VERIFY((numext::isinf)(scaledExpression(0)));
    }
  }
}

void check_arithmetic_safe_scaling_fallback() {
  using Scaling = internal::safe_scaling<double, false>;
  using Factors = internal::safe_scaling_factors<double>;

  Matrix<double, 2, 1> input;
  input << 3.0, 6.0;
  Matrix<double, 2, 1> scaled;
  const auto factors = Scaling::scale_to(scaled, input, 3.0);
  VERIFY_IS_EQUAL(factors.scale, 3.0);
  VERIFY_IS_EQUAL(factors.invScale, 1.0 / 3.0);
  VERIFY_IS_EQUAL(scaled(0), 1.0);
  VERIFY_IS_EQUAL(scaled(1), 2.0);
  Scaling::unscale_in_place(scaled, factors);
  VERIFY_IS_EQUAL(scaled, input);

  const auto check_extreme_unscale = [](const double value) {
    const Factors extremeFactors = Scaling::compute_floor_factors(value);
    double restored;
    Scaling::unscale_to(restored, 1.0, extremeFactors);
    VERIFY_IS_EQUAL(restored, value);
    restored = 1.0;
    Scaling::unscale_in_place(restored, extremeFactors);
    VERIFY_IS_EQUAL(restored, value);
  };
  check_extreme_unscale((std::numeric_limits<double>::max)());
  volatile double denormMinInput = std::numeric_limits<double>::denorm_min();
  const double denormMin = denormMinInput;
  if (denormMin > 0.0) check_extreme_unscale(denormMin);

  Factors normalReciprocalFactors;
  const double normalMin = (std::numeric_limits<double>::min)();
  VERIFY(Scaling::try_compute_ceiling_factors_with_normal_reciprocal(3.0, normalMin, normalReciprocalFactors));
  VERIFY_IS_EQUAL(normalReciprocalFactors.scale, 3.0);
  VERIFY_IS_EQUAL(normalReciprocalFactors.invScale, 1.0 / 3.0);
  VERIFY(!Scaling::try_compute_ceiling_factors_with_normal_reciprocal(std::numeric_limits<double>::denorm_min(),
                                                                      normalMin, normalReciprocalFactors));
  VERIFY(!Scaling::try_compute_ceiling_factors_with_normal_reciprocal((std::numeric_limits<double>::max)(), normalMin,
                                                                      normalReciprocalFactors));
}

template <typename Scalar>
struct scaling_test_value {
  using RealScalar = typename NumTraits<Scalar>::Real;
  static Scalar run(RealScalar real, RealScalar) { return real; }
};

template <typename RealScalar>
struct scaling_test_value<std::complex<RealScalar>> {
  static std::complex<RealScalar> run(RealScalar real, RealScalar imag) { return std::complex<RealScalar>(real, imag); }
};

template <typename Scalar>
void check_subnormal_preserving_scaling() {
  using RealScalar = typename NumTraits<Scalar>::Real;
  volatile RealScalar normalMinInput = (std::numeric_limits<RealScalar>::min)();
  volatile RealScalar denormMinInput = std::numeric_limits<RealScalar>::denorm_min();
  const long double inputScale = 256.0L * static_cast<long double>(normalMinInput);
  const RealScalar maxCoeff = RealScalar(inputScale);
  const RealScalar subnormalMaxCoeff = RealScalar(64.0L * static_cast<long double>(denormMinInput));
  const auto factors = internal::safe_scaling<RealScalar>::compute_floor_factors(maxCoeff);

  Matrix<Scalar, 2, 1> input;
  input(0) = scaling_test_value<Scalar>::run(RealScalar(inputScale), RealScalar(-inputScale));
  input(1) = scaling_test_value<Scalar>::run(RealScalar(inputScale / 512.0L), RealScalar(inputScale / 1024.0L));
  Matrix<Scalar, 1, 1> subnormalInput;
  subnormalInput(0) = scaling_test_value<Scalar>::run(subnormalMaxCoeff, RealScalar(0));
  ScopedFlushToZero flushToZero;
  Matrix<Scalar, 2, 1> scaled;
  internal::safe_scaling<RealScalar>::scale_to(scaled, input, maxCoeff, factors);

  internal::safe_scaling_factors<RealScalar> expressionFactors;
  const Matrix<Scalar, 2, 1> scaledExpression =
      internal::safe_scaling<RealScalar>::scaled_expression(input, maxCoeff, expressionFactors);

  VERIFY_IS_EQUAL(scaled(0), scaling_test_value<Scalar>::run(RealScalar(1), RealScalar(-1)));
  VERIFY_IS_EQUAL(scaled(1),
                  scaling_test_value<Scalar>::run(RealScalar(1) / RealScalar(512), RealScalar(1) / RealScalar(1024)));
  VERIFY_IS_EQUAL(expressionFactors.scale, factors.scale);
  VERIFY_IS_EQUAL(expressionFactors.invScale, factors.invScale);
  VERIFY_IS_EQUAL(scaledExpression, scaled);

  Scalar scaledValue;
  internal::safe_scaling<RealScalar>::scale_to(scaledValue, input(1), maxCoeff, factors);
  VERIFY_IS_EQUAL(scaledValue, scaled(1));

  const auto subnormalFactors = internal::safe_scaling<RealScalar>::scale_in_place(subnormalInput, subnormalMaxCoeff);
  VERIFY(subnormalFactors.scale != RealScalar(1) || subnormalFactors.invScale != RealScalar(1));
  VERIFY(numext::abs(subnormalInput(0)) > RealScalar(0));
}

EIGEN_DECLARE_TEST(safe_scaling) {
  CALL_SUBTEST(check_power_of_two_scaling_factor<half>());
  CALL_SUBTEST(check_power_of_two_scaling_factor<bfloat16>());
  CALL_SUBTEST(check_power_of_two_scaling_factor<float>());
  CALL_SUBTEST(check_power_of_two_scaling_factor<double>());
  CALL_SUBTEST(check_power_of_two_scaling_factor<long double>());
  CALL_SUBTEST(check_safe_scaling_special_values<half>());
  CALL_SUBTEST(check_safe_scaling_special_values<bfloat16>());
  CALL_SUBTEST(check_safe_scaling_special_values<float>());
  CALL_SUBTEST(check_safe_scaling_special_values<double>());
  CALL_SUBTEST(check_safe_scaling_special_value_frontends());
  CALL_SUBTEST(check_arithmetic_safe_scaling_fallback());
  CALL_SUBTEST(check_subnormal_preserving_scaling<float>());
  CALL_SUBTEST(check_subnormal_preserving_scaling<std::complex<float>>());
  CALL_SUBTEST(check_subnormal_preserving_scaling<double>());
  CALL_SUBTEST(check_subnormal_preserving_scaling<std::complex<double>>());
}
