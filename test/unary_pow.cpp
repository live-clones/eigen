// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"

#include <cmath>
#include <complex>
#include <limits>
#include <vector>

// Integer powers of binary32 and binary64 bases, real and complex, are computed by repeated squaring in
// double-word arithmetic: correctly rounded up to the rare cases where the exact power lies within about
// 7 * n * u^2 of a rounding boundary. These tests check that bound against independent references, on the
// vectorized path and on the scalar path, together with the exceptional values and signed zeros.

template <typename Scalar>
Scalar ulp_of(const Scalar& x) {
  Scalar a = numext::abs(x);
  return numext::nextafter(a, std::numeric_limits<Scalar>::infinity()) - a;
}

// |a - b| in units of ulp(b), with non-finite and zero references matched exactly (including the sign of zero).
template <typename Scalar>
bool within_ulps(const Scalar& a, const Scalar& b, double ulps) {
  if ((numext::isnan)(a) || (numext::isnan)(b)) return (numext::isnan)(a) && (numext::isnan)(b);
  if ((numext::isinf)(b) || b == Scalar(0)) return a == b && bool(numext::signbit(a)) == bool(numext::signbit(b));
  if ((numext::isinf)(a)) return false;
  return double(numext::abs(a - b)) <= ulps * double(ulp_of(b));
}

// Log-uniform magnitudes over [2^lo, 2^hi], both signs.
template <typename Scalar>
std::vector<Scalar> log_uniform_bases(int lo, int hi, int count) {
  std::vector<Scalar> bases;
  for (int i = 0; i < count; ++i) {
    double exponent = internal::random<double>(double(lo), double(hi));
    Scalar x = Scalar(std::ldexp(internal::random<double>(1.0, 2.0), int(std::floor(exponent))));
    bases.push_back(i % 2 ? -x : x);
  }
  return bases;
}

// Reference: pow in the next wider format, so its own rounding is negligible for float; for double, the
// platform pow, which is within 1 ulp on every supported libm.
template <typename Scalar>
Scalar reference_pow(const Scalar& x, long n) {
  using Wide = std::conditional_t<std::is_same<Scalar, float>::value, double, Scalar>;
  return Scalar(std::pow(Wide(x), Wide(n)));
}

template <typename Scalar, typename Exponent>
void check_real_pow(const std::vector<Scalar>& bases, const std::vector<long>& exponents, double ulps) {
  Index size = 2 * internal::packet_traits<Scalar>::size + 1;
  ArrayX<Scalar> x(size), y(size);
  for (long n : exponents) {
    Exponent exponent = Exponent(n);
    for (const Scalar& base : bases) {
      x.setConstant(base);
      y = x.pow(exponent);
      Scalar reference = reference_pow(base, n);
      for (Index i = 0; i < size; ++i) {
        bool ok = within_ulps(y(i), reference, ulps);
        if (!ok)
          std::cout << "pow(" << base << ", " << n << ") = " << y(i) << " != " << reference << " (" << ulps << " ulp)"
                    << std::endl;
        VERIFY(ok);
      }
    }
  }
}

template <typename Scalar>
void real_pow_test() {
  // Exponents on both sides of the squaring cutoffs: the plain-squaring range [-3, 7] of other scalar types, the
  // fallback to generic_pow above 2^12 (float) or 2^20 (double), and integer exponents beyond the significand,
  // which generic_pow could not represent.
  long cutoff = long(internal::unary_pow::max_squaring_exponent<Scalar>());
  const std::vector<long> exponents = {2,  3,  4,  5,  7,  8,  9,  15,  16,   17,   31,    100,  127,    1000,
                                       -2, -3, -4, -5, -7, -8, -9, -15, -100, -127, -1000, 4096, cutoff, -cutoff};
  const std::vector<long> huge_exponents = {cutoff + 1,
                                            -cutoff - 1,
                                            1 << 24,
                                            -(1 << 24),
                                            (1 << 24) + 1,
                                            -((1 << 24) + 1),
                                            (std::numeric_limits<int>::max)(),
                                            (std::numeric_limits<int>::min)()};
  constexpr int min_exponent = std::numeric_limits<Scalar>::min_exponent;
  constexpr int max_exponent = std::numeric_limits<Scalar>::max_exponent;

  // Correctly rounded within 2 ulps only holds for double-word repeated squaring, whose fallback -- plain
  // squaring -- is just "a few ulps" and exercised elsewhere. Only the MSA and HVX packets lack the path, so
  // require it everywhere else rather than skipping on the trait that selects it: losing it has to fail here.
  using Packet = typename internal::packet_traits<Scalar>::type;
  constexpr bool has_double_word = internal::unary_pow::use_double_word<Packet>::value;
#if !defined(EIGEN_VECTORIZE_MSA) && !defined(EIGEN_VECTORIZE_HVX)
  VERIFY(has_double_word);
#endif
  if (has_double_word) {
    for (long n : exponents) {
      int limit = numext::mini((max_exponent - 2) / int(numext::abs(n)), max_exponent - 2);
      const std::vector<Scalar> bases = log_uniform_bases<Scalar>(-limit, limit, 64);
      check_real_pow<Scalar, int>(bases, {n}, 2.0);
      check_real_pow<Scalar, Scalar>(bases, {n}, 2.0);
    }
  }
  // Results near the underflow and overflow thresholds: the final scaling rounds once more into the subnormal
  // range, and the reference rounds too.
  {
    const std::vector<Scalar> bases = log_uniform_bases<Scalar>(min_exponent - 4, 4, 32);
    check_real_pow<Scalar, int>(bases, {2, 3, 8, -2, -3, -8}, 3.0);
    check_real_pow<Scalar, Scalar>(bases, {2, 3, 8, -2, -3, -8}, 3.0);
    const std::vector<Scalar> tiny = log_uniform_bases<Scalar>(min_exponent - 20, min_exponent / 8, 32);
    check_real_pow<Scalar, int>(tiny, {8, 9, 100, -8, -9, -100}, 3.0);
    check_real_pow<Scalar, Scalar>(tiny, {8, 9, 100, -8, -9, -100}, 3.0);
  }
  // Exponents beyond the squaring cutoff, on bases within a few ulps of one, where the power stays finite for
  // every exponent up to INT_MAX. The reference pow then carries the full rounding error of its own exponent
  // conversion, which is why the double reference is allowed a wider margin.
  {
    std::vector<Scalar> bases;
    for (int i = -16; i <= 16; ++i) bases.push_back(Scalar(1) + Scalar(i) * NumTraits<Scalar>::epsilon());
    check_real_pow<Scalar, int>(bases, huge_exponents, 3.0);
    for (long n : huge_exponents)
      if (n == long(Scalar(n))) check_real_pow<Scalar, Scalar>(bases, {n}, 3.0);
  }
  // Exceptional values and signed zeros, exactly as std::pow.
  {
    Scalar inf = std::numeric_limits<Scalar>::infinity();
    Scalar nan = std::numeric_limits<Scalar>::quiet_NaN();
    Scalar subnormal = std::numeric_limits<Scalar>::denorm_min() * Scalar(3);
    const std::vector<Scalar> specials = {Scalar(0),
                                          -Scalar(0),
                                          inf,
                                          -inf,
                                          nan,
                                          subnormal,
                                          -subnormal,
                                          (std::numeric_limits<Scalar>::max)(),
                                          (std::numeric_limits<Scalar>::min)(),
                                          Scalar(1),
                                          Scalar(-1)};
    const std::vector<long> small = {1, 2, 3, 4, 5, 8, 9, 100, -1, -2, -3, -4, -5, -8, -9, -100};
    check_real_pow<Scalar, int>(specials, small, 1.0);
    check_real_pow<Scalar, Scalar>(specials, small, 1.0);
  }
}

// Complex reference: componentwise double-word repeated multiplication, no scaling, so the bases are kept where
// nothing under- or overflows.
template <typename Real>
struct double_word {
  Real hi, lo;
  static double_word product(const double_word& a, const double_word& b) {
    Real p = a.hi * b.hi;
    Real e = numext::fma(a.hi, b.hi, -p) + (a.hi * b.lo + a.lo * b.hi);
    Real s = p + e;
    return {s, e - (s - p)};
  }
  static double_word sum(const double_word& a, const double_word& b) {
    Real s = a.hi + b.hi;
    Real bb = s - a.hi;
    Real e = ((a.hi - (s - bb)) + (b.hi - bb)) + a.lo + b.lo;
    Real t = s + e;
    return {t, e - (t - s)};
  }
  static double_word negate(const double_word& a) { return {-a.hi, -a.lo}; }
};

template <typename Real>
std::complex<Real> reference_complex_pow(const std::complex<Real>& z, long n) {
  using DW = double_word<Real>;
  DW re{Real(1), Real(0)}, im{Real(0), Real(0)};
  DW zr{numext::real(z), Real(0)}, zi{numext::imag(z), Real(0)};
  for (long k = 0; k < numext::abs(n); ++k) {
    DW new_re = DW::sum(DW::product(re, zr), DW::negate(DW::product(im, zi)));
    im = DW::sum(DW::product(re, zi), DW::product(im, zr));
    re = new_re;
  }
  if (n < 0) {
    // 1/(re + im i) = (re - im i)/(re^2 + im^2) in double-word arithmetic, then rounded.
    using LongDouble = long double;
    DW norm = DW::sum(DW::product(re, re), DW::product(im, im));
    LongDouble inv = 1.0L / (LongDouble(norm.hi) + LongDouble(norm.lo));
    return std::complex<Real>(Real((LongDouble(re.hi) + LongDouble(re.lo)) * inv),
                              Real(-(LongDouble(im.hi) + LongDouble(im.lo)) * inv));
  }
  return std::complex<Real>(Real(re.hi + re.lo), Real(im.hi + im.lo));
}

// The error of a complex result is measured against the magnitude of the reference: a component that nearly
// cancels is not required to be accurate on its own.
template <typename Real>
bool complex_within_ulps(const std::complex<Real>& a, const std::complex<Real>& b, double ulps) {
  double magnitude = double(numext::abs(b));
  double error = double(numext::abs(a - b));
  return error <= ulps * double(ulp_of(Real(magnitude)));
}

template <typename Real, typename Exponent>
void check_complex_pow(const std::vector<std::complex<Real>>& bases, const std::vector<long>& exponents, double ulps) {
  using Complex = std::complex<Real>;
  Index size = 2 * internal::packet_traits<Complex>::size + 1;
  ArrayX<Complex> z(size), y(size);
  for (long n : exponents) {
    Exponent exponent = Exponent(n);
    for (const Complex& base : bases) {
      z.setConstant(base);
      y = z.pow(exponent);
      Complex reference = reference_complex_pow(base, n);
      for (Index i = 0; i < size; ++i) {
        bool ok = complex_within_ulps(y(i), reference, ulps);
        if (!ok)
          std::cout << "pow(" << base << ", " << n << ") = " << y(i) << " != " << reference << " (" << ulps << " ulp)"
                    << std::endl;
        VERIFY(ok);
      }
    }
  }
}

template <typename Real>
void complex_pow_test() {
  using Complex = std::complex<Real>;
  Index size = 2 * internal::packet_traits<Complex>::size + 1;

  // Small integer inputs have exactly representable powers.
  {
    ArrayX<Complex> z = ArrayX<Complex>::Constant(size, Complex(3, 4));
    VERIFY((z.pow(3) == Complex(-117, 44)).all());
    VERIFY((z.pow(Real(3)) == Complex(-117, 44)).all());
    VERIFY((z.pow(4) == Complex(-527, -336)).all());
    z.setConstant(Complex(1, 1));
    VERIFY((z.pow(2) == Complex(0, 2)).all());
    VERIFY((z.pow(Real(2)) == Complex(0, 2)).all());
    VERIFY((z.pow(4) == Complex(-4, 0)).all());
    VERIFY((z.pow(-2) == Complex(0, Real(-0.5))).all());
    VERIFY((z.pow(-4) == Complex(Real(-0.25), 0)).all());
    VERIFY((z.pow(0) == Complex(1, 0)).all());
    VERIFY((z.pow(1) == Complex(1, 1)).all());
    // (1+i)^(8k+7) = 2^(4k+3) (1 - i): exact, with a magnitude of sqrt(2) times a finite component. The unscaled
    // loop must not admit it once the exponent budget is exhausted, or its intermediates overflow.
    constexpr int k = (std::numeric_limits<Real>::max_exponent - 8) / 4;
    Real big = Real(std::ldexp(1.0, 4 * k + 3));
    VERIFY((z.pow(8 * k + 7) == Complex(big, -big)).all());
    VERIFY((z.pow(Real(8 * k + 7)) == Complex(big, -big)).all());
  }
  // Components so far apart that the smaller one underflows when both take the larger one's scale, while the
  // power's imaginary part, 2ab, is normal: each component is checked on its own, as the magnitude-relative check
  // would not notice the imaginary part missing.
  {
    Real big = Real(std::ldexp(1.0, std::numeric_limits<Real>::max_exponent / 2 - 2));
    Real tiny = Real(std::ldexp(1.0, std::numeric_limits<Real>::min_exponent + std::numeric_limits<Real>::digits));
    ArrayX<Complex> z = ArrayX<Complex>::Constant(size, Complex(big, tiny));
    ArrayX<Complex> y = z.pow(2);
    for (Index k = 0; k < size; ++k) {
      VERIFY(within_ulps(numext::real(y(k)), big * big, 1.0));
      VERIFY(within_ulps(numext::imag(y(k)), Real(2) * big * tiny, 1.0));
    }
  }
  // The same separation near |z| = 1, with a large |n|: (a + bi)^n = a^n (1 + i n b/a) and
  // (b + ai)^n = i^n a^n (1 - i n b/a) to working precision, both components to double-word accuracy.
  {
    Real a = Real(1) + Real(std::ldexp(1.0, -10));
    Real b = Real(std::ldexp(1.0, std::numeric_limits<Real>::min_exponent - 6));
    const Complex i_powers[4] = {Complex(1, 0), Complex(0, 1), Complex(-1, 0), Complex(0, -1)};
    for (long n : {100L, 101L, 1000L, 1003L, -99L, -100L, -1000L, -1001L}) {
      Real power = numext::real(reference_complex_pow(Complex(a), n));
      Real term = Real(n) * numext::real(reference_complex_pow(Complex(a), n - 1)) * b;
      for (bool swapped : {false, true}) {
        Complex expected = swapped ? i_powers[((n % 4) + 4) % 4] * Complex(power, -term) : Complex(power, term);
        ArrayX<Complex> z = ArrayX<Complex>::Constant(size, swapped ? Complex(b, a) : Complex(a, b));
        ArrayX<Complex> y = z.pow(int(n));
        for (Index k = 0; k < size; ++k) {
          VERIFY(within_ulps(numext::real(y(k)), numext::real(expected), 2.0));
          VERIFY(within_ulps(numext::imag(y(k)), numext::imag(expected), 2.0));
        }
      }
    }
  }
  // An element's power does not depend on its packet neighbours, one of which here forces the scaled loop: the
  // unscaled loop, whose residuals would underflow in the smaller component (here 8 a b^7, subnormal), requires
  // both components in range, so this base takes the scaled loop either way.
  {
    constexpr int min_exponent = std::numeric_limits<Real>::min_exponent;
    constexpr int e = min_exponent / 9;
    for (int i = 0; i < 16; ++i) {
      Real a = std::ldexp(internal::random<Real>(Real(1), Real(2)), min_exponent - 20 - 7 * e);
      Real b = std::ldexp(internal::random<Real>(Real(-2), Real(-1)), e);
      ArrayX<Complex> alone = ArrayX<Complex>::Constant(size, Complex(a, b)), mixed = alone;
      for (Index k = 1; k < size; k += 2) mixed(k) = Complex((std::numeric_limits<Real>::max)() / 4, 1);
      ArrayX<Complex> y_alone = alone.pow(8), y_mixed = mixed.pow(8);
      for (Index k = 0; k < size; k += 2) VERIFY(y_alone(k) == y_mixed(k));
    }
  }
  // Random bases at magnitudes where the reference cannot under- or overflow; the exponent runs past the
  // renormalization interval of the implementation.
  {
    const std::vector<long> exponents = {2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 40, -2, -3, -4, -5, -8, -9, -17, -40};
    std::vector<Complex> bases;
    for (int i = 0; i < 64; ++i) {
      Real magnitude = internal::random<Real>(Real(0.5), Real(2));
      Real angle = internal::random<Real>(Real(-3.14159), Real(3.14159));
      bases.push_back(Complex(magnitude * std::cos(angle), magnitude * std::sin(angle)));
    }
    check_complex_pow<Real, int>(bases, exponents, 2.0);
    check_complex_pow<Real, Real>(bases, exponents, 2.0);
  }
  // Magnitudes across the range, against the same base scaled by a power of two, which only moves the exponent.
  // The scale runs through the range in steps, ends included: at its extremes |z| exceeds what the reciprocal's
  // norm can represent.
  {
    constexpr int max_exponent = std::numeric_limits<Real>::max_exponent;
    for (long n : {1L, 3L, 8L, 100L, -1L, -2L, -3L, -8L, -100L}) {
      int extreme = (max_exponent - 4) / int(numext::abs(n)) - 1;  // |base|^n within 2^(+-|n|) stays finite
      for (int i = 0; i <= 32; ++i) {
        int e = -extreme + (2 * extreme * i) / 32;
        Real scale = Real(std::ldexp(1.0, e));
        Real magnitude = internal::random<Real>(Real(0.5), Real(1.5));
        Real angle = internal::random<Real>(Real(-3.14159), Real(3.14159));
        Complex base(magnitude * std::cos(angle), magnitude * std::sin(angle));
        ArrayX<Complex> z = ArrayX<Complex>::Constant(size, base), w = ArrayX<Complex>::Constant(size, base * scale);
        ArrayX<Complex> zn = z.pow(int(n)), wn = w.pow(int(n));
        Real power_of_scale = Real(std::pow(double(scale), double(n)));
        for (Index k = 0; k < size; ++k) {
          Complex expected = zn(k) * power_of_scale;
          VERIFY(complex_within_ulps(wn(k), expected, 2.0));
        }
      }
    }
  }
  // Exceptional values follow the plain complex product; a NaN base gives NaN. Complex division by zero or
  // infinity is implementation-defined, so negative exponents are only checked for NaN bases.
  {
    Real inf = std::numeric_limits<Real>::infinity();
    Real nan = std::numeric_limits<Real>::quiet_NaN();
    for (const Complex& base : {Complex(0, 0), Complex(inf, 0), Complex(0, inf), Complex(nan, 1), Complex(1, nan)}) {
      for (int n : {2, 3, 4}) {
        ArrayX<Complex> z = ArrayX<Complex>::Constant(size, base);
        ArrayX<Complex> y = z.pow(n);
        Complex plain = base;
        for (int k = 1; k < n; ++k) plain = internal::pmul(plain, base);
        for (Index k = 0; k < size; ++k) {
          bool both_nan = (numext::isnan)(y(k)) && (numext::isnan)(plain);
          VERIFY(both_nan || y(k) == plain);
        }
      }
    }
    for (const Complex& base : {Complex(nan, 1), Complex(1, nan)}) {
      for (int n : {-1, -2, -3}) {
        ArrayX<Complex> z = ArrayX<Complex>::Constant(size, base);
        VERIFY(z.pow(n).isNaN().all());
      }
    }
  }
}

// half and bfloat16 keep plain repeated squaring, whose error grows like n ulp, for integer exponents and small
// floating-point ones, and generic_pow (pow through float) beyond; with F16C their packets take the vectorized
// path. Check both against float.
template <typename Scalar>
void narrow_pow_test() {
  Index size = 2 * internal::packet_traits<Scalar>::size + 1;
  ArrayX<Scalar> x(size), y(size);
  for (long n : {1L, 2L, 3L, 5L, 7L, 8L, 16L, 100L, 4097L, -1L, -2L, -3L, -8L, -100L}) {
    // Powers within the normal range of the type: |x|^n in 2^(+-(max_exponent - 2)).
    double limit = double(std::numeric_limits<Scalar>::max_exponent - 2) / double(numext::abs(n));
    for (int i = 0; i < 32; ++i) {
      float base = float(std::exp2(internal::random<double>(-limit, limit)));
      x.setConstant(Scalar(i % 2 ? -base : base));
      Scalar reference = Scalar(std::pow(float(x(0)), float(n)));
      double ulps = 2.0 + double(numext::abs(n));
      y = x.pow(int(n));
      for (Index k = 0; k < size; ++k) {
        if (!within_ulps(y(k), reference, ulps))
          std::cout << "pow(" << x(0) << ", " << n << ") = " << y(k) << " != " << reference << std::endl;
        VERIFY(within_ulps(y(k), reference, ulps));
      }
      if (n == long(Scalar(float(n)))) {
        y = x.pow(Scalar(float(n)));
        for (Index k = 0; k < size; ++k) VERIFY(within_ulps(y(k), reference, ulps));
      }
    }
  }
}

// complex<long double> keeps plain squaring; its reciprocal must stay range-safe, as the double-word path's
// conj(z)/|z|^2 would not be for long double's wider exponent range.
void complex_long_double_range_test() {
  using Complex = std::complex<long double>;
  constexpr int max_exponent = std::numeric_limits<long double>::max_exponent;
  if (max_exponent <= std::numeric_limits<double>::max_exponent) return;
  ArrayX<Complex> z(9), y(9);
  auto close = [](const Complex& a, const Complex& b, long double ulps) {
    return std::abs(a - b) <= ulps * std::numeric_limits<long double>::epsilon() * std::abs(b);
  };
  // Built with ldexp rather than written as literals: 1e3000L and the like are out of range, and so a
  // compile-time error, wherever long double is binary64, which the run-time guard above cannot prevent.
  for (int e : {(4 * max_exponent) / 5, -(4 * max_exponent) / 5, (2 * max_exponent) / 5, -(2 * max_exponent) / 5}) {
    long double magnitude = std::ldexp(1.0L, e);
    z.setConstant(Complex(magnitude, magnitude / 3));
    y = z.pow(-1);
    Complex expected = Complex(1) / z(0);
    for (Index k = 0; k < 9; ++k) VERIFY(close(y(k), expected, 4.0L));
    if (2 * numext::abs(e) >= max_exponent) continue;  // the square would leave the range
    y = z.pow(-2);
    expected = Complex(1) / (z(0) * z(0));
    for (Index k = 0; k < 9; ++k) VERIFY(close(y(k), expected, 8.0L));
  }
}

EIGEN_DECLARE_TEST(unary_pow) {
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(real_pow_test<float>());
    CALL_SUBTEST_2(real_pow_test<double>());
    CALL_SUBTEST_3(complex_pow_test<float>());
    CALL_SUBTEST_4(complex_pow_test<double>());
    CALL_SUBTEST_5(narrow_pow_test<half>());
    CALL_SUBTEST_6(narrow_pow_test<bfloat16>());
    CALL_SUBTEST_7(complex_long_double_range_test());
  }
}
