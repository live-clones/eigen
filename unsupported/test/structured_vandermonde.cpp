// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"

#include <unsupported/Eigen/StructuredMatrices>

using namespace Eigen;

// Reference dense Vandermonde built entry-wise from the nodes.
template <typename Scalar>
Matrix<Scalar, Dynamic, Dynamic> reference_vandermonde(const Matrix<Scalar, Dynamic, 1>& x, Index n) {
  const Index m = x.size();
  Matrix<Scalar, Dynamic, Dynamic> dense(m, n);
  for (Index i = 0; i < m; ++i) {
    Scalar p(1);
    for (Index j = 0; j < n; ++j) {
      dense(i, j) = p;
      p *= x[i];
    }
  }
  return dense;
}

template <typename Scalar>
void test_vandermonde_product(Index m, Index n) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Vec x = Vec::Random(m);
  Vandermonde<Scalar> V(x, n);
  Mat dense = reference_vandermonde<Scalar>(x, n);

  Mat Vd = V;
  VERIFY_IS_APPROX(Vd, dense);
  for (Index t = 0; t < 5; ++t) {
    Index i = internal::random<Index>(0, m - 1), j = internal::random<Index>(0, n - 1);
    VERIFY_IS_APPROX(V.coeff(i, j), dense(i, j));
  }

  Vec a = Vec::Random(n);
  VERIFY_IS_APPROX((V * a).eval(), (dense * a).eval());

  Mat A = Mat::Random(n, 3);
  VERIFY_IS_APPROX((V * A).eval(), (dense * A).eval());

  // Accumulation form exercised by the iterative solvers.
  Vec y = Vec::Random(m);
  Vec y0 = y;
  y.noalias() += V * a;
  VERIFY_IS_APPROX(y, (y0 + dense * a).eval());
}

// Well-conditioned interpolation: Chebyshev-like nodes in [-1,1] keep the
// conditioning around (1+sqrt(2))^n, so for moderate n a modest tolerance wide
// of roundoff verifies the primal and dual solves against the exact solution.
template <typename Scalar>
void test_bjorck_pereyra(Index n) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Vec x(n);
  for (Index i = 0; i < n; ++i)
    x[i] = Scalar(RealScalar(std::cos(EIGEN_PI * double(2 * i + 1) / double(2 * n))));  // Chebyshev nodes
  Vandermonde<Scalar> V(x);
  Mat dense = reference_vandermonde<Scalar>(x, n);

  BjorckPereyra<Scalar> bp(V);
  VERIFY(bp.info() == Success);

  // Forward-error bound for the (moderately conditioned) Chebyshev-node system.
  const RealScalar tol = RealScalar(5e7) * NumTraits<RealScalar>::epsilon();  // ~1e-8 in double

  // Primal: interpolate values of a known polynomial and recover its coefficients.
  Vec aTrue = Vec::Random(n);
  Vec f = dense * aTrue;
  Vec a = bp.solve(f);
  VERIFY((a - aTrue).norm() <= tol * aTrue.norm());

  // Dual (moment) system through the SolverBase transpose idiom.
  Vec wTrue = Vec::Random(n);
  Vec b = dense.transpose() * wTrue;
  Vec w = bp.transpose().solve(b);
  VERIFY((w - wTrue).norm() <= tol * wTrue.norm());

  // Adjoint solve.
  Vec c = dense.adjoint() * wTrue;
  Vec u = bp.adjoint().solve(c);
  VERIFY((u - wTrue).norm() <= tol * wTrue.norm());

  // Multiple right-hand sides.
  Mat F = Mat::Random(n, 3);
  Mat A = bp.solve(F);
  VERIFY_IS_APPROX((dense * A).eval(), F);
}

// The n-th roots of unity make V/sqrt(n) exactly unitary: the solver must reach
// full precision, verified against the analytic inverse a = V^H f / n.
void test_bjorck_pereyra_roots_of_unity(Index n) {
  typedef std::complex<double> Scalar;
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Vec x(n);
  for (Index i = 0; i < n; ++i) x[i] = std::polar(1.0, double(2 * EIGEN_PI) * double(i) / double(n));
  Vandermonde<Scalar> V(x);
  Mat dense = reference_vandermonde<Scalar>(x, n);

  BjorckPereyra<Scalar> bp(V);
  VERIFY(bp.info() == Success);

  Vec f = Vec::Random(n);
  Vec a = bp.solve(f);
  Vec aRef = dense.adjoint() * f / Scalar(double(n));
  VERIFY_IS_APPROX(a, aRef);
}

// Björck-Pereyra's celebrated accuracy property (Higham, ASNA ch. 22): for
// monotone nodes and an alternating-sign right-hand side the forward error is
// tiny even though the matrix conditioning is astronomical. Deterministic.
void test_bjorck_pereyra_higham() {
  typedef Matrix<double, Dynamic, 1> Vec;

  const Index n = 20;
  Vec x(n);
  for (Index i = 0; i < n; ++i) x[i] = double(i + 1) / double(n);  // monotone in (0,1]
  Vandermonde<double> V(x);

  // f_i = (-1)^i: alternating signs.
  Vec f(n);
  for (Index i = 0; i < n; ++i) f[i] = (i % 2 == 0) ? 1.0 : -1.0;
  BjorckPereyra<double> bp(V);
  Vec a = bp.solve(f);

  // Residual check through the operator's O(n^2)-free Horner product: the
  // interpolant must reproduce the alternating values to near machine precision
  // relative to the coefficient scale, despite cond(V) >> 1/eps.
  Vec r = (V * a).eval() - f;
  const double tol = 5e5 * NumTraits<double>::epsilon();  // ~1e-10
  VERIFY(r.norm() <= tol * a.norm());
}

// Repeated nodes make the matrix exactly singular and must be reported.
void test_bjorck_pereyra_singular() {
  typedef Matrix<double, Dynamic, 1> Vec;
  Vec x(5);
  x << 0.1, 0.7, 0.3, 0.7, 0.9;
  Vandermonde<double> V(x);
  BjorckPereyra<double> bp(V);
  VERIFY(bp.info() == NumericalIssue);
}

// Aliased products: the products carry the default product tag, so assignment
// materializes a temporary exactly like a dense product and x = V * x,
// x += V * x must come out as if the right-hand side had been copied first.
template <typename Scalar>
void test_vandermonde_aliased_product(Index n) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Vec x = Vec::Random(n);
  Vandermonde<Scalar> V(x);  // square, so x = V * x type-checks
  Mat dense = reference_vandermonde<Scalar>(x, n);

  Vec b = Vec::Random(n);
  Vec y = b;
  y = V * y;
  VERIFY_IS_APPROX(y, (dense * b).eval());

  y = b;
  y += V * y;
  VERIFY_IS_APPROX(y, (b + dense * b).eval());

  y = b;
  y -= V * y;
  VERIFY_IS_APPROX(y, (b - dense * b).eval());

  Mat B = Mat::Random(n, 3);
  Mat Y = B;
  Y = V * Y;
  VERIFY_IS_APPROX(Y, (dense * B).eval());
}

// Aliasing beyond the same-object case: the default-product temporary must also
// resolve right-hand-side expressions that reference the destination, overlapping
// views of one buffer, and rectangular self-assignments where the destination is
// resized by the assignment.
template <typename Scalar>
void test_vandermonde_aliased_expression(Index n) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Vec nodes = Vec::Random(n);
  Vandermonde<Scalar> V(nodes);  // square
  Mat dense = reference_vandermonde<Scalar>(nodes, n);

  // Right-hand-side expression referencing the destination.
  Vec x = Vec::Random(n), x0 = x;
  x = V * (x + Vec::Ones(n));
  VERIFY_IS_APPROX(x, (dense * (x0 + Vec::Ones(n))).eval());

  // Overlapping (shifted) segments of one buffer.
  Vec buf = Vec::Random(n + 1);
  Vec expected = dense * buf.tail(n);
  buf.head(n) = V * buf.tail(n);
  VERIFY_IS_APPROX(buf.head(n).eval(), expected);

  // Rectangular self-assignment (the reviewer's 5x4 case): x = V * x resizes
  // the destination from 4 to 5, so the product must be captured before the
  // destination storage is touched.
  Vec tallNodes = Vec::Random(5);
  Vandermonde<Scalar> Vt(tallNodes, 4);
  Mat denseT = reference_vandermonde<Scalar>(tallNodes, 4);
  Vec z = Vec::Random(4), z0 = z;
  z = Vt * z;
  VERIFY_IS_EQUAL(z.size(), 5);
  VERIFY_IS_APPROX(z, (denseT * z0).eval());
}

// A delayed-evaluated product expression must keep its structured factor alive:
// the makeVandermonde() factory returns an owning temporary, so Product has to
// nest the operator by value (no NestByRefBit) or `expr` dangles.
template <typename Scalar>
void test_vandermonde_delayed_product(Index m, Index n) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  STATIC_CHECK(!std::is_reference<typename internal::ref_selector<Vandermonde<Scalar>>::type>::value);

  Vec x = Vec::Random(m), a = Vec::Random(n);
  Mat dense = reference_vandermonde<Scalar>(x, n);

  auto expr = makeVandermonde(x, n) * a;  // the factory temporary dies with the full expression
  Vec scribble = Vec::Random(2 * m);      // reuses the temporary's freed heap storage
  Vec y = expr;
  VERIFY_IS_APPROX(y, (dense * a).eval());
  VERIFY_IS_EQUAL(scribble.size(), 2 * m);  // keep the scribble alive across the evaluation
}

// Mixed-scalar products: a real operator applied to a complex right-hand side
// (and a complex operator applied to a real one) promotes to the complex
// product scalar, so alpha and the Horner accumulation must run in the promoted
// type rather than the operator scalar.
template <typename RealScalar>
void test_vandermonde_mixed_scalar(Index m, Index n) {
  typedef std::complex<RealScalar> Complex;
  typedef Matrix<RealScalar, Dynamic, 1> RVec;
  typedef Matrix<Complex, Dynamic, 1> CVec;
  typedef Matrix<Complex, Dynamic, Dynamic> CMat;

  RVec x = RVec::Random(m);
  Vandermonde<RealScalar> V(x, n);
  CMat dense = reference_vandermonde<RealScalar>(x, n).template cast<Complex>();

  CVec a = CVec::Random(n);
  CVec y = V * a;
  VERIFY_IS_APPROX(y, (dense * a).eval());

  CVec y0 = CVec::Random(m);
  y = y0;
  y.noalias() += V * a;
  VERIFY_IS_APPROX(y, (y0 + dense * a).eval());

  CVec xc = CVec::Random(m);
  Vandermonde<Complex> Vc(xc, n);
  CMat denseC = reference_vandermonde<Complex>(xc, n);
  RVec ar = RVec::Random(n);
  CVec z = Vc * ar;
  VERIFY_IS_APPROX(z, (denseC * ar).eval());
}

template <typename Scalar>
void test_vandermonde_determinant(Index n) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Vec x = Vec::Random(n);
  Vandermonde<Scalar> V(x);
  Mat dense = reference_vandermonde<Scalar>(x, n);
  VERIFY_IS_APPROX(V.determinant(), dense.determinant());
}

// Reference for the wide-dynamic-range determinant tests: the same factor
// sequence accumulated with explicit exponent tracking, including the halving of
// factors whose node difference overflows (halving finite nodes is exact there:
// a difference only overflows for huge, normal operands). The power-of-two
// rescaling is exact, so this reproduces the exact product of the computed
// factors up to one rounding per multiplication -- representable even when
// naive partial products leave the double range. Finite nodes only.
double reference_vandermonde_det(const Matrix<double, Dynamic, 1>& x) {
  double mantissa = 1.0;
  Index exponent = 0;
  for (Index j = 1; j < x.size(); ++j)
    for (Index i = 0; i < j; ++i) {
      double diff = x[j] - x[i];
      if (!(numext::isfinite)(diff)) {
        diff = 0.5 * x[j] - 0.5 * x[i];
        ++exponent;
      }
      int e;
      mantissa *= std::frexp(diff, &e);
      exponent += e;
      mantissa = std::frexp(mantissa, &e);
      exponent += e;
    }
  return std::ldexp(mantissa, static_cast<int>(exponent));
}

// Balanced determinant accumulation (reviewer repro on MR 2691): the running
// product is kept as mantissa * 2^e with exact frexp/ldexp renormalization, so
// partial products that leave the representable range cannot destroy a
// representable determinant, while zeros and genuine overflow still propagate.
void test_vandermonde_determinant_scaled() {
  typedef Matrix<double, Dynamic, 1> Vec;
  typedef std::complex<double> Complex;
  // The implementation accumulates the identical factor sequence as the
  // reference above, so the two sides differ only by the exactness of the
  // power-of-two renormalization.
  const double kBalancedTol = 16 * NumTraits<double>::epsilon();
  // The analytic values below additionally absorb the representation error of
  // the decimal node literals and of pow(10, 27.5), amplified by the 15-factor
  // product.
  const double kAnalyticTol = 100 * NumTraits<double>::epsilon();

  const double a = 1e-110, M = std::pow(10.0, 27.5);
  {
    // Underflow side (the reviewer's exact node set): the a-spaced factors alone
    // reach 2e-330, below the smallest subnormal, so the naive running product
    // flushes to an exact zero -- yet the determinant is 864 * a^3 * M^12 ~ 864.
    Vec x(6);
    x << 0.0, a, 2 * a, M, 2 * M, 3 * M;
    Vandermonde<double> V(x);
    const double det = V.determinant();
    const double ref = reference_vandermonde_det(x);
    VERIFY((numext::isfinite)(det));
    VERIFY(numext::abs(det / ref - 1.0) <= kBalancedTol);
    VERIFY(numext::abs(det / 864.0 - 1.0) <= kAnalyticTol);
  }
  {
    // Overflow-side analogue: with the large nodes first the naive running
    // product tops out at ~4.3e327 (infinity) three factors before the end, yet
    // the determinant is a representable -864 * B^12 * s^3 = -8.64e305.
    const double B = 1e28, s = 1e-11;
    Vec x(6);
    x << B, 2 * B, 3 * B, 0.0, s, 2 * s;
    Vandermonde<double> V(x);
    const double det = V.determinant();
    const double ref = reference_vandermonde_det(x);
    VERIFY((numext::isfinite)(det));
    VERIFY(numext::abs(det / ref - 1.0) <= kBalancedTol);
    VERIFY(numext::abs(det / -8.64e305 - 1.0) <= kAnalyticTol);
  }
  {
    // Genuinely overflowing determinant, ~3.5e424: must saturate to +infinity.
    const double T = 1e28;
    Vec x(6);
    x << 0.0, T, 2 * T, 3 * T, 4 * T, 5 * T;
    Vandermonde<double> V(x);
    const double det = V.determinant();
    VERIFY((numext::isinf)(det) && det > 0.0);
  }
  {
    // Genuinely underflowing determinant: subnormal-spaced nodes give 15 factors
    // below 2^-1071, a product near 2^-16000, which must saturate to zero with
    // the sign of the factored form: +0 for ascending nodes (all factors
    // positive), -0 for the descending permutation (15 negative factors).
    const double d = std::numeric_limits<double>::denorm_min();
    Vec x(6);
    x << 0.0, d, 2 * d, 3 * d, 4 * d, 5 * d;
    const double det = Vandermonde<double>(x).determinant();
    VERIFY(det == 0.0 && !std::signbit(det));
    const double detNeg = Vandermonde<double>(Vec(x.reverse())).determinant();
    VERIFY(detNeg == 0.0 && std::signbit(detNeg));
  }
  {
    // A repeated node makes the matrix exactly singular: the zero factor must
    // propagate to an exact 0 even though the naive running product overflows
    // beforehand (which would end in 0 * inf = NaN).
    Vec x(5);
    x << 0.0, 1e300, 1e-300, 1e300, 5.0;
    Vandermonde<double> V(x);
    VERIFY(V.determinant() == 0.0);
  }
  {
    // Complex path of the balanced accumulation: purely imaginary nodes i*x
    // turn every factor of the underflow-side case into i*(x_j - x_i), so
    // det = i^15 * 864 = -864i.
    Vec xr(6);
    xr << 0.0, a, 2 * a, M, 2 * M, 3 * M;
    Matrix<Complex, Dynamic, 1> x = Complex(0, 1) * xr.cast<Complex>();
    Vandermonde<Complex> V(x);
    const Complex det = V.determinant();
    const Complex ref(0.0, -reference_vandermonde_det(xr));
    VERIFY((numext::isfinite)(numext::abs(det)));
    VERIFY(numext::abs(det - ref) <= kBalancedTol * numext::abs(ref));
  }
}

// Reviewer reproducer on MR 2691: a node difference can overflow even though the
// determinant is representable. For nodes [-DBL_MAX, DBL_MAX, d, 2d, ..., 5d]
// (d the smallest subnormal) the factor DBL_MAX - (-DBL_MAX) is 2^1025, yet
// det = -576 * DBL_MAX^11 * d^10 ~ -3.1633e160. The implementation must halve
// the overflowing difference (exact for the huge, normal operands involved) and
// carry the factor of two in the running exponent.
void test_vandermonde_determinant_overflowing_differences() {
  typedef Matrix<double, Dynamic, 1> Vec;
  typedef std::complex<double> Complex;
  const double M = (std::numeric_limits<double>::max)();
  const double d = std::numeric_limits<double>::denorm_min();
  const double kBalancedTol = 16 * NumTraits<double>::epsilon();
  const double kAnalyticTol = 100 * NumTraits<double>::epsilon();

  Vec x(7);
  x << -M, M, d, 2 * d, 3 * d, 4 * d, 5 * d;
  const double det = Vandermonde<double>(x).determinant();
  VERIFY((numext::isfinite)(det));

  // Scaled reference computation with the identical factor sequence.
  const double ref = reference_vandermonde_det(x);
  VERIFY(numext::abs(det / ref - 1.0) <= kBalancedTol);

  // Analytic factored form, accumulated with the same exponent bookkeeping:
  // every huge factor rounds to +-DBL_MAX (and the halved leading factor is
  // exactly DBL_MAX * 2), the d-spaced block contributes exactly 288 * 2^-10740,
  // so det = -2 * 288 * DBL_MAX^11 * 2^-10740.
  double mantissa = -576.0;
  Index exponent = -10740;
  for (int t = 0; t < 11; ++t) {
    int e;
    mantissa *= std::frexp(M, &e);
    exponent += e;
    mantissa = std::frexp(mantissa, &e);
    exponent += e;
  }
  const double expected = std::ldexp(mantissa, static_cast<int>(exponent));
  VERIFY(numext::abs(det / expected - 1.0) <= kAnalyticTol);
  VERIFY(numext::abs(det / -3.1633e160 - 1.0) <= 1e-3);  // the reviewer's quoted value

  // Complex nodes i*x: every factor becomes i*(x_j - x_i) -- the overflow now
  // sits in the imaginary components, which the component-wise finiteness check
  // must catch -- and det picks up i^21 = i.
  Matrix<Complex, Dynamic, 1> xc = Complex(0, 1) * x.cast<Complex>();
  const Complex detc = Vandermonde<Complex>(xc).determinant();
  const Complex refc = Complex(0, 1) * Complex(ref);
  VERIFY((numext::isfinite)(detc));
  VERIFY(numext::abs(detc - refc) <= kBalancedTol * numext::abs(refc));
}

// Reviewer reproducer on MR 2691: Horner intermediates can overflow even though
// the polynomial value is representable. At the node 1/2 with coefficients
// [0, DBL_MAX, DBL_MAX] the intermediate is 1.5 * DBL_MAX -> Inf, but the value
// is exactly fl(0.75 * DBL_MAX). The scaled Horner path must return that value
// bit-exactly, the plain path must remain bit-identical for moderate data, and
// genuinely unrepresentable values must still saturate.
void test_vandermonde_scaled_horner() {
  typedef Matrix<double, Dynamic, 1> Vec;
  typedef std::complex<double> Complex;
  typedef Matrix<Complex, Dynamic, 1> CVec;
  const double M = (std::numeric_limits<double>::max)();

  {
    // The reviewer's exact reproducer, in each accumulation form.
    Vec x(1);
    x << 0.5;
    Vandermonde<double> V(x, 3);
    Vec a(3);
    a << 0.0, M, M;
    Vec y = V * a;
    VERIFY_IS_EQUAL(y[0], 0.75 * M);
    y.setZero();
    y.noalias() += V * a;
    VERIFY_IS_EQUAL(y[0], 0.75 * M);
    // Two identical columns exercise the per-column screening.
    Matrix<double, Dynamic, Dynamic> A(3, 2);
    A.col(0) = a;
    A.col(1) = a;
    Matrix<double, Dynamic, Dynamic> Y = V * A;
    VERIFY_IS_EQUAL(Y(0, 0), 0.75 * M);
    VERIFY_IS_EQUAL(Y(0, 1), 0.75 * M);
  }
  {
    // Complex path: the same coefficients rotated by i keep the overflow in a
    // single component; the value is i * fl(0.75 * DBL_MAX).
    CVec xc(1);
    xc << Complex(0.5, 0.0);
    Vandermonde<Complex> Vc(xc, 3);
    CVec ac(3);
    ac << Complex(0), Complex(0, M), Complex(0, M);
    CVec yc = Vc * ac;
    VERIFY_IS_EQUAL(numext::real(yc[0]), 0.0);
    VERIFY_IS_EQUAL(numext::imag(yc[0]), 0.75 * M);
  }
  {
    // Genuine overflow must still saturate: value 3 * DBL_MAX at the node 2.
    Vec x(1);
    x << 2.0;
    Vandermonde<double> V(x, 2);
    Vec a(2);
    a << M, M;
    Vec y = V * a;
    VERIFY((numext::isinf)(y[0]) && y[0] > 0.0);
    a << M, -M;  // value -DBL_MAX: representable again, sign preserved
    y = V * a;
    VERIFY_IS_EQUAL(y[0], -M);
  }
  {
    // Moderate data keeps the plain path: bit-identical to a naive Horner loop.
    const Index m = 7, n = 6;
    Vec x = Vec::Random(m), a = Vec::Random(n);
    Vandermonde<double> V(x, n);
    Vec y = V * a;
    for (Index i = 0; i < m; ++i) {
      double acc = a[n - 1];
      for (Index j = n - 2; j >= 0; --j) acc = acc * x[i] + a[j];
      VERIFY_IS_EQUAL(y[i], acc);
    }
  }
}

template <typename Scalar, int M, int N>
void test_vandermonde_fixed() {
  typedef Matrix<Scalar, M, 1> NodeVec;
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, M, N> MatMN;

  NodeVec x = NodeVec::Random();
  Vandermonde<Scalar, M, N> V(x, N);
  STATIC_CHECK((Vandermonde<Scalar, M, N>::RowsAtCompileTime == M));
  STATIC_CHECK((Vandermonde<Scalar, M, N>::ColsAtCompileTime == N));
  STATIC_CHECK((internal::remove_all_t<decltype(makeVandermonde(x))>::ColsAtCompileTime == M));

  MatMN dense = V;
  VERIFY_IS_APPROX(dense, MatMN(reference_vandermonde<Scalar>(Vec(x), N)));

  Matrix<Scalar, N, 1> a = Matrix<Scalar, N, 1>::Random();
  Matrix<Scalar, M, 1> y = V * a;
  VERIFY_IS_APPROX(y, (dense * a).eval());
}

EIGEN_DECLARE_TEST(structured_vandermonde) {
  for (int i = 0; i < g_repeat; ++i) {
    // Horner products, dense assignment, coefficient access.
    CALL_SUBTEST_1((test_vandermonde_product<double>(1, 1)));
    CALL_SUBTEST_1((test_vandermonde_product<double>(8, 8)));
    CALL_SUBTEST_1((test_vandermonde_product<double>(20, 12)));  // tall
    CALL_SUBTEST_1((test_vandermonde_product<double>(12, 20)));  // wide
    CALL_SUBTEST_1((test_vandermonde_product<float>(10, 10)));
    CALL_SUBTEST_1((test_vandermonde_product<std::complex<double>>(9, 7)));
    CALL_SUBTEST_1((test_vandermonde_product<std::complex<float>>(7, 9)));

    // Björck-Pereyra primal/dual/adjoint solves.
    CALL_SUBTEST_2((test_bjorck_pereyra<double>(1)));
    CALL_SUBTEST_2((test_bjorck_pereyra<double>(2)));
    CALL_SUBTEST_2((test_bjorck_pereyra<double>(10)));
    CALL_SUBTEST_2((test_bjorck_pereyra<double>(14)));
    CALL_SUBTEST_2((test_bjorck_pereyra<std::complex<double>>(10)));
    CALL_SUBTEST_2(test_bjorck_pereyra_roots_of_unity(16));
    CALL_SUBTEST_2(test_bjorck_pereyra_roots_of_unity(25));
    CALL_SUBTEST_2(test_bjorck_pereyra_higham());
    CALL_SUBTEST_2(test_bjorck_pereyra_singular());

    // Closed-form determinant and fixed sizes.
    CALL_SUBTEST_3((test_vandermonde_determinant<double>(8)));
    CALL_SUBTEST_3((test_vandermonde_determinant<std::complex<double>>(7)));
    CALL_SUBTEST_3(test_vandermonde_determinant_scaled());
    CALL_SUBTEST_3(test_vandermonde_determinant_overflowing_differences());
    CALL_SUBTEST_3(test_vandermonde_scaled_horner());
    CALL_SUBTEST_3((test_vandermonde_fixed<double, 6, 4>()));
    CALL_SUBTEST_3((test_vandermonde_fixed<double, 5, 5>()));
    CALL_SUBTEST_3((test_vandermonde_fixed<std::complex<float>, 4, 4>()));

    // Product regressions: aliasing, operand lifetime, mixed scalars.
    CALL_SUBTEST_4((test_vandermonde_aliased_product<double>(8)));
    CALL_SUBTEST_4((test_vandermonde_aliased_product<double>(17)));
    CALL_SUBTEST_4((test_vandermonde_aliased_product<std::complex<double>>(9)));
    CALL_SUBTEST_4((test_vandermonde_aliased_expression<double>(8)));
    CALL_SUBTEST_4((test_vandermonde_aliased_expression<std::complex<double>>(9)));
    CALL_SUBTEST_4((test_vandermonde_delayed_product<double>(12, 9)));
    CALL_SUBTEST_4((test_vandermonde_delayed_product<std::complex<double>>(8, 8)));
    CALL_SUBTEST_4((test_vandermonde_mixed_scalar<double>(10, 8)));
    CALL_SUBTEST_4((test_vandermonde_mixed_scalar<float>(7, 9)));
  }
}
