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

// Aliased products: the structured products are tagged AliasFreeProduct, so no
// temporary shields the destination -- the product implementation itself must
// copy an aliased right-hand side (x = V * x, x += V * x) before writing.
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
// sequence accumulated with explicit exponent tracking. The power-of-two
// rescaling is exact, so this reproduces the exact product of the computed
// factors up to one rounding per multiplication -- representable even when
// naive partial products leave the double range. Finite factors only.
double reference_vandermonde_det(const Matrix<double, Dynamic, 1>& x) {
  double mantissa = 1.0;
  Index exponent = 0;
  for (Index j = 1; j < x.size(); ++j)
    for (Index i = 0; i < j; ++i) {
      int e;
      mantissa *= std::frexp(x[j] - x[i], &e);
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
    CALL_SUBTEST_3((test_vandermonde_fixed<double, 6, 4>()));
    CALL_SUBTEST_3((test_vandermonde_fixed<double, 5, 5>()));
    CALL_SUBTEST_3((test_vandermonde_fixed<std::complex<float>, 4, 4>()));

    // Product regressions: aliasing, operand lifetime, mixed scalars.
    CALL_SUBTEST_4((test_vandermonde_aliased_product<double>(8)));
    CALL_SUBTEST_4((test_vandermonde_aliased_product<double>(17)));
    CALL_SUBTEST_4((test_vandermonde_aliased_product<std::complex<double>>(9)));
    CALL_SUBTEST_4((test_vandermonde_delayed_product<double>(12, 9)));
    CALL_SUBTEST_4((test_vandermonde_delayed_product<std::complex<double>>(8, 8)));
    CALL_SUBTEST_4((test_vandermonde_mixed_scalar<double>(10, 8)));
    CALL_SUBTEST_4((test_vandermonde_mixed_scalar<float>(7, 9)));
  }
}
