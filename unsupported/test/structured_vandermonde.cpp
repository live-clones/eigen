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

  // Primal: interpolate values of a known polynomial and recover its coefficients.
  Vec aTrue = Vec::Random(n);
  Vec f = dense * aTrue;
  Vec a = bp.solve(f);
  VERIFY((a - aTrue).norm() <= RealScalar(1e-8) * aTrue.norm());

  // Dual (moment) system through the SolverBase transpose idiom.
  Vec wTrue = Vec::Random(n);
  Vec b = dense.transpose() * wTrue;
  Vec w = bp.transpose().solve(b);
  VERIFY((w - wTrue).norm() <= RealScalar(1e-8) * wTrue.norm());

  // Adjoint solve.
  Vec c = dense.adjoint() * wTrue;
  Vec u = bp.adjoint().solve(c);
  VERIFY((u - wTrue).norm() <= RealScalar(1e-8) * wTrue.norm());

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
  VERIFY(r.norm() <= 1e-10 * a.norm());
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

template <typename Scalar>
void test_vandermonde_determinant(Index n) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Vec x = Vec::Random(n);
  Vandermonde<Scalar> V(x);
  Mat dense = reference_vandermonde<Scalar>(x, n);
  VERIFY_IS_APPROX(V.determinant(), dense.determinant());
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
    CALL_SUBTEST_3((test_vandermonde_fixed<double, 6, 4>()));
    CALL_SUBTEST_3((test_vandermonde_fixed<double, 5, 5>()));
    CALL_SUBTEST_3((test_vandermonde_fixed<std::complex<float>, 4, 4>()));
  }
}
