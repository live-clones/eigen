// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <complex>

#include "main.h"
#include <contrib/Eigen/NumericalDiff>

using Eigen::Array;
using Eigen::makeStencil;
using Eigen::Stencil;

// Independent oracle (see .agents/numerics.md): for order-Derivative weights at x0 = 0 over points
// p_0..p_{Size-1}, sum_i weights[i] * p_i^k == (k == Derivative ? Derivative! : 0) for every
// k = 0..Size-1. This holds for any polynomial reproduced exactly by an (Size-1)-degree
// interpolant, independently of how Stencil's recursion is implemented.
template <unsigned Derivative, typename Scalar, int Size>
void verify_polynomial_exactness(const Array<Scalar, Size, 1> &points) {
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;

  const Stencil<Derivative, Scalar, Size> s(points);
  const auto w = s.weights();
  const auto p = s.points();

  for (std::size_t i = 0; i < Size; ++i) {
    VERIFY_IS_APPROX(p[i], points[i]);
  }

  Scalar fact(1);
  for (unsigned i = 2; i <= Derivative; ++i) fact *= Scalar(i);

  Scalar term[Size];
  for (std::size_t i = 0; i < Size; ++i) term[i] = Scalar(1);

  for (std::size_t k = 0; k < Size; ++k) {
    Scalar acc(0);
    RealScalar magSum(0);
    for (std::size_t i = 0; i < Size; ++i) {
      const Scalar contribution = w[i] * term[i];
      acc += contribution;
      magSum += numext::abs(contribution);
    }
    const Scalar expected = (k == Derivative) ? fact : Scalar(0);
    const RealScalar tol = RealScalar(50) * RealScalar(Size) * (magSum + RealScalar(1)) * Eigen::NumTraits<RealScalar>::epsilon();
    VERIFY(numext::abs(acc - expected) <= tol);
    for (std::size_t i = 0; i < Size; ++i) term[i] *= p[i];
  }
}

void test_textbook_central() {
  const Array<double, 3, 1> points{{-1.0, 0.0, 1.0}};
  const double tol = 10.0 * Eigen::NumTraits<double>::epsilon();

  const Stencil<1, double, 3> d1(points);
  const auto w1 = d1.weights();
  VERIFY(numext::abs(w1[0] - (-0.5)) <= tol);
  VERIFY(numext::abs(w1[1] - 0.0) <= tol);
  VERIFY(numext::abs(w1[2] - 0.5) <= tol);

  const Stencil<2, double, 3> d2(points);
  const auto w2 = d2.weights();
  VERIFY(numext::abs(w2[0] - 1.0) <= tol);
  VERIFY(numext::abs(w2[1] - (-2.0)) <= tol);
  VERIFY(numext::abs(w2[2] - 1.0) <= tol);
}

void test_textbook_forward() {
  const Array<double, 3, 1> points{{0.0, 1.0, 2.0}};
  const double tol = 10.0 * Eigen::NumTraits<double>::epsilon();

  const auto d1 = makeStencil<1>(points);
  const auto w1 = d1.weights();
  VERIFY(numext::abs(w1[0] - (-1.5)) <= tol);
  VERIFY(numext::abs(w1[1] - 2.0) <= tol);
  VERIFY(numext::abs(w1[2] - (-0.5)) <= tol);
}

// Derivative == 0 (interpolation), and the boundary case Derivative == Size - 1, fall out of the
// same sweep since verify_polynomial_exactness checks every k = 0..Size-1 already.
void test_polynomial_exactness_uniform() {
  const Array<double, 5, 1> points{{-2.0, -1.0, 0.0, 1.0, 2.0}};
  verify_polynomial_exactness<0>(points);
  verify_polynomial_exactness<1>(points);
  verify_polynomial_exactness<2>(points);
  verify_polynomial_exactness<3>(points);
  verify_polynomial_exactness<4>(points);
}

// Widely varying spacing: exercises the "arbitrary spacing" case without relying on any internal
// reordering to keep the recursion well conditioned.
void test_polynomial_exactness_nonuniform() {
  const Array<double, 5, 1> points{{-1000.0, -1.0, 0.0, 1.0, 1000.0}};
  verify_polynomial_exactness<0>(points);
  verify_polynomial_exactness<1>(points);
  verify_polynomial_exactness<2>(points);
  verify_polynomial_exactness<4>(points);
}

// Points that do not include the evaluation point 0.
void test_polynomial_exactness_no_zero() {
  const Array<double, 4, 1> points{{-3.0, -1.0, 2.0, 5.0}};
  verify_polynomial_exactness<0>(points);
  verify_polynomial_exactness<1>(points);
  verify_polynomial_exactness<3>(points);
}

void test_polynomial_exactness_float() {
  const Array<float, 4, 1> points{{-2.0f, -1.0f, 1.0f, 2.0f}};
  verify_polynomial_exactness<0>(points);
  verify_polynomial_exactness<1>(points);
  verify_polynomial_exactness<3>(points);
}

void test_polynomial_exactness_complex() {
  typedef std::complex<double> Scalar;
  const Array<Scalar, 4, 1> points{{Scalar(-1.0, 1.0), Scalar(0.0, 0.0), Scalar(1.0, 1.0), Scalar(2.0, -1.0)}};
  verify_polynomial_exactness<0>(points);
  verify_polynomial_exactness<1>(points);
  verify_polynomial_exactness<3>(points);
}

// No internal reordering: weights()[i] must keep tracking points()[i] as given, so permuting the
// input permutes the output identically.
void test_order_preservation() {
  const Array<double, 5, 1> points{{-1000.0, -1.0, 0.0, 1.0, 1000.0}};
  const Array<double, 5, 1> reversed{{1000.0, 1.0, 0.0, -1.0, -1000.0}};

  const Stencil<2, double, 5> s(points);
  const Stencil<2, double, 5> sReversed(reversed);
  const auto w = s.weights();
  const auto wReversed = sReversed.weights();

  for (std::size_t i = 0; i < 5; ++i) {
    const double tol = 100.0 * Eigen::NumTraits<double>::epsilon() * (numext::abs(w[i]) + 1.0);
    VERIFY(numext::abs(w[i] - wReversed[4 - i]) <= tol);
  }
}

// Empirical confirmation that Stencil/makeStencil are usable in an actual constant expression, not
// just declared constexpr, under this toolchain's C++14 mode. weights()/points() return a Map, whose
// constructor is not constexpr-evaluable on this toolchain (MapBase::checkSanity() is not constexpr),
// so the compile-time check stops at construction; value checks against weights() run at runtime.
constexpr Array<double, 3, 1> centralPoints{{-1.0, 0.0, 1.0}};
constexpr auto centralDiff1 = makeStencil<1>(centralPoints);

void test_compile_time_smoke() {
  const auto w = centralDiff1.weights();
  VERIFY(numext::abs(w[0] - (-0.5)) <= Eigen::NumTraits<double>::epsilon());
  VERIFY(numext::abs(w[1] - 0.0) <= Eigen::NumTraits<double>::epsilon());
  VERIFY(numext::abs(w[2] - 0.5) <= Eigen::NumTraits<double>::epsilon());
}

EIGEN_DECLARE_TEST(stencil) {
  CALL_SUBTEST(test_textbook_central());
  CALL_SUBTEST(test_textbook_forward());
  CALL_SUBTEST(test_polynomial_exactness_uniform());
  CALL_SUBTEST(test_polynomial_exactness_nonuniform());
  CALL_SUBTEST(test_polynomial_exactness_no_zero());
  CALL_SUBTEST(test_polynomial_exactness_float());
  CALL_SUBTEST(test_polynomial_exactness_complex());
  CALL_SUBTEST(test_order_preservation());
  CALL_SUBTEST(test_compile_time_smoke());
}
