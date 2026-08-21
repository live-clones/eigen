// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Checks the cross-arm agreement predicate in benchmarks/comparison/bench_compare.h.
//
// agreesWithEigen is the only thing standing between a broken reference kernel
// and a published GFLOP/s number, and it is memoised: one false positive marks
// the shape validated for the whole binary, so every repetition after it is
// timed without being checked again. The cases below are the ones where a
// plausible spelling of the predicate says yes to a result no one would accept
// -- NaN, which every comparison operator is false against, and infinity, which
// makes BOTH sides of a relative comparison infinite and so passes a test
// written as `error <= tolerance * magnitude`.
//
// Deliberately free of GoogleTest, like tests/test_flops.cpp beside it: the
// standalone benchmarks project must not grow that dependency. CHECK rather than
// <cassert> because benchmark targets compile with NDEBUG.

#include <complex>
#include <cstdio>
#include <limits>
#include <string>

#include "benchmarks/comparison/bench_compare.h"

namespace {

int failures = 0;

void checkImpl(bool condition, const char* expression, const char* file, int line) {
  if (!condition) {
    std::fprintf(stderr, "%s:%d: FAILED %s\n", file, line, expression);
    ++failures;
  }
}

#define CHECK(cond) checkImpl((cond), #cond, __FILE__, __LINE__)

using Eigen::Index;
using eigen_bench::agreesWithEigen;

template <typename Scalar>
using Vector = eigen_bench::ColVector<Scalar>;

template <typename Scalar>
using Matrix = eigen_bench::ColMatrix<Scalar>;

// The contraction length the harness would pass; it only scales the tolerance.
constexpr Index kLength = 16;

template <typename Scalar>
void checkAgreementRejectsNonFinite(const char* what) {
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
  const RealScalar infinity = std::numeric_limits<RealScalar>::infinity();
  const RealScalar quiet_nan = std::numeric_limits<RealScalar>::quiet_NaN();

  const Vector<Scalar> expected = Vector<Scalar>::Constant(kLength, Scalar(1));

  // The control: an identical result agrees, so a rejection below is the
  // non-finite entry and not a predicate that refuses everything.
  CHECK(agreesWithEigen(expected, Vector<Scalar>(expected), kLength));
  (void)what;

  // A single infinity. `(actual - expected).norm()` and `actual.norm()` are both
  // inf, and `inf <= tolerance * inf` is true, so a predicate that only negates
  // for NaN accepts this.
  Vector<Scalar> overflowed = expected;
  overflowed(kLength / 2) = Scalar(infinity);
  CHECK(!agreesWithEigen(expected, overflowed, kLength));

  // Infinity of the other sign, and in the value Eigen itself produced.
  Vector<Scalar> negative_overflow = expected;
  negative_overflow(0) = Scalar(-infinity);
  CHECK(!agreesWithEigen(expected, negative_overflow, kLength));
  CHECK(!agreesWithEigen(overflowed, expected, kLength));

  // Both arms infinite: the difference is NaN rather than inf, which the
  // negation already rejected, but the norms are inf either way.
  CHECK(!agreesWithEigen(overflowed, Vector<Scalar>(overflowed), kLength));

  // NaN, the case the negation was written for; it must stay rejected.
  Vector<Scalar> not_a_number = expected;
  not_a_number(kLength - 1) = Scalar(quiet_nan);
  CHECK(!agreesWithEigen(expected, not_a_number, kLength));

  // A wrong-but-finite result is still rejected, and a result differing only by
  // rounding is still accepted, so the new guard did not swallow the tolerance.
  Vector<Scalar> wrong = expected;
  wrong(1) = Scalar(2);
  CHECK(!agreesWithEigen(expected, wrong, kLength));

  Vector<Scalar> rounded = expected;
  rounded(1) = Scalar(RealScalar(1) + Eigen::NumTraits<RealScalar>::epsilon());
  CHECK(agreesWithEigen(expected, rounded, kLength));
}

// Matrix operands take the same path; POTRF and GEMM validate through this
// predicate with matrices, not vectors.
template <typename Scalar>
void checkAgreementRejectsNonFiniteMatrix() {
  const Scalar infinity = Scalar(std::numeric_limits<typename Eigen::NumTraits<Scalar>::Real>::infinity());
  const Matrix<Scalar> expected = Matrix<Scalar>::Identity(4, 4);
  Matrix<Scalar> overflowed = expected;
  overflowed(2, 1) = infinity;
  CHECK(agreesWithEigen(expected, Matrix<Scalar>(expected), kLength));
  CHECK(!agreesWithEigen(expected, overflowed, kLength));
}

// An all-zero pair has magnitude 0 and error 0; `0 <= 0` must stay true, or
// every operation whose validation operand is legitimately zero would be
// reported as a kernel disagreement.
template <typename Scalar>
void checkZeroAgreesWithZero() {
  const Vector<Scalar> zero = Vector<Scalar>::Zero(kLength);
  CHECK(agreesWithEigen(zero, Vector<Scalar>(zero), kLength));
}

}  // namespace

int main() {
  checkAgreementRejectsNonFinite<float>("float");
  checkAgreementRejectsNonFinite<double>("double");
  checkAgreementRejectsNonFinite<std::complex<float>>("complex<float>");
  checkAgreementRejectsNonFinite<std::complex<double>>("complex<double>");

  checkAgreementRejectsNonFiniteMatrix<float>();
  checkAgreementRejectsNonFiniteMatrix<double>();
  checkAgreementRejectsNonFiniteMatrix<std::complex<double>>();

  checkZeroAgreesWithZero<float>();
  checkZeroAgreesWithZero<double>();
  checkZeroAgreesWithZero<std::complex<double>>();

  if (failures != 0) {
    std::fprintf(stderr, "%d validation check(s) failed\n", failures);
    return 1;
  }
  std::fprintf(stderr, "all validation checks passed\n");
  return 0;
}
