// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Pavel Guzenfeld
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <Eigen/QR>

template <typename Scalar>
void dgks_basic_orthogonality() {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, Dynamic> MatrixType;
  typedef Matrix<Scalar, Dynamic, 1> VectorType;

  Index n = 8;
  Index k = 5;

  // Build an orthonormal basis Q from a random matrix.
  MatrixType A = MatrixType::Random(n, k);
  HouseholderQR<MatrixType> qr(A);
  MatrixType Q = qr.householderQ() * MatrixType::Identity(n, k);

  // Random vector to orthogonalize.
  VectorType v = VectorType::Random(n);
  VectorType v_copy = v;

  RealScalar norm = internal::dgks_orthogonalize(Q, v, k);

  // v should be orthogonal to all columns of Q.
  VectorType projections = Q.adjoint() * v;
  VERIFY(projections.stableNorm() < RealScalar(n) * NumTraits<RealScalar>::epsilon() * v_copy.stableNorm());

  // Returned norm should match actual norm.
  VERIFY_IS_APPROX(norm, v.stableNorm());

  // Norm should be positive (random vector unlikely to be in span of Q).
  VERIFY(norm > RealScalar(0));
}

template <typename Scalar>
void dgks_coefficient_accumulation() {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, Dynamic> MatrixType;
  typedef Matrix<Scalar, Dynamic, 1> VectorType;

  Index n = 6;
  Index k = 4;

  MatrixType A = MatrixType::Random(n, k);
  HouseholderQR<MatrixType> qr(A);
  MatrixType Q = qr.householderQ() * MatrixType::Identity(n, k);

  VectorType v = VectorType::Random(n);
  VectorType v_orig = v;
  VectorType coeffs = VectorType::Zero(k);

  RealScalar norm = internal::dgks_orthogonalize(Q, v, k, &coeffs);

  // v_orig = Q * coeffs + v (residual), so Q * coeffs + v should reconstruct v_orig.
  VectorType reconstructed = Q * coeffs + v;
  VERIFY_IS_APPROX(reconstructed, v_orig);
  (void)norm;
}

template <typename Scalar>
void dgks_vector_in_span() {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, Dynamic> MatrixType;
  typedef Matrix<Scalar, Dynamic, 1> VectorType;

  Index n = 6;
  Index k = 4;

  MatrixType A = MatrixType::Random(n, k);
  HouseholderQR<MatrixType> qr(A);
  MatrixType Q = qr.householderQ() * MatrixType::Identity(n, k);

  // v is in the span of Q => should be orthogonalized to ~zero.
  VectorType c = VectorType::Random(k);
  VectorType v = Q * c;

  RealScalar norm = internal::dgks_orthogonalize(Q, v, k);

  VERIFY(norm < RealScalar(n) * NumTraits<RealScalar>::epsilon() * c.stableNorm());
}

template <typename Scalar>
void dgks_already_orthogonal() {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, Dynamic> MatrixType;
  typedef Matrix<Scalar, Dynamic, 1> VectorType;

  Index n = 8;
  Index k = 5;

  // Build full orthonormal basis.
  MatrixType A = MatrixType::Random(n, n);
  HouseholderQR<MatrixType> qr(A);
  MatrixType Qfull = qr.householderQ();
  MatrixType Q = Qfull.leftCols(k);

  // Take a vector already orthogonal to Q.
  VectorType v = Qfull.col(k);
  VectorType v_orig = v;

  RealScalar norm = internal::dgks_orthogonalize(Q, v, k);

  // Should be essentially unchanged (up to rounding).
  VERIFY_IS_APPROX(norm, v_orig.stableNorm());
  // Should still be orthogonal.
  VectorType projections = Q.adjoint() * v;
  VERIFY(projections.stableNorm() < RealScalar(n) * NumTraits<RealScalar>::epsilon());
}

template <typename Scalar>
void dgks_nearly_dependent() {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, Dynamic> MatrixType;
  typedef Matrix<Scalar, Dynamic, 1> VectorType;

  Index n = 8;
  Index k = 4;

  MatrixType A = MatrixType::Random(n, k);
  HouseholderQR<MatrixType> qr(A);
  MatrixType Q = qr.householderQ() * MatrixType::Identity(n, k);

  // v is nearly in span of Q: small orthogonal component.
  VectorType c = VectorType::Random(k);
  VectorType perp = VectorType::Random(n);
  perp -= Q * (Q.adjoint() * perp);
  perp.normalize();

  RealScalar epsilon = RealScalar(1e-10);
  VectorType v = Q * c + epsilon * perp;

  RealScalar norm = internal::dgks_orthogonalize(Q, v, k);

  // Should recover approximately the small orthogonal component.
  // Tolerance scales with the magnitude of the input projection.
  RealScalar tol = RealScalar(n) * NumTraits<RealScalar>::epsilon() * c.stableNorm();
  VERIFY(numext::abs(norm - epsilon) < tol + epsilon * tol);

  // Should be orthogonal to Q.
  VectorType projections = Q.adjoint() * v;
  VERIFY(projections.stableNorm() < tol);
}

template <typename Scalar>
void dgks_extreme_magnitudes() {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, Dynamic> MatrixType;
  typedef Matrix<Scalar, Dynamic, 1> VectorType;

  Index n = 6;
  Index k = 4;

  MatrixType A = MatrixType::Random(n, k);
  HouseholderQR<MatrixType> qr(A);
  MatrixType Q = qr.householderQ() * MatrixType::Identity(n, k);

  // Large magnitude vector.
  {
    RealScalar big = numext::sqrt(NumTraits<RealScalar>::highest()) * RealScalar(0.1);
    VectorType v = VectorType::Random(n) * big;
    RealScalar norm = internal::dgks_orthogonalize(Q, v, k);
    VERIFY((numext::isfinite)(norm));
    VectorType projections = Q.adjoint() * v;
    VERIFY(projections.stableNorm() < RealScalar(n) * NumTraits<RealScalar>::epsilon() * big);
  }

  // Tiny magnitude vector.
  {
    RealScalar tiny = numext::sqrt(NumTraits<RealScalar>::lowest()) * RealScalar(10);
    VectorType v = VectorType::Random(n) * tiny;
    RealScalar norm = internal::dgks_orthogonalize(Q, v, k);
    VERIFY((numext::isfinite)(norm));
    VectorType projections = Q.adjoint() * v;
    VERIFY(projections.stableNorm() < RealScalar(n) * NumTraits<RealScalar>::epsilon() * tiny);
  }
}

template <typename Scalar>
void dgks_single_column() {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, Dynamic> MatrixType;
  typedef Matrix<Scalar, Dynamic, 1> VectorType;

  Index n = 6;

  VectorType q = VectorType::Random(n).normalized();
  MatrixType Q(n, 1);
  Q.col(0) = q;

  VectorType v = VectorType::Random(n);
  internal::dgks_orthogonalize(Q, v, 1);

  RealScalar dot = numext::abs(q.dot(v));
  VERIFY(dot < RealScalar(n) * NumTraits<RealScalar>::epsilon() * v.stableNorm());
}

template <typename Scalar>
void dgks_zero_columns() {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, Dynamic> MatrixType;
  typedef Matrix<Scalar, Dynamic, 1> VectorType;

  Index n = 6;

  MatrixType Q(n, 0);
  VectorType v = VectorType::Random(n);
  VectorType v_orig = v;

  RealScalar norm = internal::dgks_orthogonalize(Q, v, 0);

  // With zero columns, v should be unchanged.
  VERIFY_IS_APPROX(v, v_orig);
  VERIFY_IS_APPROX(norm, v.stableNorm());
}

EIGEN_DECLARE_TEST(dgks_reorthogonalization) {
  for (int i = 0; i < g_repeat; ++i) {
    CALL_SUBTEST_1(dgks_basic_orthogonality<float>());
    CALL_SUBTEST_2(dgks_basic_orthogonality<double>());
    CALL_SUBTEST_3(dgks_basic_orthogonality<std::complex<double>>());
  }

  for (int i = 0; i < g_repeat; ++i) {
    CALL_SUBTEST_4(dgks_coefficient_accumulation<double>());
    CALL_SUBTEST_5(dgks_coefficient_accumulation<std::complex<double>>());
  }

  CALL_SUBTEST_6(dgks_vector_in_span<double>());
  CALL_SUBTEST_7(dgks_vector_in_span<float>());

  CALL_SUBTEST_8(dgks_already_orthogonal<double>());
  CALL_SUBTEST_9(dgks_nearly_dependent<double>());
  CALL_SUBTEST_10(dgks_extreme_magnitudes<double>());
  CALL_SUBTEST_11(dgks_extreme_magnitudes<float>());
  CALL_SUBTEST_12(dgks_single_column<double>());
  CALL_SUBTEST_13(dgks_zero_columns<double>());
}
