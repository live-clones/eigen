// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// Shared header for split qr_colpivoting tests.

#ifndef EIGEN_TEST_QR_COLPIVOTING_HELPERS_H
#define EIGEN_TEST_QR_COLPIVOTING_HELPERS_H

#include "main.h"
#include <Eigen/QR>
#include <Eigen/SVD>
#include "solverbase.h"

template <typename MatrixType>
void qr() {
  using std::sqrt;

  Index rows = internal::random<Index>(2, EIGEN_TEST_MAX_SIZE), cols = internal::random<Index>(2, EIGEN_TEST_MAX_SIZE),
        cols2 = internal::random<Index>(2, EIGEN_TEST_MAX_SIZE);
  Index rank = internal::random<Index>(1, (std::min)(rows, cols) - 1);

  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime> MatrixQType;
  MatrixType m1;
  createRandomPIMatrixOfRank(rank, rows, cols, m1);
  ColPivHouseholderQR<MatrixType> qr(m1);
  VERIFY_IS_EQUAL(rank, qr.rank());
  VERIFY_IS_EQUAL(cols - qr.rank(), qr.dimensionOfKernel());
  VERIFY(!qr.isInjective());
  VERIFY(!qr.isInvertible());
  VERIFY(!qr.isSurjective());

  MatrixQType q = qr.householderQ();
  VERIFY_IS_UNITARY(q);

  MatrixType r = qr.matrixQR().template triangularView<Upper>();
  MatrixType c = q * r * qr.colsPermutation().inverse();
  VERIFY_IS_APPROX(m1, c);

  // Verify that the absolute value of the diagonal elements in R are
  // non-increasing until they reach the singularity threshold.
  RealScalar threshold = sqrt(RealScalar(rows)) * numext::abs(r(0, 0)) * NumTraits<Scalar>::epsilon();
  for (Index i = 0; i < (std::min)(rows, cols) - 1; ++i) {
    RealScalar x = numext::abs(r(i, i));
    RealScalar y = numext::abs(r(i + 1, i + 1));
    if (x < threshold && y < threshold) continue;
    if (!test_isApproxOrLessThan(y, x)) {
      for (Index j = 0; j < (std::min)(rows, cols); ++j) {
        std::cout << "i = " << j << ", |r_ii| = " << numext::abs(r(j, j)) << std::endl;
      }
      std::cout << "Failure at i=" << i << ", rank=" << rank << ", threshold=" << threshold << std::endl;
    }
    VERIFY_IS_APPROX_OR_LESS_THAN(y, x);
  }

  check_solverbase<MatrixType, MatrixType>(m1, qr, rows, cols, cols2);

  {
    MatrixType m2, m3;
    Index size = rows;
    // Create a random diagonally dominant (thus invertible) matrix.
    m1 = MatrixType::Random(size, size);
    m1.diagonal().array() += Scalar(2 * size);
    qr.compute(m1);
    MatrixType m1_inv = qr.inverse();
    m3 = m1 * MatrixType::Random(size, cols2);
    m2 = qr.solve(m3);
    VERIFY_IS_APPROX(m2, m1_inv * m3);
  }
}

template <typename MatrixType, int Cols2>
void qr_fixedsize() {
  using std::abs;
  using std::sqrt;
  enum { Rows = MatrixType::RowsAtCompileTime, Cols = MatrixType::ColsAtCompileTime };
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  int rank = internal::random<int>(1, (std::min)(int(Rows), int(Cols)) - 1);
  Matrix<Scalar, Rows, Cols> m1;
  createRandomPIMatrixOfRank(rank, Rows, Cols, m1);
  ColPivHouseholderQR<Matrix<Scalar, Rows, Cols> > qr(m1);
  VERIFY_IS_EQUAL(rank, qr.rank());
  VERIFY_IS_EQUAL(Cols - qr.rank(), qr.dimensionOfKernel());
  VERIFY_IS_EQUAL(qr.isInjective(), (rank == Rows));
  VERIFY_IS_EQUAL(qr.isSurjective(), (rank == Cols));
  VERIFY_IS_EQUAL(qr.isInvertible(), (qr.isInjective() && qr.isSurjective()));

  Matrix<Scalar, Rows, Cols> r = qr.matrixQR().template triangularView<Upper>();
  Matrix<Scalar, Rows, Cols> c = qr.householderQ() * r * qr.colsPermutation().inverse();
  VERIFY_IS_APPROX(m1, c);

  check_solverbase<Matrix<Scalar, Cols, Cols2>, Matrix<Scalar, Rows, Cols2> >(m1, qr, Rows, Cols, Cols2);

  RealScalar threshold = sqrt(RealScalar(Rows)) * (std::abs)(r(0, 0)) * NumTraits<Scalar>::epsilon();
  for (Index i = 0; i < (std::min)(int(Rows), int(Cols)) - 1; ++i) {
    RealScalar x = numext::abs(r(i, i));
    RealScalar y = numext::abs(r(i + 1, i + 1));
    if (x < threshold && y < threshold) continue;
    if (!test_isApproxOrLessThan(y, x)) {
      for (Index j = 0; j < (std::min)(int(Rows), int(Cols)); ++j) {
        std::cout << "i = " << j << ", |r_ii| = " << numext::abs(r(j, j)) << std::endl;
      }
      std::cout << "Failure at i=" << i << ", rank=" << rank << ", threshold=" << threshold << std::endl;
    }
    VERIFY_IS_APPROX_OR_LESS_THAN(y, x);
  }
}

template <typename MatrixType>
void qr_kahan_matrix() {
  using std::abs;
  using std::sqrt;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;

  Index rows = 300, cols = rows;

  MatrixType m1;
  m1.setZero(rows, cols);
  RealScalar s = std::pow(NumTraits<RealScalar>::epsilon(), 1.0 / rows);
  RealScalar c = std::sqrt(1 - s * s);
  RealScalar pow_s_i(1.0);
  for (Index i = 0; i < rows; ++i) {
    m1(i, i) = pow_s_i;
    m1.row(i).tail(rows - i - 1) = -pow_s_i * c * MatrixType::Ones(1, rows - i - 1);
    pow_s_i *= s;
  }
  m1 = (m1 + m1.transpose()).eval();
  ColPivHouseholderQR<MatrixType> qr(m1);
  MatrixType r = qr.matrixQR().template triangularView<Upper>();

  RealScalar threshold = std::sqrt(RealScalar(rows)) * numext::abs(r(0, 0)) * NumTraits<Scalar>::epsilon();
  for (Index i = 0; i < (std::min)(rows, cols) - 1; ++i) {
    RealScalar x = numext::abs(r(i, i));
    RealScalar y = numext::abs(r(i + 1, i + 1));
    if (x < threshold && y < threshold) continue;
    if (!test_isApproxOrLessThan(y, x)) {
      for (Index j = 0; j < (std::min)(rows, cols); ++j) {
        std::cout << "i = " << j << ", |r_ii| = " << numext::abs(r(j, j)) << std::endl;
      }
      std::cout << "Failure at i=" << i << ", rank=" << qr.rank() << ", threshold=" << threshold << std::endl;
    }
    VERIFY_IS_APPROX_OR_LESS_THAN(y, x);
  }
}

template <typename MatrixType>
void qr_invertible() {
  using std::abs;
  using std::log;
  typedef typename NumTraits<typename MatrixType::Scalar>::Real RealScalar;
  typedef typename MatrixType::Scalar Scalar;

  int size = internal::random<int>(10, 50);

  MatrixType m1(size, size), m2(size, size), m3(size, size);
  m1 = MatrixType::Random(size, size);

  if (internal::is_same<RealScalar, float>::value) {
    MatrixType a = MatrixType::Random(size, size * 2);
    m1 += a * a.adjoint();
  }

  ColPivHouseholderQR<MatrixType> qr(m1);

  check_solverbase<MatrixType, MatrixType>(m1, qr, size, size, size);

  m1.setZero();
  for (int i = 0; i < size; i++) m1(i, i) = internal::random<Scalar>();
  Scalar det = m1.diagonal().prod();
  RealScalar absdet = abs(det);
  m3 = qr.householderQ();
  m1 = m3 * m1 * m3.adjoint();
  qr.compute(m1);
  VERIFY_IS_APPROX(det, qr.determinant());
  VERIFY_IS_APPROX(absdet, qr.absDeterminant());
  VERIFY_IS_APPROX(log(absdet), qr.logAbsDeterminant());
  VERIFY_IS_APPROX(numext::sign(det), qr.signDeterminant());
}

template <typename MatrixType>
void qr_verify_assert() {
  MatrixType tmp;

  ColPivHouseholderQR<MatrixType> qr;
  VERIFY_RAISES_ASSERT(qr.matrixQR())
  VERIFY_RAISES_ASSERT(qr.solve(tmp))
  VERIFY_RAISES_ASSERT(qr.transpose().solve(tmp))
  VERIFY_RAISES_ASSERT(qr.adjoint().solve(tmp))
  VERIFY_RAISES_ASSERT(qr.householderQ())
  VERIFY_RAISES_ASSERT(qr.dimensionOfKernel())
  VERIFY_RAISES_ASSERT(qr.isInjective())
  VERIFY_RAISES_ASSERT(qr.isSurjective())
  VERIFY_RAISES_ASSERT(qr.isInvertible())
  VERIFY_RAISES_ASSERT(qr.inverse())
  VERIFY_RAISES_ASSERT(qr.determinant())
  VERIFY_RAISES_ASSERT(qr.absDeterminant())
  VERIFY_RAISES_ASSERT(qr.logAbsDeterminant())
  VERIFY_RAISES_ASSERT(qr.signDeterminant())
}

// Stress test: verify rank detection on partial isometries (SVs = 0 or 1) across
// many random trials and various aspect ratios (square, tall, wide).
// This tests ROBUSTNESS: the threshold must be large enough that roundoff noise
// in the null-space R diagonal elements does not cause rank overestimation.
template <typename MatrixType>
void qr_rank_detection_stress() {
  // Test a range of matrix sizes and aspect ratios.
  const Index sizes[][2] = {{10, 10}, {20, 20}, {50, 50}, {100, 100}, {40, 10}, {100, 10}, {10, 40}, {10, 100}};
  for (const auto& sz : sizes) {
    const Index rows = sz[0], cols = sz[1];
    const Index min_dim = (std::min)(rows, cols);
    // Test several rank values: 1, half, and min_dim - 1.
    for (Index rank : {Index(1), (std::max)(Index(1), min_dim / 2), min_dim - 1}) {
      if (rank >= min_dim) continue;
      for (int trial = 0; trial < 20; ++trial) {
        MatrixType m1;
        createRandomPIMatrixOfRank(rank, rows, cols, m1);
        ColPivHouseholderQR<MatrixType> qr(m1);
        VERIFY_IS_EQUAL(rank, qr.rank());
      }
    }
  }
}

// Efficiency test: verify the threshold is not so large that it causes false
// rank deficiency. Creates matrices with smallest singular value well above
// the backward error bound, and verifies they are detected as full rank.
template <typename MatrixType>
void qr_threshold_efficiency() {
  typedef typename MatrixType::RealScalar RealScalar;
  typedef Matrix<RealScalar, Dynamic, 1> RealVectorType;

  const Index sizes[][2] = {{10, 10}, {50, 50}, {100, 100}, {40, 10}, {10, 40}};
  for (const auto& sz : sizes) {
    const Index rows = sz[0], cols = sz[1];
    const Index min_dim = (std::min)(rows, cols);
    // Create a matrix with prescribed singular values: smallest SV = 100 * threshold.
    // The threshold is 4*min(m,n)*eps, so we set sigma_min = 400*min(m,n)*eps.
    // This must be detected as full rank.
    RealScalar sigma_min = RealScalar(400) * RealScalar(min_dim) * NumTraits<RealScalar>::epsilon();
    RealVectorType svs = setupRangeSvs<RealVectorType>(min_dim, sigma_min, RealScalar(1));
    MatrixType m1;
    generateRandomMatrixSvs(svs, rows, cols, m1);
    ColPivHouseholderQR<MatrixType> qr(m1);
    // sigma_min is 100x the threshold — must be detected as full rank.
    VERIFY_IS_EQUAL(min_dim, qr.rank());

    // Also check with FullPivHouseholderQR.
    FullPivHouseholderQR<MatrixType> fpqr(m1);
    VERIFY_IS_EQUAL(min_dim, fpqr.rank());
  }
}

// Test rank detection on matrices with geometrically distributed singular values
// and a clear gap at the desired rank. This mimics real-world matrices better
// than partial isometries and stresses the threshold from both sides.
template <typename MatrixType>
void qr_rank_gap_test() {
  typedef typename MatrixType::RealScalar RealScalar;
  typedef Matrix<RealScalar, Dynamic, 1> RealVectorType;

  const Index sizes[][2] = {{20, 20}, {50, 50}, {100, 100}, {50, 20}, {20, 50}};
  for (const auto& sz : sizes) {
    const Index rows = sz[0], cols = sz[1];
    const Index min_dim = (std::min)(rows, cols);
    const Index rank = (std::max)(Index(1), min_dim / 2);

    // Singular values: [1, ..., sigma_rank] for the "signal" part,
    // then [eps_level, ..., eps_level] for the "noise" part.
    // The gap ratio is sigma_rank / eps_level >> 1.
    RealScalar sigma_rank = RealScalar(0.1);
    RealScalar eps_level = NumTraits<RealScalar>::epsilon();

    RealVectorType svs(min_dim);
    // Signal part: geometrically spaced from 1 to sigma_rank.
    for (Index i = 0; i < rank; ++i) {
      RealScalar t = (rank > 1) ? RealScalar(i) / RealScalar(rank - 1) : RealScalar(0);
      svs(i) = std::pow(sigma_rank, t);  // svs(0) = 1, svs(rank-1) = sigma_rank
    }
    // Noise part: near machine epsilon (well below any reasonable threshold).
    for (Index i = rank; i < min_dim; ++i) {
      svs(i) = eps_level * RealScalar(min_dim - i);
    }

    MatrixType m1;
    generateRandomMatrixSvs(svs, rows, cols, m1);

    ColPivHouseholderQR<MatrixType> qr(m1);
    VERIFY_IS_EQUAL(rank, qr.rank());

    FullPivHouseholderQR<MatrixType> fpqr(m1);
    VERIFY_IS_EQUAL(rank, fpqr.rank());
  }
}

#endif  // EIGEN_TEST_QR_COLPIVOTING_HELPERS_H
