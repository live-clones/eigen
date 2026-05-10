// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include <Eigen/QR>
#include <Eigen/SVD>
#include "solverbase.h"

// Use a small fixed block size in the tests so the blocked path actually
// triggers on the modest matrix sizes the unit tests exercise.
template <typename QRType>
void configure_small(QRType& qr) {
  qr.setBlockSize(4).setOversampling(2);
}

template <typename MatrixType>
void rqr() {
  using std::sqrt;

  Index rows = internal::random<Index>(2, EIGEN_TEST_MAX_SIZE), cols = internal::random<Index>(2, EIGEN_TEST_MAX_SIZE),
        cols2 = internal::random<Index>(2, EIGEN_TEST_MAX_SIZE);
  Index rank = internal::random<Index>(1, (std::min)(rows, cols) - 1);

  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime> MatrixQType;
  MatrixType m1;
  createRandomPIMatrixOfRank(rank, rows, cols, m1);
  RandColPivHouseholderQR<MatrixType> qr;
  configure_small(qr);
  qr.compute(m1);
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
  // non-increasing within each block of size b (matches the rank-revealing
  // expectation from the within-panel pivoting in HQR_P_UNB).
  RealScalar threshold = sqrt(RealScalar(rows)) * numext::abs(r(0, 0)) * NumTraits<Scalar>::epsilon();
  Index b = qr.blockSize();
  for (Index i = 0; i < (std::min)(rows, cols) - 1; ++i) {
    if ((i + 1) % b == 0) continue;  // diag may jump up at block boundaries
    RealScalar x = numext::abs(r(i, i));
    RealScalar y = numext::abs(r(i + 1, i + 1));
    if (x < threshold && y < threshold) continue;
    VERIFY_IS_APPROX_OR_LESS_THAN(y, x);
  }

  check_solverbase<MatrixType, MatrixType>(m1, qr, rows, cols, cols2);

  {
    MatrixType m2, m3;
    Index size = rows;
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
void rqr_fixedsize() {
  enum { Rows = MatrixType::RowsAtCompileTime, Cols = MatrixType::ColsAtCompileTime };
  typedef typename MatrixType::Scalar Scalar;
  int rank = internal::random<int>(1, (std::min)(int(Rows), int(Cols)) - 1);
  Matrix<Scalar, Rows, Cols> m1;
  createRandomPIMatrixOfRank(rank, Rows, Cols, m1);
  RandColPivHouseholderQR<Matrix<Scalar, Rows, Cols> > qr;
  configure_small(qr);
  qr.compute(m1);
  VERIFY_IS_EQUAL(rank, qr.rank());
  VERIFY_IS_EQUAL(Cols - qr.rank(), qr.dimensionOfKernel());
  VERIFY_IS_EQUAL(qr.isInjective(), (rank == Rows));
  VERIFY_IS_EQUAL(qr.isSurjective(), (rank == Cols));
  VERIFY_IS_EQUAL(qr.isInvertible(), (qr.isInjective() && qr.isSurjective()));

  Matrix<Scalar, Rows, Cols> r = qr.matrixQR().template triangularView<Upper>();
  Matrix<Scalar, Rows, Cols> c = qr.householderQ() * r * qr.colsPermutation().inverse();
  VERIFY_IS_APPROX(m1, c);

  check_solverbase<Matrix<Scalar, Cols, Cols2>, Matrix<Scalar, Rows, Cols2> >(m1, qr, Rows, Cols, Cols2);
}

// Round-trip verification on the Kahan matrix (the canonical
// counter-example for naive QR pivoting). The randomized strategy does
// not promise diagonal monotonicity across block boundaries, so we check
// reconstruction only.
template <typename MatrixType>
void rqr_kahan_matrix() {
  using std::sqrt;
  typedef typename MatrixType::RealScalar RealScalar;

  Index rows = 200, cols = rows;

  MatrixType m1;
  m1.setZero(rows, cols);
  RealScalar s = std::pow(NumTraits<RealScalar>::epsilon(), 1.0 / rows);
  RealScalar c_kahan = std::sqrt(1 - s * s);
  RealScalar pow_s_i(1.0);
  for (Index i = 0; i < rows; ++i) {
    m1(i, i) = pow_s_i;
    m1.row(i).tail(rows - i - 1) = -pow_s_i * c_kahan * MatrixType::Ones(1, rows - i - 1);
    pow_s_i *= s;
  }
  m1 = (m1 + m1.transpose()).eval();

  RandColPivHouseholderQR<MatrixType> qr;
  configure_small(qr);
  qr.setSeed(0xC0FFEE);  // fix seed for reproducibility
  qr.compute(m1);
  MatrixType r = qr.matrixQR().template triangularView<Upper>();

  // Reconstruction round-trip is the strongest correctness check.
  MatrixType q = qr.householderQ();
  MatrixType reconstructed = q * r * qr.colsPermutation().inverse();
  VERIFY_IS_APPROX(m1, reconstructed);
}

template <typename MatrixType>
void rqr_invertible() {
  using std::abs;
  using std::log;
  typedef typename NumTraits<typename MatrixType::Scalar>::Real RealScalar;
  typedef typename MatrixType::Scalar Scalar;

  int size = internal::random<int>(10, 50);

  MatrixType m1(size, size), m2(size, size), m3(size, size);
  m1 = MatrixType::Random(size, size);

  if (std::is_same<RealScalar, float>::value) {
    MatrixType a = MatrixType::Random(size, size * 2);
    m1 += a * a.adjoint();
  }

  RandColPivHouseholderQR<MatrixType> qr;
  configure_small(qr);
  qr.compute(m1);

  check_solverbase<MatrixType, MatrixType>(m1, qr, size, size, size);

  // Now construct a matrix with prescribed determinant and verify det/sign.
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
void rqr_verify_assert() {
  MatrixType tmp;

  RandColPivHouseholderQR<MatrixType> qr;
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

// Exercise the blocked path by using a matrix sufficiently larger than 2*b.
// At b=4 we need at least 8 columns to enter the blocked branch.
// Stress the rank-detection threshold under the blocked path. With b=4
// we run multiple blocks; the rank-revealing column lands at index 8, 12,
// or 20 — i.e. in the third block or later. A panel-local threshold
// would deflate as norms shrink across blocks and could miscount the
// rank. Only a global threshold (precomputed from the original column
// norms, as ColPivHouseholderQR does) is correct.
template <typename MatrixType>
void rqr_rank_in_late_block() {
  const Index rows = 30;
  const Index cols = 24;
  for (Index target_rank : {Index(8), Index(12), Index(20)}) {
    MatrixType m1;
    createRandomPIMatrixOfRank(target_rank, rows, cols, m1);

    RandColPivHouseholderQR<MatrixType> qr;
    configure_small(qr);
    qr.compute(m1);
    VERIFY_IS_EQUAL(target_rank, qr.rank());

    MatrixType r = qr.matrixQR().template triangularView<Upper>();
    MatrixType q = qr.householderQ();
    VERIFY_IS_APPROX(m1, MatrixType(q * r * qr.colsPermutation().inverse()));

    // Cross-check against classical column pivoting on the same input.
    ColPivHouseholderQR<MatrixType> cpqr(m1);
    VERIFY_IS_EQUAL(qr.rank(), cpqr.rank());
  }
}

template <typename MatrixType>
void rqr_blocked_path() {
  Index rows = 40, cols = 30;
  MatrixType m1 = MatrixType::Random(rows, cols);
  RandColPivHouseholderQR<MatrixType> qr;
  configure_small(qr);
  qr.setSeed(42);
  qr.compute(m1);

  MatrixType r = qr.matrixQR().template triangularView<Upper>();
  MatrixType q = qr.householderQ();
  VERIFY_IS_UNITARY(q);
  MatrixType reconstructed = q * r * qr.colsPermutation().inverse();
  VERIFY_IS_APPROX(m1, reconstructed);

  // Same input + same seed must reproduce the exact same factorization.
  RandColPivHouseholderQR<MatrixType> qr2;
  configure_small(qr2);
  qr2.setSeed(42);
  qr2.compute(m1);
  VERIFY_IS_EQUAL(qr.colsPermutation().indices(), qr2.colsPermutation().indices());
  VERIFY_IS_APPROX(qr.matrixQR(), qr2.matrixQR());
}

// Mirrors qr_rank_detection_stress from qr_colpivoting.cpp: many random
// partial-isometry trials across aspect ratios. With configure_small (b=4)
// most of these matrices engage the blocked path.
template <typename MatrixType>
void rqr_rank_detection_stress() {
  const Index sizes[][2] = {{10, 10}, {20, 20}, {50, 50}, {100, 100}, {40, 10}, {100, 10}, {10, 40}, {10, 100}};
  for (const auto& sz : sizes) {
    const Index rows = sz[0], cols = sz[1];
    const Index min_dim = (std::min)(rows, cols);
    for (Index rank : {Index(1), (std::max)(Index(1), min_dim / 2), min_dim - 1}) {
      if (rank >= min_dim) continue;
      for (int trial = 0; trial < 10; ++trial) {
        MatrixType m1;
        createRandomPIMatrixOfRank(rank, rows, cols, m1);
        RandColPivHouseholderQR<MatrixType> qr;
        configure_small(qr);
        qr.compute(m1);
        VERIFY_IS_EQUAL(rank, qr.rank());
      }
    }
  }
}

// Mirrors qr_threshold_efficiency: matrices with smallest SV well above
// the rank-detection threshold must come back as full-rank.
template <typename MatrixType>
void rqr_threshold_efficiency() {
  typedef typename MatrixType::RealScalar RealScalar;
  typedef Matrix<RealScalar, Dynamic, 1> RealVectorType;
  const Index sizes[][2] = {{10, 10}, {50, 50}, {100, 100}, {40, 10}, {10, 40}};
  for (const auto& sz : sizes) {
    const Index rows = sz[0], cols = sz[1];
    const Index min_dim = (std::min)(rows, cols);
    RealScalar sigma_min = RealScalar(400) * RealScalar(min_dim) * NumTraits<RealScalar>::epsilon();
    RealVectorType svs = setupRangeSvs<RealVectorType>(min_dim, sigma_min, RealScalar(1));
    MatrixType m1;
    generateRandomMatrixSvs(svs, rows, cols, m1);
    RandColPivHouseholderQR<MatrixType> qr;
    configure_small(qr);
    qr.compute(m1);
    VERIFY_IS_EQUAL(min_dim, qr.rank());
  }
}

// Mirrors qr_rank_gap_test: geometric signal SVs decaying to sigma_rank,
// then noise SVs near eps. The clear gap at `rank` should be detected.
template <typename MatrixType>
void rqr_rank_gap_test() {
  typedef typename MatrixType::RealScalar RealScalar;
  typedef Matrix<RealScalar, Dynamic, 1> RealVectorType;
  const Index sizes[][2] = {{20, 20}, {50, 50}, {100, 100}, {50, 20}, {20, 50}};
  for (const auto& sz : sizes) {
    const Index rows = sz[0], cols = sz[1];
    const Index min_dim = (std::min)(rows, cols);
    const Index rank = (std::max)(Index(1), min_dim / 2);
    RealScalar sigma_rank = RealScalar(0.1);
    RealScalar eps_level = NumTraits<RealScalar>::epsilon();
    RealVectorType svs(min_dim);
    for (Index i = 0; i < rank; ++i) {
      RealScalar t = (rank > 1) ? RealScalar(i) / RealScalar(rank - 1) : RealScalar(0);
      svs(i) = std::pow(sigma_rank, t);
    }
    for (Index i = rank; i < min_dim; ++i) svs(i) = eps_level * RealScalar(min_dim - i);
    MatrixType m1;
    generateRandomMatrixSvs(svs, rows, cols, m1);
    RandColPivHouseholderQR<MatrixType> qr;
    configure_small(qr);
    qr.compute(m1);
    VERIFY_IS_EQUAL(rank, qr.rank());
  }
}

// SV-decay classes from the RQRCP/HQRRP papers' section 4.2: slow linear
// decay, fast exponential decay, and a low-rank case. The papers' central
// empirical claim is that the randomized strategy reports the same rank as
// classical column pivoting on these distributions; we verify that here.
template <typename MatrixType>
void rqr_sv_decay_classes() {
  typedef typename MatrixType::RealScalar RealScalar;
  typedef Matrix<RealScalar, Dynamic, 1> RealVectorType;
  const Index n = 60;

  // Slow decay: linear from 1 to 0.01.
  {
    RealVectorType svs(n);
    for (Index i = 0; i < n; ++i) svs(i) = RealScalar(1) - (RealScalar(0.99) * RealScalar(i)) / RealScalar(n - 1);
    MatrixType m1;
    generateRandomMatrixSvs(svs, n, n, m1);
    RandColPivHouseholderQR<MatrixType> qr;
    configure_small(qr);
    qr.compute(m1);
    ColPivHouseholderQR<MatrixType> cp(m1);
    VERIFY_IS_EQUAL(qr.rank(), cp.rank());
  }

  // Fast exponential decay: sigma_i = exp(-c * i), c chosen so the tail
  // hits roughly e^-20.
  {
    RealVectorType svs(n);
    RealScalar c = RealScalar(20) / RealScalar(n);
    for (Index i = 0; i < n; ++i) svs(i) = std::exp(-c * RealScalar(i));
    MatrixType m1;
    generateRandomMatrixSvs(svs, n, n, m1);
    RandColPivHouseholderQR<MatrixType> qr;
    configure_small(qr);
    qr.compute(m1);
    ColPivHouseholderQR<MatrixType> cp(m1);
    VERIFY_IS_EQUAL(qr.rank(), cp.rank());
  }

  // Low-rank: rank-12 signal block at [1, 0.5], rest at machine eps.
  {
    const Index r = 12;
    RealVectorType svs(n);
    for (Index i = 0; i < r; ++i) svs(i) = RealScalar(1) - (RealScalar(0.5) * i) / RealScalar(r - 1);
    for (Index i = r; i < n; ++i) svs(i) = NumTraits<RealScalar>::epsilon();
    MatrixType m1;
    generateRandomMatrixSvs(svs, n, n, m1);
    RandColPivHouseholderQR<MatrixType> qr;
    configure_small(qr);
    qr.compute(m1);
    VERIFY_IS_EQUAL(r, qr.rank());
  }
}

EIGEN_DECLARE_TEST(qr_rand_colpivoting) {
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(rqr<MatrixXf>());
    CALL_SUBTEST_2(rqr<MatrixXd>());
    CALL_SUBTEST_3(rqr<MatrixXcd>());
    CALL_SUBTEST_4((rqr_fixedsize<Matrix<float, 8, 10>, 4>()));
    CALL_SUBTEST_5((rqr_fixedsize<Matrix<double, 12, 6>, 3>()));
  }

  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(rqr_invertible<MatrixXf>());
    CALL_SUBTEST_2(rqr_invertible<MatrixXd>());
    CALL_SUBTEST_6(rqr_invertible<MatrixXcf>());
    CALL_SUBTEST_3(rqr_invertible<MatrixXcd>());
  }

  CALL_SUBTEST_7(rqr_verify_assert<Matrix3f>());
  CALL_SUBTEST_8(rqr_verify_assert<Matrix3d>());
  CALL_SUBTEST_1(rqr_verify_assert<MatrixXf>());
  CALL_SUBTEST_2(rqr_verify_assert<MatrixXd>());
  CALL_SUBTEST_6(rqr_verify_assert<MatrixXcf>());
  CALL_SUBTEST_3(rqr_verify_assert<MatrixXcd>());

  // Test problem-size constructor.
  CALL_SUBTEST_9(RandColPivHouseholderQR<MatrixXf>(10, 20));

  CALL_SUBTEST_1(rqr_kahan_matrix<MatrixXf>());
  CALL_SUBTEST_2(rqr_kahan_matrix<MatrixXd>());

  CALL_SUBTEST_2(rqr_blocked_path<MatrixXd>());
  CALL_SUBTEST_3(rqr_blocked_path<MatrixXcd>());

  CALL_SUBTEST_1(rqr_rank_in_late_block<MatrixXf>());
  CALL_SUBTEST_2(rqr_rank_in_late_block<MatrixXd>());
  CALL_SUBTEST_3(rqr_rank_in_late_block<MatrixXcd>());

  CALL_SUBTEST_1(rqr_rank_detection_stress<MatrixXf>());
  CALL_SUBTEST_2(rqr_rank_detection_stress<MatrixXd>());

  CALL_SUBTEST_1(rqr_threshold_efficiency<MatrixXf>());
  CALL_SUBTEST_2(rqr_threshold_efficiency<MatrixXd>());

  CALL_SUBTEST_1(rqr_rank_gap_test<MatrixXf>());
  CALL_SUBTEST_2(rqr_rank_gap_test<MatrixXd>());

  CALL_SUBTEST_2(rqr_sv_decay_classes<MatrixXd>());
  CALL_SUBTEST_3(rqr_sv_decay_classes<MatrixXcd>());
}
