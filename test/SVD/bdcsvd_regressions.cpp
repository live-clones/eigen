// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/
// SPDX-License-Identifier: MPL-2.0

// bdcsvd split: bidiagonal hard cases and numerical-scale regressions.

#include <cstdlib>

#include "bdcsvd_helpers.h"
#include "Eigenvalues/tridiag_test_matrices.h"

// Verify SVD of bidiagonal matrix given as diagonal + superdiagonal vectors.
template <typename RealScalar>
void verify_bidiagonal_svd(const Matrix<RealScalar, Dynamic, 1>& diag,
                           const Matrix<RealScalar, Dynamic, 1>& superdiag) {
  typedef Matrix<RealScalar, Dynamic, Dynamic> MatrixXr;
  typedef Matrix<RealScalar, Dynamic, 1> VectorXr;
  const Index n = diag.size();

  BDCSVD<MatrixXr, ComputeFullU | ComputeFullV> bdcsvd(diag, superdiag);
  VERIFY(bdcsvd.info() == Success);

  const VectorXr& sv = bdcsvd.singularValues();

  // Singular values must be non-negative.
  for (Index i = 0; i < sv.size(); ++i) {
    VERIFY(sv(i) >= RealScalar(0));
  }

  // Singular values must be sorted descending.
  for (Index i = 1; i < sv.size(); ++i) {
    VERIFY(sv(i - 1) >= sv(i));
  }

  // Orthogonality of U and V.
  VERIFY_IS_APPROX(bdcsvd.matrixU().transpose() * bdcsvd.matrixU(), MatrixXr::Identity(n, n));
  VERIFY_IS_APPROX(bdcsvd.matrixV().transpose() * bdcsvd.matrixV(), MatrixXr::Identity(n, n));

  // Reconstruction: U * S * V^T should equal the original bidiagonal.
  MatrixXr B = MatrixXr::Zero(n, n);
  B.diagonal() = diag;
  if (n > 1) B.diagonal(1) = superdiag;
  MatrixXr recon = bdcsvd.matrixU() * sv.asDiagonal() * bdcsvd.matrixV().transpose();
  VERIFY_IS_APPROX(recon, B);

  // Cross-validate singular values against JacobiSVD.
  JacobiSVD<MatrixXr> jacobi(B);
  VERIFY_IS_APPROX(sv, jacobi.singularValues());
}

// Verify that bidiagonal API and matrix API produce matching singular values.
template <typename RealScalar>
void verify_bidiagonal_vs_matrix_svd(const Matrix<RealScalar, Dynamic, 1>& diag,
                                     const Matrix<RealScalar, Dynamic, 1>& superdiag) {
  typedef Matrix<RealScalar, Dynamic, Dynamic> MatrixXr;
  const Index n = diag.size();

  // Build dense bidiagonal matrix.
  MatrixXr B = MatrixXr::Zero(n, n);
  B.diagonal() = diag;
  if (n > 1) B.diagonal(1) = superdiag;

  BDCSVD<MatrixXr> bidiag_svd(diag, superdiag);
  BDCSVD<MatrixXr> matrix_svd(B);

  VERIFY(bidiag_svd.info() == Success);
  VERIFY(matrix_svd.info() == Success);
  VERIFY_IS_APPROX(bidiag_svd.singularValues(), matrix_svd.singularValues());
}

template <typename RealScalar>
void bdcsvd_bidiagonal_hard_cases() {
  Eigen::internal::set_is_malloc_allowed(true);

  // Use the shared tridiagonal test matrix generators.
  // Each generator fills (diag, offdiag) which we treat as (diagonal, superdiagonal)
  // of a bidiagonal matrix.
  test::for_all_tridiag_test_matrices<RealScalar>(
      [](const auto& diag, const auto& offdiag) { verify_bidiagonal_svd<RealScalar>(diag, offdiag); });

  // Additional SVD-specific test: identity with cross-validation against full matrix SVD.
  test::for_tridiag_sizes<RealScalar>([](auto& diag, auto& offdiag) {
    test::tridiag_identity(diag, offdiag);
    verify_bidiagonal_vs_matrix_svd<RealScalar>(diag, offdiag);
  });

  // Additional SVD-specific test: scalar for n=1.
  {
    typedef Matrix<RealScalar, Dynamic, 1> VectorXr;
    VectorXr diag(1), offdiag(0);
    diag(0) = RealScalar(3.14);
    verify_bidiagonal_svd<RealScalar>(diag, offdiag);
  }
}

TEST(BDCSVDRegressionsTest, BidiagonalHardCases) {
  (bdcsvd_bidiagonal_hard_cases<float>());
  (bdcsvd_bidiagonal_hard_cases<double>());
}

void bdcsvd_extreme_scale_regressions() {
  typedef Matrix<double, 6, 6> Matrix6d;
  const double kTolerance = 16 * Matrix6d::RowsAtCompileTime * NumTraits<double>::epsilon();

  const auto verify_decomposition = [kTolerance](const Matrix6d& matrix) {
    BDCSVD<Matrix6d, ComputeFullU | ComputeFullV> svd;
    svd.setSwitchSize(3);
    svd.compute(matrix);

    VERIFY(svd.info() == Success);

    const Matrix6d reconstruction = svd.matrixU() * svd.singularValues().asDiagonal() * svd.matrixV().transpose();
    VERIFY((reconstruction - matrix).stableNorm() <= kTolerance * matrix.stableNorm());

    const Matrix6d identity = Matrix6d::Identity();
    VERIFY((svd.matrixU().transpose() * svd.matrixU() - identity).stableNorm() <= kTolerance);
    VERIFY((svd.matrixV().transpose() * svd.matrixV() - identity).stableNorm() <= kTolerance);

    // Also exercise the values-only path, which uses a compact m_naiveU and
    // linear workspace during divide-and-conquer merges.
    BDCSVD<Matrix6d> valuesOnlySvd;
    valuesOnlySvd.setSwitchSize(3);
    valuesOnlySvd.compute(matrix);
    VERIFY(valuesOnlySvd.info() == Success);
    VERIFY_IS_APPROX(valuesOnlySvd.singularValues(), svd.singularValues());
  };

  Matrix6d matrix = Matrix6d::Zero();
  const double kSubnormal1040 = std::numeric_limits<double>::denorm_min() * 17179869184.0;  // 2^-1040
  const double kSubnormal1060 = std::numeric_limits<double>::denorm_min() * 16384.0;        // 2^-1060
  const double kSmallestNormal = (std::numeric_limits<double>::min)();                      // 2^-1022
  const double kNormal1000 = kSmallestNormal * 4194304.0;                                   // 2^-1000

  // The merge combines normal and subnormal couplings. Squaring the two
  // coupling terms directly used to underflow, which later left perturbCol0
  // without a predecessor and made the decomposition report NumericalIssue.
  matrix.diagonal() << kSubnormal1040, -kSubnormal1060, kSmallestNormal, 0.5, 1.0, kSmallestNormal;
  matrix.diagonal(1) << -kNormal1000, kNormal1000, kSubnormal1040, kSubnormal1060, -8.0;
  verify_decomposition(matrix);

  // A singular-vector coefficient grows to about 2^570 here. Its squared
  // norm overflows even though the vector has a finite, well-scaled
  // normalization.
  using std::ldexp;
  matrix.setZero();
  matrix.diagonal() << 0.0, 0.0, ldexp(1.0, -487), -1.0, 0.0, 0.0;
  matrix.diagonal(1) << 0.0, ldexp(1.0, -453), -ldexp(1.0, -627), 0.0, 0.0;
  verify_decomposition(matrix);
}

TEST(BDCSVDRegressionsTest, ExtremeScaleRegressions) { bdcsvd_extreme_scale_regressions(); }

void bdcsvd_fast_math_regression_1588() {
  const Index n = 500;
  MatrixXd matrix = MatrixXd::Zero(n, n);

  std::srand(1);
  for (Index k = 0; k < 5000; ++k) {
    const Index row = std::rand() % n;
    const Index col = std::rand() % n;
    matrix(row, col) = static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX);
  }
  matrix = matrix * matrix;

  BDCSVD<MatrixXd, ComputeThinU | ComputeThinV> svd(matrix);
  VERIFY(svd.info() == Success);

  MatrixXd reconstruction = svd.matrixU() * svd.singularValues().asDiagonal() * svd.matrixV().transpose();
  const double relative_error = (reconstruction - matrix).norm() / matrix.norm();
  VERIFY(relative_error < 1e-10);
}

TEST(BDCSVDRegressionsTest, FastMathRegression1588) { bdcsvd_fast_math_regression_1588(); }
