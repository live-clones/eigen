// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

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
    do {
      m1 = MatrixType::Random(size, size);
      qr.compute(m1);
    } while (!qr.isInvertible());
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

TEST(QRColpivotingTest, Basic) {
  for (int i = 0; i < g_repeat; i++) {
    qr<MatrixXf>();
    qr<MatrixXd>();
    qr<MatrixXcd>();
    (qr_fixedsize<Matrix<float, 3, 5>, 4>());
    (qr_fixedsize<Matrix<double, 6, 2>, 3>());
    (qr_fixedsize<Matrix<double, 1, 1>, 1>());
  }

  for (int i = 0; i < g_repeat; i++) {
    qr_invertible<MatrixXf>();
    qr_invertible<MatrixXd>();
    qr_invertible<MatrixXcf>();
    qr_invertible<MatrixXcd>();
  }

  qr_verify_assert<Matrix3f>();
  qr_verify_assert<Matrix3d>();
  qr_verify_assert<MatrixXf>();
  qr_verify_assert<MatrixXd>();
  qr_verify_assert<MatrixXcf>();
  qr_verify_assert<MatrixXcd>();

  ColPivHouseholderQR<MatrixXf>(10, 20);

  qr_kahan_matrix<MatrixXf>();
  qr_kahan_matrix<MatrixXd>();
}
