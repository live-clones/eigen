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
#include <Eigen/LU>

template <typename MatrixType>
void qr_gram_schmidt(const MatrixType& m) {
  Index rows = m.rows();
  Index cols = m.cols();
  Index size = (std::min)(rows, cols);

  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime> MatrixQType;

  MatrixType a = MatrixType::Random(rows, cols);

  GramSchmidtQR<MatrixType> qr(a);

  // Q should be unitary.
  MatrixQType q = qr.matrixQ();
  VERIFY_IS_UNITARY(q);

  // R should be upper triangular (below diagonal should be zero).
  MatrixType r = qr.matrixR();
  for (Index i = 1; i < size; ++i)
    for (Index j = 0; j < (std::min)(i, cols); ++j) VERIFY_IS_MUCH_SMALLER_THAN(r(i, j), Scalar(1));

  // A = Q * R
  VERIFY_IS_APPROX(a, q * r);
}

template <typename MatrixType>
void qr_gram_schmidt_solve(const MatrixType& m) {
  Index rows = m.rows();
  Index cols = m.cols();
  EIGEN_UNUSED_VARIABLE(cols);

  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> VectorType;

  // Square matrix solve.
  MatrixType a = MatrixType::Random(rows, rows);
  VectorType b = VectorType::Random(rows);

  GramSchmidtQR<MatrixType> qr(a);
  VectorType x = qr.solve(b);
  VERIFY_IS_APPROX(a * x, b);
}

template <typename MatrixType>
void qr_gram_schmidt_convenience(const MatrixType& m) {
  Index rows = m.rows();
  Index cols = m.cols();

  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime> MatrixQType;

  MatrixType a = MatrixType::Random(rows, cols);

  // Test convenience method.
  auto qr = a.gramSchmidtQr();

  MatrixQType q = qr.matrixQ();
  MatrixType r = qr.matrixR();
  VERIFY_IS_UNITARY(q);
  VERIFY_IS_APPROX(a, q * r);
}

template <typename MatrixType>
void qr_gram_schmidt_determinant(const MatrixType& m) {
  Index size = m.rows();

  MatrixType a = MatrixType::Random(size, size);
  GramSchmidtQR<MatrixType> qr(a);

  typename MatrixType::RealScalar abs_det = qr.absDeterminant();
  typename MatrixType::RealScalar log_abs_det = qr.logAbsDeterminant();

  typename MatrixType::RealScalar ref_abs_det = numext::abs(a.determinant());

  VERIFY_IS_APPROX(abs_det, ref_abs_det);
  VERIFY_IS_APPROX(log_abs_det, numext::log(ref_abs_det));
}

template <typename MatrixType>
void qr_gram_schmidt_edge_cases() {
  typedef typename MatrixType::Scalar Scalar;

  // Identity matrix.
  {
    Index n = 4;
    MatrixType id = MatrixType::Identity(n, n);
    GramSchmidtQR<MatrixType> qr(id);
    VERIFY_IS_APPROX(qr.matrixQ(), MatrixType::Identity(n, n));
    VERIFY_IS_APPROX(qr.matrixR(), MatrixType::Identity(n, n));
  }

  // Tall matrix (more rows than cols).
  {
    Index rows = 6, cols = 3;
    MatrixType a = MatrixType::Random(rows, cols);
    GramSchmidtQR<MatrixType> qr(a);
    VERIFY_IS_UNITARY(qr.matrixQ());
    VERIFY_IS_APPROX(a, qr.matrixQ() * qr.matrixR());
  }

  // Single column.
  {
    Index rows = 5;
    Matrix<Scalar, Dynamic, 1> v = Matrix<Scalar, Dynamic, 1>::Random(rows);
    GramSchmidtQR<Matrix<Scalar, Dynamic, 1>> qr(v);
    auto q = qr.matrixQ();
    auto r = qr.matrixR();
    VERIFY_IS_UNITARY(q);
    VERIFY_IS_APPROX(v, q * r);
  }

  // 1x1 matrix.
  {
    Matrix<Scalar, 1, 1> s;
    s(0, 0) = Scalar(3);
    GramSchmidtQR<Matrix<Scalar, 1, 1>> qr(s);
    VERIFY_IS_APPROX(numext::abs(qr.matrixQ()(0, 0)), typename MatrixType::RealScalar(1));
    VERIFY_IS_APPROX(qr.matrixQ() * qr.matrixR(), s);
  }
}

EIGEN_DECLARE_TEST(qr_gram_schmidt) {
  for (int i = 0; i < g_repeat; ++i) {
    CALL_SUBTEST_1(qr_gram_schmidt(
        MatrixXf(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE))));
    CALL_SUBTEST_2(qr_gram_schmidt(
        MatrixXd(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE))));
    CALL_SUBTEST_3(qr_gram_schmidt(MatrixXcd(internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 2),
                                             internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 2))));
  }

  for (int i = 0; i < g_repeat; ++i) {
    CALL_SUBTEST_4(qr_gram_schmidt_solve(
        MatrixXd(internal::random<int>(2, EIGEN_TEST_MAX_SIZE), internal::random<int>(2, EIGEN_TEST_MAX_SIZE))));
  }

  CALL_SUBTEST_5(qr_gram_schmidt_convenience(
      MatrixXd(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE))));

  CALL_SUBTEST_6(qr_gram_schmidt_determinant(MatrixXd(internal::random<int>(2, 10), internal::random<int>(2, 10))));

  CALL_SUBTEST_7((qr_gram_schmidt_edge_cases<MatrixXd>()));
}
