// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include <Eigen/Cholesky>

template <typename MatrixType>
void inverse_for_fixed_size(const MatrixType&, std::enable_if_t<MatrixType::SizeAtCompileTime == Dynamic>* = 0) {}

template <typename MatrixType>
void inverse_for_fixed_size(const MatrixType& m1, std::enable_if_t<MatrixType::SizeAtCompileTime != Dynamic>* = 0) {
  using std::abs;

  MatrixType m2, identity = MatrixType::Identity();

  typedef typename MatrixType::Scalar Scalar;

  m2.setZero();
  m2 = Eigen::internal::inverse_cholesky(m1);
  VERIFY_IS_APPROX(identity, m1 * m2);

  // check with submatrices
  {
    Matrix<Scalar, MatrixType::RowsAtCompileTime + 1, MatrixType::RowsAtCompileTime + 1, MatrixType::Options> m5;
    m5.setRandom();
    m5.topLeftCorner(m1.rows(), m1.rows()) = m1;
    m2 = m5.template topLeftCorner<MatrixType::RowsAtCompileTime, MatrixType::ColsAtCompileTime>().inverse();
    VERIFY_IS_APPROX((m5.template topLeftCorner<MatrixType::RowsAtCompileTime, MatrixType::ColsAtCompileTime>()),
                      Eigen::internal::inverse_cholesky(m2));
  }
}

template <typename MatrixType>
void inverse_cholesky(const MatrixType& m) {
  /* this test covers the following files:
     Cholesky/InverseImpl.h
  */
  Index rows = m.rows();
  Index cols = m.cols();

  typedef typename MatrixType::Scalar Scalar;

  MatrixType m1(rows, cols), m2(rows, cols), identity = MatrixType::Identity(rows, rows);
  MatrixType A = MatrixType::Random(rows, cols);
  m1 = A * A.adjoint();
  m1 += MatrixType::Identity(rows, cols) * Scalar(0.1);

  m2 = Eigen::internal::inverse_cholesky(m1);
  VERIFY_IS_APPROX(m1, Eigen::internal::inverse_cholesky(m2));

  VERIFY_IS_APPROX(Eigen::internal::inverse_cholesky((Scalar(2) * m2)), Eigen::internal::inverse_cholesky(m2) * Scalar(0.5));

  VERIFY_IS_APPROX(identity, Eigen::internal::inverse_cholesky(m1) * m1);
  VERIFY_IS_APPROX(identity, m1 * Eigen::internal::inverse_cholesky(m1));

  // since for the general case we implement separately row-major and col-major, test that
  VERIFY_IS_APPROX(m1, Eigen::internal::inverse_cholesky(Eigen::internal::inverse_cholesky(m1)));

  inverse_for_fixed_size(m1);
}

EIGEN_DECLARE_TEST(inverse_cholesky) {
  int s = 0;
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(inverse_cholesky(Matrix<double, 1, 1>()));
    CALL_SUBTEST_2(inverse_cholesky(Matrix2d()));
    CALL_SUBTEST_3(inverse_cholesky(Matrix3f()));
    CALL_SUBTEST_4(inverse_cholesky(Matrix4f()));
    CALL_SUBTEST_4(inverse_cholesky(Matrix<float, 4, 4, DontAlign>()));

    s = internal::random<int>(50, 320);
    CALL_SUBTEST_5(inverse_cholesky(MatrixXf(s, s)));
    TEST_SET_BUT_UNUSED_VARIABLE(s);
    CALL_SUBTEST_5(inverse_cholesky(MatrixXf(1, 1)));

    s = internal::random<int>(25, 100);
    CALL_SUBTEST_6(inverse_cholesky(MatrixXcd(s, s)));
    TEST_SET_BUT_UNUSED_VARIABLE(s);

    CALL_SUBTEST_7(inverse_cholesky(Matrix4d()));
    CALL_SUBTEST_7(inverse_cholesky(Matrix<double, 4, 4, DontAlign>()));

    // TODO: Double-inversion is numerically unstable for Matrix4cd due to rounding errors
    // CALL_SUBTEST_8(inverse_cholesky(Matrix4cd()));
  }
}
