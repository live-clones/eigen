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
void inverse_for_fixed_size(const MatrixType& m1, std::enable_if_t<MatrixType::SizeAtCompileTime != Dynamic>* = nullptr) {
  using std::abs;
  using Scalar = typename MatrixType::Scalar;

  MatrixType m2, identity = MatrixType::Identity();

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
  using Scalar = typename MatrixType::Scalar;

  Index rows = m.rows();
  Index cols = m.cols();

  MatrixType m1(rows, cols), m2(rows, cols), identity = MatrixType::Identity(rows, rows);
  MatrixType A = MatrixType::Random(rows, cols);
  m1 = A * A.adjoint();
  m1 += MatrixType::Identity(rows, cols) * Scalar(0.1);

  m2 = Eigen::internal::inverse_cholesky(m1);
  VERIFY_IS_APPROX(m1, Eigen::internal::inverse_cholesky(m2));

  VERIFY_IS_APPROX(Eigen::internal::inverse_cholesky((Scalar(2) * m2)),
                   Eigen::internal::inverse_cholesky(m2) * Scalar(0.5));

  VERIFY_IS_APPROX(identity, Eigen::internal::inverse_cholesky(m1) * m1);
  VERIFY_IS_APPROX(identity, m1 * Eigen::internal::inverse_cholesky(m1));

  // since for the general case we implement separately row-major and col-major, test that
  VERIFY_IS_APPROX(m1, Eigen::internal::inverse_cholesky(Eigen::internal::inverse_cholesky(m1)));

  inverse_for_fixed_size(m1);
}

EIGEN_DECLARE_TEST(inverse_cholesky) {
  int s = 0;
  for (int i = 0; i < g_repeat; i++) {
    // Fixed sizes straddling the point at which the inversion switches to its blocked sweep, so
    // that both compile-time paths are instantiated: 47 is the last size taking the unblocked
    // recurrence, 48 the first taking the blocked one, and 64 makes the sweep end on a partial
    // panel (64 = 24 + 24 + 16) where 48 divides evenly into whole ones.
    CALL_SUBTEST_7(inverse_cholesky(Matrix<double, 47, 47>()));
    CALL_SUBTEST_7(inverse_cholesky(Matrix<double, 48, 48>()));
    CALL_SUBTEST_7(inverse_cholesky(Matrix<double, 64, 64>()));
    CALL_SUBTEST_8(inverse_cholesky(Matrix<float, 47, 47>()));
    CALL_SUBTEST_8(inverse_cholesky(Matrix<float, 48, 48>()));
    CALL_SUBTEST_8(inverse_cholesky(Matrix<float, 64, 64>()));
    CALL_SUBTEST_9(inverse_cholesky(Matrix<std::complex<float>, 48, 48>()));
    CALL_SUBTEST_9(inverse_cholesky(Matrix<std::complex<double>, 48, 48>()));

    CALL_SUBTEST_1(inverse_cholesky(Matrix<double, 1, 1>()));
    CALL_SUBTEST_1(inverse_cholesky(Matrix2d()));
    CALL_SUBTEST_1(inverse_cholesky(Matrix3d()));
    CALL_SUBTEST_1(inverse_cholesky(Matrix4d()));
    CALL_SUBTEST_1(inverse_cholesky(Matrix<double, 5, 5>()));
    CALL_SUBTEST_1(inverse_cholesky(Matrix<double, 6, 6>()));
    CALL_SUBTEST_1(inverse_cholesky(Matrix<double, 7, 7>()));
    CALL_SUBTEST_1(inverse_cholesky(Matrix<double, 4, 4, DontAlign>()));

    CALL_SUBTEST_2(inverse_cholesky(Matrix<float, 1, 1>()));
    CALL_SUBTEST_2(inverse_cholesky(Matrix2f()));
    CALL_SUBTEST_2(inverse_cholesky(Matrix3f()));
    CALL_SUBTEST_2(inverse_cholesky(Matrix4f()));
    CALL_SUBTEST_2(inverse_cholesky(Matrix<float, 5, 5>()));
    CALL_SUBTEST_2(inverse_cholesky(Matrix<float, 6, 6>()));
    CALL_SUBTEST_2(inverse_cholesky(Matrix<float, 7, 7>()));
    CALL_SUBTEST_2(inverse_cholesky(Matrix<float, 4, 4, DontAlign>()));

    CALL_SUBTEST_3(inverse_cholesky(Matrix<std::complex<float>, 1, 1>()));
    CALL_SUBTEST_3(inverse_cholesky(Matrix2cf()));
    CALL_SUBTEST_3(inverse_cholesky(Matrix3cf()));
    CALL_SUBTEST_3(inverse_cholesky(Matrix4cf()));
    CALL_SUBTEST_3(inverse_cholesky(Matrix<std::complex<float>, 5, 5>()));
    CALL_SUBTEST_3(inverse_cholesky(Matrix<std::complex<float>, 6, 6>()));
    CALL_SUBTEST_3(inverse_cholesky(Matrix<std::complex<float>, 7, 7>()));
    CALL_SUBTEST_3(inverse_cholesky(Matrix<std::complex<float>, 4, 4, DontAlign>()));

    CALL_SUBTEST_4(inverse_cholesky(Matrix<std::complex<double>, 1, 1>()));
    CALL_SUBTEST_4(inverse_cholesky(Matrix2cd()));
    CALL_SUBTEST_4(inverse_cholesky(Matrix3cd()));
    CALL_SUBTEST_4(inverse_cholesky(Matrix4cd()));
    CALL_SUBTEST_4(inverse_cholesky(Matrix<std::complex<double>, 5, 5>()));
    CALL_SUBTEST_4(inverse_cholesky(Matrix<std::complex<double>, 6, 6>()));
    CALL_SUBTEST_4(inverse_cholesky(Matrix<std::complex<double>, 7, 7>()));
    CALL_SUBTEST_4(inverse_cholesky(Matrix<std::complex<double>, 4, 4, DontAlign>()));

    s = internal::random<int>(50, 320);
    CALL_SUBTEST_5(inverse_cholesky(MatrixXf(s, s)));
    TEST_SET_BUT_UNUSED_VARIABLE(s);
    CALL_SUBTEST_5(inverse_cholesky(MatrixXf(1, 1)));
    TEST_SET_BUT_UNUSED_VARIABLE(s);
    CALL_SUBTEST_5(inverse_cholesky(MatrixXd(s, s)));
    TEST_SET_BUT_UNUSED_VARIABLE(s);
    CALL_SUBTEST_5(inverse_cholesky(MatrixXd(1, 1)));

    // The range above starts above the size at which the inversion switches to its blocked sweep,
    // so a second draw covers the run-time sizes that take the unblocked recurrence, and the two
    // sizes straddling the switch are pinned rather than left to the draw.
    s = internal::random<int>(2, 47);
    CALL_SUBTEST_5(inverse_cholesky(MatrixXf(s, s)));
    TEST_SET_BUT_UNUSED_VARIABLE(s);
    CALL_SUBTEST_5(inverse_cholesky(MatrixXd(s, s)));
    TEST_SET_BUT_UNUSED_VARIABLE(s);
    CALL_SUBTEST_5(inverse_cholesky(MatrixXf(47, 47)));
    CALL_SUBTEST_5(inverse_cholesky(MatrixXf(48, 48)));
    CALL_SUBTEST_5(inverse_cholesky(MatrixXd(47, 47)));
    CALL_SUBTEST_5(inverse_cholesky(MatrixXd(48, 48)));

    s = internal::random<int>(25, 100);
    CALL_SUBTEST_6(inverse_cholesky(MatrixXcf(s, s)));
    TEST_SET_BUT_UNUSED_VARIABLE(s);
    CALL_SUBTEST_6(inverse_cholesky(MatrixXcf(1, 1)));
    TEST_SET_BUT_UNUSED_VARIABLE(s);
    CALL_SUBTEST_6(inverse_cholesky(MatrixXcd(s, s)));
    TEST_SET_BUT_UNUSED_VARIABLE(s);
    CALL_SUBTEST_6(inverse_cholesky(MatrixXcd(1, 1)));

    // This draw already straddles the switch to the blocked sweep; pin the two sizes either side.
    CALL_SUBTEST_6(inverse_cholesky(MatrixXcf(47, 47)));
    CALL_SUBTEST_6(inverse_cholesky(MatrixXcf(48, 48)));
    CALL_SUBTEST_6(inverse_cholesky(MatrixXcd(47, 47)));
    CALL_SUBTEST_6(inverse_cholesky(MatrixXcd(48, 48)));
  }
}
