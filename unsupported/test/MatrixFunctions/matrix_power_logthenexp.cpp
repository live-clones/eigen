// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012, 2013 Chen-Pang He <jdh8@ms63.hinet.net>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "matrix_functions_logthenexp.h"

typedef Matrix<double, 3, 3, RowMajor> Matrix3dRowMajor;
typedef Matrix<long double, 3, 3> Matrix3e;
typedef Matrix<long double, Dynamic, Dynamic> MatrixXe;

// Fast-path codegen coverage of MatrixPower / MatrixLog / MatrixExp on
// fixed-size matrices is exercised by matrix_power_general /
// matrix_power_singular / matrix_function_* / matrix_exponential_*. This
// TU canonicalizes to Matrix<Scalar, Dynamic, Dynamic> to keep compile RSS
// bounded — the test verifies the algebraic identity pow(x) == exp(x*log)
// at runtime sizes covering each supported scalar.

TEST(MatrixPowerTest, LogThenExp) {
  testLogThenExpDynamic(MatrixXd(2, 2), 2e-13);
  testLogThenExpDynamic(MatrixXd(3, 3), 1e-13);
  testLogThenExpDynamic(MatrixXd(8, 8), 3e-12);
  testLogThenExpDynamic(Matrix<std::complex<double>, Dynamic, Dynamic>(4, 4), 1e-13);
}

TEST(MatrixPowerTest, TestLogThenExp) {
  testLogThenExp(Matrix2d(), 2e-13);
  testLogThenExp(Matrix3dRowMajor(), 1e-13);
  testLogThenExp(Matrix4cd(), 1e-13);
  testLogThenExp(MatrixXd(8, 8), 3e-12);
  testLogThenExp(Matrix2f(), 1e-4f);
  testLogThenExp(Matrix3cf(), 1e-4f);
  testLogThenExp(Matrix4f(), 1e-4f);
  testLogThenExp(MatrixXf(2, 2), 1e-3f);
  testLogThenExp(MatrixXe(7, 7), 1e-12L);
  testLogThenExp(Matrix3d(), 2e-13);
  testLogThenExp(Matrix3f(), 1e-4f);
  testLogThenExp(Matrix3e(), 1e-13L);
}
