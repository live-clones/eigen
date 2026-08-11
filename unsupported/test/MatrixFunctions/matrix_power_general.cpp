// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012, 2013 Chen-Pang He <jdh8@ms63.hinet.net>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "matrix_functions_general.h"

typedef Matrix<double, 3, 3, RowMajor> Matrix3dRowMajor;

TEST(MatrixPowerTest, General) {
  // Fixed-size samples for fast-path codegen coverage.
  testGeneral(Matrix2d(), 2e-13);
  testGeneral(Matrix3dRowMajor(), 1e-13);
  testGeneral(Matrix4cd(), 1e-13);

  // Dynamic-size coverage.
  testGeneralDynamic(MatrixXd(8, 8), 3e-12);
  testGeneralDynamic(Matrix<std::complex<double>, Dynamic, Dynamic>(3, 3), 1e-13);
}
