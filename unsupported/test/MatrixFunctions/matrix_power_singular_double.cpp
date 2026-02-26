// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012, 2013 Chen-Pang He <jdh8@ms63.hinet.net>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "matrix_power_singular.h"

typedef Matrix<double, 3, 3, RowMajor> Matrix3dRowMajor;

TEST(MatrixPowerTest, SingularDouble) {
  testSingular(Matrix2d(), 1e-13);
  testSingular(Matrix3dRowMajor(), 1e-13);
  testSingular(Matrix4cd(), 1e-13);
  testSingular(MatrixXd(8, 8), 2e-12);
  testSingular(Matrix3d(), 1e-13);
}
