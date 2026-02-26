// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012, 2013 Chen-Pang He <jdh8@ms63.hinet.net>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// matrix_power_logthenexp split: double and complex double types.

#include "matrix_power_logthenexp.h"

TEST(MatrixPowerTest, LogThenExpDouble) {
  testLogThenExp(Matrix2d(), 1e-13);
  testLogThenExp(Matrix3dRowMajor(), 1e-13);
  testLogThenExp(Matrix4cd(), 1e-13);
  testLogThenExp(MatrixXd(8, 8), 2e-12);
  testLogThenExp(Matrix3d(), 1e-13);
}
