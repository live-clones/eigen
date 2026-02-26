// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012, 2013 Chen-Pang He <jdh8@ms63.hinet.net>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "matrix_power_general.h"

TEST(MatrixPowerTest, GeneralFloat) {
  testGeneral(Matrix2f(), 1e-4f);
  testGeneral(Matrix3cf(), 1e-4f);
  testGeneral(Matrix4f(), 1e-4f);
  testGeneral(MatrixXf(2, 2), 1e-3f);  // see bug 614
  testGeneral(Matrix3f(), 1e-4f);
}
