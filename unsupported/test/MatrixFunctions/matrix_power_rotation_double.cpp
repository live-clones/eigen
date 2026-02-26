// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012, 2013 Chen-Pang He <jdh8@ms63.hinet.net>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "matrix_power_rotation.h"

TEST(MatrixPowerTest, RotationDouble) {
  test2dRotation<double>(1e-13);
  test2dHyperbolicRotation<double>(1e-14);
  test3dRotation<double>(1e-13);
}
