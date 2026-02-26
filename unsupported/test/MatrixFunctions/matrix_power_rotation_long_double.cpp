// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012, 2013 Chen-Pang He <jdh8@ms63.hinet.net>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "matrix_power_rotation.h"

TEST(MatrixPowerTest, RotationLongDouble) {
  test2dRotation<long double>(1e-13L);
  test2dHyperbolicRotation<long double>(1e-14L);
  test3dRotation<long double>(1e-13L);
}
