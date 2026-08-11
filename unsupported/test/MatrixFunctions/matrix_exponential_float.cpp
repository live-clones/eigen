// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "matrix_exponential.h"

TEST(MatrixExponentialTest, Float) {
  test2dRotation<float>(2e-5);
  test2dHyperbolicRotation<float>(1e-5);
  testPascal<float>(1e-6);
  randomTest(Matrix2f(), 1e-4);
  randomTest(Matrix3cf(), 1e-4);
  randomTest(Matrix4f(), 1e-4);
  randomTest(MatrixXf(8, 8), 1e-4);
}
