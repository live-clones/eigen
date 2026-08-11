// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "matrix_exponential.h"

TEST(MatrixExponentialTest, Double) {
  test2dRotation<double>(1e-13);
  test2dRotation<long double>(1e-13);
  test2dHyperbolicRotation<double>(1e-14);
  test2dHyperbolicRotation<long double>(1e-14);
  testPascal<double>(1e-15);
  randomTest(Matrix2d(), 1e-13);
  randomTest(Matrix<double, 3, 3, RowMajor>(), 1e-13);
  randomTest(Matrix4cd(), 1e-13);
  randomTest(MatrixXd(8, 8), 1e-13);
  randomTest(Matrix<long double, Dynamic, Dynamic>(7, 7), 1e-13);
}
