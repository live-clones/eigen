// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "matrix_function.h"

TEST(MatrixFunctionTest, Float) {
  testMatrixType(Matrix<float, 1, 1>());
  testMatrixType(Matrix3cf());
  testMatrixType(MatrixXf(8, 8));

  testMapRef(Matrix<float, 1, 1>());
  testMapRef(Matrix3cf());
  testMapRef(MatrixXf(8, 8));
}
