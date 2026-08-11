// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "matrix_function.h"

TEST(MatrixFunctionTest, Double) {
  testMatrixType(Matrix2d());
  testMatrixType(Matrix<double, 5, 5, RowMajor>());
  testMatrixType(Matrix4cd());
  testMatrixType(MatrixXd(13, 13));

  testMapRef(MatrixXd(13, 13));
}
