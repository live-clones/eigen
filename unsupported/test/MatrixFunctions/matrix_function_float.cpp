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
  // Dynamic-size coverage at varying sizes / scalars (canonicalized to share
  // instantiations). matrix_function_double covers fixed-size fast-path
  // codegen for one Matrix<double, ...>; that suffices for codegen coverage.
  testMatrixTypeDynamic(MatrixXf(4, 4));
  testMatrixTypeDynamic(MatrixXf(8, 8));
  testMatrixTypeDynamic(Matrix<std::complex<float>, Dynamic, Dynamic>(3, 3));

  testMapRef(MatrixXf(8, 8));
}
