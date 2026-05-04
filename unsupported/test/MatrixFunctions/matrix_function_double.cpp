// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "matrix_function.h"

TEST(MatrixFunctionTest, Double) {
  // Fixed-size sample for fast-path codegen coverage.
  testMatrixType(Matrix2d());

  // Dynamic-size coverage at varying sizes (canonicalized to share
  // instantiations) and a complex<double> case.
  testMatrixTypeDynamic(MatrixXd(5, 5));
  testMatrixTypeDynamic(MatrixXd(13, 13));
  testMatrixTypeDynamic(Matrix<std::complex<double>, Dynamic, Dynamic>(4, 4));

  testMapRef(MatrixXd(13, 13));
}
