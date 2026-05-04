// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "matrix_exponential.h"

TEST(MatrixExponentialTest, Double) {
  test2dRotation<double>(1e-13);
  test2dRotation<long double>(1e-13);
  test2dHyperbolicRotation<double>(1e-14);
  test2dHyperbolicRotation<long double>(1e-14);
  testPascal<double>(1e-15);

  // Fixed-size sample for fast-path codegen coverage.
  randomTest(Matrix2d(), 1e-13);

  // Dynamic-size coverage at varying sizes / scalars.
  randomTestDynamic(MatrixXd(3, 3), 1e-13);
  randomTestDynamic(MatrixXd(8, 8), 1e-13);
  randomTestDynamic(Matrix<std::complex<double>, Dynamic, Dynamic>(4, 4), 1e-13);
  randomTestDynamic(Matrix<long double, Dynamic, Dynamic>(7, 7), 1e-13);
}
