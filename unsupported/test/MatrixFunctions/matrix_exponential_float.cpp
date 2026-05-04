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

TEST(MatrixExponentialTest, Float) {
  test2dRotation<float>(2e-5);
  test2dHyperbolicRotation<float>(1e-5);
  testPascal<float>(1e-6);

  // Fixed-size sample for fast-path codegen coverage.
  randomTest(Matrix2f(), 1e-4);

  // Dynamic-size coverage at varying sizes / scalars.
  randomTestDynamic(MatrixXf(4, 4), 1e-4);
  randomTestDynamic(MatrixXf(8, 8), 1e-4);
  randomTestDynamic(Matrix<std::complex<float>, Dynamic, Dynamic>(3, 3), 1e-4);
}
