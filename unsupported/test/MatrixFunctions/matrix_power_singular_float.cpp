// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012, 2013 Chen-Pang He <jdh8@ms63.hinet.net>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "matrix_functions_singular.h"

TEST(MatrixPowerTest, SingularFloat) {
  // Fixed-size sample for fast-path codegen coverage.
  testSingular(Matrix2f(), 1e-4f);

  // Dynamic-size coverage at varying sizes / scalars.
  testSingularDynamic(MatrixXf(2, 2), 1e-3f);
  testSingularDynamic(MatrixXf(4, 4), 1e-4f);
  testSingularDynamic(Matrix<std::complex<float>, Dynamic, Dynamic>(3, 3), 1e-4f);
}
