// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012, 2013 Chen-Pang He <jdh8@ms63.hinet.net>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "matrix_functions_general.h"

typedef Matrix<long double, 3, 3> Matrix3e;
typedef Matrix<long double, Dynamic, Dynamic> MatrixXe;

TEST(MatrixPowerTest, GeneralFloat) {
  // Fixed-size samples for fast-path codegen coverage. complex<float>
  // fixed-size instantiations are too costly to compile alongside long
  // double; the dynamic call provides equivalent runtime coverage.
  testGeneral(Matrix2f(), 1e-4f);
  testGeneral(Matrix3e(), 1e-13L);

  // Dynamic-size coverage.
  testGeneralDynamic(MatrixXf(2, 2), 1e-3f);
  testGeneralDynamic(MatrixXf(4, 4), 1e-4f);
  testGeneralDynamic(Matrix<std::complex<float>, Dynamic, Dynamic>(3, 3), 1e-4f);
  testGeneralDynamic(MatrixXe(7, 7), 1e-12L);
}
