// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012, 2013 Chen-Pang He <jdh8@ms63.hinet.net>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "matrix_functions_logthenexp.h"

typedef Matrix<long double, Dynamic, Dynamic> MatrixXe;

TEST(MatrixPowerTest, LogThenExpFloat) {
  testLogThenExpDynamic(MatrixXf(2, 2), 1e-3f);
  testLogThenExpDynamic(MatrixXf(3, 3), 1e-4f);
  testLogThenExpDynamic(MatrixXf(4, 4), 1e-4f);
  testLogThenExpDynamic(Matrix<std::complex<float>, Dynamic, Dynamic>(3, 3), 1e-4f);
  testLogThenExpDynamic(MatrixXe(7, 7), 1e-12L);
  testLogThenExpDynamic(MatrixXe(3, 3), 1e-13L);
}
