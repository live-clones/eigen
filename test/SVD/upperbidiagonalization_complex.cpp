// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "upperbidiagonalization.h"

TEST(UpperbidiagonalizationTest, Complex) {
  for (int i = 0; i < g_repeat; i++) {
    upperbidiag(MatrixXcf(20, 20));
    upperbidiag(Matrix<std::complex<double>, Dynamic, Dynamic, RowMajor>(16, 15));
  }
}
