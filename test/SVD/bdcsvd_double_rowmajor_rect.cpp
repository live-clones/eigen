// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/
// SPDX-License-Identifier: MPL-2.0

// bdcsvd split: row-major double dynamic rectangular cases.
// Complex MatrixXcd cases live in bdcsvd_double_rowmajor.cpp.

#include "bdcsvd_helpers.h"

TEST(BDCSVDDoubleRowmajorTest, Rect) {
  for (int i = 0; i < g_repeat; i++) {
    (bdcsvd_thin_options<Matrix<double, Dynamic, Dynamic, RowMajor>>(
        Matrix<double, Dynamic, Dynamic, RowMajor>(20, 27)));
    (bdcsvd_full_options<Matrix<double, Dynamic, Dynamic, RowMajor>>(
        Matrix<double, Dynamic, Dynamic, RowMajor>(20, 27)));
    (bdcsvd_thin_options<Matrix<double, Dynamic, Dynamic, RowMajor>>(
        Matrix<double, Dynamic, Dynamic, RowMajor>(27, 20)));
    (bdcsvd_full_options<Matrix<double, Dynamic, Dynamic, RowMajor>>(
        Matrix<double, Dynamic, Dynamic, RowMajor>(27, 20)));
  }
}
