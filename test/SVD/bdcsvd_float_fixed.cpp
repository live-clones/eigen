// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/
// SPDX-License-Identifier: MPL-2.0

// bdcsvd split: thin/full option checks for float fixed-size matrices.

#include "bdcsvd_helpers.h"

TEST(BDCSVDFloatFixedTest, Square) {
  for (int i = 0; i < g_repeat; i++) {
    (bdcsvd_thin_full_options<Matrix3f>());
  }
}

TEST(BDCSVDFloatFixedTest, Rect) {
  for (int i = 0; i < g_repeat; i++) {
    (bdcsvd_thin_full_options<Matrix<float, 2, 3>>());
  }
}
