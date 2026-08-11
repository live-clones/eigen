// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/
// SPDX-License-Identifier: MPL-2.0

// bdcsvd split: matrixbase method checks and small fixed-size option checks.

#include "bdcsvd_helpers.h"

TEST(BDCSVDTrivialTest, Method) {
  (bdcsvd_method<Matrix2cd>());
  (bdcsvd_method<Matrix3f>());
}

TEST(BDCSVDTrivialTest, SmallFixed) {
  (bdcsvd_thin_full_options<Matrix2cd>());
  (bdcsvd_thin_full_options<Matrix2d>());
}
