// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/
// SPDX-License-Identifier: MPL-2.0

// jacobisvd split: thin/full option checks for float fixed 2x3 type.

#include "jacobisvd_helpers.h"

TEST(JacobisvdFloatFixedTest, Basic) {
  for (int i = 0; i < g_repeat; i++) {
    (jacobisvd_thin_options<Matrix<float, 2, 3>>());
    (jacobisvd_full_options<Matrix<float, 2, 3>>());
  }
}
