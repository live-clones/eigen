// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/
// SPDX-License-Identifier: MPL-2.0

// jacobisvd split: thin/full option checks for MatrixXf dynamic type.

#include "jacobisvd_helpers.h"

TEST(JacobisvdFloatDynamicTest, Basic) {
  for (int i = 0; i < g_repeat; i++) {
    int r = internal::random<int>(1, 30), c = internal::random<int>(1, 30);

    TEST_SET_BUT_UNUSED_VARIABLE(r);
    TEST_SET_BUT_UNUSED_VARIABLE(c);

    (jacobisvd_thin_options<MatrixXf>(MatrixXf(r, c)));
    (jacobisvd_full_options<MatrixXf>(MatrixXf(r, c)));
  }
}
