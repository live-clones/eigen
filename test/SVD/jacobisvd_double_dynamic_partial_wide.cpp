// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/
// SPDX-License-Identifier: MPL-2.0

// jacobisvd split: thin/full option checks for partial-dynamic (wide) double types.

#include "jacobisvd_helpers.h"

TEST(JacobisvdDoubleDynamicPartialWideTest, Basic) {
  for (int i = 0; i < g_repeat; i++) {
    int c = internal::random<int>(1, 30);

    TEST_SET_BUT_UNUSED_VARIABLE(c);

    (jacobisvd_thin_options<Matrix<double, 5, Dynamic>>(Matrix<double, 5, Dynamic>(5, c)));
    (jacobisvd_full_options<Matrix<double, 5, Dynamic>>(Matrix<double, 5, Dynamic>(5, c)));
  }
}
