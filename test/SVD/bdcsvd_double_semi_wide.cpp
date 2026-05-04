// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/
// SPDX-License-Identifier: MPL-2.0

// bdcsvd split: thin/full option checks for wide semi-fixed double types.

#include "bdcsvd_helpers.h"

TEST(BDCSVDDoubleSemiWideTest, Basic) {
  for (int i = 0; i < g_repeat; i++) {
    int c = internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 2);
    TEST_SET_BUT_UNUSED_VARIABLE(c);

    (bdcsvd_thin_options<Matrix<double, 13, Dynamic>>(Matrix<double, 13, Dynamic>(13, c)));
    (bdcsvd_full_options<Matrix<double, 13, Dynamic>>(Matrix<double, 13, Dynamic>(13, c)));
  }
}
