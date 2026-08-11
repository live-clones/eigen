// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/

// bdcsvd split: thin/full option checks for fixed-size float types.

#include "bdcsvd_helpers.h"

TEST(BDCSVDFloatFixedTest, Basic) {
  for (int i = 0; i < g_repeat; i++) {
    (bdcsvd_thin_options<Matrix3f>());
    (bdcsvd_full_options<Matrix3f>());
    (bdcsvd_thin_options<Matrix<float, 2, 3>>());
    (bdcsvd_full_options<Matrix<float, 2, 3>>());
  }
}
