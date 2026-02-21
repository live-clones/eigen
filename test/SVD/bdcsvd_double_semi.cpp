// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/

// bdcsvd split: thin/full option checks for semi-fixed double types.

#include "bdcsvd_helpers.h"

EIGEN_DECLARE_TEST(bdcsvd_double_semi) {
  for (int i = 0; i < g_repeat; i++) {
    int r = internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 2), c = internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 2);

    TEST_SET_BUT_UNUSED_VARIABLE(r)
    TEST_SET_BUT_UNUSED_VARIABLE(c)

    (bdcsvd_thin_options<Matrix<double, Dynamic, 15>>(Matrix<double, Dynamic, 15>(r, 15)));
    (bdcsvd_full_options<Matrix<double, Dynamic, 15>>(Matrix<double, Dynamic, 15>(r, 15)));
    (bdcsvd_thin_options<Matrix<double, 13, Dynamic>>(Matrix<double, 13, Dynamic>(13, c)));
    (bdcsvd_full_options<Matrix<double, 13, Dynamic>>(Matrix<double, 13, Dynamic>(13, c)));
  }
}
