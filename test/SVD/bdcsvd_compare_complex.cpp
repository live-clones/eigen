// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/
// SPDX-License-Identifier: MPL-2.0

// bdcsvd split: compare tests, complex types.

#include "bdcsvd_helpers.h"

TEST(BDCSVDCompareTest, Complex) {
  for (int i = 0; i < g_repeat; i++) {
    int r = internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 2), c = internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 2);

    TEST_SET_BUT_UNUSED_VARIABLE(r);
    TEST_SET_BUT_UNUSED_VARIABLE(c);

    (compare_bdc_jacobi<MatrixXcd>(MatrixXcd(r, c)));
  }
}
