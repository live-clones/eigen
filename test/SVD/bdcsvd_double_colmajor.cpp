// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/

// bdcsvd split: thin/full option checks for MatrixXd.

#include "bdcsvd_helpers.h"

TEST(BDCSVDDoubleColmajorTest, Basic) {
  for (int i = 0; i < g_repeat; i++) {
    int r = internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 2), c = internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 2);

    TEST_SET_BUT_UNUSED_VARIABLE(r)
    TEST_SET_BUT_UNUSED_VARIABLE(c)

    bdcsvd_thin_options<MatrixXd>(MatrixXd(20, 17));
    bdcsvd_full_options<MatrixXd>(MatrixXd(20, 17));
    bdcsvd_thin_options<MatrixXd>(MatrixXd(17, 20));
    bdcsvd_full_options<MatrixXd>(MatrixXd(17, 20));
    bdcsvd_thin_options<MatrixXd>(MatrixXd(r, c));
    bdcsvd_full_options<MatrixXd>(MatrixXd(r, c));
  }
}
