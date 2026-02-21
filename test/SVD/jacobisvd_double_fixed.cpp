// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/

// jacobisvd split: thin/full option checks for fixed-size double colmajor types.

#include "jacobisvd_helpers.h"

EIGEN_DECLARE_TEST(jacobisvd_double_fixed) {
  for (int i = 0; i < g_repeat; i++) {
    (jacobisvd_thin_options<Matrix4d>());
    (jacobisvd_full_options<Matrix4d>());
    (jacobisvd_thin_options<Matrix<double, 4, 7>>());
    (jacobisvd_full_options<Matrix<double, 4, 7>>());
    (jacobisvd_thin_options<Matrix<double, 7, 4>>());
    (jacobisvd_full_options<Matrix<double, 7, 4>>());
  }
}
