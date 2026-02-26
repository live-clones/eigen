// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/

// jacobisvd split: thin/full option checks for 5x7 double row-major type.

#include "jacobisvd_helpers.h"

TEST(JacobisvdDoubleRowmajor5x7Test, Basic) {
  for (int i = 0; i < g_repeat; i++) {
    jacobisvd_thin_options<Matrix<double, 5, 7, RowMajor>>();
    jacobisvd_full_options<Matrix<double, 5, 7, RowMajor>>();
  }
}
