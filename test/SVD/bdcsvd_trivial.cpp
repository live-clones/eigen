// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/

// bdcsvd split: 2x2 trivials for Matrix2cd and method tests.

#include "bdcsvd_helpers.h"

EIGEN_DECLARE_TEST(bdcsvd_trivial) {
  (svd_all_trivial_2x2(bdcsvd_thin_options<Matrix2cd>));
  (svd_all_trivial_2x2(bdcsvd_full_options<Matrix2cd>));

  // test matrixbase method
  (bdcsvd_method<Matrix2cd>());
  (bdcsvd_method<Matrix3f>());
}
