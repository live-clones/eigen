// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/

// bdcsvd split: 2x2 trivials for Matrix2cd and method test for Matrix3f.

#include "bdcsvd_helpers.h"

TEST(BDCSVDTrivialTest, Real) {
  // test matrixbase method
  (bdcsvd_method<Matrix3f>());
}
