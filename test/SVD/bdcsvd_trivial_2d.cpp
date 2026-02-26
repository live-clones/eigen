// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/

// bdcsvd split: 2x2 trivials for Matrix2d.

#include "bdcsvd_helpers.h"

TEST(BDCSVDTrivial2dTest, Basic) {
  svd_all_trivial_2x2(bdcsvd_thin_options<Matrix2d>);
  svd_all_trivial_2x2(bdcsvd_full_options<Matrix2d>);
}
