// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/
// SPDX-License-Identifier: MPL-2.0

// bdcsvd split: verify_assert tests for real fixed-size rectangular types.

#include "bdcsvd_helpers.h"

TEST(BDCSVDAssertRectTest, Real) {
  (bdcsvd_verify_assert<Matrix<float, 10, 7>>());
  (bdcsvd_verify_assert<Matrix<float, 7, 10>>());
}
