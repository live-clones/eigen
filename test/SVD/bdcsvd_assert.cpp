// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/

// bdcsvd split: verify_assert tests for fixed-size types.

#include "bdcsvd_helpers.h"

TEST(BDCSVDAssertTest, Basic) {
  bdcsvd_verify_assert<Matrix3f>();
  bdcsvd_verify_assert<Matrix4d>();
  bdcsvd_verify_assert<Matrix<float, 10, 7>>();
  bdcsvd_verify_assert<Matrix<float, 7, 10>>();
  bdcsvd_verify_assert<Matrix<std::complex<double>, 6, 9>>();
}
