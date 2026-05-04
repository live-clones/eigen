// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/

// bdcsvd split: verify_assert tests for complex fixed-size types.

#include "bdcsvd_helpers.h"

TEST(BDCSVDAssertTest, Complex) { (bdcsvd_verify_assert<Matrix<std::complex<double>, 6, 9>>()); }
