// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// array_cwise split: low-precision (half, bfloat16) real math functions.

#include "array_cwise_helpers.h"

TEST(ArrayCwiseRealLowprecTest, RealDynamic) {
  for (int i = 0; i < g_repeat; i++) {
    array_real(Array<Eigen::half, 32, 32>());
    array_real(Array<Eigen::bfloat16, 32, 32>());
  }
}
