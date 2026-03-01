// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Larger fixed and Dynamic-size cast tests — split from array_cwise_cast.cpp
// to reduce per-TU memory usage under ASAN+UBSAN.

#include "array_cwise_helpers.h"

TEST(ArrayCwiseCastDynamicTest, FixedLargeAndDynamic) {
  for (int i = 0; i < g_repeat; i++) {
    cast_test<9, 1>();
    cast_test<17, 1>();
    cast_test<Dynamic, 1>();
  }
}
