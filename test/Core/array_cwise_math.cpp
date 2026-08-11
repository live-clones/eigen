// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "array_cwise_helpers.h"

// =============================================================================
// Tests for array_cwise_math
// =============================================================================
TEST(ArrayCwiseMathTest, FloatPow) {
  for (int i = 0; i < g_repeat; i++) {
    float_pow_test();
  }
}

TEST(ArrayCwiseMathTest, IntPow) {
  for (int i = 0; i < g_repeat; i++) {
    int_pow_test();
  }
}

TEST(ArrayCwiseMathTest, MixedPow) {
  for (int i = 0; i < g_repeat; i++) {
    mixed_pow_test();
  }
}

TEST(ArrayCwiseMathTest, Signbit) {
  for (int i = 0; i < g_repeat; i++) {
    signbit_tests();
  }
}
