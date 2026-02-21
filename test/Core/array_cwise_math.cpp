// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "array_cwise_helpers.h"

EIGEN_DECLARE_TEST(array_cwise_math) {
  for (int i = 0; i < g_repeat; i++) {
    float_pow_test();
    int_pow_test();
    mixed_pow_test();
    signbit_tests();
  }
}
