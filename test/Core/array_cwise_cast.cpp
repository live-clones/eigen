// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// array_cwise split: type cast tests.

#include "array_cwise_helpers.h"

EIGEN_DECLARE_TEST(array_cwise_cast) {
  for (int i = 0; i < g_repeat; i++) {
    (cast_test<1, 1>());
    (cast_test<3, 1>());
    (cast_test<5, 1>());
    (cast_test<9, 1>());
    (cast_test<17, 1>());
    (cast_test<Dynamic, 1>());
  }
}
