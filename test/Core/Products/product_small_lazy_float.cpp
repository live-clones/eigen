// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// product_small split: lazy product tests for float only.
// complex<float> tests are in product_small_lazy_cplxfloat.cpp to reduce
// per-TU memory usage under ASAN+UBSAN.

#include "product_small_helpers.h"

TEST(ProductSmallLazyFloatTest, Basic) {
  for (int i = 0; i < g_repeat; i++) {
    test_lazy_l1<float>();
    test_lazy_l2<float>();
    test_lazy_l3<float>();
  }
}
