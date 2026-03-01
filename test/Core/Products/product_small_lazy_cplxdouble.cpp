// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// product_small split: lazy product tests for complex<double> only.
// Split from product_small_lazy_double.cpp to reduce per-TU memory usage
// under ASAN+UBSAN.

#include "product_small_helpers.h"

TEST(ProductSmallLazyCplxDoubleTest, Basic) {
  for (int i = 0; i < g_repeat; i++) {
    test_lazy_l1<std::complex<double> >();
    test_lazy_l2<std::complex<double> >();
    test_lazy_l3<std::complex<double> >();
  }
}
