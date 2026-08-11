// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// product_small split: lazy product tests for double and complex<double>.

#include "product_small_helpers.h"

// =============================================================================
// Tests for product_small_lazy_double
// =============================================================================
TEST(ProductSmallLazyDoubleTest, Basic) {
  for (int i = 0; i < g_repeat; i++) {
    test_lazy_l1<double>();
    test_lazy_l2<double>();
    test_lazy_l3<double>();

    test_lazy_l1<std::complex<double> >();
    test_lazy_l2<std::complex<double> >();
    test_lazy_l3<std::complex<double> >();
  }
}
