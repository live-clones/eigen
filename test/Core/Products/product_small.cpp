// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// product_small split: float product tests, regressions, and sweeps.

#include "product_small_helpers.h"

// =============================================================================
// Tests for product_small
// =============================================================================
TEST(ProductSmallTest, Basic) {
  for (int i = 0; i < g_repeat; i++) {
    product(Matrix<float, 3, 2>());
    product(Matrix4f());
    product(Matrix<bfloat16, 3, 2>());
    product1x1<0>();

    (test_linear_but_not_vectorizable<float, 2, 1, Dynamic>());
    (test_linear_but_not_vectorizable<float, 3, 1, Dynamic>());
    (test_linear_but_not_vectorizable<float, 2, 1, 16>());

    bug_1311<3>();
    bug_1311<5>();

    test_dynamic_bool();

    product_sweep<float>(10, 10, 10);
    product_sweep<Eigen::half>(10, 10, 10);
    product_sweep<Eigen::bfloat16>(10, 10, 10);
  }

  product_small_regressions<0>();
}
