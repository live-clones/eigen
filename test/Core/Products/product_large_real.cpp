// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// product_large split: float scalar types, bfloat16, aliasing.
// Double/int types and regressions are in product_large_double.cpp.

#include "product_large_helpers.h"

TEST(ProductLargeFloatTest, Basic) {
  for (int i = 0; i < g_repeat; i++) {
    product(MatrixXf(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
    product(Matrix<float, Dynamic, Dynamic, RowMajor>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE),
                                                      internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
    product(Matrix<bfloat16, Dynamic, Dynamic, RowMajor>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE),
                                                         internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));

    test_aliasing<float>();
  }
}
