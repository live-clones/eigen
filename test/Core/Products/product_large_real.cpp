// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// product_large split: real scalar types, aliasing, regressions.

#include "product_large_helpers.h"

// =============================================================================
// Tests for product_large_real
// =============================================================================
TEST(ProductLargeRealTest, Basic) {
  for (int i = 0; i < g_repeat; i++) {
    product(MatrixXf(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
    product(MatrixXd(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
    product(MatrixXd(internal::random<int>(1, 10), internal::random<int>(1, 10)));

    product(MatrixXi(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
    product(Matrix<float, Dynamic, Dynamic, RowMajor>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE),
                                                      internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
    product(Matrix<bfloat16, Dynamic, Dynamic, RowMajor>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE),
                                                         internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));

    test_aliasing<float>();

    bug_1622<1>();
  }

  product_large_regressions<0>();
}
