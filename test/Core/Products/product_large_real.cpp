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
// Typed test suite for product (real scalar types)
// =============================================================================
template <typename T>
class ProductLargeRealTest : public ::testing::Test {};

using ProductLargeRealTypes = ::testing::Types<MatrixXf, MatrixXd, MatrixXi, Matrix<float, Dynamic, Dynamic, RowMajor>,
                                               Matrix<bfloat16, Dynamic, Dynamic, RowMajor>>;
EIGEN_TYPED_TEST_SUITE(ProductLargeRealTest, ProductLargeRealTypes);

TYPED_TEST(ProductLargeRealTest, Product) {
  for (int i = 0; i < g_repeat; i++) {
    product(TypeParam(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
  }
}

// =============================================================================
// Separate tests for special sizes, aliasing, and regressions
// =============================================================================
TEST(ProductLargeRealExtraTest, MatrixXdSmall) {
  for (int i = 0; i < g_repeat; i++) {
    product(MatrixXd(internal::random<int>(1, 10), internal::random<int>(1, 10)));
  }
}

TEST(ProductLargeRealExtraTest, Aliasing) {
  for (int i = 0; i < g_repeat; i++) {
    test_aliasing<float>();
  }
}

TEST(ProductLargeRealRegressionTest, Bug1622) {
  for (int i = 0; i < g_repeat; i++) {
    bug_1622<1>();
  }
}

TEST(ProductLargeRealRegressionTest, LargeRegressions) { product_large_regressions<0>(); }
