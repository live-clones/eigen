// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// product_large split: complex scalar types and OpenMP regression.

#include "product_large_helpers.h"

// =============================================================================
// Typed test suite for product (complex ColMajor types, half max size)
// =============================================================================
template <typename T>
class ProductLargeCplxHalfTest : public ::testing::Test {};

using ProductLargeCplxHalfTypes = ::testing::Types<MatrixXcf, MatrixXcd>;
EIGEN_TYPED_TEST_SUITE(ProductLargeCplxHalfTest, ProductLargeCplxHalfTypes);

TYPED_TEST(ProductLargeCplxHalfTest, Product) {
  for (int i = 0; i < g_repeat; i++) {
    product(TypeParam(internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 2),
                      internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 2)));
  }
}

// =============================================================================
// Typed test suite for product (RowMajor types, full max size)
// =============================================================================
template <typename T>
class ProductLargeCplxRowMajorTest : public ::testing::Test {};

using ProductLargeCplxRowMajorTypes = ::testing::Types<Matrix<double, Dynamic, Dynamic, RowMajor>,
                                                       Matrix<std::complex<float>, Dynamic, Dynamic, RowMajor>,
                                                       Matrix<std::complex<double>, Dynamic, Dynamic, RowMajor>>;
EIGEN_TYPED_TEST_SUITE(ProductLargeCplxRowMajorTest, ProductLargeCplxRowMajorTypes);

TYPED_TEST(ProductLargeCplxRowMajorTest, Product) {
  for (int i = 0; i < g_repeat; i++) {
    product(TypeParam(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
  }
}

// =============================================================================
// Regression tests
// =============================================================================
#if defined EIGEN_HAS_OPENMP
TEST(ProductLargeCplxRegressionTest, Bug714_OpenMP) {
  omp_set_dynamic(1);
  for (int i = 0; i < g_repeat; i++) {
    product(Matrix<float, Dynamic, Dynamic>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE),
                                            internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
  }
}
#endif
