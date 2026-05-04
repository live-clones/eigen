// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "product_notemporary.h"

template <typename T>
class ProductNoTemporaryComplexTest : public ::testing::Test {};

using ProductNoTemporaryComplexTypes = ::testing::Types<MatrixXcf, MatrixXcd>;
EIGEN_TYPED_TEST_SUITE(ProductNoTemporaryComplexTest, ProductNoTemporaryComplexTypes);

TYPED_TEST(ProductNoTemporaryComplexTest, ProductNoTemporary) {
  using Scalar = typename TypeParam::Scalar;
  const int max_size = NumTraits<Scalar>::IsComplex ? EIGEN_TEST_MAX_SIZE / 2 : EIGEN_TEST_MAX_SIZE;
  for (int i = 0; i < g_repeat; i++) {
    int s = internal::random<int>(16, max_size);
    product_notemporary(TypeParam(s, s));
  }
}
