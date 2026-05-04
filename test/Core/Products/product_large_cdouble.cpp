// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// product_large split: complex<double> scalar types and double row-major.

#include "product_large_helpers.h"

TEST(ProductLargeCdoubleTest, Basic) {
  for (int i = 0; i < g_repeat; i++) {
    product(MatrixXcd(internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 2),
                      internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 2)));
    product(Matrix<double, Dynamic, Dynamic, RowMajor>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE),
                                                       internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
    product(Matrix<std::complex<double>, Dynamic, Dynamic, RowMajor>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE),
                                                                     internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
  }
}
