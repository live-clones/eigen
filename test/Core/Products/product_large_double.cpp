// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// product_large split: double/int scalar types and regressions.
// Float/bfloat16 types are in product_large_real.cpp.

#include "product_large_helpers.h"

TEST(ProductLargeDoubleTest, Basic) {
  for (int i = 0; i < g_repeat; i++) {
    product(MatrixXd(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
    product(MatrixXd(internal::random<int>(1, 10), internal::random<int>(1, 10)));
    product(MatrixXi(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));

    test_aliasing<double>();
    bug_1622<1>();
  }

  product_large_regressions<0>();
  bug_gemv_rowmajor_large_stride<0>();
  bug_gemv_run_small_cols<0>();
  gemv_small_cols_systematic<0>();
  gemv_rowmajor_large_stride_varied_rows<0>();
  product_extreme_aspect_ratios<0>();
}
