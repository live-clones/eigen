// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2011 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "sparse_product.h"

TEST(SparseProductTest, Basic) {
  for (int i = 0; i < g_repeat; i++) {
    (test_sparse_vector_dense_product());
    (sparse_product<SparseMatrix<double, ColMajor>>());
    (sparse_product<SparseMatrix<double, RowMajor>>());
    (bug_942<double>());
    (sparse_product<SparseMatrix<float, ColMajor, long int>>());
    (sparse_product_regression_test<SparseMatrix<double, RowMajor>, Matrix<double, Dynamic, Dynamic, RowMajor>>());

    (test_mixing_types<float>());
    (test_mixed_storage());
  }
}
