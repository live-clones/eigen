// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2011 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2008 Daniel Gomez Ferro <dgomezferro@gmail.com>
// Copyright (C) 2013 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// sparse_basic split: double and float types with default StorageIndex.
// Complex types and non-default StorageIndex are in sparse_basic_extra.cpp.

#include "sparse_basic_helpers.h"

TEST(SparseBasicTest, Basic) {
  g_dense_op_sparse_count = 0;  // Suppresses compiler warning.
  for (int i = 0; i < g_repeat; i++) {
    int r = Eigen::internal::random<int>(1, 200), c = Eigen::internal::random<int>(1, 200);
    if (Eigen::internal::random<int>(0, 3) == 0) {
      r = c;  // check square matrices in 25% of tries
    }
    EIGEN_UNUSED_VARIABLE(r + c);
    sparse_basic(SparseMatrix<double>(1, 1));
    sparse_basic(SparseMatrix<double>(8, 8));
    sparse_basic(SparseMatrix<float, RowMajor>(r, c));
    sparse_basic(SparseMatrix<float, ColMajor>(r, c));
    sparse_basic(SparseMatrix<double, ColMajor>(r, c));
    sparse_basic(SparseMatrix<double, RowMajor>(r, c));
  }

  // Regression test for bug 900: (manually insert higher values here, if you have enough RAM):
  big_sparse_triplet<SparseMatrix<float, RowMajor, int>>(10000, 10000, 0.125);
  big_sparse_triplet<SparseMatrix<double, ColMajor, long int>>(10000, 10000, 0.125);

  bug1105<0>();
}
