// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2011 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2008 Daniel Gomez Ferro <dgomezferro@gmail.com>
// Copyright (C) 2013 Desire Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// sparse_basic split: complex scalar types and long StorageIndex.

#include "sparse_basic_helpers.h"

TEST(SparseBasicCplxTest, ComplexAndLongIndex) {
  g_dense_op_sparse_count = 0;  // Suppresses compiler warning.
  for (int i = 0; i < g_repeat; i++) {
    int r = Eigen::internal::random<int>(1, 200), c = Eigen::internal::random<int>(1, 200);
    if (Eigen::internal::random<int>(0, 3) == 0) {
      r = c;  // check square matrices in 25% of tries
    }
    sparse_basic(SparseMatrix<std::complex<double>, ColMajor>(r, c));
    sparse_basic(SparseMatrix<std::complex<double>, RowMajor>(r, c));
    sparse_basic(SparseMatrix<double, ColMajor, long int>(r, c));
    sparse_basic(SparseMatrix<double, RowMajor, long int>(r, c));
  }
}
