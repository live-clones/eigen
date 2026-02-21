// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2010 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "block_helpers.h"

EIGEN_DECLARE_TEST(block_extra) {
  for (int i = 0; i < g_repeat; i++) {
    block(MatrixXf(internal::random(2, 50), internal::random(2, 50)));
    block(Matrix<int, Dynamic, Dynamic, RowMajor>(internal::random(2, 50), internal::random(2, 50)));

    block(Matrix<float, Dynamic, 4>(3, 4));
    unwind_test(MatrixXf());

#ifndef EIGEN_DEFAULT_TO_ROW_MAJOR
    data_and_stride(MatrixXf(internal::random(5, 50), internal::random(5, 50)));

    data_and_stride(Matrix<int, Dynamic, Dynamic, RowMajor>(internal::random(5, 50), internal::random(5, 50)));
#endif
  }
}
