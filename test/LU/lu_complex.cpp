// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// lu split: complex scalar types.

#include "lu_helpers.h"

TEST(LUTest, Complex) {
  for (int i = 0; i < g_repeat; i++) {
    lu_non_invertible<MatrixXcf>();
    lu_invertible<MatrixXcf>();
    lu_verify_assert<MatrixXcf>();

    lu_non_invertible<MatrixXcd>();
    lu_invertible<MatrixXcd>();
    lu_partial_piv<MatrixXcd>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE));
    lu_verify_assert<MatrixXcd>();
  }
}
