// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// lu split: float scalar type.

#include "lu_helpers.h"

TEST(LUTest, Basic) {
  for (int i = 0; i < g_repeat; i++) {
    lu_non_invertible<Matrix3f>();
    lu_invertible<Matrix3f>();
    lu_verify_assert<Matrix3f>();
    lu_partial_piv<Matrix3f>();

    lu_non_invertible<MatrixXf>();
    lu_invertible<MatrixXf>();
    lu_verify_assert<MatrixXf>();

    lu_non_invertible<Matrix<float, Dynamic, 16> >();

    // Test problem size constructors
    PartialPivLU<MatrixXf>(10);
    FullPivLU<MatrixXf>(10, 20);

    test_2889();
  }
}
