// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2010 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "block_helpers.h"

EIGEN_DECLARE_TEST(block_basic) {
  for (int i = 0; i < g_repeat; i++) {
    block(Matrix<float, 1, 1>());
    block(Matrix<float, 1, Dynamic>(internal::random(2, 50)));
    block(Matrix<float, Dynamic, 1>(internal::random(2, 50)));
    block(Matrix4d());
    block(MatrixXcf(internal::random(2, 50), internal::random(2, 50)));
    block(MatrixXi(internal::random(2, 50), internal::random(2, 50)));
    block(MatrixXcd(internal::random(2, 50), internal::random(2, 50)));
  }
}
