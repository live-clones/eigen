// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009-2010 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "householder.h"

TEST(HouseholderTest, Float) {
  for (int i = 0; i < g_repeat; i++) {
    householder(Matrix<float, 2, 3>());
    householder(Matrix<float, 4, 4>());
    householder(MatrixXf(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));

    householder_update(Matrix<float, 4, 2>());
  }
}
