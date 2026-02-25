// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009-2010 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// householder split: dynamic-size matrix types.

#include "householder_helpers.h"

TEST(HouseholderTest, Dynamic) {
  for (int i = 0; i < g_repeat; i++) {
    householder(MatrixXd(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
    householder(
        MatrixXcf(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
    householder(MatrixXf(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));

    householder_update(
        MatrixXcf(internal::random<Index>(1, EIGEN_TEST_MAX_SIZE), internal::random<Index>(1, EIGEN_TEST_MAX_SIZE)));
  }
}
