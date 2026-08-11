// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2015 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "redux_helpers.h"

// =============================================================================
// Tests for redux_vector
// =============================================================================
TEST(ReduxVectorTest, Fixed) {
  for (int i = 0; i < g_repeat; i++) {
    vectorRedux(Vector4f());
    vectorRedux(Array4f());
  }
}

TEST(ReduxVectorTest, DynamicFloat) {
  int maxsize = (std::min)(100, EIGEN_TEST_MAX_SIZE);
  for (int i = 0; i < g_repeat; i++) {
    int size = internal::random<int>(1, maxsize);
    vectorRedux(VectorXf(size));
    vectorRedux(ArrayXf(size));
  }
}

TEST(ReduxVectorTest, DynamicDouble) {
  int maxsize = (std::min)(100, EIGEN_TEST_MAX_SIZE);
  for (int i = 0; i < g_repeat; i++) {
    int size = internal::random<int>(1, maxsize);
    vectorRedux(VectorXd(size));
    vectorRedux(ArrayXd(size));
  }
}

TEST(ReduxVectorTest, DynamicInt) {
  int maxsize = (std::min)(100, EIGEN_TEST_MAX_SIZE);
  for (int i = 0; i < g_repeat; i++) {
    int size = internal::random<int>(1, maxsize);
    vectorRedux(VectorXi(size));
    vectorRedux(ArrayXi(size));
    vectorRedux(VectorX<int64_t>(size));
    vectorRedux(ArrayX<int64_t>(size));
  }
}
