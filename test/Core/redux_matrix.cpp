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
// Tests for redux_matrix
// =============================================================================
TEST(ReduxMatrixTest, Fixed) {
  for (int i = 0; i < g_repeat; i++) {
    matrixRedux(Matrix<float, 1, 1>());
    matrixRedux(Array<float, 1, 1>());
    matrixRedux(Matrix2f());
    matrixRedux(Array2f());
    matrixRedux(Array22f());
    matrixRedux(Matrix4d());
    matrixRedux(Array4d());
    matrixRedux(Array44d());
  }
}

TEST(ReduxMatrixTest, DynamicFloat) {
  int maxsize = (std::min)(100, EIGEN_TEST_MAX_SIZE);
  for (int i = 0; i < g_repeat; i++) {
    int rows = internal::random<int>(1, maxsize);
    int cols = internal::random<int>(1, maxsize);
    matrixRedux(MatrixXf(rows, cols));
    matrixRedux(ArrayXXf(rows, cols));
  }
}

TEST(ReduxMatrixTest, DynamicDouble) {
  int maxsize = (std::min)(100, EIGEN_TEST_MAX_SIZE);
  for (int i = 0; i < g_repeat; i++) {
    int rows = internal::random<int>(1, maxsize);
    int cols = internal::random<int>(1, maxsize);
    matrixRedux(MatrixXd(rows, cols));
    matrixRedux(ArrayXXd(rows, cols));
  }
}

TEST(ReduxMatrixTest, DynamicInt) {
  int maxsize = (std::min)(100, EIGEN_TEST_MAX_SIZE);
  for (int i = 0; i < g_repeat; i++) {
    int rows = internal::random<int>(1, maxsize);
    int cols = internal::random<int>(1, maxsize);
    matrixRedux(MatrixXi(rows, cols));
    matrixRedux(ArrayXXi(rows, cols));
    matrixRedux(MatrixX<int64_t>(rows, cols));
    matrixRedux(ArrayXX<int64_t>(rows, cols));
  }
}

TEST(ReduxMatrixTest, DynamicComplex) {
  int maxsize = (std::min)(100, EIGEN_TEST_MAX_SIZE);
  for (int i = 0; i < g_repeat; i++) {
    int rows = internal::random<int>(1, maxsize);
    int cols = internal::random<int>(1, maxsize);
    matrixRedux(MatrixXcf(rows, cols));
    matrixRedux(ArrayXXcf(rows, cols));
    matrixRedux(MatrixXcd(rows, cols));
    matrixRedux(ArrayXXcd(rows, cols));
  }
}
