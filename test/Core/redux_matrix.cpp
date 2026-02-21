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

EIGEN_DECLARE_TEST(redux_matrix) {
  // the max size cannot be too large, otherwise reduxion operations obviously generate large errors.
  int maxsize = (std::min)(100, EIGEN_TEST_MAX_SIZE);
  TEST_SET_BUT_UNUSED_VARIABLE(maxsize);
  for (int i = 0; i < g_repeat; i++) {
    int rows = internal::random<int>(1, maxsize);
    int cols = internal::random<int>(1, maxsize);
    EIGEN_UNUSED_VARIABLE(rows);
    EIGEN_UNUSED_VARIABLE(cols);
    matrixRedux(Matrix<float, 1, 1>());
    matrixRedux(Array<float, 1, 1>());
    matrixRedux(Matrix2f());
    matrixRedux(Array2f());
    matrixRedux(Array22f());
    matrixRedux(Matrix4d());
    matrixRedux(Array4d());
    matrixRedux(Array44d());
    matrixRedux(MatrixXf(rows, cols));
    matrixRedux(ArrayXXf(rows, cols));
    matrixRedux(MatrixXd(rows, cols));
    matrixRedux(ArrayXXd(rows, cols));
    matrixRedux(MatrixXi(rows, cols));
    matrixRedux(ArrayXXi(rows, cols));
    matrixRedux(MatrixX<int64_t>(rows, cols));
    matrixRedux(ArrayXX<int64_t>(rows, cols));
    matrixRedux(MatrixXcf(rows, cols));
    matrixRedux(ArrayXXcf(rows, cols));
    matrixRedux(MatrixXcd(rows, cols));
    matrixRedux(ArrayXXcd(rows, cols));
  }
}
