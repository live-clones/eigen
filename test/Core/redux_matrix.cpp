// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2015 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

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

TEST(ReduxMatrixTest, BoolRedux) {
  // Bool reductions (deterministic, outside g_repeat)
  boolRedux(1, 1);
  boolRedux(4, 4);
  boolRedux(7, 13);
  boolRedux(63, 63);

  // Bool reductions at vectorization boundary sizes.
  // all()/any()/count() use packet-level visitors with remainder handling.
  {
    // bool packets are typically 16 bytes (SSE) or 32 bytes (AVX).
    // Test sizes around common packet sizes to catch off-by-one in remainder loops.
    const Index bsizes[] = {1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129};
    for (int si = 0; si < 18; ++si) {
      boolRedux(bsizes[si], 1);  // column vector
      boolRedux(1, bsizes[si]);  // row vector
      boolRedux(bsizes[si], 3);  // thin matrix
    }
  }
}

TEST(ReduxMatrixTest, VecBoundary) {
  // Vectorization boundary sizes -- deterministic, run once.
  // Integer types are excluded: full-range random ints overflow in sum/prod (UB).
  redux_vec_boundary<float>();
  redux_vec_boundary<double>();
}

TEST(ReduxMatrixTest, Strided) {
  // Strided (non-contiguous) reductions.
  redux_strided<float>();
  redux_strided<double>();
  redux_strided<std::complex<float>>();
}
