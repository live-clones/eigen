// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "visitor_helpers.h"

// =============================================================================
// Tests for visitor_matrix
// =============================================================================
TEST(VisitorMatrixTest, Fixed) {
  for (int i = 0; i < g_repeat; i++) {
    matrixVisitor(Matrix<float, 1, 1>());
    matrixVisitor(Matrix2f());
    matrixVisitor(Matrix4d());
  }
}

TEST(VisitorMatrixTest, Dynamic) {
  for (int i = 0; i < g_repeat; i++) {
    matrixVisitor(MatrixXd(8, 12));
    matrixVisitor(Matrix<double, Dynamic, Dynamic, RowMajor>(20, 20));
    matrixVisitor(MatrixXi(8, 12));
  }
}
