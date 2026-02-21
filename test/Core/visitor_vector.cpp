// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "visitor_helpers.h"

EIGEN_DECLARE_TEST(visitor_vector) {
  for (int i = 0; i < g_repeat; i++) {
    vectorVisitor(Vector4f());
    vectorVisitor(Matrix<int, 12, 1>());
    vectorVisitor(VectorXd(10));
    vectorVisitor(RowVectorXd(10));
    vectorVisitor(VectorXf(33));
  }
  checkOptimalTraversal();
}
