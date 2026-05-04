// This file is triangularView of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "triangular.h"

template <typename T>
class TriangularRectTest : public ::testing::Test {};

using TriangularRectTypes = ::testing::Types<Matrix<float, 4, 5>, Matrix<double, 6, 2>, MatrixXcf, MatrixXcd,
                                             Matrix<float, Dynamic, Dynamic, RowMajor>>;
EIGEN_TYPED_TEST_SUITE(TriangularRectTest, TriangularRectTypes);

TYPED_TEST(TriangularRectTest, TriangularRect) {
  for (int i = 0; i < g_repeat; i++) {
    triangular_rect(make_test_matrix<TypeParam>());
  }
}

TEST(TriangularRectTest, Deprecated) {
  for (int i = 0; i < g_repeat; i++) {
    triangular_deprecated(Matrix<float, 5, 7>());
    const int maxsize = (std::min)(EIGEN_TEST_MAX_SIZE, 20);
    const int r = internal::random<int>(2, maxsize);
    const int c = internal::random<int>(2, maxsize);
    triangular_deprecated(MatrixXd(r, c));
  }
}

TEST(TriangularRectTest, Bug159) { bug_159(); }

// Triangular solve/product at blocking boundaries (deterministic, outside g_repeat).
TEST(TriangularRectTest, BlockingBoundaries) { triangular_at_blocking_boundaries<0>(); }
