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
class TriangularSquareTest : public ::testing::Test {};

using TriangularSquareTypes =
    ::testing::Types<Matrix<float, 1, 1>, Matrix<float, 2, 2>, Matrix3d, Matrix<std::complex<float>, 8, 8>, MatrixXcd,
                     Matrix<float, Dynamic, Dynamic, RowMajor>>;
EIGEN_TYPED_TEST_SUITE(TriangularSquareTest, TriangularSquareTypes);

TYPED_TEST(TriangularSquareTest, TriangularSquare) {
  for (int i = 0; i < g_repeat; i++) {
    triangular_square(make_square_test_matrix<TypeParam>());
  }
}
