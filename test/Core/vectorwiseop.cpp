// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2015 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "vectorwiseop.h"

TEST(VectorwiseOpTest, Array) {
  for (int i = 0; i < g_repeat; i++) {
    vectorwiseop_array(Array22cd());
    vectorwiseop_array(Array<double, 3, 2>());
    vectorwiseop_array(ArrayXXf(3, 4));
  }
}

TEST(VectorwiseOpTest, MatrixDynamic) {
  for (int i = 0; i < g_repeat; i++) {
    vectorwiseop_matrix(
        MatrixXd(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
    vectorwiseop_matrix(VectorXd(internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
    vectorwiseop_matrix(RowVectorXd(internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
  }
}

TEST(VectorwiseOpTest, MixedScalar) { vectorwiseop_mixedscalar(); }

TEST(VectorwiseOpTest, ArrayInteger) {
  for (int i = 0; i < g_repeat; i++) {
    vectorwiseop_array_integer(
        ArrayXXi(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
  }
}

// RowMajor partial reductions (deterministic, outside g_repeat).
TEST(VectorwiseOpTest, RowMajor) {
  vectorwiseop_rowmajor<float>();
  vectorwiseop_rowmajor<double>();
  vectorwiseop_rowmajor<std::complex<float>>();
}
