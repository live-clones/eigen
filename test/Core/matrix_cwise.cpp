// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "matrix_cwise.h"

TEST(MatrixCwiseTest, RealFloat) {
  for (int i = 0; i < g_repeat; i++) {
    test_cwise_real(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>(20, 20));
    test_cwise_real(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>(20, 20));
    test_cwise_real(Eigen::Matrix<Eigen::half, Eigen::Dynamic, Eigen::Dynamic>(20, 20));
    test_cwise_real(Eigen::Matrix<Eigen::bfloat16, Eigen::Dynamic, Eigen::Dynamic>(20, 20));
  }
}
