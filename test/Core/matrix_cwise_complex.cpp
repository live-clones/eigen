// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "matrix_cwise.h"

TEST(MatrixCwiseTest, Complex) {
  for (int i = 0; i < g_repeat; i++) {
    test_cwise_complex(Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic>(20, 20));
    test_cwise_complex(Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>(20, 20));
  }
}
