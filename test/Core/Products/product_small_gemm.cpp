// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// product_small split: double/int product tests and sweep.

#include "product_small_helpers.h"

EIGEN_DECLARE_TEST(product_small_gemm) {
  for (int i = 0; i < g_repeat; i++) {
    product(Matrix<int, 3, 17>());
    product(Matrix<double, 3, 17>());
    product(Matrix3d());
    product(Matrix4d());
    product_sweep<double>(10, 10, 10);
  }
}
