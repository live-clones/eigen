// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/

// jacobisvd split: misc tests (inf/nan, method, preallocate).

#include "jacobisvd_helpers.h"

TEST(JacobisvdMiscTest, Basic) {
  for (int i = 0; i < g_repeat; i++) {
    // Test on inf/nan matrix
    (svd_inf_nan<MatrixXf>());
    (svd_inf_nan<MatrixXd>());
  }

  // test matrixbase method
  (jacobisvd_method<Matrix2cd>());
  (jacobisvd_method<Matrix3f>());

  // Test problem size constructors
  JacobiSVD<MatrixXf>(10, 10);

  // Check that preallocation avoids subsequent mallocs
  svd_preallocate<void>();

  svd_underoverflow<void>();

  msvc_workaround();
}
