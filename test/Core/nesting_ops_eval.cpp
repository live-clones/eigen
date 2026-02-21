// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Hauke Heibel <hauke.heibel@gmail.com>
// Copyright (C) 2015 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// nesting_ops split: temporary counting and eval-type verification.

#include "nesting_ops_helpers.h"

EIGEN_DECLARE_TEST(nesting_ops_eval) {
  Index s = internal::random<int>(1, EIGEN_TEST_MAX_SIZE);
  run_nesting_ops_2(MatrixXf(s, s));
  run_nesting_ops_2(MatrixXcd(s, s));
  run_nesting_ops_2(Matrix4f());
  run_nesting_ops_2(Matrix2d());
  TEST_SET_BUT_UNUSED_VARIABLE(s)
}
