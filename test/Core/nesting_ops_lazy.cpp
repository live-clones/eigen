// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Hauke Heibel <hauke.heibel@gmail.com>
// Copyright (C) 2015 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// nesting_ops split: lazy evaluation correctness (no asserts/segfaults).

#include "nesting_ops_helpers.h"

// =============================================================================
// Tests for nesting_ops_lazy
// =============================================================================
TEST(NestingOpsLazyTest, Basic) {
  run_nesting_ops_1(MatrixXf::Random(25, 25));
  run_nesting_ops_1(MatrixXcd::Random(25, 25));
  run_nesting_ops_1(Matrix4f::Random());
  run_nesting_ops_1(Matrix2d::Random());
}
