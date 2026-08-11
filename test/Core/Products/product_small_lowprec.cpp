// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// product_small split: low-precision (half, bfloat16) and double sweeps.

#include "product_small_helpers.h"

TEST(ProductSmallLowprecTest, Basic) {
  for (int i = 0; i < g_repeat; i++) {
    product(Matrix<bfloat16, 3, 2>());

    product_sweep<Eigen::half>(10, 10, 10);
    product_sweep<Eigen::bfloat16>(10, 10, 10);
  }

  product_transition_sizes<double>();
}
