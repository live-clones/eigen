// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2010 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// block_basic split: fixed-size and real dynamic types.
// Complex/integer dynamic types are in block_basic_dynamic.cpp.

#include "block_helpers.h"

// =============================================================================
// Tests for block_basic
// =============================================================================
TEST(BlockBasicTest, Fixed) {
  for (int i = 0; i < g_repeat; i++) {
    block(Matrix<float, 1, 1>());
    block(Matrix4d());
  }
}

TEST(BlockBasicTest, DynamicVector) {
  for (int i = 0; i < g_repeat; i++) {
    block(Matrix<float, 1, Dynamic>(internal::random(2, 50)));
    block(Matrix<float, Dynamic, 1>(internal::random(2, 50)));
  }
}
