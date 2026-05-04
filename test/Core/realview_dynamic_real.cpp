// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2025 The Eigen Authors
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// realview split: float scalar with dynamic-size combinations.

#include "realview_helpers.h"

TEST(RealviewDynamicRealTest, Float) {
  for (int i = 0; i < g_repeat; i++) {
    (test_realview_float<Dynamic, Dynamic, Dynamic, Dynamic>());
    (test_realview_float<Dynamic, Dynamic, 17, Dynamic>());
    (test_realview_float<Dynamic, Dynamic, Dynamic, 19>());
    (test_realview_float<Dynamic, Dynamic, 17, 19>());
    (test_realview_float<17, Dynamic, 17, Dynamic>());
    (test_realview_float<Dynamic, 19, Dynamic, 19>());
  }
}
