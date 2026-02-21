// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2025 The Eigen Authors
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// realview split: real scalar types with dynamic-size combinations.

#include "realview_helpers.h"

EIGEN_DECLARE_TEST(realview_dynamic_real) {
  for (int i = 0; i < g_repeat; i++) {
    (test_realview_real_types<Dynamic, Dynamic, Dynamic, Dynamic>());
    (test_realview_real_types<Dynamic, Dynamic, 17, Dynamic>());
    (test_realview_real_types<Dynamic, Dynamic, Dynamic, 19>());
    (test_realview_real_types<Dynamic, Dynamic, 17, 19>());
    (test_realview_real_types<17, Dynamic, 17, Dynamic>());
    (test_realview_real_types<Dynamic, 19, Dynamic, 19>());
  }
}
