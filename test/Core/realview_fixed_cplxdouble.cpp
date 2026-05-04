// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2025 The Eigen Authors
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// realview split: complex<double> + complex<long double> with fixed-size combinations.

#include "realview_helpers.h"

TEST(RealviewFixedCplxDoubleTest, Basic) {
  for (int i = 0; i < g_repeat; i++) {
    (test_realview_complex_double_long<17, 19, 17, 19>());
    (test_realview_complex_double_long<Dynamic, 1>());
    (test_realview_complex_double_long<1, Dynamic>());
    (test_realview_complex_double_long<1, 1>());
  }
}
