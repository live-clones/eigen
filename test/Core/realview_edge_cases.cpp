// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2025 The Eigen Authors
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// realview split: edge cases for complex float/double.

#include "realview_helpers.h"

// =============================================================================
// Tests for realview_edge_cases
// =============================================================================
TEST(RealviewEdgeCasesTest, Basic) {
  for (int i = 0; i < g_repeat; i++) {
    test_edge_cases(std::complex<float>());
    test_edge_cases(std::complex<double>());
  }
}
