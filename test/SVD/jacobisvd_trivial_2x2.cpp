// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/
// SPDX-License-Identifier: MPL-2.0

// jacobisvd split: 2x2 trivial cases for double. Complex case in jacobisvd_trivial_2x2_complex.cpp.

#include "jacobisvd_helpers.h"

TEST(JacobisvdTrivial2x2Test, Basic) { svd_all_trivial_2x2(jacobisvd_thin_options<Matrix2d>); }
