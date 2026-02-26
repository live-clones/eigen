// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Manuel Yguel <manuel.yguel@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "polynomialsolver.h"

TEST(PolynomialsolverTest, Float) {
  for (int i = 0; i < g_repeat; i++) {
    polynomialsolver<float, 1>(1);
    polynomialsolver<float, 4>(4);
    polynomialsolver<float, 6>(6);
    polynomialsolver<float, 7>(7);
    polynomialsolver<float, Dynamic>(internal::random<int>(9, 13));
    polynomialsolver<float, Dynamic>(1);
  }
}
