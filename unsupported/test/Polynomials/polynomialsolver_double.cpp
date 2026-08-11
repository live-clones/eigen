// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Manuel Yguel <manuel.yguel@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "polynomialsolver.h"

TEST(PolynomialsolverTest, Double) {
  for (int i = 0; i < g_repeat; i++) {
    (polynomialsolver<double, 2>(2));
    (polynomialsolver<double, 3>(3));
    (polynomialsolver<double, 5>(5));
    (polynomialsolver<double, 8>(8));
    (polynomialsolver<double, Dynamic>(internal::random<int>(9, 13)));
    (polynomialsolver<std::complex<double>, Dynamic>(internal::random<int>(2, 13)));
  }
}
