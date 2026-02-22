// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Gael Guennebaud <g.gael@free.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// simplicial_cholesky split: ColMajor tests.

#include "simplicial_cholesky_helpers.h"

TEST(SimplicialCholeskyColmajorTest, Basic) {
  (test_simplicial_cholesky_T<double, int, ColMajor>());
  (test_simplicial_cholesky_T<std::complex<double>, int, ColMajor>());
  (test_simplicial_cholesky_T<double, long int, ColMajor>());
}
