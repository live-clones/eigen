// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// cholesky split: complex scalar types — split from cholesky.cpp
// to reduce per-TU memory usage during compilation.

#include "cholesky_helpers.h"

TEST(CholeskyComplexTest, Basic) {
  for (int i = 0; i < g_repeat; i++) {
    int s = internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 2);
    cholesky_cplx(MatrixXcd(s, s));
    TEST_SET_BUT_UNUSED_VARIABLE(s)
  }

  TEST_SET_BUT_UNUSED_VARIABLE(nb_temporaries)
}
