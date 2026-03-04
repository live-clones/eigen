// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// product_trsolve split: real scalar types and meta-unrollers.
// Complex types are in product_trsolve_complex.cpp.

#include "product_trsolve_helpers.h"

TEST(ProductTrsolveTest, Basic) {
  for (int i = 0; i < g_repeat; i++) {
    // matrices
    trsolve<float, Dynamic, Dynamic>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE),
                                     internal::random<int>(1, EIGEN_TEST_MAX_SIZE));
    trsolve<double, Dynamic, Dynamic>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE),
                                      internal::random<int>(1, EIGEN_TEST_MAX_SIZE));

    // vectors
    trsolve<float, Dynamic, 1>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE));
    trsolve<double, Dynamic, 1>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE));

    // meta-unrollers
    trsolve<float, 4, 1>();
    trsolve<double, 4, 1>();
    trsolve<float, 1, 1>();
    trsolve<float, 1, 2>();
    trsolve<float, 3, 1>();
  }

  // Strided solve at blocking boundaries (deterministic, outside g_repeat).
  trsolve_strided_boundary<0>();
}
