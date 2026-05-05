// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// product_trsolve split: complex scalar types — split from product_trsolve.cpp
// to reduce per-TU memory usage during compilation.

#include "product_trsolve_helpers.h"

TEST(ProductTrsolveComplexTest, Basic) {
  for (int i = 0; i < g_repeat; i++) {
    // matrices
    trsolve<std::complex<float>, Dynamic, Dynamic>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 2),
                                                   internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 2));
    trsolve<std::complex<double>, Dynamic, Dynamic>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 2),
                                                    internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 2));

    // vectors
    trsolve<std::complex<float>, Dynamic, 1>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE));
    trsolve<std::complex<double>, Dynamic, 1>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE));

    // meta-unrollers
    trsolve<std::complex<float>, 4, 1>();
  }
}
