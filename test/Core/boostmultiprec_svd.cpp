// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// boostmultiprec split: JacobiSVD, BDCSVD, and simplicial cholesky tests.

#include "boostmultiprec_helpers.h"

EIGEN_DECLARE_TEST(boostmultiprec_svd) {
  boostmp_jacobisvd(Mat(internal::random<int>(EIGEN_TEST_MAX_SIZE / 4, EIGEN_TEST_MAX_SIZE),
                        internal::random<int>(EIGEN_TEST_MAX_SIZE / 4, EIGEN_TEST_MAX_SIZE / 2)));
  boostmp_bdcsvd(Mat(internal::random<int>(EIGEN_TEST_MAX_SIZE / 4, EIGEN_TEST_MAX_SIZE),
                     internal::random<int>(EIGEN_TEST_MAX_SIZE / 4, EIGEN_TEST_MAX_SIZE / 2)));

  (boostmp_simplicial_cholesky<Real, int, ColMajor>());
}
