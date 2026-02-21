// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// boostmultiprec split: LU, QR, and eigensolver tests.

#include "boostmultiprec_helpers.h"

EIGEN_DECLARE_TEST(boostmultiprec_lu_qr) {
  for (int i = 0; i < g_repeat; i++) {
    int s = internal::random<int>(1, EIGEN_TEST_MAX_SIZE);

    boostmp_lu<Mat>(s);
    boostmp_lu<MatC>(s);

    boostmp_qr<Mat>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE));

    boostmp_eigensolver_selfadjoint(Mat(s, s));

    boostmp_eigensolver_generic(Mat(s, s));

    boostmp_generalized_eigensolver(Mat(s, s));

    TEST_SET_BUT_UNUSED_VARIABLE(s)
  }
}
