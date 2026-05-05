// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "upperbidiagonalization.h"

TEST(UpperbidiagonalizationExtraTest, Real) {
  for (int i = 0; i < g_repeat; i++) {
    // Explicit unblocked path tests.
    upperbidiag_verify_unblocked(MatrixXf(MatrixXf::Random(10, 8)));
    upperbidiag_verify_unblocked(MatrixXd(MatrixXd::Random(64, 64)));
    upperbidiag_verify_unblocked(MatrixXd(MatrixXd::Random(100, 80)));

    // Structured/ill-conditioned matrices.
    upperbidiag_structured<float>(20, 15);
    upperbidiag_structured<double>(100, 80);
    upperbidiag_structured<double>(200, 200);
  }
}
