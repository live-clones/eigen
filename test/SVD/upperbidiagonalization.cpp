// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "upperbidiagonalization.h"

TEST(UpperbidiagonalizationTest, Real) {
  for (int i = 0; i < g_repeat; i++) {
    upperbidiag(MatrixXf(3, 3));
    upperbidiag(MatrixXd(17, 12));
    upperbidiag(Matrix<float, 6, 4>());
    upperbidiag(Matrix<float, 5, 5>());
    upperbidiag(Matrix<double, 4, 3>());

    // Larger sizes that exercise the blocked path (bcols >= 48).
    upperbidiag(MatrixXf(64, 64));
    upperbidiag(MatrixXd(100, 80));
    upperbidiag(MatrixXd(200, 200));
    upperbidiag(MatrixXf(300, 100));

    // Tall-skinny matrices.
    upperbidiag(MatrixXd(500, 10));
    upperbidiag(MatrixXf(1000, 50));
  }
}
