// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// qr split: real scalar types. Complex types are in qr_complex.cpp.

#include "qr.h"

TEST(QRTest, Basic) {
  for (int i = 0; i < g_repeat; i++) {
    qr(MatrixXf(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
    (qr_fixedsize<Matrix<float, 3, 4>, 2>());
    (qr_fixedsize<Matrix<double, 6, 2>, 4>());
    (qr_fixedsize<Matrix<double, 2, 5>, 7>());
    qr(Matrix<float, 1, 1>());
  }

  for (int i = 0; i < g_repeat; i++) {
    qr_invertible<MatrixXf>();
    qr_invertible<MatrixXd>();
  }

  qr_verify_assert<Matrix3f>();
  qr_verify_assert<Matrix3d>();
  qr_verify_assert<MatrixXf>();
  qr_verify_assert<MatrixXd>();

  // Test problem size constructors
  HouseholderQR<MatrixXf>(10, 20);
}
