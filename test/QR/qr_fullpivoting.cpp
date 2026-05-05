// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "qr_fullpivoting.h"

TEST(QRFullpivotingTest, Basic) {
  for (int i = 0; i < 1; i++) {
    qr<Matrix3f>();
    qr<Matrix3d>();
    qr<Matrix2f>();
    qr<MatrixXf>();
    qr<MatrixXd>();
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
  FullPivHouseholderQR<MatrixXf>(10, 20);
  (FullPivHouseholderQR<Matrix<float, 10, 20> >(10, 20));
  (FullPivHouseholderQR<Matrix<float, 10, 20> >(Matrix<float, 10, 20>::Random()));
  (FullPivHouseholderQR<Matrix<float, 20, 10> >(20, 10));
  (FullPivHouseholderQR<Matrix<float, 20, 10> >(Matrix<float, 20, 10>::Random()));
}
