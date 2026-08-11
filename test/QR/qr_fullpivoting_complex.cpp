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

TEST(QRFullpivotingComplexTest, Basic) {
  for (int i = 0; i < 1; i++) {
    qr<MatrixXcd>();
  }

  for (int i = 0; i < g_repeat; i++) {
    qr_invertible<MatrixXcf>();
    qr_invertible<MatrixXcd>();
  }

  qr_verify_assert<MatrixXcf>();
  qr_verify_assert<MatrixXcd>();
}
