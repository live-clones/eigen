// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// qr_colpivoting split: complex float types and small double fixed-size cases.

#include "qr_colpivoting_helpers.h"

TEST(QRColpivotingComplexTest, Basic) {
  for (int i = 0; i < g_repeat; i++) {
    qr_fixedsize<Matrix<double, 6, 2>, 3>();
    qr_fixedsize<Matrix<double, 1, 1>, 1>();
  }

  for (int i = 0; i < g_repeat; i++) {
    qr_invertible<MatrixXcf>();
  }

  qr_verify_assert<MatrixXcf>();
}
