// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// CompleteOrthogonalDecomposition tests, fixed-size double types.

#include "qr_cod.h"

TEST(QRCODFixedTest, Double) {
  for (int i = 0; i < g_repeat; i++) {
    (cod_fixedsize<Matrix<double, 6, 2>, 3>());
    (cod_fixedsize<Matrix<double, 1, 1>, 1>());
  }

  cod_verify_assert<Matrix3d>();
}
