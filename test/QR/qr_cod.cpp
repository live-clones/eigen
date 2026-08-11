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

// CompleteOrthogonalDecomposition tests, double type.

#include "qr_cod.h"

TEST(QRCODTest, Double) {
  for (int i = 0; i < g_repeat; i++) {
    cod<MatrixXd>();
  }

  cod_verify_assert<MatrixXd>();
}
