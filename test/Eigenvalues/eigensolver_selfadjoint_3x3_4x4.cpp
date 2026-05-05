// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// eigensolver_selfadjoint split: 3x3 and 4x4 real fixed-size types.

#include "eigensolver_selfadjoint_helpers.h"

TEST(EigensolverSelfadjoint3x34x4Test, Real) {
  for (int i = 0; i < g_repeat; i++) {
    selfadjointeigensolver(Matrix3f());
    selfadjointeigensolver(Matrix3d());
    selfadjointeigensolver(Matrix4d());
  }

  bug_854<0>();
  bug_1014<0>();
  bug_1225<0>();
}
