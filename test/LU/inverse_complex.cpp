// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "inverse.h"

TEST(InverseTest, Complex) {
  int s = 0;
  for (int i = 0; i < g_repeat; i++) {
    s = internal::random<int>(25, 100);
    inverse(MatrixXcd(s, s));
    TEST_SET_BUT_UNUSED_VARIABLE(s);

    inverse(Matrix4cd());
  }
}
