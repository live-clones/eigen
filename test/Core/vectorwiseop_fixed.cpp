// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2015 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "vectorwiseop.h"

TEST(VectorwiseOpFixedTest, Matrix) {
  for (int i = 0; i < g_repeat; i++) {
    vectorwiseop_matrix(Matrix4cf());
    vectorwiseop_matrix(Matrix4f());
    vectorwiseop_matrix(Vector4f());
    vectorwiseop_matrix(Matrix<float, 4, 5>());
  }
}
