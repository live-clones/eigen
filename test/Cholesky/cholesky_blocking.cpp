// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "cholesky_blocking.h"

TEST(CholeskyTest, BlockingBoundaryReal) {
  cholesky_blocking_boundary<double>();
  cholesky_blocking_boundary<float>();
}

TEST(CholeskyTest, RowMajorBoundary) {
  cholesky_rowmajor_boundary<double>();
  cholesky_rowmajor_boundary<float>();
}
