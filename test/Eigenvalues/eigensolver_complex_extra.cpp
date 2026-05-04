// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2010 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// eigensolver_complex split: real-scalar Matrix3f (uses EigenSolver path) and
// custom complex scalar type test. Standard complex types are in eigensolver_complex.cpp.

#include "eigensolver_complex.h"
#include "CustomComplex.h"

TEST(EigensolverComplexTest, Extra) {
  for (int i = 0; i < g_repeat; i++) {
    eigensolver(Matrix3f());
  }
  eigensolver_verify_assert(Matrix3f());

  // Test custom complex scalar type.
  eigensolver(Matrix<CustomComplex<double>, 5, 5>());
}
