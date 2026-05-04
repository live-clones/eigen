// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2010 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// eigensolver_complex split: standard complex types (float/double).
// Real-scalar Matrix3f and CustomComplex<double> live in eigensolver_complex_extra.cpp.

#include "eigensolver_complex.h"

void custom_complex_stable_normalization() {
  typedef CustomComplex<double> Scalar;
  typedef Matrix<Scalar, 2, 1> Vector2;

  Vector2 input;
  input << Scalar(3.0, 4.0), Scalar(0.0, 0.0);
  const Vector2 normalized = input.stableNormalized();
  VERIFY_IS_APPROX(normalized(0).re, 0.6);
  VERIFY_IS_APPROX(normalized(0).im, 0.8);
  VERIFY_IS_EQUAL(normalized(1), Scalar(0.0, 0.0));

  input.stableNormalize();
  VERIFY_IS_APPROX(input(0).re, 0.6);
  VERIFY_IS_APPROX(input(0).im, 0.8);
  VERIFY_IS_EQUAL(input(1), Scalar(0.0, 0.0));
}

TEST(EigensolverComplexTest, Basic) {
  int s = 0;
  for (int i = 0; i < g_repeat; i++) {
    eigensolver(Matrix4cf());
    s = internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 4);
    eigensolver(MatrixXcd(s, s));
    eigensolver(Matrix<std::complex<float>, 1, 1>());
    TEST_SET_BUT_UNUSED_VARIABLE(s);
  }
  eigensolver_verify_assert(Matrix4cf());
  s = internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 4);
  eigensolver_verify_assert(MatrixXcd(s, s));
  eigensolver_verify_assert(Matrix<std::complex<float>, 1, 1>());

  // Test problem size constructors
  ComplexEigenSolver<MatrixXf> tmp(s);

  TEST_SET_BUT_UNUSED_VARIABLE(s);
}
