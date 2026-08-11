// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "matrix_exponential.h"

TEST(MatrixExponentialTest, Double) {
  test2dRotation<double>(1e-13);
  test2dRotation<long double>(1e-13);
  test2dHyperbolicRotation<double>(1e-14);
  test2dHyperbolicRotation<long double>(1e-14);
  testPascal<double>(1e-15);

  // Fixed-size sample for fast-path codegen coverage.
  randomTest(Matrix2d(), 1e-13);

  // Dynamic-size coverage at varying sizes / scalars.
  randomTestDynamic(MatrixXd(3, 3), 1e-13);
  randomTestDynamic(MatrixXd(8, 8), 1e-13);
  randomTestDynamic(Matrix<std::complex<double>, Dynamic, Dynamic>(4, 4), 1e-13);
  randomTestDynamic(Matrix<long double, Dynamic, Dynamic>(7, 7), 1e-13);
}

template <int Options>
void testComplexScalingPath() {
  using Scalar = std::complex<double>;
  using MatrixType = Matrix<Scalar, 3, 3, Options>;
  MatrixType A = MatrixType::Zero();
  A.diagonal() << Scalar(-1.0, 64.0), Scalar(0.5, -32.0), Scalar(-0.25, 16.0);

  MatrixType expected = MatrixType::Zero();
  for (Index i = 0; i < A.rows(); ++i) expected(i, i) = std::exp(A(i, i));

  const double tol = 100.0 * NumTraits<double>::epsilon();
  VERIFY(A.exp().isApprox(expected, tol));
}

TEST(MatrixExponentialTest, TestComplexScalingPath) {
  (testComplexScalingPath<ColMajor>());
  (testComplexScalingPath<RowMajor>());
}

void testCustomComplexScalingPath() {
  using Scalar = CustomComplex<double>;
  using MatrixType = Matrix<Scalar, 3, 3>;
  static_assert(!internal::complex_array_access<Scalar>::value, "test must exercise the scalar scaling fallback");

  MatrixType A = MatrixType::Zero();
  A.diagonal() << Scalar(-1.0, 64.0), Scalar(0.5, -32.0), Scalar(-0.25, 16.0);

  const int squarings = 4;
  const MatrixType scaled = internal::matrix_exp_scale<MatrixType>(A, squarings);
  for (Index i = 0; i < A.size(); ++i) {
    using std::ldexp;
    VERIFY_IS_EQUAL(numext::real(scaled(i)), ldexp(numext::real(A(i)), -squarings));
    VERIFY_IS_EQUAL(numext::imag(scaled(i)), ldexp(numext::imag(A(i)), -squarings));
  }
}

TEST(MatrixExponentialTest, TestCustomComplexScalingPath) { testCustomComplexScalingPath(); }
