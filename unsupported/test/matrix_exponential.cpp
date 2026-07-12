// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "matrix_functions.h"

double binom(int n, int k) {
  double res = 1;
  for (int i = 0; i < k; i++) res = res * (n - k + i + 1) / (i + 1);
  return res;
}

template <typename T>
T expfn(T x, int) {
  return std::exp(x);
}

template <typename T>
void test2dRotation(T tol) {
  Matrix<T, 2, 2> A, B, C;
  T angle;

  A << 0, 1, -1, 0;
  for (int i = 0; i <= 20; i++) {
    angle = static_cast<T>(pow(10, i / 5. - 2));
    B << std::cos(angle), std::sin(angle), -std::sin(angle), std::cos(angle);

    C = (angle * A).matrixFunction(expfn);
    std::cout << "test2dRotation: i = " << i << "   error funm = " << relerr(C, B);
    VERIFY(C.isApprox(B, tol));

    C = (angle * A).exp();
    std::cout << "   error expm = " << relerr(C, B) << "\n";
    VERIFY(C.isApprox(B, tol));
  }
}

template <typename T>
void test2dHyperbolicRotation(T tol) {
  Matrix<std::complex<T>, 2, 2> A, B, C;
  std::complex<T> imagUnit(0, 1);
  T angle, ch, sh;

  for (int i = 0; i <= 20; i++) {
    angle = static_cast<T>((i - 10) / 2.0);
    ch = std::cosh(angle);
    sh = std::sinh(angle);
    A << 0, angle * imagUnit, -angle * imagUnit, 0;
    B << ch, sh * imagUnit, -sh * imagUnit, ch;

    C = A.matrixFunction(expfn);
    std::cout << "test2dHyperbolicRotation: i = " << i << "   error funm = " << relerr(C, B);
    VERIFY(C.isApprox(B, tol));

    C = A.exp();
    std::cout << "   error expm = " << relerr(C, B) << "\n";
    VERIFY(C.isApprox(B, tol));
  }
}

template <typename T>
void testPascal(T tol) {
  for (int size = 1; size < 20; size++) {
    Matrix<T, Dynamic, Dynamic> A(size, size), B(size, size), C(size, size);
    A.setZero();
    for (int i = 0; i < size - 1; i++) A(i + 1, i) = static_cast<T>(i + 1);
    B.setZero();
    for (int i = 0; i < size; i++)
      for (int j = 0; j <= i; j++) B(i, j) = static_cast<T>(binom(i, j));

    C = A.matrixFunction(expfn);
    std::cout << "testPascal: size = " << size << "   error funm = " << relerr(C, B);
    VERIFY(C.isApprox(B, tol));

    C = A.exp();
    std::cout << "   error expm = " << relerr(C, B) << "\n";
    VERIFY(C.isApprox(B, tol));
  }
}

template <typename MatrixType>
void randomTest(const MatrixType& m,
                const typename NumTraits<typename internal::traits<MatrixType>::Scalar>::Real& tol) {
  /* this test covers the following files:
     Inverse.h
  */
  typename MatrixType::Index rows = m.rows();
  typename MatrixType::Index cols = m.cols();
  MatrixType m1(rows, cols), m2(rows, cols), identity = MatrixType::Identity(rows, cols);

  for (int i = 0; i < g_repeat; i++) {
    m1 = MatrixType::Random(rows, cols);

    m2 = m1.matrixFunction(expfn) * (-m1).matrixFunction(expfn);
    std::cout << "randomTest: error funm = " << relerr(identity, m2);
    VERIFY(identity.isApprox(m2, tol));

    m2 = m1.exp() * (-m1).exp();
    std::cout << "   error expm = " << relerr(identity, m2) << "\n";
    VERIFY(identity.isApprox(m2, tol));
  }
}

EIGEN_DECLARE_TEST(matrix_exponential) {
  // matrixFunction() dominates the largest cases. The factors retain at least 1.6x headroom over the largest relative
  // errors observed in extended GCC and Clang test runs.
  CALL_SUBTEST_2(test2dRotation<double>(256 * NumTraits<double>::epsilon()));
  CALL_SUBTEST_1(test2dRotation<float>(128 * NumTraits<float>::epsilon()));
  CALL_SUBTEST_8(test2dRotation<long double>(256 * NumTraits<long double>::epsilon()));
  CALL_SUBTEST_2(test2dHyperbolicRotation<double>(32 * NumTraits<double>::epsilon()));
  CALL_SUBTEST_1(test2dHyperbolicRotation<float>(32 * NumTraits<float>::epsilon()));
  CALL_SUBTEST_8(test2dHyperbolicRotation<long double>(32 * NumTraits<long double>::epsilon()));
  CALL_SUBTEST_6(testPascal<float>(4 * NumTraits<float>::epsilon()));
  CALL_SUBTEST_5(testPascal<double>(4 * NumTraits<double>::epsilon()));
  CALL_SUBTEST_2(randomTest(Matrix2d(), 384 * NumTraits<double>::epsilon()));
  CALL_SUBTEST_7(randomTest(Matrix<double, 3, 3, RowMajor>(), 384 * NumTraits<double>::epsilon()));
  CALL_SUBTEST_3(randomTest(Matrix4cd(), 384 * NumTraits<std::complex<double>>::epsilon()));
  CALL_SUBTEST_4(randomTest(MatrixXd(8, 8), 384 * NumTraits<double>::epsilon()));
  CALL_SUBTEST_1(randomTest(Matrix2f(), 384 * NumTraits<float>::epsilon()));
  CALL_SUBTEST_5(randomTest(Matrix3cf(), 384 * NumTraits<std::complex<float>>::epsilon()));
  CALL_SUBTEST_1(randomTest(Matrix4f(), 384 * NumTraits<float>::epsilon()));
  CALL_SUBTEST_6(randomTest(MatrixXf(8, 8), 384 * NumTraits<float>::epsilon()));
  CALL_SUBTEST_9(randomTest(Matrix<long double, Dynamic, Dynamic>(7, 7), 384 * NumTraits<long double>::epsilon()));
}
