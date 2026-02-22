// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012, 2013 Chen-Pang He <jdh8@ms63.hinet.net>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "matrix_functions.h"

template <typename T>
void test2dRotation(const T& tol) {
  Matrix<T, 2, 2> A, B, C;
  T angle, c, s;

  A << 0, 1, -1, 0;
  MatrixPower<Matrix<T, 2, 2> > Apow(A);

  for (int i = 0; i <= 20; ++i) {
    angle = std::pow(T(10), T(i - 10) / T(5.));
    c = std::cos(angle);
    s = std::sin(angle);
    B << c, s, -s, c;

    C = Apow(std::ldexp(angle, 1) / T(EIGEN_PI));
    std::cout << "test2dRotation: i = " << i << "   error powerm = " << relerr(C, B) << '\n';
    VERIFY(C.isApprox(B, tol));
  }
}

template <typename T>
void test2dHyperbolicRotation(const T& tol) {
  Matrix<std::complex<T>, 2, 2> A, B, C;
  T angle, ch = std::cosh((T)1);
  std::complex<T> ish(0, std::sinh((T)1));

  A << ch, ish, -ish, ch;
  MatrixPower<Matrix<std::complex<T>, 2, 2> > Apow(A);

  for (int i = 0; i <= 20; ++i) {
    angle = std::ldexp(static_cast<T>(i - 10), -1);
    ch = std::cosh(angle);
    ish = std::complex<T>(0, std::sinh(angle));
    B << ch, ish, -ish, ch;

    C = Apow(angle);
    std::cout << "test2dHyperbolicRotation: i = " << i << "   error powerm = " << relerr(C, B) << '\n';
    VERIFY(C.isApprox(B, tol));
  }
}

template <typename T>
void test3dRotation(const T& tol) {
  Matrix<T, 3, 1> v;
  T angle;

  for (int i = 0; i <= 20; ++i) {
    v = Matrix<T, 3, 1>::Random();
    v.normalize();
    angle = std::pow(T(10), T(i - 10) / T(5.));
    VERIFY(AngleAxis<T>(angle, v).matrix().isApprox(AngleAxis<T>(1, v).matrix().pow(angle), tol));
  }
}

TEST(MatrixPowerTest, Rotation) {
  test2dRotation<double>(1e-13);
  test2dRotation<float>(2e-5f);
  test2dRotation<long double>(1e-13L);
  test2dHyperbolicRotation<double>(1e-14);
  test2dHyperbolicRotation<float>(1e-5f);
  test2dHyperbolicRotation<long double>(1e-14L);
  test3dRotation<double>(1e-13);
  test3dRotation<float>(1e-5f);
  test3dRotation<long double>(1e-13L);
}
