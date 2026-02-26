// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012, 2013 Chen-Pang He <jdh8@ms63.hinet.net>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Shared header for split matrix_power_singular tests.

#ifndef EIGEN_TEST_MATRIX_POWER_SINGULAR_H
#define EIGEN_TEST_MATRIX_POWER_SINGULAR_H

#include "matrix_functions.h"

template <typename MatrixType>
void testSingular(const MatrixType& m_const, const typename MatrixType::RealScalar& tol) {
  // we need to pass by reference in order to prevent errors with
  // MSVC for aligned data types ...
  MatrixType& m = const_cast<MatrixType&>(m_const);

  const int IsComplex = NumTraits<typename internal::traits<MatrixType>::Scalar>::IsComplex;
  typedef std::conditional_t<IsComplex, TriangularView<MatrixType, Upper>, const MatrixType&> TriangularType;
  std::conditional_t<IsComplex, ComplexSchur<MatrixType>, RealSchur<MatrixType> > schur;
  MatrixType T;

  for (int i = 0; i < g_repeat; ++i) {
    m.setRandom();
    m.col(0).fill(0);

    schur.compute(m);
    T = schur.matrixT();
    const MatrixType& U = schur.matrixU();
    processTriangularMatrix<MatrixType>::run(m, T, U);
    MatrixPower<MatrixType> mpow(m);

    T = T.sqrt();
    VERIFY(mpow(0.5L).isApprox(U * (TriangularType(T) * U.adjoint()), tol));

    T = T.sqrt();
    VERIFY(mpow(0.25L).isApprox(U * (TriangularType(T) * U.adjoint()), tol));

    T = T.sqrt();
    VERIFY(mpow(0.125L).isApprox(U * (TriangularType(T) * U.adjoint()), tol));
  }
}

typedef Matrix<double, 3, 3, RowMajor> Matrix3dRowMajor;
typedef Matrix<long double, 3, 3> Matrix3e;
typedef Matrix<long double, Dynamic, Dynamic> MatrixXe;

#endif  // EIGEN_TEST_MATRIX_POWER_SINGULAR_H
