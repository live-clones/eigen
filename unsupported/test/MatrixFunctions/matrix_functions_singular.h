// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012, 2013 Chen-Pang He <jdh8@ms63.hinet.net>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_TEST_MATRIX_FUNCTIONS_SINGULAR_H
#define EIGEN_TEST_MATRIX_FUNCTIONS_SINGULAR_H

#include "matrix_functions.h"

// Templated path: instantiates Schur / TriangularView / MatrixPower for the
// exact MatrixType. Use sparingly to keep test compile RSS bounded — only
// for fast-path codegen coverage of representative fixed-size matrices.
template <typename MatrixType>
void testSingular(const MatrixType& m_const, const typename MatrixType::RealScalar& tol) {
  MatrixType& m = const_cast<MatrixType&>(m_const);

  typedef typename MatrixType::RealScalar RealScalar;
  const int IsComplex = NumTraits<typename internal::traits<MatrixType>::Scalar>::IsComplex;
  typedef std::conditional_t<IsComplex, TriangularView<MatrixType, Upper>, const MatrixType&> TriangularType;
  std::conditional_t<IsComplex, ComplexSchur<MatrixType>, RealSchur<MatrixType> > schur;
  MatrixType T;

  // MatrixPower uses its own Schur decomposition while the reference uses
  // repeated sqrt of the caller-provided Schur form, so errors compound
  // from two independent algorithm paths and scale with matrix dimension.
  RealScalar tol2 = tol * RealScalar(m.rows() * m.rows());

  for (int i = 0; i < g_repeat; ++i) {
    m.setRandom();
    m.col(0).fill(0);

    schur.compute(m);
    T = schur.matrixT();
    const MatrixType& U = schur.matrixU();
    processTriangularMatrix<MatrixType>::run(m, T, U);
    MatrixPower<MatrixType> mpow(m);

    T = T.sqrt();
    VERIFY(mpow(0.5L).isApprox(U * (TriangularType(T) * U.adjoint()), tol2));

    T = T.sqrt();
    VERIFY(mpow(0.25L).isApprox(U * (TriangularType(T) * U.adjoint()), tol2));

    T = T.sqrt();
    VERIFY(mpow(0.125L).isApprox(U * (TriangularType(T) * U.adjoint()), tol2));
  }
}

// Canonicalized path: copies the input into Matrix<Scalar, Dynamic, Dynamic>
// before exercising the test. Collapses to one instantiation per Scalar.
template <typename MatrixType>
void testSingularDynamic(const MatrixType& m_const, const typename MatrixType::RealScalar& tol) {
  using Scalar = typename MatrixType::Scalar;
  using DynMatrix = Matrix<Scalar, Dynamic, Dynamic>;
  DynMatrix m = m_const;
  testSingular<DynMatrix>(m, tol);
}

#endif  // EIGEN_TEST_MATRIX_FUNCTIONS_SINGULAR_H
