// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012, 2013 Chen-Pang He <jdh8@ms63.hinet.net>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_TEST_MATRIX_FUNCTIONS_LOGTHENEXP_H
#define EIGEN_TEST_MATRIX_FUNCTIONS_LOGTHENEXP_H

#include "matrix_functions.h"

// Templated path: instantiates MatrixPower / Schur / triangular expression
// chains for the exact MatrixType. Use sparingly to keep test compile RSS
// bounded — only for fast-path codegen coverage of representative fixed-size
// matrices.
template <typename MatrixType>
void testLogThenExp(const MatrixType& m_const, const typename MatrixType::RealScalar& tol) {
  MatrixType& m = const_cast<MatrixType&>(m_const);

  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  Scalar x;

  const Index n = m.rows();
  RealScalar tol2 = (std::max)(tol * RealScalar(n * n), test_precision<RealScalar>() * RealScalar(n * n));

  for (int i = 0; i < g_repeat; ++i) {
    generateTestMatrix<MatrixType>::run(m, m.rows());
    x = internal::random<Scalar>();
    VERIFY(m.pow(x).isApprox((x * m.log()).exp(), tol2));
  }
}

// Canonicalized path: copies the input into Matrix<Scalar, Dynamic, Dynamic>
// before exercising m.pow / m.log / .exp(). All call sites collapse to one
// instantiation per Scalar, drastically reducing per-TU compile RSS.
template <typename MatrixType>
void testLogThenExpDynamic(const MatrixType& m_const, const typename MatrixType::RealScalar& tol) {
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename MatrixType::RealScalar;
  using DynMatrix = Matrix<Scalar, Dynamic, Dynamic>;

  DynMatrix m = m_const;
  Scalar x;
  const Index n = m.rows();
  RealScalar tol2 = (std::max)(tol * RealScalar(n * n), test_precision<RealScalar>() * RealScalar(n * n));

  for (int i = 0; i < g_repeat; ++i) {
    generateTestMatrix<DynMatrix>::run(m, m.rows());
    x = internal::random<Scalar>();
    VERIFY(m.pow(x).isApprox((x * m.log()).exp(), tol2));
  }
}

#endif  // EIGEN_TEST_MATRIX_FUNCTIONS_LOGTHENEXP_H
