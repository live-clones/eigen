// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012, 2013 Chen-Pang He <jdh8@ms63.hinet.net>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_TEST_MATRIX_POWER_LOGTHENEXP_H
#define EIGEN_TEST_MATRIX_POWER_LOGTHENEXP_H

#include "matrix_functions.h"

template <typename MatrixType>
void testLogThenExp(const MatrixType& m_const, const typename MatrixType::RealScalar& tol) {
  // we need to pass by reference in order to prevent errors with
  // MSVC for aligned data types ...
  MatrixType& m = const_cast<MatrixType&>(m_const);

  typedef typename MatrixType::Scalar Scalar;
  Scalar x;

  for (int i = 0; i < g_repeat; ++i) {
    generateTestMatrix<MatrixType>::run(m, m.rows());
    x = internal::random<Scalar>();
    VERIFY(m.pow(x).isApprox((x * m.log()).exp(), tol));
  }
}

#endif  // EIGEN_TEST_MATRIX_POWER_LOGTHENEXP_H
