// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012, 2013 Chen-Pang He <jdh8@ms63.hinet.net>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_TEST_MATRIX_FUNCTIONS_GENERAL_H
#define EIGEN_TEST_MATRIX_FUNCTIONS_GENERAL_H

#include "matrix_functions.h"

// Templated path: instantiates MatrixPower for the exact MatrixType. Use
// sparingly to keep test compile RSS bounded — only for fast-path codegen
// coverage of representative fixed-size matrices.
template <typename MatrixType>
void testGeneral(const MatrixType& m, const typename MatrixType::RealScalar& tol) {
  typedef typename MatrixType::RealScalar RealScalar;
  MatrixType m1, m2, m3, m4, m5;
  RealScalar x, y;

  // The identities A^(xy) = (A^x)^y and (cA)^y = c^y * A^y involve two
  // independent Schur decompositions. Each decomposition + Pade + squaring
  // chain introduces rounding that scales with matrix dimension.
  RealScalar tol2 = tol * RealScalar(m.rows() * m.rows());

  for (int i = 0; i < g_repeat; ++i) {
    generateTestMatrix<MatrixType>::run(m1, m.rows());
    MatrixPower<MatrixType> mpow(m1);

    x = internal::random<RealScalar>();
    y = internal::random<RealScalar>();
    m2 = mpow(x);
    m3 = mpow(y);

    m4 = mpow(x + y);
    m5.noalias() = m2 * m3;
    VERIFY(m4.isApprox(m5, tol));

    m4 = mpow(x * y);
    m5 = m2.pow(y);
    VERIFY(m4.isApprox(m5, tol2));

    m4 = (std::abs(x) * m1).pow(y);
    m5 = std::pow(std::abs(x), y) * m3;
    VERIFY(m4.isApprox(m5, tol2));
  }
}

// Canonicalized path: copies the input into Matrix<Scalar, Dynamic, Dynamic>
// before exercising MatrixPower. Collapses to one instantiation per Scalar.
template <typename MatrixType>
void testGeneralDynamic(const MatrixType& m_const, const typename MatrixType::RealScalar& tol) {
  using Scalar = typename MatrixType::Scalar;
  using DynMatrix = Matrix<Scalar, Dynamic, Dynamic>;
  DynMatrix m = m_const;
  testGeneral<DynMatrix>(m, tol);
}

#endif  // EIGEN_TEST_MATRIX_FUNCTIONS_GENERAL_H
