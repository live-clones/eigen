// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012, 2013 Chen-Pang He <jdh8@ms63.hinet.net>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "matrix_functions.h"

template <typename MatrixType>
void testLogThenExp(const MatrixType& m_const, const typename MatrixType::RealScalar& tol) {
  // we need to pass by reference in order to prevent errors with
  // MSVC for aligned data types ...
  MatrixType& m = const_cast<MatrixType&>(m_const);

  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  Scalar x;

  // Computing exp(x*log(m)) chains three separate Pade-based algorithms
  // (matrix log, scaling, matrix exp). The accumulated error far exceeds
  // what a single matrix power computation achieves.
  const Index n = m.rows();
  RealScalar tol2 = (std::max)(tol * RealScalar(n * n), test_precision<RealScalar>() * RealScalar(n * n));

  for (int i = 0; i < g_repeat; ++i) {
    generateTestMatrix<MatrixType>::run(m, m.rows());
    x = internal::random<Scalar>();
    VERIFY(m.pow(x).isApprox((x * m.log()).exp(), tol2));
  }
}

typedef Matrix<double, 3, 3, RowMajor> Matrix3dRowMajor;
typedef Matrix<long double, 3, 3> Matrix3e;
typedef Matrix<long double, Dynamic, Dynamic> MatrixXe;

TEST(MatrixPowerTest, LogThenExp) {
  testLogThenExp(Matrix2d(), 2e-13);
  testLogThenExp(Matrix3dRowMajor(), 1e-13);
  testLogThenExp(Matrix4cd(), 1e-13);
  testLogThenExp(MatrixXd(8, 8), 3e-12);
  testLogThenExp(Matrix2f(), 1e-4f);
  testLogThenExp(Matrix3cf(), 1e-4f);
  testLogThenExp(Matrix4f(), 1e-4f);
  testLogThenExp(MatrixXf(2, 2), 1e-3f);
  testLogThenExp(MatrixXe(7, 7), 1e-12L);
  testLogThenExp(Matrix3d(), 1e-13);
  testLogThenExp(Matrix3f(), 1e-4f);
  testLogThenExp(Matrix3e(), 1e-13L);
}
