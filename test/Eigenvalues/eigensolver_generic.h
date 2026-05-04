// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2010,2012 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_TEST_EIGENSOLVER_GENERIC_H
#define EIGEN_TEST_EIGENSOLVER_GENERIC_H

#include "main.h"
#include <limits>
#include <Eigen/Eigenvalues>

template <typename EigType, typename MatType>
void check_eigensolver_for_given_mat(const EigType& eig, const MatType& a) {
  typedef typename NumTraits<typename MatType::Scalar>::Real RealScalar;
  typedef Matrix<RealScalar, MatType::RowsAtCompileTime, 1> RealVectorType;
  typedef typename std::complex<RealScalar> Complex;
  Index n = a.rows();
  VERIFY_IS_EQUAL(eig.info(), Success);
  VERIFY_IS_APPROX(a * eig.pseudoEigenvectors(), eig.pseudoEigenvectors() * eig.pseudoEigenvalueMatrix());
  VERIFY_IS_APPROX(a.template cast<Complex>() * eig.eigenvectors(),
                   eig.eigenvectors() * eig.eigenvalues().asDiagonal());
  VERIFY_IS_APPROX(eig.eigenvectors().colwise().norm(), RealVectorType::Ones(n).transpose());
  VERIFY_IS_APPROX(a.eigenvalues(), eig.eigenvalues());
}

template <typename CoeffType>
Matrix<typename CoeffType::Scalar, Dynamic, Dynamic> make_companion(const CoeffType& coeffs) {
  Index n = coeffs.size() - 1;
  Matrix<typename CoeffType::Scalar, Dynamic, Dynamic> res(n, n);
  res.setZero();
  res.row(0) = -coeffs.tail(n) / coeffs(0);
  res.diagonal(-1).setOnes();
  return res;
}

#endif  // EIGEN_TEST_EIGENSOLVER_GENERIC_H
