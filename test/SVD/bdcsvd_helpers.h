// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2013 Gauthier Brun <brun.gauthier@gmail.com>
// Copyright (C) 2013 Nicolas Carre <nicolas.carre@ensimag.fr>
// Copyright (C) 2013 Jean Ceccato <jean.ceccato@ensimag.fr>
// Copyright (C) 2013 Pierre Zoppitelli <pierre.zoppitelli@ensimag.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/

// Shared header for split bdcsvd tests.

#ifndef EIGEN_TEST_BDCSVD_HELPERS_H
#define EIGEN_TEST_BDCSVD_HELPERS_H

// discard stack allocation as that too bypasses malloc
#define EIGEN_STACK_ALLOCATION_LIMIT 0
#define EIGEN_RUNTIME_NO_MALLOC

#include "main.h"
#include <Eigen/SVD>

#define SVD_DEFAULT(M) BDCSVD<M>
#define SVD_FOR_MIN_NORM(M) BDCSVD<M>
#define SVD_STATIC_OPTIONS(M, O) BDCSVD<M, O>
#include "svd_common.h"

template <typename MatrixType>
void bdcsvd_method() {
  enum { Size = MatrixType::RowsAtCompileTime };
  typedef typename MatrixType::RealScalar RealScalar;
  typedef Matrix<RealScalar, Size, 1> RealVecType;
  MatrixType m = MatrixType::Identity();
  VERIFY_IS_APPROX(m.bdcSvd().singularValues(), RealVecType::Ones());
  VERIFY_RAISES_ASSERT(m.bdcSvd().matrixU());
  VERIFY_RAISES_ASSERT(m.bdcSvd().matrixV());
}

// compare the Singular values returned with Jacobi and Bdc
template <typename MatrixType>
void compare_bdc_jacobi(const MatrixType& a = MatrixType(), int algoswap = 16, bool random = true) {
  MatrixType m = random ? MatrixType::Random(a.rows(), a.cols()) : a;

  BDCSVD<MatrixType> bdc_svd(m.rows(), m.cols());
  bdc_svd.setSwitchSize(algoswap);
  bdc_svd.compute(m);

  JacobiSVD<MatrixType> jacobi_svd(m);
  VERIFY_IS_APPROX(bdc_svd.singularValues(), jacobi_svd.singularValues());
}

// Verifies total deflation is **not** triggered.
inline void compare_bdc_jacobi_instance(bool structure_as_m, int algoswap = 16) {
  MatrixXd m(4, 3);
  if (structure_as_m) {
    // The first 3 rows are the reduced form of Matrix 1 as shown below, and it
    // has nonzero elements in the first column and diagonals only.
    m << 1.056293, 0, 0, -0.336468, 0.907359, 0, -1.566245, 0, 0.149150, -0.1, 0, 0;
  } else {
    // Matrix 1.
    m << 0.882336, 18.3914, -26.7921, -5.58135, 17.1931, -24.0892, -20.794, 8.68496, -4.83103, -8.4981, -10.5451,
        23.9072;
  }
  compare_bdc_jacobi(m, algoswap, false);
}

template <typename MatrixType>
void bdcsvd_thin_options(const MatrixType& input = MatrixType()) {
  svd_thin_option_checks<MatrixType, 0>(input);
}

template <typename MatrixType>
void bdcsvd_full_options(const MatrixType& input = MatrixType()) {
  svd_option_checks_full_only<MatrixType, 0>(input);
}

template <typename MatrixType>
void bdcsvd_verify_assert(const MatrixType& input = MatrixType()) {
  svd_verify_assert<MatrixType>(input);
  svd_verify_constructor_options_assert<BDCSVD<MatrixType>>(input);
}

template <typename MatrixType>
void bdcsvd_check_convergence(const MatrixType& input) {
  BDCSVD<MatrixType, Eigen::ComputeThinU | Eigen::ComputeThinV> svd(input);
  VERIFY(svd.info() == Eigen::Success);
  MatrixType D = svd.matrixU() * svd.singularValues().asDiagonal() * svd.matrixV().transpose();
  VERIFY_IS_APPROX(input, D);
}

#endif  // EIGEN_TEST_BDCSVD_HELPERS_H
