// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_TEST_CHOLESKY_BLOCKING_H
#define EIGEN_TEST_CHOLESKY_BLOCKING_H

#include "cholesky_helpers.h"

// Test Cholesky decomposition at blocking and vectorization boundaries.
// LLT uses blocks of size max(8, min(size/8 rounded to 16, 128)).
// Sizes near these boundaries exercise the transition between full
// blocked and unblocked paths, including triangular solve boundaries.
template <typename Scalar>
void cholesky_blocking_boundary() {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, Dynamic> MatrixType;
  typedef Matrix<Scalar, Dynamic, 1> VectorType;

  const Index PS = internal::packet_traits<Scalar>::size;
  const Index sizes[] = {1, 2, 3,  PS - 1, PS, PS + 1, 2 * PS - 1, 2 * PS, 2 * PS + 1, 4 * PS, 4 * PS + 1, 7,
                         8, 9, 15, 16,     17, 31,     32,         33,     63,         64,     65};
  for (Index si = 0; si < Index(sizeof(sizes) / sizeof(sizes[0])); ++si) {
    Index n = sizes[si];
    if (n < 1) continue;

    // Create a symmetric positive definite matrix: A = R'*R + n*I
    MatrixType R = MatrixType::Random(n, n);
    MatrixType m = R.adjoint() * R;
    m.diagonal().array() += RealScalar(n);

    // LLT
    LLT<MatrixType> llt(m);
    VERIFY(llt.info() == Success);
    VERIFY_IS_APPROX(m, llt.reconstructedMatrix());
    VectorType rhs = VectorType::Random(n);
    VectorType x = llt.solve(rhs);
    VERIFY_IS_APPROX(m * x, rhs);

    // LDLT
    LDLT<MatrixType> ldlt(m);
    VERIFY(ldlt.info() == Success);
    VERIFY_IS_APPROX(m, ldlt.reconstructedMatrix());
    x = ldlt.solve(rhs);
    VERIFY_IS_APPROX(m * x, rhs);
  }
}

// Test Cholesky with RowMajor storage at blocking boundaries.
template <typename Scalar>
void cholesky_rowmajor_boundary() {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, Dynamic, RowMajor> RowMatrixType;

  const Index sizes[] = {7, 8, 9, 15, 16, 17, 31, 32, 33};
  for (Index si = 0; si < Index(sizeof(sizes) / sizeof(sizes[0])); ++si) {
    Index n = sizes[si];
    RowMatrixType R = RowMatrixType::Random(n, n);
    RowMatrixType m = R.adjoint() * R;
    m.diagonal().array() += RealScalar(n);

    LLT<RowMatrixType> llt(m);
    VERIFY(llt.info() == Success);
    VERIFY_IS_APPROX(m, llt.reconstructedMatrix());

    LDLT<RowMatrixType> ldlt(m);
    VERIFY(ldlt.info() == Success);
    VERIFY_IS_APPROX(m, ldlt.reconstructedMatrix());
  }
}

#endif  // EIGEN_TEST_CHOLESKY_BLOCKING_H
