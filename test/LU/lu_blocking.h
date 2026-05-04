// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_TEST_LU_BLOCKING_H
#define EIGEN_TEST_LU_BLOCKING_H

#include "lu_helpers.h"

// Test LU decomposition at blocking and vectorization boundaries.
// PartialPivLU uses blocks of size max(8, min(size/8 rounded to 16, 256)).
// Sizes near these boundaries exercise transitions between full blocks
// and remainder tails, including pivot propagation across block edges.
template <typename Scalar>
void lu_blocking_boundary() {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, Dynamic> MatrixType;

  const Index PS = internal::packet_traits<Scalar>::size;
  const Index sizes[] = {1, 2, 3,  PS - 1, PS, PS + 1, 2 * PS - 1, 2 * PS, 2 * PS + 1, 4 * PS, 4 * PS + 1, 7,
                         8, 9, 15, 16,     17, 31,     32,         33,     63,         64,     65};
  for (Index si = 0; si < Index(sizeof(sizes) / sizeof(sizes[0])); ++si) {
    Index n = sizes[si];
    if (n < 1) continue;

    // Create a diagonally dominant (invertible) matrix.
    MatrixType m = MatrixType::Random(n, n);
    m.diagonal().array() += RealScalar(2 * n);

    // PartialPivLU
    PartialPivLU<MatrixType> plu(m);
    VERIFY_IS_APPROX(m, plu.reconstructedMatrix());
    MatrixType rhs = MatrixType::Random(n, 3);
    MatrixType x = plu.solve(rhs);
    VERIFY_IS_APPROX(m * x, rhs);

    // FullPivLU
    FullPivLU<MatrixType> flu(m);
    VERIFY_IS_APPROX(m, flu.reconstructedMatrix());
    VERIFY(flu.isInvertible());
    x = flu.solve(rhs);
    VERIFY_IS_APPROX(m * x, rhs);
  }

  // Non-square matrices at boundary sizes for FullPivLU.
  const Index rect_sizes[][2] = {{PS, 2 * PS}, {2 * PS, PS}, {15, 33}, {33, 15}, {1, 5}, {5, 1}};
  for (Index si = 0; si < Index(sizeof(rect_sizes) / sizeof(rect_sizes[0])); ++si) {
    Index rows = rect_sizes[si][0];
    Index cols = rect_sizes[si][1];
    MatrixType m = MatrixType::Random(rows, cols);
    FullPivLU<MatrixType> flu(m);
    VERIFY_IS_APPROX(m, flu.reconstructedMatrix());
  }
}

// Test PartialPivLU with RowMajor storage order at blocking boundaries.
template <typename Scalar>
void lu_rowmajor_boundary() {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, Dynamic, RowMajor> RowMatrixType;

  const Index sizes[] = {7, 8, 9, 15, 16, 17, 31, 32, 33};
  for (Index si = 0; si < Index(sizeof(sizes) / sizeof(sizes[0])); ++si) {
    Index n = sizes[si];
    RowMatrixType m = RowMatrixType::Random(n, n);
    m.diagonal().array() += RealScalar(2 * n);
    PartialPivLU<RowMatrixType> plu(m);
    VERIFY_IS_APPROX(m, plu.reconstructedMatrix());
  }
}

#endif  // EIGEN_TEST_LU_BLOCKING_H
