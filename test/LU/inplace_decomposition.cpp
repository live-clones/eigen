// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <Eigen/LU>
#include <Eigen/Cholesky>
#include <Eigen/QR>

// This file test inplace decomposition through Ref<>, as supported by Cholesky, LU, and QR decompositions.

template <typename DecType, typename MatrixType>
void inplace(bool square = false, bool SPD = false) {
  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> RhsType;
  typedef Matrix<Scalar, MatrixType::ColsAtCompileTime, 1> ResType;

  Index rows = MatrixType::RowsAtCompileTime == Dynamic ? internal::random<Index>(2, EIGEN_TEST_MAX_SIZE / 2)
                                                        : Index(MatrixType::RowsAtCompileTime);
  Index cols = MatrixType::ColsAtCompileTime == Dynamic ? (square ? rows : internal::random<Index>(2, rows))
                                                        : Index(MatrixType::ColsAtCompileTime);

  MatrixType A = MatrixType::Random(rows, cols);
  RhsType b = RhsType::Random(rows);
  ResType x(cols);

  if (SPD) {
    assert(square);
    A.topRows(cols) = A.topRows(cols).adjoint() * A.topRows(cols);
    A.diagonal().array() += 1e-3;
  }

  MatrixType A0 = A;
  MatrixType A1 = A;

  DecType dec(A);

  // Check that the content of A has been modified
  VERIFY_IS_NOT_APPROX(A, A0);

  // Check that the decomposition is correct:
  if (rows == cols) {
    VERIFY_IS_APPROX(A0 * (x = dec.solve(b)), b);
  } else {
    VERIFY_IS_APPROX(A0.transpose() * A0 * (x = dec.solve(b)), A0.transpose() * b);
  }

  // Check that modifying A breaks the current dec:
  A.setRandom();
  if (rows == cols) {
    VERIFY_IS_NOT_APPROX(A0 * (x = dec.solve(b)), b);
  } else {
    VERIFY_IS_NOT_APPROX(A0.transpose() * A0 * (x = dec.solve(b)), A0.transpose() * b);
  }

  // Check that calling compute(A1) does not modify A1:
  A = A0;
  dec.compute(A1);
  VERIFY_IS_EQUAL(A0, A1);
  VERIFY_IS_NOT_APPROX(A, A0);
  if (rows == cols) {
    VERIFY_IS_APPROX(A0 * (x = dec.solve(b)), b);
  } else {
    VERIFY_IS_APPROX(A0.transpose() * A0 * (x = dec.solve(b)), A0.transpose() * b);
  }
}

TEST(InplaceDecompositionTest, Basic) {
  EIGEN_UNUSED typedef Matrix<double, 4, 3> Matrix43d;
  for (int i = 0; i < g_repeat; i++) {
    (inplace<LLT<Ref<MatrixXd> >, MatrixXd>(true, true));
    (inplace<LLT<Ref<Matrix4d> >, Matrix4d>(true, true));

    (inplace<LDLT<Ref<MatrixXd> >, MatrixXd>(true, true));
    (inplace<LDLT<Ref<Matrix4d> >, Matrix4d>(true, true));

    (inplace<PartialPivLU<Ref<MatrixXd> >, MatrixXd>(true, false));
    (inplace<PartialPivLU<Ref<Matrix4d> >, Matrix4d>(true, false));

    (inplace<FullPivLU<Ref<MatrixXd> >, MatrixXd>(true, false));
    (inplace<FullPivLU<Ref<Matrix4d> >, Matrix4d>(true, false));

    (inplace<HouseholderQR<Ref<MatrixXd> >, MatrixXd>(false, false));
    (inplace<HouseholderQR<Ref<Matrix43d> >, Matrix43d>(false, false));

    (inplace<ColPivHouseholderQR<Ref<MatrixXd> >, MatrixXd>(false, false));
    (inplace<ColPivHouseholderQR<Ref<Matrix43d> >, Matrix43d>(false, false));

    (inplace<FullPivHouseholderQR<Ref<MatrixXd> >, MatrixXd>(false, false));
    (inplace<FullPivHouseholderQR<Ref<Matrix43d> >, Matrix43d>(false, false));

    (inplace<CompleteOrthogonalDecomposition<Ref<MatrixXd> >, MatrixXd>(false, false));
    (inplace<CompleteOrthogonalDecomposition<Ref<Matrix43d> >, Matrix43d>(false, false));
  }
}
