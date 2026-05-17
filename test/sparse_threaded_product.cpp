// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#define EIGEN_USE_THREADS 1
#include "sparse.h"

// Pulls in the ThreadedSparseProduct header via the SparseCore module.
// (Eigen/Sparse -> Eigen/SparseCore conditional include under EIGEN_USE_THREADS.)
#include <Eigen/SparseCore>

template <typename SparseMatrixType, typename DenseVectorType>
void verify_threaded_spmv(Index rows, Index cols, double density) {
  typedef typename SparseMatrixType::Scalar Scalar;
  typedef Matrix<Scalar, Dynamic, Dynamic> DenseMatrix;

  DenseMatrix refMat(rows, cols);
  SparseMatrixType A(rows, cols);
  initSparse<Scalar>(density, refMat, A);
  A.makeCompressed();

  DenseVectorType x_fwd = DenseVectorType::Random(cols);
  DenseVectorType x_adj = DenseVectorType::Random(rows);

  // Reference results via the existing dense path.
  DenseVectorType y_fwd_ref = refMat * x_fwd;
  DenseVectorType y_adj_ref = refMat.adjoint() * x_adj;

  ThreadedSparseProduct<SparseMatrixType> op(A);
  VERIFY(op.rows() == A.rows());
  VERIFY(op.cols() == A.cols());

  // Overwriting forms.
  {
    DenseVectorType y(rows);
    y.setRandom();
    op.apply(x_fwd, y);
    VERIFY_IS_APPROX(y, y_fwd_ref);
  }
  {
    DenseVectorType y(cols);
    y.setRandom();
    op.applyAdjoint(x_adj, y);
    VERIFY_IS_APPROX(y, y_adj_ref);
  }

  // Accumulating forms: y += alpha * A * x, with both alpha = 1 and a
  // non-trivial alpha.
  for (Scalar alpha : {Scalar(1), Scalar(internal::random<Scalar>())}) {
    {
      DenseVectorType y0 = DenseVectorType::Random(rows);
      DenseVectorType y_ref = y0 + alpha * y_fwd_ref;
      DenseVectorType y = y0;
      op.applyAddTo(x_fwd, y, alpha);
      VERIFY_IS_APPROX(y, y_ref);
    }
    {
      DenseVectorType y0 = DenseVectorType::Random(cols);
      DenseVectorType y_ref = y0 + alpha * y_adj_ref;
      DenseVectorType y = y0;
      op.applyAdjointAddTo(x_adj, y, alpha);
      VERIFY_IS_APPROX(y, y_ref);
    }
  }

  // Calling adjoint must have materialized the mirror; calling it again must
  // not rebuild (no easy assertion for "not rebuilt", but verify the path is
  // consistent across repeated calls).
  VERIFY(op.hasMirror());
  {
    DenseVectorType y(cols);
    op.applyAdjoint(x_adj, y);
    VERIFY_IS_APPROX(y, y_adj_ref);
  }

  // Re-running analyzePattern must invalidate the mirror.
  op.analyzePattern(A);
  VERIFY(!op.hasMirror());
  {
    DenseVectorType y(cols);
    op.applyAdjoint(x_adj, y);
    VERIFY_IS_APPROX(y, y_adj_ref);
  }
}

template <typename Scalar>
void run_grid() {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  // Cover both storage orders, square and rectangular, small (serial path,
  // nnz < threshold) and larger (threaded path).
  for (int order = 0; order < 2; ++order) {
    if (order == 0) {
      verify_threaded_spmv<SparseMatrix<Scalar, ColMajor>, Vec>(8, 8, 0.5);
      verify_threaded_spmv<SparseMatrix<Scalar, ColMajor>, Vec>(13, 21, 0.3);
      verify_threaded_spmv<SparseMatrix<Scalar, ColMajor>, Vec>(500, 500, 0.15);
      verify_threaded_spmv<SparseMatrix<Scalar, ColMajor>, Vec>(800, 400, 0.10);
      // Large enough that nnz > kThreadingThreshold (20000) to exercise the
      // multi-threaded dispatch path.
      verify_threaded_spmv<SparseMatrix<Scalar, ColMajor>, Vec>(1500, 1500, 0.02);
    } else {
      verify_threaded_spmv<SparseMatrix<Scalar, RowMajor>, Vec>(8, 8, 0.5);
      verify_threaded_spmv<SparseMatrix<Scalar, RowMajor>, Vec>(21, 13, 0.3);
      verify_threaded_spmv<SparseMatrix<Scalar, RowMajor>, Vec>(500, 500, 0.15);
      verify_threaded_spmv<SparseMatrix<Scalar, RowMajor>, Vec>(400, 800, 0.10);
      verify_threaded_spmv<SparseMatrix<Scalar, RowMajor>, Vec>(1500, 1500, 0.02);
    }
  }
}

EIGEN_DECLARE_TEST(sparse_threaded_product) {
  for (int i = 0; i < g_repeat; ++i) {
    CALL_SUBTEST_1(run_grid<float>());
    CALL_SUBTEST_2(run_grid<double>());
    CALL_SUBTEST_3(run_grid<std::complex<double> >());
  }
}
