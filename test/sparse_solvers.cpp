// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "sparse.h"

template <typename Scalar>
void initSPD(double density, Matrix<Scalar, Dynamic, Dynamic>& refMat, SparseMatrix<Scalar>& sparseMat) {
  Matrix<Scalar, Dynamic, Dynamic> aux(refMat.rows(), refMat.cols());
  initSparse(density, refMat, sparseMat);
  refMat = refMat * refMat.adjoint();
  for (int k = 0; k < 2; ++k) {
    initSparse(density, aux, sparseMat, ForceNonZeroDiag);
    refMat += aux * aux.adjoint();
  }
  sparseMat.setZero();
  for (int j = 0; j < sparseMat.cols(); ++j)
    for (int i = j; i < sparseMat.rows(); ++i)
      if (refMat(i, j) != Scalar(0)) sparseMat.insert(i, j) = refMat(i, j);
  sparseMat.finalize();
}

template <typename Scalar>
void sparse_solvers(int rows, int cols) {
  double density = (std::max)(8. / (rows * cols), 0.01);
  typedef Matrix<Scalar, Dynamic, Dynamic> DenseMatrix;
  typedef Matrix<Scalar, Dynamic, 1> DenseVector;
  // Scalar eps = 1e-6;

  DenseVector vec1 = DenseVector::Random(rows);

  std::vector<Vector2i> zeroCoords;
  std::vector<Vector2i> nonzeroCoords;

  // test triangular solver
  {
    DenseVector vec2 = vec1, vec3 = vec1;
    SparseMatrix<Scalar> m2(rows, cols);
    DenseMatrix refMat2 = DenseMatrix::Zero(rows, cols);

    // lower - dense
    initSparse<Scalar>(density, refMat2, m2, ForceNonZeroDiag | MakeLowerTriangular, &zeroCoords, &nonzeroCoords);
    VERIFY_IS_APPROX(refMat2.template triangularView<Lower>().solve(vec2),
                     m2.template triangularView<Lower>().solve(vec3));

    // upper - dense
    initSparse<Scalar>(density, refMat2, m2, ForceNonZeroDiag | MakeUpperTriangular, &zeroCoords, &nonzeroCoords);
    VERIFY_IS_APPROX(refMat2.template triangularView<Upper>().solve(vec2),
                     m2.template triangularView<Upper>().solve(vec3));
    VERIFY_IS_APPROX(refMat2.conjugate().template triangularView<Upper>().solve(vec2),
                     m2.conjugate().template triangularView<Upper>().solve(vec3));
    {
      SparseMatrix<Scalar> cm2(m2);
      // Index rows, Index cols, Index nnz, Index* outerIndexPtr, Index* innerIndexPtr, Scalar* valuePtr
      Map<SparseMatrix<Scalar> > mm2(rows, cols, cm2.nonZeros(), cm2.outerIndexPtr(), cm2.innerIndexPtr(),
                                     cm2.valuePtr());
      VERIFY_IS_APPROX(refMat2.conjugate().template triangularView<Upper>().solve(vec2),
                       mm2.conjugate().template triangularView<Upper>().solve(vec3));
    }

    // lower - transpose
    initSparse<Scalar>(density, refMat2, m2, ForceNonZeroDiag | MakeLowerTriangular, &zeroCoords, &nonzeroCoords);
    VERIFY_IS_APPROX(refMat2.transpose().template triangularView<Upper>().solve(vec2),
                     m2.transpose().template triangularView<Upper>().solve(vec3));

    // upper - transpose
    initSparse<Scalar>(density, refMat2, m2, ForceNonZeroDiag | MakeUpperTriangular, &zeroCoords, &nonzeroCoords);
    VERIFY_IS_APPROX(refMat2.transpose().template triangularView<Lower>().solve(vec2),
                     m2.transpose().template triangularView<Lower>().solve(vec3));

    SparseMatrix<Scalar> matB(rows, rows);
    DenseMatrix refMatB = DenseMatrix::Zero(rows, rows);

    // lower - sparse
    initSparse<Scalar>(density, refMat2, m2, ForceNonZeroDiag | MakeLowerTriangular);
    initSparse<Scalar>(density, refMatB, matB);
    refMat2.template triangularView<Lower>().solveInPlace(refMatB);
    m2.template triangularView<Lower>().solveInPlace(matB);
    VERIFY_IS_APPROX(matB.toDense(), refMatB);

    // upper - sparse
    initSparse<Scalar>(density, refMat2, m2, ForceNonZeroDiag | MakeUpperTriangular);
    initSparse<Scalar>(density, refMatB, matB);
    refMat2.template triangularView<Upper>().solveInPlace(refMatB);
    m2.template triangularView<Upper>().solveInPlace(matB);
    VERIFY_IS_APPROX(matB, refMatB);

    // A triangularView is a view of the triangular PART of a possibly-general matrix,
    // so the stored matrix need not be strictly triangular. Exercise a general lhs, a
    // SparseVector rhs, a mismatched-StorageIndex rhs, an uncompressed lhs, an
    // expression lhs, and a unit diagonal -- none of which the checks above cover.
    {
      SparseMatrix<Scalar> mg(rows, rows);
      DenseMatrix refMatG = DenseMatrix::Zero(rows, rows);
      initSparse<Scalar>(density, refMatG, mg, ForceNonZeroDiag);  // GENERAL (both triangles stored)
      initSparse<Scalar>(density, refMatB, matB);

      // general matrix through a lower / upper / unit-upper view, sparse rhs
      for (int mode = 0; mode < 3; ++mode) {
        DenseMatrix rb = refMatB;
        SparseMatrix<Scalar> mb = matB;
        if (mode == 0) {
          refMatG.template triangularView<Lower>().solveInPlace(rb);
          mg.template triangularView<Lower>().solveInPlace(mb);
        } else if (mode == 1) {
          refMatG.template triangularView<Upper>().solveInPlace(rb);
          mg.template triangularView<Upper>().solveInPlace(mb);
        } else {
          refMatG.template triangularView<UnitUpper>().solveInPlace(rb);
          mg.template triangularView<UnitUpper>().solveInPlace(mb);
        }
        VERIFY_IS_APPROX(mb.toDense(), rb);
      }

      // expression lhs (no raw storage -> iterator path)
      {
        DenseMatrix rb = refMatB;
        SparseMatrix<Scalar> mb = matB;
        refMatG.template triangularView<Upper>().solveInPlace(rb);
        (Scalar(1) * mg).template triangularView<Upper>().solveInPlace(mb);
        VERIFY_IS_APPROX(mb.toDense(), rb);
      }

      // uncompressed lhs (innerNonZeroPtr != null)
      {
        DenseMatrix rb = refMatB;
        SparseMatrix<Scalar> mb = matB, mu = mg;
        mu.reserve(Matrix<int, Dynamic, 1>::Constant(mu.cols(), rows));  // -> uncompressed
        refMatG.template triangularView<Lower>().solveInPlace(rb);
        mu.template triangularView<Lower>().solveInPlace(mb);
        VERIFY_IS_APPROX(mb.toDense(), rb);
      }

      // mismatched-StorageIndex rhs (-> InnerIterator fallback)
      {
        DenseMatrix rb = refMatB;
        SparseMatrix<Scalar, ColMajor, long> mbl = matB;
        refMatG.template triangularView<Lower>().solveInPlace(rb);
        mg.template triangularView<Lower>().solveInPlace(mbl);
        VERIFY_IS_APPROX(DenseMatrix(mbl), rb);
      }

      // SparseVector rhs (sets CompressedAccessBit but its outerIndexPtr() is null)
      {
        DenseVector rv = DenseVector::Zero(rows);
        SparseVector<Scalar> vb(rows);
        for (Index i = 0; i < rows; ++i)
          if (internal::random<int>(0, 2) == 0) {
            Scalar s = internal::random<Scalar>();
            vb.coeffRef(i) = s;
            rv(i) = s;
          }
        DenseVector rref = refMatG.template triangularView<Lower>().solve(rv);
        SparseVector<Scalar> vx = vb;
        mg.template triangularView<Lower>().solveInPlace(vx);
        VERIFY_IS_APPROX(DenseVector(vx), rref);
      }
    }

    // test deprecated API
    initSparse<Scalar>(density, refMat2, m2, ForceNonZeroDiag | MakeLowerTriangular, &zeroCoords, &nonzeroCoords);
    VERIFY_IS_APPROX(refMat2.template triangularView<Lower>().solve(vec2),
                     m2.template triangularView<Lower>().solve(vec3));

    // test empty triangular matrix
    {
      m2.resize(0, 0);
      refMatB.resize(0, refMatB.cols());
      DenseMatrix res = m2.template triangularView<Lower>().solve(refMatB);
      VERIFY_IS_EQUAL(res.rows(), 0);
      VERIFY_IS_EQUAL(res.cols(), refMatB.cols());
      res = refMatB;
      m2.template triangularView<Lower>().solveInPlace(res);
      VERIFY_IS_EQUAL(res.rows(), 0);
      VERIFY_IS_EQUAL(res.cols(), refMatB.cols());
    }
  }
}

EIGEN_DECLARE_TEST(sparse_solvers) {
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(sparse_solvers<double>(8, 8));
    int s = internal::random<int>(1, 300);
    CALL_SUBTEST_2(sparse_solvers<std::complex<double> >(s, s));
    CALL_SUBTEST_1(sparse_solvers<double>(s, s));
  }
}
