// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include <Eigen/LU>
#include <Eigen/Cholesky>
#include <Eigen/QR>
#include <Eigen/Eigenvalues>

// This file test inplace decomposition through Ref<>, as supported by the Cholesky, LU, and QR decompositions, the
// Hessenberg, tridiagonal, Schur and QZ reductions, and the dense eigensolvers.

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

template <typename MatrixType>
MatrixType random_selfadjoint(Index n) {
  MatrixType a = MatrixType::Random(n, n);
  return a + a.adjoint();
}

// The Ref<> and plain instantiations perform the same operations on the same memory layout only for dynamic-size
// column-major types. For fixed sizes and row-major storage the compile-time shapes differ (a Ref<> has a dynamic
// outer stride), and with them the compiler's floating-point contraction, so their results agree only to rounding.
template <typename MatrixType>
constexpr bool same_operations() {
  return MatrixType::SizeAtCompileTime == Dynamic && !MatrixType::IsRowMajor;
}

template <typename MatrixType, typename LhsType, typename RhsType>
void verify_same_result(const LhsType& actual, const RhsType& expected) {
  if (same_operations<MatrixType>()) {
    VERIFY_IS_EQUAL(actual, expected);
  } else {
    VERIFY_IS_APPROX(actual, expected);
  }
}

// The reductions and eigensolvers below check that the Ref<> instantiation works within the memory of its input,
// that compute() then reads its argument without rebinding, and that the results are those of the plain
// instantiation, which runs the same operations on a copy.

template <typename MatrixType>
void inplace_reductions(Index size) {
  using Scalar = typename MatrixType::Scalar;
  {
    MatrixType A = MatrixType::Random(size, size), A0 = A;
    HessenbergDecomposition<Ref<MatrixType> > hess(A);
    HessenbergDecomposition<MatrixType> hess0(A0);
    VERIFY(internal::is_same_dense(hess.packedMatrix(), A));
    verify_same_result<MatrixType>(hess.packedMatrix(), hess0.packedMatrix());
    MatrixType Q = hess.matrixQ(), H = hess.matrixH();
    VERIFY_IS_APPROX(A0, Q * H * Q.adjoint());

    MatrixType A1 = MatrixType::Random(size, size), A1c = A1;
    hess.compute(A1);
    VERIFY_IS_EQUAL(A1, A1c);
    verify_same_result<MatrixType>(A, HessenbergDecomposition<MatrixType>(A1).packedMatrix());
  }
  {
    MatrixType A = random_selfadjoint<MatrixType>(size), A0 = A;
    Tridiagonalization<Ref<MatrixType> > tri(A);
    Tridiagonalization<MatrixType> tri0(A0);
    VERIFY(internal::is_same_dense(tri.packedMatrix(), A));
    verify_same_result<MatrixType>(tri.packedMatrix(), tri0.packedMatrix());
    MatrixType Q = tri.matrixQ(), T = tri.matrixT().eval().template cast<Scalar>();
    VERIFY_IS_APPROX(A0, Q * T * Q.adjoint());
  }
}

template <template <typename> class Schur, typename MatrixType>
void inplace_schur(Index size) {
  MatrixType A = MatrixType::Random(size, size), A0 = A;
  Schur<Ref<MatrixType> > schur(A);
  Schur<MatrixType> schur0(A0);
  VERIFY_IS_EQUAL(schur.info(), Success);
  VERIFY(internal::is_same_dense(schur.matrixT(), A));
  verify_same_result<MatrixType>(schur.matrixT(), schur0.matrixT());
  verify_same_result<MatrixType>(schur.matrixU(), schur0.matrixU());
  VERIFY_IS_APPROX(A0, schur.matrixU() * A * schur.matrixU().adjoint());

  MatrixType A1 = MatrixType::Random(size, size), A1c = A1;
  schur.compute(A1, false);
  VERIFY_IS_EQUAL(A1, A1c);
  verify_same_result<MatrixType>(A, Schur<MatrixType>(A1, false).matrixT());
}

template <typename MatrixType>
void inplace_selfadjoint_eigensolver(Index size) {
  using RealScalar = typename MatrixType::RealScalar;
  MatrixType A = random_selfadjoint<MatrixType>(size), A0 = A;
  SelfAdjointEigenSolver<Ref<MatrixType> > es(A);
  SelfAdjointEigenSolver<MatrixType> es0(A0);
  VERIFY_IS_EQUAL(es.info(), Success);
  VERIFY(internal::is_same_dense(es.eigenvectors(), A));
  verify_same_result<MatrixType>(es.eigenvalues(), es0.eigenvalues());
  if (same_operations<MatrixType>()) VERIFY_IS_EQUAL(es.eigenvectors(), es0.eigenvectors());
  VERIFY_IS_APPROX(A0 * A, A * es.eigenvalues().asDiagonal());

  MatrixType A1 = random_selfadjoint<MatrixType>(size), A1c = A1;
  es.compute(A1, EigenvaluesOnly);
  VERIFY_IS_EQUAL(A1, A1c);
  verify_same_result<MatrixType>(es.eigenvalues(),
                                 SelfAdjointEigenSolver<MatrixType>(A1, EigenvaluesOnly).eigenvalues());

  // A Ref<> whose outer stride exceeds its number of rows.
  if (MatrixType::RowsAtCompileTime == Dynamic) {
    MatrixType big = MatrixType::Random(size + 3, size + 2);
    Ref<MatrixType> B = big.block(2, 1, size, size);
    B = random_selfadjoint<MatrixType>(size);
    MatrixType Bin = B;
    SelfAdjointEigenSolver<Ref<MatrixType> > esb(B);
    VERIFY(internal::is_same_dense(esb.eigenvectors(), B));
    verify_same_result<MatrixType>(esb.eigenvalues(), SelfAdjointEigenSolver<MatrixType>(Bin).eigenvalues());
    VERIFY_IS_APPROX(Bin * B, B * esb.eigenvalues().asDiagonal());
  }

  MatrixType Ag = random_selfadjoint<MatrixType>(size);
  MatrixType Bg = MatrixType::Random(size, size);
  Bg = Bg * Bg.adjoint() + RealScalar(size) * MatrixType::Identity(size, size);
  for (int type : {int(Ax_lBx), int(ABx_lx), int(BAx_lx)}) {
    MatrixType A2 = Ag, B2 = Bg;
    GeneralizedSelfAdjointEigenSolver<Ref<MatrixType> > ges(A2, B2, ComputeEigenvectors | type);
    GeneralizedSelfAdjointEigenSolver<MatrixType> ges0(Ag, Bg, ComputeEigenvectors | type);
    VERIFY_IS_EQUAL(ges.info(), Success);
    VERIFY(internal::is_same_dense(ges.eigenvectors(), A2));
    verify_same_result<MatrixType>(ges.eigenvalues(), ges0.eigenvalues());
    if (same_operations<MatrixType>()) VERIFY_IS_EQUAL(ges.eigenvectors(), ges0.eigenvectors());
    const MatrixType& V = ges.eigenvectors();
    MatrixType VD = V * ges.eigenvalues().asDiagonal();
    if (type == Ax_lBx) {
      VERIFY_IS_APPROX(Ag * V, Bg * VD);
    } else if (type == ABx_lx) {
      VERIFY_IS_APPROX(Ag * (Bg * V), VD);
    } else {
      VERIFY_IS_APPROX(Bg * (Ag * V), VD);
    }

    MatrixType A3 = Ag, B3 = Bg;
    GeneralizedSelfAdjointEigenSolver<Ref<MatrixType> > gesv(A3, B3, EigenvaluesOnly | type);
    verify_same_result<MatrixType>(gesv.eigenvalues(), ges0.eigenvalues());
  }
}

// A zero matrix takes the early exits and a NaN entry the failure paths of the real Schur-based solvers; both must
// end in the state the plain instantiation reaches.
template <typename MatrixType>
void inplace_special_values(Index size) {
  using RealScalar = typename MatrixType::RealScalar;
  for (int kind = 0; kind < 2; ++kind) {
    MatrixType A0 = MatrixType::Zero(size, size);
    if (kind == 1) A0(0, 0) = std::numeric_limits<RealScalar>::quiet_NaN();

    MatrixType A = A0;
    RealSchur<Ref<MatrixType> > schur(A);
    RealSchur<MatrixType> schur0(A0);
    VERIFY_IS_EQUAL(schur.info(), schur0.info());
    if (schur.info() == Success) {
      VERIFY_IS_EQUAL(schur.matrixT(), schur0.matrixT());
      VERIFY_IS_EQUAL(schur.matrixU(), schur0.matrixU());
    }

    MatrixType B = A0;
    SelfAdjointEigenSolver<Ref<MatrixType> > saes(B);
    SelfAdjointEigenSolver<MatrixType> saes0(A0);
    VERIFY_IS_EQUAL(saes.info(), saes0.info());
    if (saes.info() == Success) {
      VERIFY_IS_EQUAL(saes.eigenvalues(), saes0.eigenvalues());
      VERIFY_IS_EQUAL(saes.eigenvectors(), saes0.eigenvectors());
    }

    MatrixType C = A0;
    EigenSolver<Ref<MatrixType> > es(C);
    EigenSolver<MatrixType> es0(A0);
    VERIFY_IS_EQUAL(es.info(), es0.info());
    if (es.info() == Success) {
      VERIFY_IS_EQUAL(es.eigenvalues(), es0.eigenvalues());
      VERIFY_IS_EQUAL(es.pseudoEigenvectors(), es0.pseudoEigenvectors());
    }
  }
}

template <typename MatrixType>
void inplace_eigensolver(Index size) {
  using EigenvectorsType = typename EigenSolver<MatrixType>::EigenvectorsType;
  using ComplexScalar = typename EigenvectorsType::Scalar;
  MatrixType A = MatrixType::Random(size, size), A0 = A;
  EigenSolver<Ref<MatrixType> > es(A);
  EigenSolver<MatrixType> es0(A0);
  VERIFY_IS_EQUAL(es.info(), Success);
  verify_same_result<MatrixType>(es.eigenvalues(), es0.eigenvalues());
  verify_same_result<MatrixType>(es.eigenvectors(), es0.eigenvectors());
  verify_same_result<MatrixType>(es.pseudoEigenvectors(), es0.pseudoEigenvectors());
  EigenvectorsType V = es.eigenvectors();
  VERIFY_IS_APPROX(A0.template cast<ComplexScalar>() * V, V * es.eigenvalues().asDiagonal());
  VERIFY_IS_APPROX(A0 * es.pseudoEigenvectors(), es.pseudoEigenvectors() * es.pseudoEigenvalueMatrix());

  MatrixType A1 = MatrixType::Random(size, size), A1c = A1;
  es.compute(A1, false);
  VERIFY_IS_EQUAL(A1, A1c);
  verify_same_result<MatrixType>(es.eigenvalues(), EigenSolver<MatrixType>(A1, false).eigenvalues());
}

template <typename MatrixType>
void inplace_complex_eigensolver(Index size) {
  MatrixType A = MatrixType::Random(size, size), A0 = A;
  ComplexEigenSolver<Ref<MatrixType> > es(A);
  ComplexEigenSolver<MatrixType> es0(A0);
  VERIFY_IS_EQUAL(es.info(), Success);
  verify_same_result<MatrixType>(es.eigenvalues(), es0.eigenvalues());
  verify_same_result<MatrixType>(es.eigenvectors(), es0.eigenvectors());
  VERIFY_IS_APPROX(A0 * es.eigenvectors(), es.eigenvectors() * es.eigenvalues().asDiagonal());

  MatrixType A1 = MatrixType::Random(size, size), A1c = A1;
  es.compute(A1, false);
  VERIFY_IS_EQUAL(A1, A1c);
  verify_same_result<MatrixType>(es.eigenvalues(), ComplexEigenSolver<MatrixType>(A1, false).eigenvalues());
}

template <typename MatrixType>
void inplace_real_qz(Index size) {
  using EigenvectorsType = typename GeneralizedEigenSolver<MatrixType>::EigenvectorsType;
  using ComplexScalar = typename EigenvectorsType::Scalar;
  MatrixType A = MatrixType::Random(size, size), B = MatrixType::Random(size, size), A0 = A, Bin = B;
  RealQZ<Ref<MatrixType> > qz(A, B);
  RealQZ<MatrixType> qz0(A0, Bin);
  VERIFY_IS_EQUAL(qz.info(), Success);
  VERIFY(internal::is_same_dense(qz.matrixS(), A));
  VERIFY(internal::is_same_dense(qz.matrixT(), B));
  verify_same_result<MatrixType>(qz.matrixS(), qz0.matrixS());
  verify_same_result<MatrixType>(qz.matrixT(), qz0.matrixT());
  verify_same_result<MatrixType>(qz.matrixQ(), qz0.matrixQ());
  verify_same_result<MatrixType>(qz.matrixZ(), qz0.matrixZ());
  VERIFY_IS_APPROX(A0, qz.matrixQ() * A * qz.matrixZ());
  VERIFY_IS_APPROX(Bin, qz.matrixQ() * B * qz.matrixZ());

  MatrixType A1 = MatrixType::Random(size, size), B1 = MatrixType::Random(size, size), A1c = A1, B1c = B1;
  qz.compute(A1, B1, false);
  VERIFY_IS_EQUAL(A1, A1c);
  VERIFY_IS_EQUAL(B1, B1c);
  verify_same_result<MatrixType>(A, RealQZ<MatrixType>(A1, B1, false).matrixS());

  MatrixType A2 = A0, B2 = Bin;
  GeneralizedEigenSolver<Ref<MatrixType> > ges(A2, B2);
  GeneralizedEigenSolver<MatrixType> ges0(A0, Bin);
  VERIFY_IS_EQUAL(ges.info(), Success);
  verify_same_result<MatrixType>(ges.alphas(), ges0.alphas());
  verify_same_result<MatrixType>(ges.betas(), ges0.betas());
  verify_same_result<MatrixType>(ges.eigenvectors(), ges0.eigenvectors());
  EigenvectorsType V = ges.eigenvectors();
  VERIFY_IS_APPROX(A0.template cast<ComplexScalar>() * V * ges.betas().template cast<ComplexScalar>().asDiagonal(),
                   Bin.template cast<ComplexScalar>() * V * ges.alphas().asDiagonal());
}

template <typename MatrixType>
void inplace_complex_qz(Index size) {
  MatrixType A = MatrixType::Random(size, size), B = MatrixType::Random(size, size), A0 = A, Bin = B;
  ComplexQZ<Ref<MatrixType> > qz(A, B);
  ComplexQZ<MatrixType> qz0(A0, Bin);
  VERIFY_IS_EQUAL(qz.info(), Success);
  VERIFY(internal::is_same_dense(qz.matrixS(), A));
  VERIFY(internal::is_same_dense(qz.matrixT(), B));
  verify_same_result<MatrixType>(qz.matrixS(), qz0.matrixS());
  verify_same_result<MatrixType>(qz.matrixT(), qz0.matrixT());
  verify_same_result<MatrixType>(qz.matrixQ(), qz0.matrixQ());
  verify_same_result<MatrixType>(qz.matrixZ(), qz0.matrixZ());
  VERIFY_IS_APPROX(A0, qz.matrixQ() * A * qz.matrixZ());
  VERIFY_IS_APPROX(Bin, qz.matrixQ() * B * qz.matrixZ());

  MatrixType A1 = MatrixType::Random(size, size), B1 = MatrixType::Random(size, size), A1c = A1, B1c = B1;
  qz.compute(A1, B1, false);
  VERIFY_IS_EQUAL(A1, A1c);
  VERIFY_IS_EQUAL(B1, B1c);
  verify_same_result<MatrixType>(A, ComplexQZ<MatrixType>(A1, B1, false).matrixS());
}

EIGEN_DECLARE_TEST(inplace_decomposition) {
  EIGEN_UNUSED typedef Matrix<double, 4, 3> Matrix43d;
  for (int i = 0; i < g_repeat; i++) {
    Index size = internal::random<Index>(2, EIGEN_TEST_MAX_SIZE / 4);

    CALL_SUBTEST_1((inplace<LLT<Ref<MatrixXd> >, MatrixXd>(true, true)));
    CALL_SUBTEST_1((inplace<LLT<Ref<Matrix4d> >, Matrix4d>(true, true)));

    CALL_SUBTEST_2((inplace<LDLT<Ref<MatrixXd> >, MatrixXd>(true, true)));
    CALL_SUBTEST_2((inplace<LDLT<Ref<Matrix4d> >, Matrix4d>(true, true)));

    CALL_SUBTEST_3((inplace<PartialPivLU<Ref<MatrixXd> >, MatrixXd>(true, false)));
    CALL_SUBTEST_3((inplace<PartialPivLU<Ref<Matrix4d> >, Matrix4d>(true, false)));

    CALL_SUBTEST_4((inplace<FullPivLU<Ref<MatrixXd> >, MatrixXd>(true, false)));
    CALL_SUBTEST_4((inplace<FullPivLU<Ref<Matrix4d> >, Matrix4d>(true, false)));

    CALL_SUBTEST_5((inplace<HouseholderQR<Ref<MatrixXd> >, MatrixXd>(false, false)));
    CALL_SUBTEST_5((inplace<HouseholderQR<Ref<Matrix43d> >, Matrix43d>(false, false)));

    CALL_SUBTEST_6((inplace<ColPivHouseholderQR<Ref<MatrixXd> >, MatrixXd>(false, false)));
    CALL_SUBTEST_6((inplace<ColPivHouseholderQR<Ref<Matrix43d> >, Matrix43d>(false, false)));

    CALL_SUBTEST_7((inplace<FullPivHouseholderQR<Ref<MatrixXd> >, MatrixXd>(false, false)));
    CALL_SUBTEST_7((inplace<FullPivHouseholderQR<Ref<Matrix43d> >, Matrix43d>(false, false)));

    CALL_SUBTEST_8((inplace<CompleteOrthogonalDecomposition<Ref<MatrixXd> >, MatrixXd>(false, false)));
    CALL_SUBTEST_8((inplace<CompleteOrthogonalDecomposition<Ref<Matrix43d> >, Matrix43d>(false, false)));

    CALL_SUBTEST_2((inplace<BunchKaufman<Ref<MatrixXd> >, MatrixXd>(true, true)));
    CALL_SUBTEST_2((inplace<BunchKaufman<Ref<Matrix4d> >, Matrix4d>(true, true)));

    CALL_SUBTEST_9((inplace_reductions<MatrixXd>(size)));
    CALL_SUBTEST_9((inplace_reductions<MatrixXcd>(size)));
    CALL_SUBTEST_9((inplace_reductions<MatrixXd>(1)));
    CALL_SUBTEST_9((inplace_reductions<Matrix4f>(4)));

    CALL_SUBTEST_10((inplace_schur<RealSchur, MatrixXd>(size)));
    CALL_SUBTEST_10((inplace_schur<RealSchur, MatrixXd>(1)));
    CALL_SUBTEST_10((inplace_schur<RealSchur, Matrix4f>(4)));
    CALL_SUBTEST_10((inplace_schur<ComplexSchur, MatrixXcd>(size)));
    CALL_SUBTEST_10((inplace_schur<ComplexSchur, MatrixXcd>(1)));
    CALL_SUBTEST_10((inplace_schur<ComplexSchur, Matrix4cf>(4)));
    CALL_SUBTEST_10((inplace_special_values<MatrixXd>(size)));
    CALL_SUBTEST_10((inplace_special_values<Matrix4f>(4)));

    CALL_SUBTEST_11((inplace_selfadjoint_eigensolver<MatrixXd>(size)));
    CALL_SUBTEST_11((inplace_selfadjoint_eigensolver<MatrixXcd>(size)));
    CALL_SUBTEST_11((inplace_selfadjoint_eigensolver<MatrixXd>(1)));
    // Fixed sizes whose columns are packet-aligned take the vectorized rotation kernel.
    CALL_SUBTEST_11((inplace_selfadjoint_eigensolver<Matrix2d>(2)));
    CALL_SUBTEST_11((inplace_selfadjoint_eigensolver<Matrix3d>(3)));
    CALL_SUBTEST_11((inplace_selfadjoint_eigensolver<Matrix4f>(4)));
    CALL_SUBTEST_11((inplace_selfadjoint_eigensolver<Matrix4d>(4)));
    CALL_SUBTEST_11((inplace_selfadjoint_eigensolver<Matrix<float, Dynamic, Dynamic, RowMajor> >(size)));
    // Large enough for the blocked tridiagonalization, on strided and row-major storage.
    CALL_SUBTEST_11((inplace_selfadjoint_eigensolver<MatrixXd>(100)));
    CALL_SUBTEST_11((inplace_selfadjoint_eigensolver<Matrix<float, Dynamic, Dynamic, RowMajor> >(100)));

    CALL_SUBTEST_12((inplace_eigensolver<MatrixXd>(size)));
    CALL_SUBTEST_12((inplace_eigensolver<MatrixXd>(1)));
    CALL_SUBTEST_12((inplace_eigensolver<Matrix4f>(4)));
    CALL_SUBTEST_12((inplace_complex_eigensolver<MatrixXcd>(size)));
    CALL_SUBTEST_12((inplace_complex_eigensolver<MatrixXcd>(1)));
    CALL_SUBTEST_12((inplace_complex_eigensolver<Matrix4cf>(4)));

    CALL_SUBTEST_13((inplace_real_qz<MatrixXd>(size)));
    CALL_SUBTEST_13((inplace_real_qz<MatrixXd>(1)));
    CALL_SUBTEST_13((inplace_real_qz<Matrix4d>(4)));
    CALL_SUBTEST_13((inplace_complex_qz<MatrixXcd>(size)));
    CALL_SUBTEST_13((inplace_complex_qz<MatrixXcd>(1)));
  }
}
