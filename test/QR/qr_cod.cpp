// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// CompleteOrthogonalDecomposition tests, split from qr_colpivoting.

#include "main.h"
#include <Eigen/QR>
#include <Eigen/SVD>
#include "solverbase.h"

template <typename MatrixType>
void cod() {
  Index rows = internal::random<Index>(2, EIGEN_TEST_MAX_SIZE);
  Index cols = internal::random<Index>(2, EIGEN_TEST_MAX_SIZE);
  Index cols2 = internal::random<Index>(2, EIGEN_TEST_MAX_SIZE);
  Index rank = internal::random<Index>(1, (std::min)(rows, cols) - 1);

  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime> MatrixQType;
  MatrixType matrix;
  createRandomPIMatrixOfRank(rank, rows, cols, matrix);
  CompleteOrthogonalDecomposition<MatrixType> cod(matrix);
  VERIFY(rank == cod.rank());
  VERIFY(cols - cod.rank() == cod.dimensionOfKernel());
  VERIFY(!cod.isInjective());
  VERIFY(!cod.isInvertible());
  VERIFY(!cod.isSurjective());

  MatrixQType q = cod.householderQ();
  VERIFY_IS_UNITARY(q);

  MatrixType z = cod.matrixZ();
  VERIFY_IS_UNITARY(z);

  MatrixType t;
  t.setZero(rows, cols);
  t.topLeftCorner(rank, rank) = cod.matrixT().topLeftCorner(rank, rank).template triangularView<Upper>();

  MatrixType c = q * t * z * cod.colsPermutation().inverse();
  VERIFY_IS_APPROX(matrix, c);

  check_solverbase<MatrixType, MatrixType>(matrix, cod, rows, cols, cols2);

  // Verify that we get the same minimum-norm solution as the SVD.
  MatrixType exact_solution = MatrixType::Random(cols, cols2);
  MatrixType rhs = matrix * exact_solution;
  MatrixType cod_solution = cod.solve(rhs);
  JacobiSVD<MatrixType, ComputeThinU | ComputeThinV> svd(matrix);
  MatrixType svd_solution = svd.solve(rhs);
  VERIFY_IS_APPROX(cod_solution, svd_solution);

  MatrixType pinv = cod.pseudoInverse();
  VERIFY_IS_APPROX(cod_solution, pinv * rhs);

  // now construct a (square) matrix with prescribed determinant
  Index size = internal::random<Index>(2, 20);
  matrix.setZero(size, size);
  for (int i = 0; i < size; i++) {
    matrix(i, i) = internal::random<Scalar>();
  }
  Scalar det = matrix.diagonal().prod();
  RealScalar absdet = numext::abs(det);
  CompleteOrthogonalDecomposition<MatrixType> cod2(matrix);
  cod2.compute(matrix);
  q = cod2.householderQ();
  matrix = q * matrix * q.adjoint();
  VERIFY_IS_APPROX(det, cod2.determinant());
  VERIFY_IS_APPROX(absdet, cod2.absDeterminant());
  VERIFY_IS_APPROX(numext::log(absdet), cod2.logAbsDeterminant());
  VERIFY_IS_APPROX(numext::sign(det), cod2.signDeterminant());
}

template <typename MatrixType, int Cols2>
void cod_fixedsize() {
  enum { Rows = MatrixType::RowsAtCompileTime, Cols = MatrixType::ColsAtCompileTime };
  typedef typename MatrixType::Scalar Scalar;
  typedef CompleteOrthogonalDecomposition<Matrix<Scalar, Rows, Cols> > COD;
  int rank = internal::random<int>(1, (std::min)(int(Rows), int(Cols)) - 1);
  Matrix<Scalar, Rows, Cols> matrix;
  createRandomPIMatrixOfRank(rank, Rows, Cols, matrix);
  COD cod(matrix);
  VERIFY(rank == cod.rank());
  VERIFY(Cols - cod.rank() == cod.dimensionOfKernel());
  VERIFY(cod.isInjective() == (rank == Rows));
  VERIFY(cod.isSurjective() == (rank == Cols));
  VERIFY(cod.isInvertible() == (cod.isInjective() && cod.isSurjective()));

  check_solverbase<Matrix<Scalar, Cols, Cols2>, Matrix<Scalar, Rows, Cols2> >(matrix, cod, Rows, Cols, Cols2);

  // Verify that we get the same minimum-norm solution as the SVD.
  Matrix<Scalar, Cols, Cols2> exact_solution;
  exact_solution.setRandom(Cols, Cols2);
  Matrix<Scalar, Rows, Cols2> rhs = matrix * exact_solution;
  Matrix<Scalar, Cols, Cols2> cod_solution = cod.solve(rhs);
  JacobiSVD<MatrixType, ComputeFullU | ComputeFullV> svd(matrix);
  Matrix<Scalar, Cols, Cols2> svd_solution = svd.solve(rhs);
  VERIFY_IS_APPROX(cod_solution, svd_solution);

  typename Inverse<COD>::PlainObject pinv = cod.pseudoInverse();
  VERIFY_IS_APPROX(cod_solution, pinv * rhs);
}

template <typename MatrixType>
void cod_verify_assert() {
  MatrixType tmp;

  CompleteOrthogonalDecomposition<MatrixType> cod;
  VERIFY_RAISES_ASSERT(cod.matrixQTZ())
  VERIFY_RAISES_ASSERT(cod.solve(tmp))
  VERIFY_RAISES_ASSERT(cod.transpose().solve(tmp))
  VERIFY_RAISES_ASSERT(cod.adjoint().solve(tmp))
  VERIFY_RAISES_ASSERT(cod.householderQ())
  VERIFY_RAISES_ASSERT(cod.dimensionOfKernel())
  VERIFY_RAISES_ASSERT(cod.isInjective())
  VERIFY_RAISES_ASSERT(cod.isSurjective())
  VERIFY_RAISES_ASSERT(cod.isInvertible())
  VERIFY_RAISES_ASSERT(cod.pseudoInverse())
  VERIFY_RAISES_ASSERT(cod.determinant())
  VERIFY_RAISES_ASSERT(cod.absDeterminant())
  VERIFY_RAISES_ASSERT(cod.logAbsDeterminant())
  VERIFY_RAISES_ASSERT(cod.signDeterminant())
}

// =============================================================================
// Typed test suite for cod (dynamic types)
// =============================================================================
template <typename T>
class CODTest : public ::testing::Test {};

using CODTypes = ::testing::Types<MatrixXf, MatrixXd, MatrixXcd>;
TYPED_TEST_SUITE(CODTest, CODTypes);

TYPED_TEST(CODTest, COD) {
  for (int i = 0; i < g_repeat; i++) {
    cod<TypeParam>();
  }
}

// =============================================================================
// Fixed-size tests
// =============================================================================
TEST(CODTest, FixedSize) {
  for (int i = 0; i < g_repeat; i++) {
    cod_fixedsize<Matrix<float, 3, 5>, 4>();
    cod_fixedsize<Matrix<double, 6, 2>, 3>();
    cod_fixedsize<Matrix<double, 1, 1>, 1>();
  }
}

// =============================================================================
// Verify assert tests
// =============================================================================
TEST(CODTest, VerifyAssert) {
  cod_verify_assert<Matrix3f>();
  cod_verify_assert<Matrix3d>();
  cod_verify_assert<MatrixXf>();
  cod_verify_assert<MatrixXd>();
  cod_verify_assert<MatrixXcf>();
  cod_verify_assert<MatrixXcd>();
}
