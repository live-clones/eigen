// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <Eigen/QR>
#include "solverbase.h"

template<typename MatrixType, typename PermutationIndex> void qr()
{

  static const int Rows = MatrixType::RowsAtCompileTime, Cols = MatrixType::ColsAtCompileTime;
  Index max_size = EIGEN_TEST_MAX_SIZE;
  Index min_size = numext::maxi(1,EIGEN_TEST_MAX_SIZE/10);
  Index rows  = Rows == Dynamic ? internal::random<Index>(min_size,max_size) : Rows,
        cols  = Cols == Dynamic ? internal::random<Index>(min_size,max_size) : Cols,
        cols2 = Cols == Dynamic ? internal::random<Index>(min_size,max_size) : Cols,
        rank  = internal::random<Index>(1, (std::min)(rows, cols)-1);

  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime> MatrixQType;
  MatrixType m1;
  createRandomPIMatrixOfRank(rank,rows,cols,m1);
  FullPivHouseholderQR<MatrixType, PermutationIndex> qr(m1);
  VERIFY_IS_EQUAL(rank, qr.rank());
  VERIFY_IS_EQUAL(cols - qr.rank(), qr.dimensionOfKernel());
  VERIFY(!qr.isInjective());
  VERIFY(!qr.isInvertible());
  VERIFY(!qr.isSurjective());

  MatrixType r = qr.matrixQR();
  
  MatrixQType q = qr.matrixQ();
  VERIFY_IS_UNITARY(q);
  
  // FIXME need better way to construct trapezoid
  for(int i = 0; i < rows; i++) for(int j = 0; j < cols; j++) if(i>j) r(i,j) = Scalar(0);

  MatrixType c = qr.matrixQ() * r * qr.colsPermutation().inverse();

  VERIFY_IS_APPROX(m1, c);
  
  // stress the ReturnByValue mechanism
  MatrixType tmp;
  VERIFY_IS_APPROX(tmp.noalias() = qr.matrixQ() * r, (qr.matrixQ() * r).eval());
  
  check_solverbase<MatrixType, MatrixType>(m1, qr, rows, cols, cols2);

  {
    MatrixType m2, m3;
    Index size = rows;
    do {
      m1 = MatrixType::Random(size,size);
      qr.compute(m1);
    } while(!qr.isInvertible());
    MatrixType m1_inv = qr.inverse();
    m3 = m1 * MatrixType::Random(size,cols2);
    m2 = qr.solve(m3);
    VERIFY_IS_APPROX(m2, m1_inv*m3);
  }
}

template<typename MatrixType, typename PermutationIndex> void qr_invertible()
{
  using std::log;
  using std::abs;
  typedef typename NumTraits<typename MatrixType::Scalar>::Real RealScalar;
  typedef typename MatrixType::Scalar Scalar;

  Index max_size = numext::mini(50,EIGEN_TEST_MAX_SIZE);
  Index min_size = numext::maxi(1,EIGEN_TEST_MAX_SIZE/10);
  Index size = internal::random<Index>(min_size,max_size);

  MatrixType m1(size, size), m2(size, size), m3(size, size);
  m1 = MatrixType::Random(size,size);

  if (internal::is_same<RealScalar,float>::value)
  {
    // let's build a matrix more stable to inverse
    MatrixType a = MatrixType::Random(size,size*2);
    m1 += a * a.adjoint();
  }

  FullPivHouseholderQR<MatrixType, PermutationIndex> qr(m1);
  VERIFY(qr.isInjective());
  VERIFY(qr.isInvertible());
  VERIFY(qr.isSurjective());

  check_solverbase<MatrixType, MatrixType>(m1, qr, size, size, size);

  // now construct a matrix with prescribed determinant
  m1.setZero();
  for(int i = 0; i < size; i++) m1(i,i) = internal::random<Scalar>();
  Scalar det = m1.diagonal().prod();
  RealScalar absdet = abs(det);
  m3 = qr.matrixQ(); // get a unitary
  m1 = m3 * m1 * m3.adjoint();
  qr.compute(m1);
  VERIFY_IS_APPROX(det, qr.determinant());
  VERIFY_IS_APPROX(absdet, qr.absDeterminant());
  VERIFY_IS_APPROX(log(absdet), qr.logAbsDeterminant());
}

template<typename MatrixType, typename PermutationIndex> void qr_verify_assert()
{
  MatrixType tmp;

  FullPivHouseholderQR<MatrixType, PermutationIndex> qr;
  VERIFY_RAISES_ASSERT(qr.matrixQR())
  VERIFY_RAISES_ASSERT(qr.solve(tmp))
  VERIFY_RAISES_ASSERT(qr.transpose().solve(tmp))
  VERIFY_RAISES_ASSERT(qr.adjoint().solve(tmp))
  VERIFY_RAISES_ASSERT(qr.matrixQ())
  VERIFY_RAISES_ASSERT(qr.dimensionOfKernel())
  VERIFY_RAISES_ASSERT(qr.isInjective())
  VERIFY_RAISES_ASSERT(qr.isSurjective())
  VERIFY_RAISES_ASSERT(qr.isInvertible())
  VERIFY_RAISES_ASSERT(qr.inverse())
  VERIFY_RAISES_ASSERT(qr.determinant())
  VERIFY_RAISES_ASSERT(qr.absDeterminant())
  VERIFY_RAISES_ASSERT(qr.logAbsDeterminant())
}

EIGEN_DECLARE_TEST(qr_fullpivoting)
{
  typedef int PermutationIndex;

  for(int i = 0; i < 1; i++) {
    CALL_SUBTEST_5( (qr<Matrix3f, PermutationIndex>()) );
    CALL_SUBTEST_6( (qr<Matrix3d, PermutationIndex>()) );
    CALL_SUBTEST_8( (qr<Matrix2f, PermutationIndex>()) );
    CALL_SUBTEST_1( (qr<MatrixXf, PermutationIndex>()) );
    CALL_SUBTEST_2( (qr<MatrixXd, PermutationIndex>()) );
    CALL_SUBTEST_3( (qr<MatrixXcd, PermutationIndex>()) );
  }

  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( (qr_invertible<MatrixXf, PermutationIndex>()) );
    CALL_SUBTEST_2( (qr_invertible<MatrixXd, PermutationIndex>()) );
    CALL_SUBTEST_4( (qr_invertible<MatrixXcf, PermutationIndex>()) );
    CALL_SUBTEST_3( (qr_invertible<MatrixXcd, PermutationIndex>()) );
  }

  CALL_SUBTEST_5( (qr_verify_assert<Matrix3f, PermutationIndex>()) );
  CALL_SUBTEST_6( (qr_verify_assert<Matrix3d, PermutationIndex>()) );
  CALL_SUBTEST_1( (qr_verify_assert<MatrixXf, PermutationIndex>()) );
  CALL_SUBTEST_2( (qr_verify_assert<MatrixXd, PermutationIndex>()) );
  CALL_SUBTEST_4( (qr_verify_assert<MatrixXcf, PermutationIndex>()) );
  CALL_SUBTEST_3( (qr_verify_assert<MatrixXcd, PermutationIndex>()) );

  // Test problem size constructors
  CALL_SUBTEST_7( (FullPivHouseholderQR<MatrixXf,PermutationIndex>(10, 20)));
  CALL_SUBTEST_7( (FullPivHouseholderQR<Matrix<float, 10, 20>, PermutationIndex>(10, 20)));
  CALL_SUBTEST_7( (FullPivHouseholderQR<Matrix<float, 10, 20>, PermutationIndex>(Matrix<float,10,20>::Random())));
  CALL_SUBTEST_7( (FullPivHouseholderQR<Matrix<float, 20, 10>, PermutationIndex>(20, 10)));
  CALL_SUBTEST_7( (FullPivHouseholderQR<Matrix<float, 20, 10>, PermutationIndex>(Matrix<float,20,10>::Random())));
}
