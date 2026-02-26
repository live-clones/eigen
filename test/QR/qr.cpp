// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <Eigen/QR>
#include "solverbase.h"

template <typename MatrixType>
void qr(const MatrixType& m) {
  Index rows = m.rows();
  Index cols = m.cols();

  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime> MatrixQType;

  MatrixType a = MatrixType::Random(rows, cols);
  HouseholderQR<MatrixType> qrOfA(a);

  MatrixQType q = qrOfA.householderQ();
  VERIFY_IS_UNITARY(q);

  MatrixType r = qrOfA.matrixQR().template triangularView<Upper>();
  VERIFY_IS_APPROX(a, qrOfA.householderQ() * r);
}

template <typename MatrixType, int Cols2>
void qr_fixedsize() {
  enum { Rows = MatrixType::RowsAtCompileTime, Cols = MatrixType::ColsAtCompileTime };
  typedef typename MatrixType::Scalar Scalar;
  Matrix<Scalar, Rows, Cols> m1 = Matrix<Scalar, Rows, Cols>::Random();
  HouseholderQR<Matrix<Scalar, Rows, Cols>> qr(m1);

  Matrix<Scalar, Rows, Cols> r = qr.matrixQR();
  // FIXME need better way to construct trapezoid
  for (int i = 0; i < Rows; i++)
    for (int j = 0; j < Cols; j++)
      if (i > j) r(i, j) = Scalar(0);

  VERIFY_IS_APPROX(m1, qr.householderQ() * r);

  check_solverbase<Matrix<Scalar, Cols, Cols2>, Matrix<Scalar, Rows, Cols2>>(m1, qr, Rows, Cols, Cols2);
}

template <typename MatrixType>
void qr_invertible() {
  using std::abs;
  using std::log;
  using std::max;
  using std::pow;
  typedef typename NumTraits<typename MatrixType::Scalar>::Real RealScalar;
  typedef typename MatrixType::Scalar Scalar;

  STATIC_CHECK((internal::is_same<typename HouseholderQR<MatrixType>::StorageIndex, int>::value));

  int size = internal::random<int>(10, 50);

  MatrixType m1(size, size), m2(size, size), m3(size, size);
  m1 = MatrixType::Random(size, size);

  if (internal::is_same<RealScalar, float>::value) {
    // let's build a matrix more stable to inverse
    MatrixType a = MatrixType::Random(size, size * 4);
    m1 += a * a.adjoint();
  }

  HouseholderQR<MatrixType> qr(m1);

  check_solverbase<MatrixType, MatrixType>(m1, qr, size, size, size);

  // now construct a matrix with prescribed determinant
  m1.setZero();
  for (int i = 0; i < size; i++) m1(i, i) = internal::random<Scalar>();
  Scalar det = m1.diagonal().prod();
  RealScalar absdet = abs(det);
  m3 = qr.householderQ();  // get a unitary
  m1 = m3 * m1 * m3.adjoint();
  qr.compute(m1);
  VERIFY_IS_APPROX(log(absdet), qr.logAbsDeterminant());
  VERIFY_IS_APPROX(numext::sign(det), qr.signDeterminant());
  // This test is tricky if the determinant becomes too small.
  // Since we generate random numbers with magnitude range [0,1], the average determinant is 0.5^size
  RealScalar tol =
      numext::maxi(RealScalar(pow(0.5, size)), numext::maxi<RealScalar>(abs(absdet), abs(qr.absDeterminant())));
  VERIFY_IS_MUCH_SMALLER_THAN(abs(det - qr.determinant()), tol);
  VERIFY_IS_MUCH_SMALLER_THAN(abs(absdet - qr.absDeterminant()), tol);
}

template <typename MatrixType>
void qr_verify_assert() {
  MatrixType tmp;

  HouseholderQR<MatrixType> qr;
  VERIFY_RAISES_ASSERT(qr.matrixQR())
  VERIFY_RAISES_ASSERT(qr.solve(tmp))
  VERIFY_RAISES_ASSERT(qr.transpose().solve(tmp))
  VERIFY_RAISES_ASSERT(qr.adjoint().solve(tmp))
  VERIFY_RAISES_ASSERT(qr.householderQ())
  VERIFY_RAISES_ASSERT(qr.determinant())
  VERIFY_RAISES_ASSERT(qr.absDeterminant())
  VERIFY_RAISES_ASSERT(qr.signDeterminant())
}

// =============================================================================
// Config struct + typed test suite for qr
// =============================================================================
template <typename MatrixType_, int MaxRows_, int MaxCols_>
struct QRConfig {
  using MatrixType = MatrixType_;
  static constexpr int MaxRows = MaxRows_;
  static constexpr int MaxCols = MaxCols_;
};

template <typename T>
class QRTest : public ::testing::Test {};

using QRTypes = ::testing::Types<QRConfig<MatrixXf, EIGEN_TEST_MAX_SIZE, EIGEN_TEST_MAX_SIZE>,
                                 QRConfig<MatrixXcd, EIGEN_TEST_MAX_SIZE / 2, EIGEN_TEST_MAX_SIZE / 2>,
                                 QRConfig<Matrix<float, 1, 1>, 1, 1>>;
TYPED_TEST_SUITE(QRTest, QRTypes);

TYPED_TEST(QRTest, QR) {
  using MatrixType = typename TypeParam::MatrixType;
  constexpr int MaxRows = TypeParam::MaxRows;
  constexpr int MaxCols = TypeParam::MaxCols;
  for (int i = 0; i < g_repeat; i++) {
    int rows =
        MatrixType::RowsAtCompileTime == Dynamic ? internal::random<int>(1, MaxRows) : MatrixType::RowsAtCompileTime;
    int cols =
        MatrixType::ColsAtCompileTime == Dynamic ? internal::random<int>(1, MaxCols) : MatrixType::ColsAtCompileTime;
    qr(MatrixType(rows, cols));
  }
}

// =============================================================================
// Typed test suite for qr_invertible
// =============================================================================
template <typename T>
class QRInvertibleTest : public ::testing::Test {};

using QRInvertibleTypes = ::testing::Types<MatrixXf, MatrixXd, MatrixXcf, MatrixXcd>;
TYPED_TEST_SUITE(QRInvertibleTest, QRInvertibleTypes);

TYPED_TEST(QRInvertibleTest, Invertible) {
  for (int i = 0; i < g_repeat; i++) {
    qr_invertible<TypeParam>();
  }
}

// =============================================================================
// Fixed-size tests
// =============================================================================
TEST(QRTest, FixedSize) {
  for (int i = 0; i < g_repeat; i++) {
    qr_fixedsize<Matrix<float, 3, 4>, 2>();
    qr_fixedsize<Matrix<double, 6, 2>, 4>();
    qr_fixedsize<Matrix<double, 2, 5>, 7>();
  }
}

// =============================================================================
// Verify assert tests
// =============================================================================
TEST(QRTest, VerifyAssert) {
  qr_verify_assert<Matrix3f>();
  qr_verify_assert<Matrix3d>();
  qr_verify_assert<MatrixXf>();
  qr_verify_assert<MatrixXd>();
  qr_verify_assert<MatrixXcf>();
  qr_verify_assert<MatrixXcd>();
}

// =============================================================================
// Regression / misc tests
// =============================================================================
TEST(QRTest, ProblemSizeConstructor) {
  // Test problem size constructors
  HouseholderQR<MatrixXf>(10, 20);
}
