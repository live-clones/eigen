// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen (rmlarsen@gmail.com)
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <Eigen/Dense>

template <typename MatrixType>
typename MatrixType::RealScalar matrix_l1_norm(const MatrixType& m) {
  return m.cwiseAbs().colwise().sum().maxCoeff();
}

template <typename MatrixType>
void rcond_partial_piv_lu() {
  typedef typename MatrixType::RealScalar RealScalar;
  Index size = MatrixType::RowsAtCompileTime;
  if (size == Dynamic) size = internal::random<Index>(2, EIGEN_TEST_MAX_SIZE);

  MatrixType m = MatrixType::Random(size, size);
  m.diagonal().array() += RealScalar(2 * size);

  PartialPivLU<MatrixType> lu(m);
  MatrixType m_inverse = lu.inverse();
  RealScalar rcond = (RealScalar(1) / matrix_l1_norm(m)) / matrix_l1_norm(m_inverse);
  RealScalar rcond_est = lu.rcond();
  VERIFY(rcond_est > rcond / 10 && rcond_est < rcond * 10);
}

template <typename MatrixType>
void rcond_full_piv_lu() {
  typedef typename MatrixType::RealScalar RealScalar;
  Index size = MatrixType::RowsAtCompileTime;
  if (size == Dynamic) size = internal::random<Index>(2, EIGEN_TEST_MAX_SIZE);

  MatrixType m = MatrixType::Random(size, size);
  m.diagonal().array() += RealScalar(2 * size);

  FullPivLU<MatrixType> lu(m);
  MatrixType m_inverse = lu.inverse();
  RealScalar rcond = (RealScalar(1) / matrix_l1_norm(m)) / matrix_l1_norm(m_inverse);
  RealScalar rcond_est = lu.rcond();
  VERIFY(rcond_est > rcond / 10 && rcond_est < rcond * 10);
}

template <typename MatrixType>
void rcond_llt() {
  typedef typename MatrixType::RealScalar RealScalar;
  Index size = MatrixType::RowsAtCompileTime;
  if (size == Dynamic) size = internal::random<Index>(2, EIGEN_TEST_MAX_SIZE);

  MatrixType a = MatrixType::Random(size, size);
  MatrixType m = a.adjoint() * a + MatrixType::Identity(size, size);

  LLT<MatrixType> llt(m);
  VERIFY(llt.info() == Success);
  MatrixType m_inverse = llt.solve(MatrixType::Identity(size, size));
  RealScalar rcond = (RealScalar(1) / matrix_l1_norm(m)) / matrix_l1_norm(m_inverse);
  RealScalar rcond_est = llt.rcond();
  VERIFY(rcond_est > rcond / 10 && rcond_est < rcond * 10);
}

template <typename MatrixType>
void rcond_ldlt() {
  typedef typename MatrixType::RealScalar RealScalar;
  Index size = MatrixType::RowsAtCompileTime;
  if (size == Dynamic) size = internal::random<Index>(2, EIGEN_TEST_MAX_SIZE);

  MatrixType a = MatrixType::Random(size, size);
  MatrixType m = a.adjoint() * a + MatrixType::Identity(size, size);

  LDLT<MatrixType> ldlt(m);
  VERIFY(ldlt.info() == Success);
  MatrixType m_inverse = ldlt.solve(MatrixType::Identity(size, size));
  RealScalar rcond = (RealScalar(1) / matrix_l1_norm(m)) / matrix_l1_norm(m_inverse);
  RealScalar rcond_est = ldlt.rcond();
  VERIFY(rcond_est > rcond / 10 && rcond_est < rcond * 10);
}

template <typename MatrixType>
void rcond_singular() {
  typedef typename MatrixType::Scalar Scalar;
  Index size = MatrixType::RowsAtCompileTime;
  if (size == Dynamic) size = internal::random<Index>(2, EIGEN_TEST_MAX_SIZE);

  MatrixType m = MatrixType::Random(size, size);
  m.row(0).setZero();

  FullPivLU<MatrixType> lu(m);
  VERIFY_IS_EQUAL(lu.rcond(), Scalar(0));
}

template <typename MatrixType>
void rcond_identity() {
  typedef typename MatrixType::RealScalar RealScalar;
  Index size = MatrixType::RowsAtCompileTime;
  if (size == Dynamic) size = internal::random<Index>(2, EIGEN_TEST_MAX_SIZE);

  MatrixType m = MatrixType::Identity(size, size);

  {
    PartialPivLU<MatrixType> lu(m);
    VERIFY(lu.rcond() > RealScalar(0.5));
  }
  {
    FullPivLU<MatrixType> lu(m);
    VERIFY(lu.rcond() > RealScalar(0.5));
  }
  {
    LLT<MatrixType> llt(m);
    VERIFY(llt.rcond() > RealScalar(0.5));
  }
  {
    LDLT<MatrixType> ldlt(m);
    VERIFY(ldlt.rcond() > RealScalar(0.5));
  }
}

template <typename MatrixType>
void rcond_ill_conditioned() {
  typedef typename MatrixType::RealScalar RealScalar;
  Index size = MatrixType::RowsAtCompileTime;
  if (size == Dynamic) size = internal::random<Index>(4, EIGEN_TEST_MAX_SIZE);

  MatrixType m = MatrixType::Zero(size, size);
  m(0, 0) = RealScalar(1);
  for (Index i = 1; i < size; ++i) {
    m(i, i) = RealScalar(1e-3);
  }

  {
    PartialPivLU<MatrixType> lu(m);
    RealScalar rcond_est = lu.rcond();
    VERIFY(rcond_est < RealScalar(1e-1));
    VERIFY(rcond_est > RealScalar(1e-5));
  }
  {
    FullPivLU<MatrixType> lu(m);
    RealScalar rcond_est = lu.rcond();
    VERIFY(rcond_est < RealScalar(1e-1));
    VERIFY(rcond_est > RealScalar(1e-5));
  }
}

template <typename MatrixType>
void rcond_1x1() {
  typedef typename MatrixType::RealScalar RealScalar;
  typedef Matrix<typename MatrixType::Scalar, 1, 1> Mat1;
  Mat1 m;
  m(0, 0) = internal::random<RealScalar>(RealScalar(1), RealScalar(100));

  {
    PartialPivLU<Mat1> lu(m);
    VERIFY_IS_APPROX(lu.rcond(), RealScalar(1));
  }
  {
    FullPivLU<Mat1> lu(m);
    VERIFY_IS_APPROX(lu.rcond(), RealScalar(1));
  }
  {
    LLT<Mat1> llt(m);
    VERIFY_IS_APPROX(llt.rcond(), RealScalar(1));
  }
  {
    LDLT<Mat1> ldlt(m);
    VERIFY_IS_APPROX(ldlt.rcond(), RealScalar(1));
  }
}

template <typename MatrixType>
void rcond_2x2() {
  typedef typename MatrixType::RealScalar RealScalar;
  typedef Matrix<typename MatrixType::Scalar, 2, 2> Mat2;

  Mat2 m;
  m << RealScalar(2), RealScalar(1), RealScalar(1), RealScalar(3);

  {
    PartialPivLU<Mat2> lu(m);
    Mat2 m_inverse = lu.inverse();
    RealScalar rcond = (RealScalar(1) / matrix_l1_norm(m)) / matrix_l1_norm(m_inverse);
    RealScalar rcond_est = lu.rcond();
    VERIFY(rcond_est > rcond / 10 && rcond_est < rcond * 10);
  }
  {
    FullPivLU<Mat2> lu(m);
    Mat2 m_inverse = lu.inverse();
    RealScalar rcond = (RealScalar(1) / matrix_l1_norm(m)) / matrix_l1_norm(m_inverse);
    RealScalar rcond_est = lu.rcond();
    VERIFY(rcond_est > rcond / 10 && rcond_est < rcond * 10);
  }
  {
    LLT<Mat2> llt(m);
    Mat2 m_inverse = llt.solve(Mat2::Identity());
    RealScalar rcond = (RealScalar(1) / matrix_l1_norm(m)) / matrix_l1_norm(m_inverse);
    RealScalar rcond_est = llt.rcond();
    VERIFY(rcond_est > rcond / 10 && rcond_est < rcond * 10);
  }
}

// =============================================================================
// Fixed-size tests (float and double)
// =============================================================================
template <typename T>
class ConditionEstimatorFixedTest : public ::testing::Test {};

using ConditionEstimatorFixedTypes = ::testing::Types<Matrix3f, Matrix4d>;
EIGEN_TYPED_TEST_SUITE(ConditionEstimatorFixedTest, ConditionEstimatorFixedTypes);

TYPED_TEST(ConditionEstimatorFixedTest, PartialPivLU) {
  for (int i = 0; i < g_repeat; i++) rcond_partial_piv_lu<TypeParam>();
}
TYPED_TEST(ConditionEstimatorFixedTest, FullPivLU) {
  for (int i = 0; i < g_repeat; i++) rcond_full_piv_lu<TypeParam>();
}
TYPED_TEST(ConditionEstimatorFixedTest, LLT) {
  for (int i = 0; i < g_repeat; i++) rcond_llt<TypeParam>();
}
TYPED_TEST(ConditionEstimatorFixedTest, LDLT) {
  for (int i = 0; i < g_repeat; i++) rcond_ldlt<TypeParam>();
}
TYPED_TEST(ConditionEstimatorFixedTest, Singular) {
  for (int i = 0; i < g_repeat; i++) rcond_singular<TypeParam>();
}
TYPED_TEST(ConditionEstimatorFixedTest, Identity) {
  for (int i = 0; i < g_repeat; i++) rcond_identity<TypeParam>();
}
TYPED_TEST(ConditionEstimatorFixedTest, OneByOne) {
  for (int i = 0; i < g_repeat; i++) rcond_1x1<TypeParam>();
}
TYPED_TEST(ConditionEstimatorFixedTest, TwoByTwo) {
  for (int i = 0; i < g_repeat; i++) rcond_2x2<TypeParam>();
}

// =============================================================================
// Dynamic-size real tests (float and double)
// =============================================================================
template <typename T>
class ConditionEstimatorDynamicRealTest : public ::testing::Test {};

using ConditionEstimatorDynamicRealTypes = ::testing::Types<MatrixXf, MatrixXd>;
EIGEN_TYPED_TEST_SUITE(ConditionEstimatorDynamicRealTest, ConditionEstimatorDynamicRealTypes);

TYPED_TEST(ConditionEstimatorDynamicRealTest, PartialPivLU) {
  for (int i = 0; i < g_repeat; i++) rcond_partial_piv_lu<TypeParam>();
}
TYPED_TEST(ConditionEstimatorDynamicRealTest, FullPivLU) {
  for (int i = 0; i < g_repeat; i++) rcond_full_piv_lu<TypeParam>();
}
TYPED_TEST(ConditionEstimatorDynamicRealTest, LLT) {
  for (int i = 0; i < g_repeat; i++) rcond_llt<TypeParam>();
}
TYPED_TEST(ConditionEstimatorDynamicRealTest, LDLT) {
  for (int i = 0; i < g_repeat; i++) rcond_ldlt<TypeParam>();
}
TYPED_TEST(ConditionEstimatorDynamicRealTest, Singular) {
  for (int i = 0; i < g_repeat; i++) rcond_singular<TypeParam>();
}
TYPED_TEST(ConditionEstimatorDynamicRealTest, Identity) {
  for (int i = 0; i < g_repeat; i++) rcond_identity<TypeParam>();
}
TYPED_TEST(ConditionEstimatorDynamicRealTest, IllConditioned) {
  for (int i = 0; i < g_repeat; i++) rcond_ill_conditioned<TypeParam>();
}

// =============================================================================
// Dynamic-size complex tests
// =============================================================================
template <typename T>
class ConditionEstimatorDynamicComplexTest : public ::testing::Test {};

using ConditionEstimatorDynamicComplexTypes = ::testing::Types<MatrixXcf, MatrixXcd>;
EIGEN_TYPED_TEST_SUITE(ConditionEstimatorDynamicComplexTest, ConditionEstimatorDynamicComplexTypes);

TYPED_TEST(ConditionEstimatorDynamicComplexTest, PartialPivLU) {
  for (int i = 0; i < g_repeat; i++) rcond_partial_piv_lu<TypeParam>();
}
TYPED_TEST(ConditionEstimatorDynamicComplexTest, FullPivLU) {
  for (int i = 0; i < g_repeat; i++) rcond_full_piv_lu<TypeParam>();
}
TYPED_TEST(ConditionEstimatorDynamicComplexTest, LLT) {
  for (int i = 0; i < g_repeat; i++) rcond_llt<TypeParam>();
}
TYPED_TEST(ConditionEstimatorDynamicComplexTest, LDLT) {
  for (int i = 0; i < g_repeat; i++) rcond_ldlt<TypeParam>();
}
TYPED_TEST(ConditionEstimatorDynamicComplexTest, Singular) {
  for (int i = 0; i < g_repeat; i++) rcond_singular<TypeParam>();
}
TYPED_TEST(ConditionEstimatorDynamicComplexTest, Identity) {
  for (int i = 0; i < g_repeat; i++) rcond_identity<TypeParam>();
}
