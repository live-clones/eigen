// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// lu split: real scalar types (float and double).

#include "lu_helpers.h"

// =============================================================================
// Typed test suite for lu_non_invertible (types shared with other helpers)
// =============================================================================
template <typename T>
class LURealNonInvertibleTest : public ::testing::Test {};

using LURealNonInvertibleTypes =
    ::testing::Types<Matrix3f, Matrix<double, 4, 6>, MatrixXf, MatrixXd, Matrix<float, Dynamic, 16>>;
TYPED_TEST_SUITE(LURealNonInvertibleTest, LURealNonInvertibleTypes);

TYPED_TEST(LURealNonInvertibleTest, NonInvertible) {
  for (int i = 0; i < g_repeat; i++) {
    lu_non_invertible<TypeParam>();
  }
}

// =============================================================================
// Typed test suite for lu_invertible
// =============================================================================
template <typename T>
class LURealInvertibleTest : public ::testing::Test {};

using LURealInvertibleTypes = ::testing::Types<Matrix3f, MatrixXf, MatrixXd>;
TYPED_TEST_SUITE(LURealInvertibleTest, LURealInvertibleTypes);

TYPED_TEST(LURealInvertibleTest, Invertible) {
  for (int i = 0; i < g_repeat; i++) {
    lu_invertible<TypeParam>();
  }
}

// =============================================================================
// Typed test suite for lu_verify_assert
// =============================================================================
template <typename T>
class LURealVerifyAssertTest : public ::testing::Test {};

using LURealVerifyAssertTypes = ::testing::Types<Matrix3f, Matrix<double, 4, 6>, MatrixXf, MatrixXd>;
TYPED_TEST_SUITE(LURealVerifyAssertTest, LURealVerifyAssertTypes);

TYPED_TEST(LURealVerifyAssertTest, VerifyAssert) { lu_verify_assert<TypeParam>(); }

// =============================================================================
// Typed test suite for lu_partial_piv (fixed-size square types)
// =============================================================================
template <typename T>
class LURealPartialPivFixedTest : public ::testing::Test {};

using LURealPartialPivFixedTypes = ::testing::Types<Matrix3f, Matrix2d, Matrix4d, Matrix<double, 6, 6>>;
TYPED_TEST_SUITE(LURealPartialPivFixedTest, LURealPartialPivFixedTypes);

TYPED_TEST(LURealPartialPivFixedTest, PartialPiv) {
  for (int i = 0; i < g_repeat; i++) {
    lu_partial_piv<TypeParam>();
  }
}

// =============================================================================
// Dynamic-size partial pivot test (MatrixXd with random size in [1, MAX_SIZE])
// =============================================================================
TEST(LURealTest, PartialPivDynamic) {
  for (int i = 0; i < g_repeat; i++) {
    lu_partial_piv<MatrixXd>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE));
  }
}

// =============================================================================
// Problem size constructors
// =============================================================================
TEST(LURealTest, ProblemSizeConstructors) {
  PartialPivLU<MatrixXf>(10);
  FullPivLU<MatrixXf>(10, 20);
}

// =============================================================================
// Regression test for Bug 2889
// =============================================================================
TEST(LURealRegressionTest, Bug2889) {
  for (int i = 0; i < g_repeat; i++) {
    test_2889();
  }
}
