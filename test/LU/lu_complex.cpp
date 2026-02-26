// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// lu split: complex scalar types.

#include "lu_helpers.h"

// =============================================================================
// Typed test suite for complex LU tests (non_invertible, invertible, verify_assert)
// =============================================================================
template <typename T>
class LUComplexTest : public ::testing::Test {};

using LUComplexTypes = ::testing::Types<MatrixXcf, MatrixXcd>;
EIGEN_TYPED_TEST_SUITE(LUComplexTest, LUComplexTypes);

TYPED_TEST(LUComplexTest, NonInvertible) {
  for (int i = 0; i < g_repeat; i++) {
    lu_non_invertible<TypeParam>();
  }
}

TYPED_TEST(LUComplexTest, Invertible) {
  for (int i = 0; i < g_repeat; i++) {
    lu_invertible<TypeParam>();
  }
}

TYPED_TEST(LUComplexTest, VerifyAssert) { lu_verify_assert<TypeParam>(); }

// =============================================================================
// Dynamic-size partial pivot test (MatrixXcd with random size in [1, MAX_SIZE])
// =============================================================================
TEST(LUComplexPartialPivTest, DynamicXcd) {
  for (int i = 0; i < g_repeat; i++) {
    lu_partial_piv<MatrixXcd>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE));
  }
}
