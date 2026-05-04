// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen (rmlarsen@gmail.com)
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// condition_estimator split: real scalar types. Complex types in condition_estimator_complex.cpp.

#include "condition_estimator.h"

template <typename T>
class ConditionEstimatorTest : public ::testing::Test {};

using ConditionEstimatorTypes = ::testing::Types<Matrix3f, Matrix4d, MatrixXf, MatrixXd>;
EIGEN_TYPED_TEST_SUITE(ConditionEstimatorTest, ConditionEstimatorTypes);

TYPED_TEST(ConditionEstimatorTest, PartialPivLU) {
  for (int i = 0; i < g_repeat; i++) {
    rcond_partial_piv_lu<TypeParam>();
  }
}

TYPED_TEST(ConditionEstimatorTest, FullPivLU) {
  for (int i = 0; i < g_repeat; i++) {
    rcond_full_piv_lu<TypeParam>();
  }
}

TYPED_TEST(ConditionEstimatorTest, LLT) {
  for (int i = 0; i < g_repeat; i++) {
    rcond_llt<TypeParam>();
  }
}

TYPED_TEST(ConditionEstimatorTest, LDLT) {
  for (int i = 0; i < g_repeat; i++) {
    rcond_ldlt<TypeParam>();
  }
}

TYPED_TEST(ConditionEstimatorTest, Singular) {
  for (int i = 0; i < g_repeat; i++) {
    rcond_singular<TypeParam>();
  }
}

TYPED_TEST(ConditionEstimatorTest, Identity) {
  for (int i = 0; i < g_repeat; i++) {
    rcond_identity<TypeParam>();
  }
}

TEST(ConditionEstimatorTest, IllConditioned) {
  for (int i = 0; i < g_repeat; i++) {
    rcond_ill_conditioned<MatrixXf>();
    rcond_ill_conditioned<MatrixXd>();
  }
}

TEST(ConditionEstimatorTest, SmallFixed) {
  for (int i = 0; i < g_repeat; i++) {
    rcond_1x1<Matrix3f>();
    rcond_2x2<Matrix3f>();
    rcond_2x2<Matrix4d>();
  }
}
