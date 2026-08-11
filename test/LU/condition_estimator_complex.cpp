// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen (rmlarsen@gmail.com)
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// condition_estimator split: complex scalar types. Real types in condition_estimator.cpp.

#include "condition_estimator.h"

template <typename T>
class ConditionEstimatorComplexTest : public ::testing::Test {};

using ConditionEstimatorComplexTypes = ::testing::Types<MatrixXcf, MatrixXcd>;
EIGEN_TYPED_TEST_SUITE(ConditionEstimatorComplexTest, ConditionEstimatorComplexTypes);

TYPED_TEST(ConditionEstimatorComplexTest, PartialPivLU) {
  for (int i = 0; i < g_repeat; i++) {
    rcond_partial_piv_lu<TypeParam>();
  }
}

TYPED_TEST(ConditionEstimatorComplexTest, FullPivLU) {
  for (int i = 0; i < g_repeat; i++) {
    rcond_full_piv_lu<TypeParam>();
  }
}

TYPED_TEST(ConditionEstimatorComplexTest, LLT) {
  for (int i = 0; i < g_repeat; i++) {
    rcond_llt<TypeParam>();
  }
}

TYPED_TEST(ConditionEstimatorComplexTest, LDLT) {
  for (int i = 0; i < g_repeat; i++) {
    rcond_ldlt<TypeParam>();
  }
}

TYPED_TEST(ConditionEstimatorComplexTest, Singular) {
  for (int i = 0; i < g_repeat; i++) {
    rcond_singular<TypeParam>();
  }
}

TYPED_TEST(ConditionEstimatorComplexTest, Identity) {
  for (int i = 0; i < g_repeat; i++) {
    rcond_identity<TypeParam>();
  }
}
