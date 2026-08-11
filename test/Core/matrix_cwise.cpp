// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "matrix_cwise.h"

TEST(MatrixCwiseTest, RealFloat) {
  for (int i = 0; i < g_repeat; i++) {
    test_cwise_real(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>(20, 20));
    test_cwise_real(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>(20, 20));
    test_cwise_real(Eigen::Matrix<Eigen::half, Eigen::Dynamic, Eigen::Dynamic>(20, 20));
    test_cwise_real(Eigen::Matrix<Eigen::bfloat16, Eigen::Dynamic, Eigen::Dynamic>(20, 20));
  }
}

void test_cwise_pointer_result() {
  pointer_result_payload p0 = {0.0};
  pointer_result_payload p1 = {1.0};
  pointer_result_payload p2 = {2.0};
  pointer_result_payload p3 = {3.0};

  Matrix<pointer_result_holder, 2, 2> holders;
  holders(0, 0).ptr = &p0;
  holders(1, 0).ptr = &p1;
  holders(0, 1).ptr = &p2;
  holders(1, 1).ptr = &p3;

  const auto unary = holders.unaryExpr(unary_pointer_result_op());
  VERIFY((std::is_same<typename Eigen::internal::traits<decltype(unary)>::Scalar, pointer_result_payload*>::value));
  Matrix<pointer_result_payload*, 2, 2> unary_out = unary;
  VERIFY_IS_EQUAL(unary_out(0, 0), &p0);
  VERIFY_IS_EQUAL(unary_out(1, 0), &p1);
  VERIFY_IS_EQUAL(unary_out(0, 1), &p2);
  VERIFY_IS_EQUAL(unary_out(1, 1), &p3);

  const auto binary = holders.binaryExpr(holders, binary_pointer_result_op());
  VERIFY((std::is_same<typename Eigen::internal::traits<decltype(binary)>::Scalar, pointer_result_payload*>::value));
  Matrix<pointer_result_payload*, 2, 2> binary_out = binary;
  VERIFY_IS_EQUAL(binary_out(0, 0), &p0);
  VERIFY_IS_EQUAL(binary_out(1, 0), &p1);
  VERIFY_IS_EQUAL(binary_out(0, 1), &p2);
  VERIFY_IS_EQUAL(binary_out(1, 1), &p3);

  Array<bool, 2, 2> mask;
  mask(0, 0) = true;
  mask(1, 0) = false;
  mask(0, 1) = false;
  mask(1, 1) = true;
  Array<pointer_result_payload*, 2, 2> then_values;
  then_values(0, 0) = &p0;
  then_values(1, 0) = &p1;
  then_values(0, 1) = &p2;
  then_values(1, 1) = &p3;
  Array<pointer_result_payload*, 2, 2> else_values;
  else_values(0, 0) = &p3;
  else_values(1, 0) = &p2;
  else_values(0, 1) = &p1;
  else_values(1, 1) = &p0;

  const auto selected = mask.select(then_values, else_values);
  VERIFY((std::is_same<typename Eigen::internal::traits<decltype(selected)>::Scalar, pointer_result_payload*>::value));
  Array<pointer_result_payload*, 2, 2> selected_out = selected;
  VERIFY_IS_EQUAL(selected_out(0, 0), &p0);
  VERIFY_IS_EQUAL(selected_out(1, 0), &p2);
  VERIFY_IS_EQUAL(selected_out(0, 1), &p1);
  VERIFY_IS_EQUAL(selected_out(1, 1), &p3);
}

TEST(MatrixCwiseTest, TestCwisePointerResult) {
  for (int i = 0; i < g_repeat; i++) {
    test_cwise_pointer_result();
  }
}
