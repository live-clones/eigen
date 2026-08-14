// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// This test is compiled with -DEIGEN_DEFAULT_DENSE_INDEX_TYPE=int (see test/CMakeLists.txt) so that it fails if
// Eigen::Index ever stops tracking that macro. Because Index is ABI-affecting, the override has to be applied to the
// whole translation unit rather than #defined here.

#include "main.h"

static_assert(std::is_same<Eigen::Index, EIGEN_DEFAULT_DENSE_INDEX_TYPE>::value,
              "Eigen::Index must follow EIGEN_DEFAULT_DENSE_INDEX_TYPE");
static_assert(std::is_same<Eigen::DenseIndex, EIGEN_DEFAULT_DENSE_INDEX_TYPE>::value,
              "Eigen::DenseIndex must follow EIGEN_DEFAULT_DENSE_INDEX_TYPE");

// The narrowed index has to survive into the expression types users actually see, not just the Eigen::Index alias.
static_assert(std::is_same<Eigen::MatrixXd::Index, EIGEN_DEFAULT_DENSE_INDEX_TYPE>::value,
              "PlainObjectBase::Index must follow EIGEN_DEFAULT_DENSE_INDEX_TYPE");

void check_narrowed_index() {
  const Index rows = 7, cols = 5;
  MatrixXd m = MatrixXd::Random(rows, cols);

  VERIFY_IS_EQUAL(m.rows(), rows);
  VERIFY_IS_EQUAL(m.cols(), cols);
  VERIFY_IS_EQUAL(m.size(), rows * cols);

  // Index-returning APIs must agree with the narrowed type.
  Index r = -1, c = -1;
  m.maxCoeff(&r, &c);
  VERIFY(r >= 0 && r < rows && c >= 0 && c < cols);
  VERIFY_IS_EQUAL(m.block(1, 1, rows - 2, cols - 2).rows(), rows - 2);
  VERIFY_IS_EQUAL(m.transpose().rows(), cols);

  // Blocks, products and reductions all thread Index through the evaluators.
  MatrixXd p = m.transpose() * m;
  VERIFY_IS_EQUAL(p.rows(), cols);
  VERIFY_IS_APPROX(p, (m.transpose() * m).eval());
}

EIGEN_DECLARE_TEST(index_type) { CALL_SUBTEST(check_narrowed_index()); }
