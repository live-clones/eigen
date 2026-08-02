// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Charlie Schlosser <cs.schlosser@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "main.h"

// Assigning a dense view to another view of the very same type must assign coefficients through to the underlying
// data. It must not degenerate into a memberwise copy of the view itself, which would rebind the view -- or, when
// the view holds no data of its own, do nothing at all.
//
// Every one of these types gets its operator=(const Derived&) from EIGEN_INHERIT_DENSE_ASSIGNMENT_OPERATORS. Should
// one of them be switched to the non-dense EIGEN_INHERIT_ASSIGNMENT_OPERATORS by mistake, its operator= would
// delegate to the defaulted copy-assignment of an empty base and silently become a no-op. That is not detectable by
// any type trait -- the operator is user-provided either way, only its body differs -- so it has to be caught here.

template <typename Lhs, typename Rhs>
void check_assign(const char* /*what*/, Lhs dst, Rhs src, const MatrixXd& before, const MatrixXd& expected,
                  MatrixXd& target) {
  VERIFY_IS_NOT_APPROX(before, expected);  // otherwise the test would pass even if nothing were assigned
  dst = src;
  VERIFY_IS_APPROX(target, expected);
}

void test_dense_views() {
  const Index n = 4;
  MatrixXd b(n, n);
  for (Index j = 0; j < n; ++j)
    for (Index i = 0; i < n; ++i) b(i, j) = double(1 + i + n * j);

  // Block
  {
    MatrixXd a = MatrixXd::Zero(n, n), e = MatrixXd::Zero(n, n);
    e.block(0, 0, 2, 2) = b.block(0, 0, 2, 2);
    a.block(0, 0, 2, 2) = b.block(0, 0, 2, 2);
    VERIFY_IS_APPROX(a, e);
    VERIFY_IS_NOT_APPROX(a, MatrixXd::Zero(n, n).eval());
  }
  // VectorBlock
  {
    VectorXd a = VectorXd::Zero(n), s = b.col(0);
    a.segment(0, 3) = s.segment(0, 3);
    VERIFY_IS_APPROX(a.segment(0, 3).eval(), s.segment(0, 3).eval());
  }
  // Map
  {
    MatrixXd a = MatrixXd::Zero(n, n);
    Map<MatrixXd> ma(a.data(), n, n), mb(b.data(), n, n);
    ma = mb;
    VERIFY_IS_APPROX(a, b);
    VERIFY(ma.data() == a.data());  // assignment must not rebind the Map
  }
  // Ref
  {
    MatrixXd a = MatrixXd::Zero(n, n);
    Ref<MatrixXd> ra(a), rb(b);
    ra = rb;
    VERIFY_IS_APPROX(a, b);
    VERIFY(ra.data() == a.data());
  }
  // Transpose
  {
    MatrixXd a = MatrixXd::Zero(n, n);
    a.transpose() = b.transpose();
    VERIFY_IS_APPROX(a, b);
  }
  // Reverse
  {
    MatrixXd a = MatrixXd::Zero(n, n);
    a.reverse() = b.reverse();
    VERIFY_IS_APPROX(a, b);
  }
  // Diagonal
  {
    MatrixXd a = MatrixXd::Zero(n, n);
    a.diagonal() = b.diagonal();
    VERIFY_IS_APPROX(a.diagonal().eval(), b.diagonal().eval());
  }
  // Reshaped
  {
    MatrixXd a = MatrixXd::Zero(n, n);
    a.reshaped(n * n, 1) = b.reshaped(n * n, 1);
    VERIFY_IS_APPROX(a, b);
  }
  // IndexedView
  {
    MatrixXd a = MatrixXd::Zero(n, n);
    std::vector<Index> idx{0, 2};
    a(idx, idx) = b(idx, idx);
    VERIFY_IS_APPROX(a(idx, idx).eval(), b(idx, idx).eval());
    VERIFY_IS_NOT_APPROX(a, MatrixXd::Zero(n, n).eval());
  }
  // ArrayWrapper
  {
    MatrixXd a = MatrixXd::Zero(n, n);
    a.array() = b.array();
    VERIFY_IS_APPROX(a, b);
  }
  // MatrixWrapper
  {
    ArrayXXd a = ArrayXXd::Zero(n, n), bb = b.array();
    a.matrix() = bb.matrix();
    VERIFY_IS_APPROX(a.matrix().eval(), bb.matrix().eval());
  }
  // CwiseUnaryView (real/imag of a complex matrix)
  {
    MatrixXcd a = MatrixXcd::Zero(n, n), c = MatrixXcd::Zero(n, n);
    c.real() = b;
    c.imag() = b;
    a.real() = c.real();
    VERIFY_IS_APPROX(a.real().eval(), b);
    VERIFY_IS_APPROX(a.imag().eval(), MatrixXd::Zero(n, n).eval());
    a.imag() = c.imag();
    VERIFY_IS_APPROX(a.imag().eval(), b);
  }
}

EIGEN_DECLARE_TEST(dense_view_assignment) {
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST(test_dense_views());
  }
}
