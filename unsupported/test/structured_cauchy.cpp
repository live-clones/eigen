// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"

#include <unsupported/Eigen/StructuredMatrices>

using namespace Eigen;

// Reference dense Cauchy built entry-wise from the node vectors.
template <typename Scalar>
Matrix<Scalar, Dynamic, Dynamic> reference_cauchy(const Matrix<Scalar, Dynamic, 1>& x,
                                                  const Matrix<Scalar, Dynamic, 1>& y) {
  Matrix<Scalar, Dynamic, Dynamic> dense(x.size(), y.size());
  for (Index j = 0; j < y.size(); ++j)
    for (Index i = 0; i < x.size(); ++i) dense(i, j) = Scalar(1) / (x[i] - y[j]);
  return dense;
}

// Separated node sets: x in [2,3], y in [0,1], so all denominators are in [1,3].
template <typename Scalar>
void separated_nodes(Index m, Index n, Matrix<Scalar, Dynamic, 1>& x, Matrix<Scalar, Dynamic, 1>& y) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  x = Matrix<Scalar, Dynamic, 1>::Random(m);
  y = Matrix<Scalar, Dynamic, 1>::Random(n);
  x = (x * Scalar(RealScalar(0.5))).array() + Scalar(RealScalar(2.5));
  y = (y * Scalar(RealScalar(0.5))).array() + Scalar(RealScalar(0.5));
}

template <typename Scalar>
void test_cauchy_product(Index m, Index n) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Vec x, y;
  separated_nodes<Scalar>(m, n, x, y);
  Cauchy<Scalar> C(x, y);
  Mat dense = reference_cauchy<Scalar>(x, y);

  Mat Cd = C;
  VERIFY_IS_APPROX(Cd, dense);
  for (Index t = 0; t < 5; ++t) {
    Index i = internal::random<Index>(0, m - 1), j = internal::random<Index>(0, n - 1);
    VERIFY_IS_APPROX(C.coeff(i, j), dense(i, j));
  }

  Vec v = Vec::Random(n);
  VERIFY_IS_APPROX((C * v).eval(), (dense * v).eval());

  Mat V = Mat::Random(n, 3);
  VERIFY_IS_APPROX((C * V).eval(), (dense * V).eval());

  // Accumulation form exercised by the iterative solvers.
  Vec w = Vec::Random(m);
  Vec w0 = w;
  w.noalias() += C * v;
  VERIFY_IS_APPROX(w, (w0 + dense * v).eval());
}

template <typename Scalar>
void test_cauchy_transpose(Index m, Index n) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Vec x, y;
  separated_nodes<Scalar>(m, n, x, y);
  Cauchy<Scalar> C(x, y);
  Mat dense = reference_cauchy<Scalar>(x, y);

  Mat Td = C.transpose();
  VERIFY_IS_APPROX(Td, Mat(dense.transpose()));
  Mat Ad = C.adjoint();
  VERIFY_IS_APPROX(Ad, Mat(dense.adjoint()));
  Mat Kd = C.conjugate();
  VERIFY_IS_APPROX(Kd, Mat(dense.conjugate()));

  Vec w = Vec::Random(m);
  VERIFY_IS_APPROX((C.transpose() * w).eval(), (dense.transpose() * w).eval());
  VERIFY_IS_APPROX((C.adjoint() * w).eval(), (dense.adjoint() * w).eval());
}

// GKO solve on separated random nodes, verified through residuals (Cauchy
// matrices are exponentially ill-conditioned, so forward-error comparisons
// between different algorithms would not be meaningful).
template <typename Scalar>
void test_cauchy_lu(Index n) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Vec x, y;
  separated_nodes<Scalar>(n, n, x, y);
  Cauchy<Scalar> C(x, y);
  Mat dense = reference_cauchy<Scalar>(x, y);

  CauchyLU<Scalar> lu(C);
  VERIFY(lu.info() == Success);

  // Backward-stability-style residual bound, scaled by the scalar's epsilon (the
  // separated node sets bound the GKO generators by 1, so no growth term).
  const RealScalar tol = RealScalar(1000) * RealScalar(n) * NumTraits<RealScalar>::epsilon();

  Vec b = Vec::Random(n);
  Vec u = lu.solve(b);
  VERIFY((dense * u - b).norm() <= tol * (dense.norm() * u.norm() + b.norm()));

  // Multiple right-hand sides.
  Mat B = Mat::Random(n, 3);
  Mat U = lu.solve(B);
  VERIFY((dense * U - B).norm() <= tol * (dense.norm() * U.norm() + B.norm()));

  // Transposed and adjoint systems reuse the same factorization.
  Vec vt = lu.transpose().solve(b);
  VERIFY((dense.transpose() * vt - b).norm() <= tol * (dense.norm() * vt.norm() + b.norm()));
  Vec va = lu.adjoint().solve(b);
  VERIFY((dense.adjoint() * va - b).norm() <= tol * (dense.norm() * va.norm() + b.norm()));
}

// The Hilbert matrix as a Cauchy matrix (x_i = i+1, y_j = -j): deterministic
// residual bounds far below what the astronomical conditioning would allow a
// forward-error test, plus the closed-form determinant against the dense LU one.
void test_cauchy_hilbert() {
  typedef Matrix<double, Dynamic, 1> Vec;
  typedef Matrix<double, Dynamic, Dynamic> Mat;

  auto residual = [](Index n) {
    Vec x(n), y(n);
    for (Index i = 0; i < n; ++i) {
      x[i] = double(i + 1);
      y[i] = -double(i);
    }
    Cauchy<double> C(x, y);
    Mat dense = reference_cauchy<double>(x, y);
    Vec b = dense * Vec::Ones(n);
    CauchyLU<double> lu(C);
    VERIFY(lu.info() == Success);
    Vec u = lu.solve(b);
    return (dense * u - b).norm() / b.norm();
  };
  VERIFY(residual(8) <= 1e-10);
  VERIFY(residual(12) <= 1e-8);

  Vec x(5), y(5);
  for (Index i = 0; i < 5; ++i) {
    x[i] = double(i + 1);
    y[i] = -double(i);
  }
  Cauchy<double> H5(x, y);
  Mat dense = reference_cauchy<double>(x, y);
  VERIFY_IS_APPROX(H5.determinant(), dense.determinant());
}

// Clustered row nodes make leading minors nearly singular: partial pivoting must
// keep the factorization backward stable (residual check).
void test_cauchy_lu_pivoting(Index n) {
  typedef Matrix<double, Dynamic, 1> Vec;
  typedef Matrix<double, Dynamic, Dynamic> Mat;

  Vec x(n), y(n);
  for (Index i = 0; i < n; ++i) {
    x[i] = 2.0 + 1e-13 * double(i * i);  // tight cluster
    y[i] = double(i) / double(n);        // spread out
  }
  Cauchy<double> C(x, y);
  Mat dense = reference_cauchy<double>(x, y);
  CauchyLU<double> lu(C);
  VERIFY(lu.info() == Success);
  Vec b = Vec::Random(n);
  Vec u = lu.solve(b);
  VERIFY((dense * u - b).norm() <= 1e-8 * (dense.norm() * u.norm() + b.norm()));
}

// A duplicated row node makes two rows identical, hence the matrix exactly
// singular; a zero pivot must survive partial pivoting and be reported.
void test_cauchy_lu_singular() {
  typedef Matrix<double, Dynamic, 1> Vec;
  Vec x(5), y(5);
  x << 2.0, 2.5, 2.0, 2.75, 2.25;  // x[2] duplicates x[0]
  y << 0.1, 0.3, 0.5, 0.7, 0.9;
  Cauchy<double> C(x, y);
  CauchyLU<double> lu(C);
  VERIFY(lu.info() == NumericalIssue);
}

// y = -x gives the symmetric generalized Hilbert matrix 1/(x_i + x_j): verify
// the symmetry closure of the transpose through the node identity C^T = C(-y,-x).
template <typename Scalar>
void test_cauchy_symmetric(Index n) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Vec x(n);
  for (Index i = 0; i < n; ++i) x[i] = Scalar(1) + Scalar(i);
  Vec negx = -x;
  Cauchy<Scalar> C(x, negx);
  Mat dense = reference_cauchy<Scalar>(x, negx);
  VERIFY_IS_APPROX(dense, Mat(dense.transpose()));
  Cauchy<Scalar> Ct = C.transpose();
  VERIFY_IS_EQUAL(Ct.rowNodes(), Vec(x));  // -(-x) round-trips exactly
  Mat Ctd = Ct;
  VERIFY_IS_APPROX(Ctd, dense);
}

// The closed form is a product of exact node differences (relative error
// O(n^2 eps)); the dense LU determinant it is compared against carries a
// cond(C)*eps error, so keep n small and the tolerance loose: the reference is
// the less accurate side of this comparison.
template <typename Scalar>
void test_cauchy_determinant(Index n) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Vec x, y;
  separated_nodes<Scalar>(n, n, x, y);
  Cauchy<Scalar> C(x, y);
  Mat dense = reference_cauchy<Scalar>(x, y);
  VERIFY(numext::abs(C.determinant() - dense.determinant()) <= RealScalar(1e-6) * numext::abs(dense.determinant()));
}

template <typename Scalar, int M, int N>
void test_cauchy_fixed() {
  typedef Matrix<Scalar, M, 1> XVec;
  typedef Matrix<Scalar, N, 1> YVec;
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, M, N> MatMN;

  Vec xd, yd;
  separated_nodes<Scalar>(M, N, xd, yd);
  XVec x = xd;
  YVec y = yd;
  Cauchy<Scalar, M, N> C(x, y);
  STATIC_CHECK((Cauchy<Scalar, M, N>::RowsAtCompileTime == M));
  STATIC_CHECK((Cauchy<Scalar, M, N>::ColsAtCompileTime == N));
  STATIC_CHECK((internal::remove_all_t<decltype(makeCauchy(x, y))>::ColsAtCompileTime == N));
  STATIC_CHECK((internal::remove_all_t<decltype(C.transpose())>::RowsAtCompileTime == N));
  STATIC_CHECK((internal::remove_all_t<decltype(C.transpose())>::ColsAtCompileTime == M));

  MatMN dense = C;
  VERIFY_IS_APPROX(dense, MatMN(reference_cauchy<Scalar>(xd, yd)));

  YVec v = YVec::Random();
  Matrix<Scalar, M, 1> w = C * v;
  VERIFY_IS_APPROX(w, (dense * v).eval());
}

EIGEN_DECLARE_TEST(structured_cauchy) {
  for (int i = 0; i < g_repeat; ++i) {
    // Products, dense assignment, coefficient access.
    CALL_SUBTEST_1((test_cauchy_product<double>(1, 1)));
    CALL_SUBTEST_1((test_cauchy_product<double>(8, 8)));
    CALL_SUBTEST_1((test_cauchy_product<double>(20, 12)));  // tall
    CALL_SUBTEST_1((test_cauchy_product<double>(12, 20)));  // wide
    CALL_SUBTEST_1((test_cauchy_product<float>(10, 10)));
    CALL_SUBTEST_1((test_cauchy_product<std::complex<double>>(9, 7)));
    CALL_SUBTEST_1((test_cauchy_product<std::complex<float>>(7, 9)));
    CALL_SUBTEST_1((test_cauchy_transpose<double>(10, 14)));
    CALL_SUBTEST_1((test_cauchy_transpose<std::complex<double>>(8, 6)));

    // GKO pivoted LU solves.
    CALL_SUBTEST_2((test_cauchy_lu<double>(1)));
    CALL_SUBTEST_2((test_cauchy_lu<double>(2)));
    CALL_SUBTEST_2((test_cauchy_lu<double>(12)));
    CALL_SUBTEST_2((test_cauchy_lu<double>(30)));
    CALL_SUBTEST_2((test_cauchy_lu<std::complex<double>>(16)));
    CALL_SUBTEST_2((test_cauchy_lu<float>(10)));
    CALL_SUBTEST_2(test_cauchy_hilbert());
    CALL_SUBTEST_2(test_cauchy_lu_pivoting(20));
    CALL_SUBTEST_2(test_cauchy_lu_singular());

    // Closed-form determinant, symmetric generalized Hilbert, fixed sizes.
    CALL_SUBTEST_3((test_cauchy_determinant<double>(4)));
    CALL_SUBTEST_3((test_cauchy_determinant<std::complex<double>>(4)));
    CALL_SUBTEST_3((test_cauchy_symmetric<double>(9)));
    CALL_SUBTEST_3((test_cauchy_fixed<double, 6, 4>()));
    CALL_SUBTEST_3((test_cauchy_fixed<std::complex<float>, 4, 5>()));
  }
}
