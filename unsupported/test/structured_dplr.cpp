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

// Well-conditioned random operator: diagonal bounded away from zero and a
// contractive correction, so the capacitance matrix is safely invertible.
template <typename Scalar>
void random_dplr(Index n, Index k, Matrix<Scalar, Dynamic, 1>& d, Matrix<Scalar, Dynamic, Dynamic>& U,
                 Matrix<Scalar, Dynamic, Dynamic>& V) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  d = Matrix<Scalar, Dynamic, 1>::Random(n);
  d.array() += Scalar(RealScalar(3));
  U = Scalar(RealScalar(0.5)) * Matrix<Scalar, Dynamic, Dynamic>::Random(n, k);
  V = Scalar(RealScalar(0.5)) * Matrix<Scalar, Dynamic, Dynamic>::Random(n, k);
}

// Reference dense matrix built independently, entry by entry.
template <typename Scalar>
Matrix<Scalar, Dynamic, Dynamic> reference_dplr(const Matrix<Scalar, Dynamic, 1>& d,
                                                const Matrix<Scalar, Dynamic, Dynamic>& U,
                                                const Matrix<Scalar, Dynamic, Dynamic>& V) {
  const Index n = d.size();
  Matrix<Scalar, Dynamic, Dynamic> dense(n, n);
  for (Index j = 0; j < n; ++j)
    for (Index i = 0; i < n; ++i) {
      Scalar s = (i == j) ? d[i] : Scalar(0);
      for (Index t = 0; t < U.cols(); ++t) s += U(i, t) * numext::conj(V(j, t));
      dense(i, j) = s;
    }
  return dense;
}

template <typename Scalar>
void test_dplr_product(Index n, Index k) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Vec d;
  Mat U, V;
  random_dplr<Scalar>(n, k, d, U, V);
  DiagonalPlusLowRank<Scalar> A(d, U, V);
  Mat dense = reference_dplr<Scalar>(d, U, V);
  VERIFY_IS_EQUAL(A.correctionRank(), k);

  Mat Ad = A;
  VERIFY_IS_APPROX(Ad, dense);
  for (Index t = 0; t < 5; ++t) {
    Index i = internal::random<Index>(0, n - 1), j = internal::random<Index>(0, n - 1);
    VERIFY_IS_APPROX(A.coeff(i, j), dense(i, j));
  }

  Vec x = Vec::Random(n);
  VERIFY_IS_APPROX((A * x).eval(), (dense * x).eval());

  Mat X = Mat::Random(n, 3);
  VERIFY_IS_APPROX((A * X).eval(), (dense * X).eval());

  // Accumulation form exercised by the iterative solvers.
  Vec y = Vec::Random(n);
  Vec y0 = y;
  y.noalias() += A * x;
  VERIFY_IS_APPROX(y, (y0 + dense * x).eval());
}

template <typename Scalar>
void test_dplr_transpose(Index n, Index k) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Vec d;
  Mat U, V;
  random_dplr<Scalar>(n, k, d, U, V);
  DiagonalPlusLowRank<Scalar> A(d, U, V);
  Mat dense = reference_dplr<Scalar>(d, U, V);

  Mat Td = A.transpose();
  VERIFY_IS_APPROX(Td, Mat(dense.transpose()));
  Mat Ad = A.adjoint();
  VERIFY_IS_APPROX(Ad, Mat(dense.adjoint()));
  Mat Kd = A.conjugate();
  VERIFY_IS_APPROX(Kd, Mat(dense.conjugate()));

  Vec x = Vec::Random(n);
  VERIFY_IS_APPROX((A.transpose() * x).eval(), (dense.transpose() * x).eval());
  VERIFY_IS_APPROX((A.adjoint() * x).eval(), (dense.adjoint() * x).eval());
}

template <typename Scalar>
void test_dplr_solve(Index n, Index k) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Vec d;
  Mat U, V;
  random_dplr<Scalar>(n, k, d, U, V);
  DiagonalPlusLowRank<Scalar> A(d, U, V);
  Mat dense = reference_dplr<Scalar>(d, U, V);

  Vec b = Vec::Random(n);
  Vec x = A.solve(b);
  VERIFY_IS_APPROX((dense * x).eval(), b);
  VERIFY_IS_APPROX(x, dense.partialPivLu().solve(b).eval());

  Mat B = Mat::Random(n, 3);
  Mat X = A.solve(B);
  VERIFY_IS_APPROX(X, dense.partialPivLu().solve(B).eval());
}

template <typename Scalar>
void test_dplr_inverse_determinant(Index n, Index k) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Vec d;
  Mat U, V;
  random_dplr<Scalar>(n, k, d, U, V);
  DiagonalPlusLowRank<Scalar> A(d, U, V);
  Mat dense = reference_dplr<Scalar>(d, U, V);

  // The inverse is itself diagonal-plus-low-rank; materialize and check.
  DiagonalPlusLowRank<Scalar> Ainv = A.inverse();
  VERIFY_IS_EQUAL(Ainv.correctionRank(), k);
  Mat invd = Ainv;
  VERIFY_IS_APPROX((invd * dense).eval(), Mat(Mat::Identity(n, n)));

  // Inverting twice returns to the original operator.
  Mat back = Ainv.inverse();
  VERIFY_IS_APPROX(back, dense);

  VERIFY_IS_APPROX(A.determinant(), dense.determinant());
}

template <typename Scalar>
void test_dplr_matrix_free_gmres(Index n, Index k) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Vec d;
  Mat U, V;
  random_dplr<Scalar>(n, k, d, U, V);
  DiagonalPlusLowRank<Scalar> A(d, U, V);
  Mat dense = reference_dplr<Scalar>(d, U, V);

  Vec b = Vec::Random(n);
  GMRES<DiagonalPlusLowRank<Scalar>, IdentityPreconditioner> gmres;
  gmres.setTolerance(RealScalar(1e-12));
  gmres.compute(A);
  Vec x = gmres.solve(b);
  VERIFY(gmres.info() == Success);
  const RealScalar tol = RealScalar(5e6) * NumTraits<RealScalar>::epsilon();  // ~1e-9 in double
  VERIFY((dense * x - b).norm() <= tol * b.norm());
}

template <typename Scalar, int N, int K>
void test_dplr_fixed() {
  typedef Matrix<Scalar, N, 1> VecN;
  typedef Matrix<Scalar, N, K> FacNK;
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;
  typedef typename NumTraits<Scalar>::Real RealScalar;

  VecN d = VecN::Random();
  d.array() += Scalar(RealScalar(3));
  FacNK U = FacNK::Random(), V = FacNK::Random();
  DiagonalPlusLowRank<Scalar, N, K> A(d, U, V);
  STATIC_CHECK((DiagonalPlusLowRank<Scalar, N, K>::RowsAtCompileTime == N));
  STATIC_CHECK((internal::remove_all_t<decltype(makeDiagonalPlusLowRank(d, U, V))>::RowsAtCompileTime == N));

  Mat dense = reference_dplr<Scalar>(Vec(d), Mat(U), Mat(V));
  Matrix<Scalar, N, N> Adn = A;
  VERIFY_IS_APPROX(Mat(Adn), dense);

  VecN x = VecN::Random();
  VecN y = A * x;
  VERIFY_IS_APPROX(y, (dense * x).eval());

  VecN b = VecN::Random();
  VecN xs = A.solve(b);
  VERIFY_IS_APPROX((dense * xs).eval(), b);
}

EIGEN_DECLARE_TEST(structured_dplr) {
  for (int i = 0; i < g_repeat; ++i) {
    // Products, dense assignment, coefficient access; k = 0 is a plain diagonal
    // and k = n a full-rank correction.
    CALL_SUBTEST_1((test_dplr_product<double>(1, 0)));
    CALL_SUBTEST_1((test_dplr_product<double>(10, 0)));
    CALL_SUBTEST_1((test_dplr_product<double>(10, 1)));
    CALL_SUBTEST_1((test_dplr_product<double>(20, 3)));
    CALL_SUBTEST_1((test_dplr_product<double>(8, 8)));
    CALL_SUBTEST_1((test_dplr_product<float>(12, 2)));
    CALL_SUBTEST_1((test_dplr_product<std::complex<double>>(9, 2)));
    CALL_SUBTEST_1((test_dplr_product<std::complex<float>>(7, 3)));
    CALL_SUBTEST_1((test_dplr_transpose<double>(10, 2)));
    CALL_SUBTEST_1((test_dplr_transpose<std::complex<double>>(8, 3)));

    // Woodbury solves against the dense LU.
    CALL_SUBTEST_2((test_dplr_solve<double>(1, 0)));
    CALL_SUBTEST_2((test_dplr_solve<double>(10, 0)));
    CALL_SUBTEST_2((test_dplr_solve<double>(10, 1)));
    CALL_SUBTEST_2((test_dplr_solve<double>(30, 4)));
    CALL_SUBTEST_2((test_dplr_solve<double>(8, 8)));
    CALL_SUBTEST_2((test_dplr_solve<float>(12, 2)));
    CALL_SUBTEST_2((test_dplr_solve<std::complex<double>>(14, 3)));
    CALL_SUBTEST_2((test_dplr_solve<std::complex<float>>(8, 2)));

    // Inverse closure, determinant lemma, matrix-free GMRES, fixed sizes.
    CALL_SUBTEST_3((test_dplr_inverse_determinant<double>(10, 2)));
    CALL_SUBTEST_3((test_dplr_inverse_determinant<double>(6, 0)));
    CALL_SUBTEST_3((test_dplr_inverse_determinant<std::complex<double>>(8, 3)));
    CALL_SUBTEST_3((test_dplr_matrix_free_gmres<double>(24, 3)));
    CALL_SUBTEST_3((test_dplr_fixed<double, 6, 2>()));
    CALL_SUBTEST_3((test_dplr_fixed<std::complex<float>, 5, 1>()));
  }
}
