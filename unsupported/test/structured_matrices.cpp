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

// Reference dense circulant built straight from the generating column, so that the
// fast (FFT) product is validated against an independent construction of the matrix.
template <typename Scalar>
Matrix<Scalar, Dynamic, Dynamic> reference_circulant(const Matrix<Scalar, Dynamic, 1>& c) {
  const Index n = c.size();
  Matrix<Scalar, Dynamic, Dynamic> dense(n, n);
  for (Index j = 0; j < n; ++j)
    for (Index i = 0; i < n; ++i) {
      Index k = i - j;
      if (k < 0) k += n;
      dense(i, j) = c[k];
    }
  return dense;
}

template <typename Scalar>
Matrix<Scalar, Dynamic, Dynamic> reference_toeplitz(const Matrix<Scalar, Dynamic, 1>& c,
                                                    const Matrix<Scalar, Dynamic, 1>& r) {
  const Index m = c.size(), n = r.size();
  Matrix<Scalar, Dynamic, Dynamic> dense(m, n);
  for (Index j = 0; j < n; ++j)
    for (Index i = 0; i < m; ++i) dense(i, j) = (i >= j) ? c[i - j] : r[j - i];
  return dense;
}

template <typename Scalar>
void test_circulant_product(Index n) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Vec c = Vec::Random(n);
  Circulant<Scalar> C(c);
  Mat dense = reference_circulant<Scalar>(c);

  // The operator agrees with the independently-built dense matrix, both as a
  // dense expansion and through coeff access.
  VERIFY_IS_APPROX(C.toDense(), dense);
  for (Index t = 0; t < (std::min)(n, Index(5)); ++t) {
    Index i = internal::random<Index>(0, n - 1), j = internal::random<Index>(0, n - 1);
    VERIFY_IS_APPROX(C.coeff(i, j), dense(i, j));
  }

  // Fast matrix-vector and matrix-matrix products.
  Vec x = Vec::Random(n);
  VERIFY_IS_APPROX((C * x).eval(), (dense * x).eval());

  Mat X = Mat::Random(n, 3);
  VERIFY_IS_APPROX((C * X).eval(), (dense * X).eval());

  // Accumulation forms exercised by the iterative solvers.
  Vec y = Vec::Random(n);
  Vec y0 = y;
  y.noalias() += C * x;
  VERIFY_IS_APPROX(y, (y0 + dense * x).eval());
}

template <typename Scalar>
void test_circulant_solve(Index n) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  // Diagonally dominant => well conditioned, so the direct FFT solve is accurate.
  Vec c = Vec::Random(n);
  c[0] += Scalar(RealScalar(2 * n));
  Circulant<Scalar> C(c);
  Mat dense = reference_circulant<Scalar>(c);

  Vec b = Vec::Random(n);
  Vec x = C.solve(b);
  VERIFY_IS_APPROX((dense * x).eval(), b);

  // Multiple right-hand sides at once.
  Mat B = Mat::Random(n, 4);
  Mat Xs = C.solve(B);
  VERIFY_IS_APPROX((dense * Xs).eval(), B);
}

template <typename Scalar>
void test_toeplitz_product(Index m, Index n) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Vec c = Vec::Random(m), r = Vec::Random(n);
  r[0] = c[0];  // diagonal entry; r[0] is ignored anyway
  Toeplitz<Scalar> T(c, r);
  Mat dense = reference_toeplitz<Scalar>(c, r);

  VERIFY_IS_APPROX(T.toDense(), dense);

  Vec x = Vec::Random(n);
  VERIFY_IS_APPROX((T * x).eval(), (dense * x).eval());

  Mat X = Mat::Random(n, 3);
  VERIFY_IS_APPROX((T * X).eval(), (dense * X).eval());
}

template <typename Scalar>
void test_matrix_free_cg(Index n) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  // Symmetric, strongly diagonally dominant circulant => SPD.
  Vec c = Vec::Zero(n);
  c[0] = Scalar(4);
  c[1] = Scalar(-1);
  c[n - 1] = Scalar(-1);
  Circulant<Scalar> C(c);
  Mat dense = reference_circulant<Scalar>(c);

  Vec b = Vec::Random(n);
  ConjugateGradient<Circulant<Scalar>, Lower | Upper, IdentityPreconditioner> cg;
  cg.compute(C);
  Vec x = cg.solve(b);
  VERIFY(cg.info() == Success);
  VERIFY_IS_APPROX((dense * x).eval(), b);
}

template <typename Scalar>
void test_matrix_free_gmres(Index n) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  // Strongly diagonally dominant (nonsymmetric) Toeplitz.
  Vec c = Vec::Random(n) * Scalar(RealScalar(0.1));
  Vec r = Vec::Random(n) * Scalar(RealScalar(0.1));
  c[0] = Scalar(3);
  r[0] = Scalar(3);
  Toeplitz<Scalar> T(c, r);
  Mat dense = reference_toeplitz<Scalar>(c, r);

  Vec b = Vec::Random(n);
  GMRES<Toeplitz<Scalar>, IdentityPreconditioner> gmres;
  gmres.compute(T);
  Vec x = gmres.solve(b);
  VERIFY(gmres.info() == Success);
  VERIFY_IS_APPROX((dense * x).eval(), b);
}

EIGEN_DECLARE_TEST(structured_matrices) {
  for (int i = 0; i < g_repeat; ++i) {
    // Circulant: direct path (small), FFT path (composite and prime sizes), edge cases.
    CALL_SUBTEST_1((test_circulant_product<double>(1)));
    CALL_SUBTEST_1((test_circulant_product<double>(2)));
    CALL_SUBTEST_1((test_circulant_product<double>(8)));
    CALL_SUBTEST_1((test_circulant_product<double>(64)));
    CALL_SUBTEST_1((test_circulant_product<double>(97)));  // prime, FFT path
    CALL_SUBTEST_1((test_circulant_product<float>(48)));
    CALL_SUBTEST_1((test_circulant_product<std::complex<double>>(50)));
    CALL_SUBTEST_1((test_circulant_product<std::complex<float>>(40)));
    CALL_SUBTEST_1((test_circulant_solve<double>(50)));
    CALL_SUBTEST_1((test_circulant_solve<std::complex<double>>(40)));
    CALL_SUBTEST_1((test_circulant_solve<float>(32)));

    // Toeplitz: square, tall, wide, small (direct), real and complex.
    CALL_SUBTEST_2((test_toeplitz_product<double>(1, 1)));
    CALL_SUBTEST_2((test_toeplitz_product<double>(2, 2)));
    CALL_SUBTEST_2((test_toeplitz_product<double>(10, 10)));
    CALL_SUBTEST_2((test_toeplitz_product<double>(64, 64)));
    CALL_SUBTEST_2((test_toeplitz_product<double>(96, 48)));
    CALL_SUBTEST_2((test_toeplitz_product<double>(48, 96)));
    CALL_SUBTEST_2((test_toeplitz_product<float>(50, 50)));
    CALL_SUBTEST_2((test_toeplitz_product<std::complex<double>>(48, 64)));
    CALL_SUBTEST_2((test_toeplitz_product<std::complex<float>>(40, 40)));

    // Matrix-free iterative solves through the existing solvers.
    CALL_SUBTEST_3((test_matrix_free_cg<double>(80)));
    CALL_SUBTEST_3((test_matrix_free_gmres<double>(80)));
  }
}
