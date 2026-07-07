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

// Reference dense BCCB built entry-wise from the generating array, independently
// of the operator under test: entry (i,j) with i = b1*n2+i2, j = c1*n2+j2 is
// G((i2-j2) mod n2, (b1-c1) mod n1).
template <typename Scalar>
Matrix<Scalar, Dynamic, Dynamic> reference_bccb(const Matrix<Scalar, Dynamic, Dynamic>& G) {
  const Index n2 = G.rows(), n1 = G.cols(), N = n1 * n2;
  Matrix<Scalar, Dynamic, Dynamic> dense(N, N);
  for (Index j = 0; j < N; ++j)
    for (Index i = 0; i < N; ++i) {
      Index k2 = i % n2 - j % n2;
      if (k2 < 0) k2 += n2;
      Index k1 = i / n2 - j / n2;
      if (k1 < 0) k1 += n1;
      dense(i, j) = G(k2, k1);
    }
  return dense;
}

template <typename Scalar>
void test_bccb_product(Index n2, Index n1) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Mat G = Mat::Random(n2, n1);
  Bccb<Scalar> C(G);
  Mat dense = reference_bccb<Scalar>(G);
  const Index N = n1 * n2;
  VERIFY_IS_EQUAL(C.rows(), N);

  Mat Cd = C;
  VERIFY_IS_APPROX(Cd, dense);
  for (Index t = 0; t < 5; ++t) {
    Index i = internal::random<Index>(0, N - 1), j = internal::random<Index>(0, N - 1);
    VERIFY_IS_APPROX(C.coeff(i, j), dense(i, j));
  }

  Vec x = Vec::Random(N);
  VERIFY_IS_APPROX((C * x).eval(), (dense * x).eval());

  Mat X = Mat::Random(N, 3);
  VERIFY_IS_APPROX((C * X).eval(), (dense * X).eval());

  // Accumulation form exercised by the iterative solvers.
  Vec y = Vec::Random(N);
  Vec y0 = y;
  y.noalias() += C * x;
  VERIFY_IS_APPROX(y, (y0 + dense * x).eval());
}

template <typename Scalar>
void test_bccb_transpose(Index n2, Index n1) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Mat G = Mat::Random(n2, n1);
  Bccb<Scalar> C(G);
  Mat dense = reference_bccb<Scalar>(G);
  const Index N = n1 * n2;

  Mat Td = C.transpose();
  VERIFY_IS_APPROX(Td, Mat(dense.transpose()));
  Mat Ad = C.adjoint();
  VERIFY_IS_APPROX(Ad, Mat(dense.adjoint()));
  Mat Kd = C.conjugate();
  VERIFY_IS_APPROX(Kd, Mat(dense.conjugate()));

  Vec x = Vec::Random(N);
  VERIFY_IS_APPROX((C.transpose() * x).eval(), (dense.transpose() * x).eval());
  VERIFY_IS_APPROX((C.adjoint() * x).eval(), (dense.adjoint() * x).eval());

  // Exact round trips: generators and symbols are pure permutations/conjugations.
  Bccb<Scalar> Ctt = C.transpose().transpose();
  VERIFY_IS_EQUAL(Ctt.generator(), G);
  VERIFY_IS_EQUAL(Ctt.symbol(), C.symbol());
  Bccb<Scalar> Caa = C.adjoint().adjoint();
  VERIFY_IS_EQUAL(Caa.generator(), G);
  VERIFY_IS_EQUAL(Caa.symbol(), C.symbol());
}

template <typename Scalar>
void test_bccb_solve(Index n2, Index n1) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  // Diagonal dominance (through the (0,0) generator entry) keeps the symbol away
  // from zero, so the direct 2-D FFT solve is exact up to roundoff.
  Mat G = Mat::Random(n2, n1);
  G(0, 0) += Scalar(RealScalar(2 * n1 * n2));
  Bccb<Scalar> C(G);
  Mat dense = reference_bccb<Scalar>(G);
  const Index N = n1 * n2;

  Vec b = Vec::Random(N);
  Vec x = C.solve(b);
  VERIFY_IS_APPROX((dense * x).eval(), b);

  Mat B = Mat::Random(N, 3);
  Mat Xs = C.solve(B);
  VERIFY_IS_APPROX((dense * Xs).eval(), B);
}

// Rank-deficient BCCB synthesized by zeroing 2-D symbol entries: the rank counts
// the surviving entries and solve() matches the SVD pseudo-inverse.
template <typename Scalar>
void test_bccb_rank_deficient(Index n2, Index n1, Index defect) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef std::complex<RealScalar> Complex;
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;
  typedef Matrix<Complex, Dynamic, Dynamic> CMat;

  const Index N = n1 * n2;
  CMat S = CMat::Random(n2, n1);
  S.array() += Complex(2);  // keep the surviving moduli away from the threshold
  for (Index k = 0; k < defect; ++k) S((2 * k + 1) % n2, (3 * k) % n1) = Complex(0);

  // Build the generator by an inverse 2-D DFT of the symbol (row-column, using
  // the dense DFT matrices for independence from the implementation under test).
  CMat F2(n2, n2), F1(n1, n1);
  for (Index a = 0; a < n2; ++a)
    for (Index c = 0; c < n2; ++c)
      F2(a, c) = std::polar(RealScalar(1) / RealScalar(n2),
                            RealScalar(2 * EIGEN_PI) * RealScalar((a * c) % n2) / RealScalar(n2));
  for (Index a = 0; a < n1; ++a)
    for (Index c = 0; c < n1; ++c)
      F1(a, c) = std::polar(RealScalar(1) / RealScalar(n1),
                            RealScalar(2 * EIGEN_PI) * RealScalar((a * c) % n1) / RealScalar(n1));
  Mat G = (F2 * S * F1.transpose()).eval();  // Scalar is complex here
  Bccb<Scalar> C(G);
  VERIFY_IS_EQUAL(C.rank(), N - defect);

  Mat dense = reference_bccb<Scalar>(G);
  JacobiSVD<Mat> svd(dense, ComputeThinU | ComputeThinV);
  Vec b = Vec::Random(N);
  VERIFY_IS_APPROX(C.solve(b), svd.solve(b).eval());
}

template <typename Scalar>
void test_bccb_zero(Index n2, Index n1) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Bccb<Scalar> C(Mat(Mat::Zero(n2, n1)));
  VERIFY_IS_EQUAL(C.rank(), 0);
  VERIFY(C.singularValues().isZero());
  Vec b = Vec::Random(n1 * n2);
  VERIFY(C.solve(b).isZero());
}

void test_bccb_nan_propagation(Index n2, Index n1) {
  typedef Matrix<double, Dynamic, 1> Vec;
  typedef Matrix<double, Dynamic, Dynamic> Mat;

  Mat G = Mat::Random(n2, n1);
  G(n2 / 2, n1 / 2) = std::numeric_limits<double>::quiet_NaN();
  Bccb<double> C(G);
  VERIFY_IS_EQUAL(C.rank(), n1 * n2);
  Vec b = Vec::Random(n1 * n2);
  Vec x = C.solve(b);
  VERIFY(!(x.array() == x.array()).all());
}

template <typename Scalar>
void test_bccb_eigen(Index n2, Index n1) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef std::complex<RealScalar> Complex;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;
  typedef Matrix<Complex, Dynamic, Dynamic> CMat;

  Mat G = Mat::Random(n2, n1);
  Bccb<Scalar> C(G);
  CMat denseC = reference_bccb<Scalar>(G).template cast<Complex>();
  const Index N = n1 * n2;

  Matrix<Complex, Dynamic, 1> lam = C.eigenvalues();
  CMat V = C.eigenvectors();
  VERIFY_IS_APPROX((denseC * V).eval(), (V * lam.asDiagonal()).eval());
  VERIFY_IS_APPROX((V.adjoint() * V).eval(), CMat(CMat::Identity(N, N)));
}

template <typename Scalar>
void test_bccb_svd(Index n2, Index n1) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef std::complex<RealScalar> Complex;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;
  typedef Matrix<Complex, Dynamic, Dynamic> CMat;

  Mat G = Mat::Random(n2, n1);
  Bccb<Scalar> C(G);
  Mat dense = reference_bccb<Scalar>(G);
  const Index N = n1 * n2;

  Matrix<RealScalar, Dynamic, 1> sv = C.singularValues();
  JacobiSVD<Mat> svd(dense);
  VERIFY_IS_APPROX(sv, svd.singularValues());

  CMat U = C.matrixU(), V = C.matrixV();
  VERIFY_IS_APPROX((U * sv.template cast<Complex>().asDiagonal() * V.adjoint()).eval(),
                   CMat(dense.template cast<Complex>()));
  VERIFY_IS_APPROX((U.adjoint() * U).eval(), CMat(CMat::Identity(N, N)));
  VERIFY_IS_APPROX((V.adjoint() * V).eval(), CMat(CMat::Identity(N, N)));
}

template <typename Scalar>
void test_bccb_inverse_determinant(Index n2, Index n1) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Mat G = Mat::Random(n2, n1);
  G(0, 0) += Scalar(RealScalar(2 * n1 * n2));  // safely invertible, tame determinant scale
  Bccb<Scalar> C(G);
  Mat dense = reference_bccb<Scalar>(G);
  const Index N = n1 * n2;

  Mat inv = C.inverse();
  VERIFY_IS_APPROX((inv * dense).eval(), Mat(Mat::Identity(N, N)));

  Vec b = Vec::Random(N);
  VERIFY_IS_APPROX((C.inverse() * b).eval(), C.solve(b));
}

template <typename Scalar>
void test_bccb_determinant(Index n2, Index n1) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Mat G = Mat::Random(n2, n1);
  G(0, 0) += Scalar(RealScalar(2 * n1 * n2));
  Bccb<Scalar> C(G);
  Mat dense = reference_bccb<Scalar>(G);
  VERIFY_IS_APPROX(C.determinant(), dense.determinant());
}

template <typename Scalar>
void test_bccb_matrix_free_cg(Index n2, Index n1) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  // Symmetrizing the generator under the two-dimensional index reversal makes
  // the BCCB matrix symmetric; diagonal dominance then makes it positive
  // definite.
  Mat G = Mat::Random(n2, n1);
  Mat Gs = G;
  for (Index k1 = 0; k1 < n1; ++k1)
    for (Index k2 = 0; k2 < n2; ++k2) Gs(k2, k1) = (G(k2, k1) + G((n2 - k2) % n2, (n1 - k1) % n1)) / Scalar(2);
  Gs(0, 0) += Scalar(RealScalar(2 * n1 * n2));
  Bccb<Scalar> C(Gs);
  Mat dense = reference_bccb<Scalar>(Gs);
  VERIFY_IS_APPROX(dense, Mat(dense.transpose()));

  const Index N = n1 * n2;
  Vec b = Vec::Random(N);
  ConjugateGradient<Bccb<Scalar>, Lower | Upper, IdentityPreconditioner> cg;
  cg.compute(C);
  Vec x = cg.solve(b);
  VERIFY(cg.info() == Success);
  VERIFY_IS_APPROX((dense * x).eval(), b);
}

template <typename Scalar, int N2, int N1>
void test_bccb_fixed() {
  typedef Matrix<Scalar, N2, N1> GenMat;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  GenMat G = GenMat::Random();
  Bccb<Scalar, N2, N1> C(G);
  STATIC_CHECK((Bccb<Scalar, N2, N1>::RowsAtCompileTime == N1 * N2));
  STATIC_CHECK((internal::remove_all_t<decltype(makeBccb(G))>::RowsAtCompileTime == N1 * N2));

  Mat dense = reference_bccb<Scalar>(Mat(G));
  Matrix<Scalar, N1 * N2, N1 * N2> Cd = C;
  VERIFY_IS_APPROX(Mat(Cd), dense);

  Matrix<Scalar, N1 * N2, 1> x = Matrix<Scalar, N1 * N2, 1>::Random();
  Matrix<Scalar, N1 * N2, 1> y = C * x;
  VERIFY_IS_APPROX(y, (dense * x).eval());
}

EIGEN_DECLARE_TEST(structured_bccb) {
  for (int i = 0; i < g_repeat; ++i) {
    // Products, dense assignment, coefficient access: scalar tier (N <= 32) and
    // 2-D FFT tier, including single-row/column-of-blocks degenerate shapes.
    CALL_SUBTEST_1((test_bccb_product<double>(1, 1)));
    CALL_SUBTEST_1((test_bccb_product<double>(3, 4)));
    CALL_SUBTEST_1((test_bccb_product<double>(5, 5)));
    CALL_SUBTEST_1((test_bccb_product<double>(8, 6)));
    CALL_SUBTEST_1((test_bccb_product<double>(7, 5)));  // odd prime dimensions
    CALL_SUBTEST_1((test_bccb_product<double>(1, 40)));
    CALL_SUBTEST_1((test_bccb_product<double>(40, 1)));
    CALL_SUBTEST_1((test_bccb_product<float>(6, 8)));
    CALL_SUBTEST_1((test_bccb_product<std::complex<double>>(4, 3)));
    CALL_SUBTEST_1((test_bccb_product<std::complex<double>>(9, 6)));
    CALL_SUBTEST_1((test_bccb_product<std::complex<float>>(6, 7)));

    // Transposition family with exact symbol round trips.
    CALL_SUBTEST_2((test_bccb_transpose<double>(3, 4)));
    CALL_SUBTEST_2((test_bccb_transpose<double>(8, 6)));
    CALL_SUBTEST_2((test_bccb_transpose<double>(1, 12)));
    CALL_SUBTEST_2((test_bccb_transpose<std::complex<double>>(6, 8)));
    CALL_SUBTEST_2((test_bccb_transpose<std::complex<float>>(5, 7)));

    // Direct and pseudo-inverse solves, degenerate operators.
    CALL_SUBTEST_3((test_bccb_solve<double>(1, 1)));
    CALL_SUBTEST_3((test_bccb_solve<double>(6, 8)));
    CALL_SUBTEST_3((test_bccb_solve<float>(5, 6)));
    CALL_SUBTEST_3((test_bccb_solve<std::complex<double>>(6, 6)));
    CALL_SUBTEST_3((test_bccb_rank_deficient<std::complex<double>>(6, 8, 4)));
    CALL_SUBTEST_3((test_bccb_rank_deficient<std::complex<float>>(4, 5, 2)));
    CALL_SUBTEST_3((test_bccb_zero<double>(4, 5)));
    CALL_SUBTEST_3(test_bccb_nan_propagation(6, 7));

    // Closed-form eigendecomposition, SVD, inverse, determinant.
    CALL_SUBTEST_4((test_bccb_eigen<double>(4, 5)));
    CALL_SUBTEST_4((test_bccb_eigen<double>(1, 8)));
    CALL_SUBTEST_4((test_bccb_eigen<std::complex<double>>(3, 5)));
    CALL_SUBTEST_4((test_bccb_svd<double>(4, 5)));
    CALL_SUBTEST_4((test_bccb_svd<std::complex<double>>(4, 4)));
    CALL_SUBTEST_4((test_bccb_svd<float>(3, 4)));
    CALL_SUBTEST_4((test_bccb_inverse_determinant<double>(6, 8)));
    CALL_SUBTEST_4((test_bccb_inverse_determinant<std::complex<double>>(5, 5)));
    CALL_SUBTEST_4((test_bccb_determinant<double>(3, 4)));
    CALL_SUBTEST_4((test_bccb_determinant<std::complex<double>>(3, 3)));

    // Matrix-free iterative solve and fixed-size generators.
    CALL_SUBTEST_5((test_bccb_matrix_free_cg<double>(8, 10)));
    CALL_SUBTEST_5((test_bccb_fixed<double, 3, 4>()));
    CALL_SUBTEST_5((test_bccb_fixed<std::complex<float>, 4, 3>()));
  }
}
