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

// Reference dense Kronecker product, built entry-wise and independently of both
// the operator under test and the KroneckerProduct module.
template <typename Scalar>
Matrix<Scalar, Dynamic, Dynamic> reference_kron(const Matrix<Scalar, Dynamic, Dynamic>& A,
                                                const Matrix<Scalar, Dynamic, Dynamic>& B) {
  Matrix<Scalar, Dynamic, Dynamic> K(A.rows() * B.rows(), A.cols() * B.cols());
  for (Index i = 0; i < K.rows(); ++i)
    for (Index j = 0; j < K.cols(); ++j) K(i, j) = A(i / B.rows(), j / B.cols()) * B(i % B.rows(), j % B.cols());
  return K;
}

template <typename Scalar>
void test_kron_product(Index m1, Index n1, Index m2, Index n2) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Mat A = Mat::Random(m1, n1), B = Mat::Random(m2, n2);
  KroneckerOperator<Mat, Mat> K(A, B);
  Mat dense = reference_kron<Scalar>(A, B);
  VERIFY_IS_EQUAL(K.rows(), m1 * m2);
  VERIFY_IS_EQUAL(K.cols(), n1 * n2);

  // Dense assignment through the evaluator and coefficient access.
  Mat Kd = K;
  VERIFY_IS_APPROX(Kd, dense);
  for (Index t = 0; t < 5; ++t) {
    Index i = internal::random<Index>(0, K.rows() - 1), j = internal::random<Index>(0, K.cols() - 1);
    VERIFY_IS_APPROX(K.coeff(i, j), dense(i, j));
  }

  // Fast matrix-vector and matrix-matrix products via the vec identity.
  Vec x = Vec::Random(n1 * n2);
  VERIFY_IS_APPROX((K * x).eval(), (dense * x).eval());
  Mat X = Mat::Random(n1 * n2, 3);
  VERIFY_IS_APPROX((K * X).eval(), (dense * X).eval());

  // Accumulation form exercised by the iterative solvers.
  Vec y = Vec::Random(m1 * m2);
  Vec y0 = y;
  y.noalias() += K * x;
  VERIFY_IS_APPROX(y, (y0 + dense * x).eval());
}

template <typename Scalar>
void test_kron_transpose(Index m1, Index n1, Index m2, Index n2) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Mat A = Mat::Random(m1, n1), B = Mat::Random(m2, n2);
  KroneckerOperator<Mat, Mat> K(A, B);
  Mat dense = reference_kron<Scalar>(A, B);

  Mat Td = K.transpose();
  VERIFY_IS_APPROX(Td, Mat(dense.transpose()));
  Mat Ad = K.adjoint();
  VERIFY_IS_APPROX(Ad, Mat(dense.adjoint()));
  Mat Cd = K.conjugate();
  VERIFY_IS_APPROX(Cd, Mat(dense.conjugate()));

  Vec y = Vec::Random(m1 * m2);
  VERIFY_IS_APPROX((K.transpose() * y).eval(), (dense.transpose() * y).eval());
  VERIFY_IS_APPROX((K.adjoint() * y).eval(), (dense.adjoint() * y).eval());
}

template <typename Scalar>
void test_kron_solve(Index n1, Index n2) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  // Diagonal boosts keep both factors (hence the product) well conditioned.
  Mat A = Mat::Random(n1, n1) + RealScalar(2 * n1) * Mat::Identity(n1, n1);
  Mat B = Mat::Random(n2, n2) + RealScalar(2 * n2) * Mat::Identity(n2, n2);
  KroneckerOperator<Mat, Mat> K(A, B);
  Mat dense = reference_kron<Scalar>(A, B);

  Vec b = Vec::Random(n1 * n2);
  Vec x = K.solve(b);
  VERIFY_IS_APPROX((dense * x).eval(), b);
  VERIFY_IS_APPROX(x, dense.partialPivLu().solve(b).eval());

  Mat Bm = Mat::Random(n1 * n2, 3);
  Mat Xm = K.solve(Bm);
  VERIFY_IS_APPROX(Xm, dense.partialPivLu().solve(Bm).eval());
}

template <typename Scalar>
void test_kron_least_squares(Index m1, Index n1, Index m2, Index n2) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Mat A = Mat::Random(m1, n1), B = Mat::Random(m2, n2);
  KroneckerOperator<Mat, Mat> K(A, B);
  Mat dense = reference_kron<Scalar>(A, B);

  // The minimum-norm least-squares solution is unique, so the factored
  // pseudo-inverse must match the dense complete orthogonal decomposition.
  Vec b = Vec::Random(m1 * m2);
  Vec x = K.leastSquaresSolve(b);
  VERIFY_IS_APPROX(x, dense.completeOrthogonalDecomposition().solve(b).eval());
}

template <typename Scalar>
void test_kron_least_squares_rank_deficient(Index m1, Index n1, Index m2, Index n2) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  // A rank-one left factor makes the product rank deficient; the minimum-norm
  // least-squares solution stays unique and comparable.
  Mat A = Vec::Random(m1) * Vec::Random(n1).transpose();
  Mat B = Mat::Random(m2, n2);
  KroneckerOperator<Mat, Mat> K(A, B);
  Mat dense = reference_kron<Scalar>(A, B);
  VERIFY_IS_EQUAL(K.rank(), dense.completeOrthogonalDecomposition().rank());

  Vec b = Vec::Random(m1 * m2);
  Vec x = K.leastSquaresSolve(b);
  VERIFY_IS_APPROX(x, dense.completeOrthogonalDecomposition().solve(b).eval());
}

template <typename Scalar>
void test_kron_eigen(Index n1, Index n2) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef std::complex<RealScalar> Complex;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;
  typedef Matrix<Complex, Dynamic, Dynamic> CMat;
  typedef Matrix<Complex, Dynamic, 1> CVec;

  Mat A = Mat::Random(n1, n1), B = Mat::Random(n2, n2);
  KroneckerOperator<Mat, Mat> K(A, B);
  CMat dense = reference_kron<Scalar>(A, B).template cast<Complex>();

  CVec lambda = K.eigenvalues();
  CMat V = K.eigenvectors();  // materializes the Kronecker eigenvector operator
  VERIFY_IS_APPROX((dense * V).eval(), (V * lambda.asDiagonal()).eval());
}

template <typename Scalar>
void test_kron_svd(Index m1, Index n1, Index m2, Index n2) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Mat A = Mat::Random(m1, n1), B = Mat::Random(m2, n2);
  KroneckerOperator<Mat, Mat> K(A, B);
  Mat dense = reference_kron<Scalar>(A, B);

  Matrix<RealScalar, Dynamic, 1> sv = K.singularValues();
  Mat U = K.matrixU(), V = K.matrixV();
  const Index k = sv.size();
  VERIFY_IS_EQUAL(U.cols(), k);
  VERIFY_IS_EQUAL(V.cols(), k);

  // Thin SVD in Kronecker order: U and V have orthonormal columns and
  // U * diag(sv) * V^H reconstructs the matrix.
  VERIFY_IS_APPROX((U.adjoint() * U).eval(), Mat(Mat::Identity(k, k)));
  VERIFY_IS_APPROX((V.adjoint() * V).eval(), Mat(Mat::Identity(k, k)));
  VERIFY_IS_APPROX((U * sv.template cast<Scalar>().asDiagonal() * V.adjoint()).eval(), dense);

  // The value multiset matches the dense SVD (compare sorted).
  Matrix<RealScalar, Dynamic, 1> svSorted = sv;
  std::sort(svSorted.data(), svSorted.data() + svSorted.size(), std::greater<RealScalar>());
  JacobiSVD<Mat> svd(dense);
  VERIFY_IS_APPROX(svSorted, svd.singularValues().head(k).eval());
  if (svd.singularValues().size() > k)  // rectangular shapes pad with structural zeros
    VERIFY(svd.singularValues().tail(svd.singularValues().size() - k).norm() <=
           RealScalar(100) * NumTraits<RealScalar>::epsilon() * svd.singularValues()[0]);
}

template <typename Scalar>
void test_kron_inverse_determinant(Index n1, Index n2) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  // Modest diagonal boost: invertible, with the determinant's magnitude tame
  // enough for a relative comparison in double precision.
  Mat A = Mat::Random(n1, n1) + RealScalar(2) * Mat::Identity(n1, n1);
  Mat B = Mat::Random(n2, n2) + RealScalar(2) * Mat::Identity(n2, n2);
  KroneckerOperator<Mat, Mat> K(A, B);
  Mat dense = reference_kron<Scalar>(A, B);

  Mat inv = K.inverse();
  VERIFY_IS_APPROX((inv * dense).eval(), Mat(Mat::Identity(dense.rows(), dense.cols())));
  VERIFY_IS_APPROX(K.determinant(), dense.determinant());
}

template <typename Scalar>
void test_kron_matrix_free_cg(Index n1, Index n2) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  // SPD factors make an SPD Kronecker product.
  Mat Ar = Mat::Random(n1, n1), Br = Mat::Random(n2, n2);
  Mat A = Ar * Ar.adjoint() + RealScalar(n1) * Mat::Identity(n1, n1);
  Mat B = Br * Br.adjoint() + RealScalar(n2) * Mat::Identity(n2, n2);
  KroneckerOperator<Mat, Mat> K(A, B);
  Mat dense = reference_kron<Scalar>(A, B);

  Vec b = Vec::Random(n1 * n2);
  ConjugateGradient<KroneckerOperator<Mat, Mat>, Lower | Upper, IdentityPreconditioner> cg;
  cg.compute(K);
  Vec x = cg.solve(b);
  VERIFY(cg.info() == Success);
  VERIFY_IS_APPROX((dense * x).eval(), b);
}

template <typename Scalar, int M1, int N1, int M2, int N2>
void test_kron_fixed() {
  typedef Matrix<Scalar, M1, N1> MatA;
  typedef Matrix<Scalar, M2, N2> MatB;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  MatA A = MatA::Random();
  MatB B = MatB::Random();
  KroneckerOperator<MatA, MatB> K(A, B);
  STATIC_CHECK((KroneckerOperator<MatA, MatB>::RowsAtCompileTime == M1 * M2));
  STATIC_CHECK((KroneckerOperator<MatA, MatB>::ColsAtCompileTime == N1 * N2));
  STATIC_CHECK((internal::remove_all_t<decltype(makeKroneckerOperator(A, B))>::RowsAtCompileTime == M1 * M2));
  STATIC_CHECK((internal::remove_all_t<decltype(K.transpose())>::RowsAtCompileTime == N1 * N2));
  STATIC_CHECK((internal::remove_all_t<decltype(K.transpose())>::ColsAtCompileTime == M1 * M2));

  Mat dense = reference_kron<Scalar>(Mat(A), Mat(B));
  Matrix<Scalar, M1 * M2, N1 * N2> Kd = K;
  VERIFY_IS_APPROX(Mat(Kd), dense);

  Matrix<Scalar, N1 * N2, 1> x = Matrix<Scalar, N1 * N2, 1>::Random();
  Matrix<Scalar, M1 * M2, 1> y = K * x;
  VERIFY_IS_APPROX(y, (dense * x).eval());
}

EIGEN_DECLARE_TEST(structured_kronecker) {
  for (int i = 0; i < g_repeat; ++i) {
    // Products, dense assignment, coefficient access across factor shapes.
    CALL_SUBTEST_1((test_kron_product<double>(1, 1, 1, 1)));
    CALL_SUBTEST_1((test_kron_product<double>(2, 3, 3, 2)));
    CALL_SUBTEST_1((test_kron_product<double>(4, 4, 5, 5)));
    CALL_SUBTEST_1((test_kron_product<double>(8, 3, 2, 7)));
    CALL_SUBTEST_1((test_kron_product<double>(1, 6, 7, 1)));  // vector-shaped factors
    CALL_SUBTEST_1((test_kron_product<float>(5, 4, 4, 5)));
    CALL_SUBTEST_1((test_kron_product<std::complex<double>>(3, 5, 4, 2)));
    CALL_SUBTEST_1((test_kron_product<std::complex<float>>(4, 3, 3, 4)));

    // Transposition family.
    CALL_SUBTEST_2((test_kron_transpose<double>(4, 6, 5, 3)));
    CALL_SUBTEST_2((test_kron_transpose<std::complex<double>>(3, 4, 5, 2)));

    // Direct solves through factor LU.
    CALL_SUBTEST_3((test_kron_solve<double>(1, 1)));
    CALL_SUBTEST_3((test_kron_solve<double>(4, 7)));
    CALL_SUBTEST_3((test_kron_solve<double>(9, 5)));
    CALL_SUBTEST_3((test_kron_solve<float>(5, 4)));
    CALL_SUBTEST_3((test_kron_solve<std::complex<double>>(6, 4)));

    // Minimum-norm least squares through factor pseudo-inverses.
    CALL_SUBTEST_3((test_kron_least_squares<double>(8, 5, 7, 4)));  // tall x tall
    CALL_SUBTEST_3((test_kron_least_squares<double>(8, 5, 3, 6)));  // tall x wide
    CALL_SUBTEST_3((test_kron_least_squares<std::complex<double>>(6, 4, 5, 3)));
    CALL_SUBTEST_3((test_kron_least_squares_rank_deficient<double>(6, 4, 5, 3)));
    CALL_SUBTEST_3((test_kron_least_squares_rank_deficient<std::complex<double>>(5, 3, 4, 4)));

    // Eigendecomposition, SVD, inverse, determinant.
    CALL_SUBTEST_4((test_kron_eigen<double>(4, 5)));
    CALL_SUBTEST_4((test_kron_eigen<double>(1, 6)));
    CALL_SUBTEST_4((test_kron_eigen<std::complex<double>>(3, 4)));
    CALL_SUBTEST_4((test_kron_svd<double>(4, 4, 5, 5)));
    CALL_SUBTEST_4((test_kron_svd<double>(6, 4, 3, 5)));  // rectangular: structural zeros
    CALL_SUBTEST_4((test_kron_svd<std::complex<double>>(4, 3, 4, 4)));
    CALL_SUBTEST_4((test_kron_svd<float>(4, 4, 4, 4)));
    CALL_SUBTEST_4((test_kron_inverse_determinant<double>(4, 5)));
    CALL_SUBTEST_4((test_kron_inverse_determinant<std::complex<double>>(3, 4)));

    // Matrix-free iterative solve and fixed-size factors.
    CALL_SUBTEST_5((test_kron_matrix_free_cg<double>(6, 7)));
    CALL_SUBTEST_5((test_kron_fixed<double, 3, 4, 2, 5>()));
    CALL_SUBTEST_5((test_kron_fixed<std::complex<float>, 2, 3, 3, 2>()));
  }
}
