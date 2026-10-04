// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"

#include <contrib/Eigen/StructuredMatrices>

using namespace Eigen;

// Dense references built entry-wise, independently of the operators under test.
template <typename Scalar>
Matrix<Scalar, Dynamic, Dynamic> reference_kron(const Matrix<Scalar, Dynamic, Dynamic>& A,
                                                const Matrix<Scalar, Dynamic, Dynamic>& B) {
  Matrix<Scalar, Dynamic, Dynamic> K(A.rows() * B.rows(), A.cols() * B.cols());
  for (Index i = 0; i < K.rows(); ++i)
    for (Index j = 0; j < K.cols(); ++j) K(i, j) = A(i / B.rows(), j / B.cols()) * B(i % B.rows(), j % B.cols());
  return K;
}

template <typename Scalar>
Matrix<Scalar, Dynamic, Dynamic> reference_ksum(const Matrix<Scalar, Dynamic, Dynamic>& A,
                                                const Matrix<Scalar, Dynamic, Dynamic>& B) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  return reference_kron<Scalar>(A, Mat::Identity(B.rows(), B.rows())) +
         reference_kron<Scalar>(Mat::Identity(A.rows(), A.rows()), B);
}

template <typename Scalar>
SparseMatrix<Scalar> random_sparse(Index n) {
  Matrix<Scalar, Dynamic, Dynamic> dense = Matrix<Scalar, Dynamic, Dynamic>::Random(n, n);
  for (Index j = 0; j < n; ++j)
    for (Index i = 0; i < n; ++i)
      if (internal::random<int>(0, 1) == 0) dense(i, j) = Scalar(0);
  return dense.sparseView();
}

// The 1-D second-difference matrix tridiag(-1, 2, -1) of order n, in sparse form.
SparseMatrix<double> second_difference(Index n) {
  SparseMatrix<double> D(n, n);
  D.reserve(VectorXi::Constant(int(n), 3));
  for (Index j = 0; j < n; ++j) {
    if (j > 0) D.insert(j - 1, j) = -1.0;
    D.insert(j, j) = 2.0;
    if (j + 1 < n) D.insert(j + 1, j) = -1.0;
  }
  D.makeCompressed();
  return D;
}

template <typename Op, typename Mat>
void check_products(const Op& K, const Mat& ref) {
  using Scalar = typename Mat::Scalar;
  using Complex = std::complex<typename NumTraits<Scalar>::Real>;
  using Vec = Matrix<Scalar, Dynamic, 1>;
  using CVec = Matrix<Complex, Dynamic, 1>;
  const Vec x = Vec::Random(ref.cols());
  VERIFY_IS_APPROX((K * x).eval(), (ref * x).eval());
  const Mat X = Mat::Random(ref.cols(), 3);
  VERIFY_IS_APPROX((K * X).eval(), (ref * X).eval());
  Vec y = Vec::Random(ref.rows());
  const Vec y0 = y;
  y.noalias() += K * x;
  VERIFY_IS_APPROX(y, (y0 + ref * x).eval());
  y = y0;
  y.noalias() -= K * x;
  VERIFY_IS_APPROX(y, (y0 - ref * x).eval());
  const CVec xc = CVec::Random(ref.cols());
  VERIFY_IS_APPROX((K * xc).eval(), (ref.template cast<Complex>() * xc).eval());
}

// The number of structurally stored entries of A (+) B: A(i,j) on the diagonal of
// block (i,j), B on every diagonal block, counted once where both land.
template <typename Mat>
Index ksum_pattern_size(const Mat& storedA, const Mat& storedB) {
  using Scalar = typename Mat::Scalar;
  const Mat pattern = reference_ksum<Scalar>(storedA, storedB);
  return Index((pattern.array() != Scalar(0)).count());
}

// Products, materialization, coefficients and the transposition family of
// A (+) B for every factor-kind mix, against the dense reference.
template <typename Scalar>
void test_ksum_product(Index n1, Index n2) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using Vec = Matrix<Scalar, Dynamic, 1>;
  using Sparse = SparseMatrix<Scalar>;
  using RowSparse = SparseMatrix<Scalar, RowMajor>;
  using Diag = DiagonalMatrix<Scalar, Dynamic>;
  using Id = internal::kron_identity_factor<Scalar, Dynamic, Dynamic>;

  const Mat A = Mat::Random(n1, n1);
  const Sparse S = random_sparse<Scalar>(n2);
  const Mat Sd = S;
  const Diag D(Vec::Random(n2));
  const Mat Dd = D.toDenseMatrix();

  auto K = makeKroneckerSum(A, S);
  STATIC_CHECK((std::is_same<decltype(K), KroneckerSum<Mat, Sparse>>::value));
  const Mat ref = reference_ksum<Scalar>(A, Sd);
  VERIFY_IS_EQUAL(K.rows(), n1 * n2);
  VERIFY_IS_EQUAL(K.cols(), n1 * n2);
  check_products(K, ref);
  check_products(makeKroneckerSum(S, A), reference_ksum<Scalar>(Sd, A));
  check_products(makeKroneckerSum(A, D), reference_ksum<Scalar>(A, Dd));
  check_products(makeKroneckerSum(D, Mat::Identity(n1, n1)), reference_ksum<Scalar>(Dd, Mat::Identity(n1, n1)));
  STATIC_CHECK((std::is_same<decltype(makeKroneckerSum(A, Mat::Identity(n2, n2))), KroneckerSum<Mat, Id>>::value));

  // Dense assignment and accumulation, and coefficient access.
  VERIFY_IS_APPROX(Mat(K), ref);
  Mat acc = Mat::Random(ref.rows(), ref.cols());
  const Mat acc0 = acc;
  acc += K;
  VERIFY_IS_APPROX(acc, (acc0 + ref).eval());
  acc -= K;
  VERIFY_IS_APPROX(acc, acc0);
  for (Index i = 0; i < ref.rows(); ++i)
    for (Index j = 0; j < ref.cols(); ++j) VERIFY_IS_APPROX(K.coeff(i, j), ref(i, j));

  // Sparse assignment stores the union of the two terms' patterns, compressed.
  const Mat storedA = Mat::Ones(n1, n1);
  const Mat storedS = (Sd.array() != Scalar(0)).template cast<Scalar>();  // sparseView() stores no zeros
  Sparse M;
  M = K;
  VERIFY(M.isCompressed());
  VERIFY_IS_EQUAL(M.nonZeros(), ksum_pattern_size(storedA, storedS));
  VERIFY_IS_APPROX(Mat(M), ref);
  RowSparse MR;
  MR = makeKroneckerSum(S, D);
  VERIFY_IS_APPROX(Mat(MR), reference_ksum<Scalar>(Sd, Dd));
  Sparse Macc = M;
  Macc += K;
  VERIFY_IS_APPROX(Mat(Macc), (2 * ref).eval());
  Macc -= K;
  VERIFY_IS_APPROX(Mat(Macc), ref);

  // The transposition family: (A (+) B)^T = A^T (+) B^T.
  check_products(K.transpose(), Mat(ref.transpose()));
  check_products(K.adjoint(), Mat(ref.adjoint()));
  check_products(K.conjugate(), Mat(ref.conjugate()));
  VERIFY_IS_APPROX(Mat(K.adjoint()), Mat(ref.adjoint()));
}

// Kronecker sums of three factors, nested on either side, and cross-nesting
// with KroneckerOperator.
template <typename Scalar>
void test_ksum_nested(Index n1, Index n2, Index n3) {
  using RealScalar = typename NumTraits<Scalar>::Real;
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using Vec = Matrix<Scalar, Dynamic, 1>;
  using Sparse = SparseMatrix<Scalar>;
  using Id = internal::kron_identity_factor<Scalar, Dynamic, Dynamic>;

  const Mat A = Mat::Random(n1, n1), C = Mat::Random(n3, n3);
  const Sparse B = random_sparse<Scalar>(n2);
  const Mat Bd = B;
  const Mat ref = reference_ksum<Scalar>(A, reference_ksum<Scalar>(Bd, C));

  auto KR = makeKroneckerSum(A, B, C);
  STATIC_CHECK((std::is_same<decltype(KR), KroneckerSum<Mat, KroneckerSum<Sparse, Mat>>>::value));
  const KroneckerSum<KroneckerSum<Mat, Sparse>, Mat> KL(makeKroneckerSum(A, B), C);
  check_products(KR, ref);
  check_products(KL, ref);
  VERIFY_IS_APPROX(Mat(KR), ref);
  VERIFY_IS_APPROX(Mat(KL), ref);
  Sparse M;
  M = KL;
  VERIFY_IS_APPROX(Mat(M), ref);
  for (Index t = 0; t < 10; ++t) {
    const Index i = internal::random<Index>(0, ref.rows() - 1), j = internal::random<Index>(0, ref.cols() - 1);
    VERIFY_IS_APPROX(KR.coeff(i, j), ref(i, j));
    VERIFY_IS_APPROX(KL.coeff(i, j), ref(i, j));
  }
  check_products(KR.transpose(), Mat(ref.transpose()));
  check_products(KL.adjoint(), Mat(ref.adjoint()));

  // I_p (x) (A (+) C): a Kronecker sum as a KroneckerOperator factor.
  const Index p = n2;
  const Mat Ad = A + RealScalar(4) * Mat::Identity(n1, n1), Cd = C + RealScalar(4) * Mat::Identity(n3, n3);
  auto IS = makeKroneckerOperator(Mat::Identity(p, p), makeKroneckerSum(Ad, Cd));
  STATIC_CHECK((std::is_same<decltype(IS), KroneckerOperator<Id, KroneckerSum<Mat, Mat>>>::value));
  const Mat refIS = reference_kron<Scalar>(Mat::Identity(p, p), reference_ksum<Scalar>(Ad, Cd));
  check_products(IS, refIS);
  Sparse MIS;
  MIS = IS;
  VERIFY_IS_APPROX(Mat(MIS), refIS);
  VERIFY_IS_APPROX(Mat(IS), refIS);
  const Vec b = Vec::Random(refIS.rows());
  VERIFY_IS_APPROX((refIS * IS.solve(b)).eval(), b);
  VERIFY_IS_APPROX(IS.determinant(), refIS.determinant());
  VERIFY_IS_APPROX((Mat(IS.inverse()) * refIS).eval(), Mat(Mat::Identity(refIS.rows(), refIS.cols())));
  check_products(IS.adjoint(), Mat(refIS.adjoint()));
  // (A (+) C) (x) I_p and (A (+) C) (x) B apply the sum from the right.
  check_products(makeKroneckerOperator(makeKroneckerSum(Ad, Cd), Mat::Identity(p, p)),
                 reference_kron<Scalar>(reference_ksum<Scalar>(Ad, Cd), Mat::Identity(p, p)));
  check_products(makeKroneckerOperator(makeKroneckerSum(Ad, Cd), B),
                 reference_kron<Scalar>(reference_ksum<Scalar>(Ad, Cd), Bd));

  // A (+) (B (x) C): a KroneckerOperator as a Kronecker-sum factor.
  auto SK = makeKroneckerSum(Ad, makeKroneckerOperator(B, Cd));
  const Mat refSK = reference_ksum<Scalar>(Ad, reference_kron<Scalar>(Bd, Cd));
  check_products(SK, refSK);
  VERIFY_IS_APPROX(Mat(SK), refSK);
  const Vec bs = Vec::Random(refSK.rows());
  VERIFY_IS_APPROX((refSK * SK.solve(bs)).eval(), bs);
}

// The Bartels-Stewart residual bound for (A_1 (+) ... (+) A_d) x = b,
// ||b - K x|| <= c (n_1 + ... + n_d) eps (||A_1||_F + ... + ||A_d||_F) ||x||:
// the Schur forms and the transforms are backward stable factor by factor, and
// the triangular substitution is backward stable in each shifted block.
template <typename Mat, typename Vec>
void check_ksum_residual(const Mat& ref, const Vec& x, const Vec& b, Index sizeSum,
                         typename NumTraits<typename Mat::Scalar>::Real normSum) {
  using RealScalar = typename NumTraits<typename Mat::Scalar>::Real;
  const RealScalar tol = RealScalar(4) * RealScalar(sizeSum) * NumTraits<RealScalar>::epsilon() * normSum * x.norm();
  const RealScalar residual = (b - ref * x).norm();
  VERIFY((numext::isfinite)(residual));
  VERIFY(residual <= tol);
}

// An exactly Hermitian positive definite matrix: R R^H alone need not be exactly
// Hermitian (an FMA complex product rounds the two triangles differently).
template <typename Mat>
Mat hermitian_spd(Index n) {
  using RealScalar = typename NumTraits<typename Mat::Scalar>::Real;
  const Mat R = Mat::Random(n, n);
  const Mat G = R * R.adjoint();
  return (G + G.adjoint()) * RealScalar(0.5) + Mat::Identity(n, n);
}

// The upper bidiagonal Jordan-type block lambda I + s N, far from normal for
// |s| >> |lambda|: its Schur form is itself.
template <typename Mat>
Mat jordan_block(Index n, const typename Mat::Scalar& lambda, const typename Mat::Scalar& s) {
  Mat J = lambda * Mat::Identity(n, n);
  J.diagonal(1).setConstant(s);
  return J;
}

template <typename Mat>
Mat hilbert(Index n) {
  Mat H(n, n);
  for (Index i = 0; i < n; ++i)
    for (Index j = 0; j < n; ++j) H(i, j) = typename Mat::Scalar(1) / typename Mat::Scalar(i + j + 1);
  return H;
}

template <typename Mat>
void check_ksum_solves(const Mat& A, const Mat& B) {
  using Vec = Matrix<typename Mat::Scalar, Dynamic, 1>;
  const Mat ref = reference_ksum<typename Mat::Scalar>(A, B);
  const auto K = makeKroneckerSum(A, B);
  const BartelsStewart<KroneckerSum<Mat, Mat>> solver(K);
  VERIFY_IS_EQUAL(solver.info(), Success);
  VERIFY_IS_EQUAL(solver.isHermitian(), A == A.adjoint() && B == B.adjoint());
  const Mat Bm = Mat::Random(ref.rows(), 3);
  const Mat X = solver.solve(Bm);
  for (Index j = 0; j < Bm.cols(); ++j)
    check_ksum_residual(ref, Vec(X.col(j)), Vec(Bm.col(j)), A.rows() + B.rows(), A.norm() + B.norm());
}

template <typename Scalar>
void test_ksum_solve(Index n1, Index n2) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using Vec = Matrix<Scalar, Dynamic, 1>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  // General factors (the Schur path) and Hermitian ones (fast diagonalization),
  // several right-hand sides each.
  const Mat A = Mat::Random(n1, n1) + RealScalar(3) * Mat::Identity(n1, n1);
  const Mat B = Mat::Random(n2, n2) + RealScalar(3) * Mat::Identity(n2, n2);
  const Mat H1 = hermitian_spd<Mat>(n1), H2 = hermitian_spd<Mat>(n2);
  check_ksum_solves(A, B);
  check_ksum_solves(H1, H2);
  check_ksum_solves(H1, B);  // a mixed pair takes the Schur path
  const Vec b = Vec::Random(n1 * n2);
  check_ksum_residual(reference_ksum<Scalar>(A, B), Vec(makeKroneckerSum(A, B).solve(b)), b, n1 + n2,
                      A.norm() + B.norm());

  // The residual bound holds whatever the conditioning: nearly singular
  // (eigenvalue sums down to delta) on both paths, far from normal, and with
  // factor norms 10^6 apart.
  const RealScalar delta = numext::sqrt(NumTraits<RealScalar>::epsilon());
  check_ksum_solves(A, Mat(delta * Mat::Identity(n1, n1) - A.transpose()));
  check_ksum_solves(hilbert<Mat>(n1), Mat(delta * Mat::Identity(n1, n1) - hilbert<Mat>(n1)));
  check_ksum_solves(jordan_block<Mat>(n1, Scalar(1), Scalar(1000)), jordan_block<Mat>(n2, Scalar(1), Scalar(-1000)));
  check_ksum_solves(Mat(RealScalar(1e6) * A), B);
}

// Three factors through the nested sum, either path, and non-finite factors,
// also as a KroneckerOperator factor; separate from test_ksum_solve to keep
// each part's compile small.
template <typename Scalar>
void test_ksum_solve_nested(Index n1, Index n2) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using Vec = Matrix<Scalar, Dynamic, 1>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  const Mat A = Mat::Random(n1, n1) + RealScalar(3) * Mat::Identity(n1, n1);
  const Mat B = Mat::Random(n2, n2) + RealScalar(3) * Mat::Identity(n2, n2);
  const Mat H1 = hermitian_spd<Mat>(n1), H2 = hermitian_spd<Mat>(n2);
  const Vec b = Vec::Random(n1 * n2);
  const Index n3 = 3;
  const Mat C = jordan_block<Mat>(n3, Scalar(2), Scalar(100)), H3 = hermitian_spd<Mat>(n3);
  const Vec b3 = Vec::Random(n1 * n2 * n3);
  const Mat J1 = jordan_block<Mat>(n1, Scalar(1), Scalar(100)), J2 = jordan_block<Mat>(n2, Scalar(-1), Scalar(-100));
  check_ksum_residual(reference_ksum<Scalar>(J1, reference_ksum<Scalar>(J2, C)),
                      Vec(makeKroneckerSum(J1, J2, C).solve(b3)), b3, n1 + n2 + n3, J1.norm() + J2.norm() + C.norm());
  const KroneckerSum<KroneckerSum<Mat, Mat>, Mat> KH3(makeKroneckerSum(H1, H2), H3);
  const BartelsStewart<KroneckerSum<KroneckerSum<Mat, Mat>, Mat>> h3solver(KH3);
  VERIFY(h3solver.isHermitian());
  check_ksum_residual(reference_ksum<Scalar>(reference_ksum<Scalar>(H1, H2), H3), Vec(h3solver.solve(b3)), b3,
                      n1 + n2 + n3, H1.norm() + H2.norm() + H3.norm());

  // A non-finite factor is rejected, and every solve through it returns NaN.
  Mat Abad = A;
  Abad(0, 0) = Scalar(NumTraits<RealScalar>::quiet_NaN());
  const BartelsStewart<KroneckerSum<Mat, Mat>> bad(makeKroneckerSum(Abad, B));
  VERIFY_IS_EQUAL(bad.info(), InvalidInput);
  VERIFY((Vec(bad.solve(b)).array().isNaN()).all());
  Mat Hbad = H1;
  Hbad(0, 0) = Scalar(NumTraits<RealScalar>::infinity());  // still exactly Hermitian
  VERIFY((Vec(makeKroneckerSum(Hbad, H2).solve(b)).array().isNaN()).all());
  const Vec bn = Vec::Random(n2 * n1 * n2);
  VERIFY(!Vec(makeKroneckerOperator(Mat::Identity(n2, n2), makeKroneckerSum(Abad, B)).solve(bn)).allFinite());
}

// A real n x n matrix far from normal whose eigenvalues are alpha +- i beta_j:
// Q (blockdiag([alpha, beta_j; -beta_j, alpha]) + U) Q^T with U strictly upper
// triangular and Q orthogonal, so the real Schur form has 2x2 blocks coupled by
// a nonzero upper part (plus a 1x1 block alpha when n is odd).
template <typename Mat>
Mat rotational_nonnormal(Index n, typename Mat::Scalar alpha) {
  using RealScalar = typename Mat::Scalar;
  Mat B = alpha * Mat::Identity(n, n);
  for (Index i = 0; i + 1 < n; i += 2) {
    const RealScalar beta = RealScalar(1) + internal::random<RealScalar>(RealScalar(0), RealScalar(1));
    B(i, i + 1) = beta;
    B(i + 1, i) = -beta;
  }
  for (Index j = 0; j < n; ++j)
    for (Index i = 0; i < j; ++i)
      if (!(i % 2 == 0 && j == i + 1)) B(i, j) = RealScalar(2) * internal::random<RealScalar>();
  const Mat Q = HouseholderQR<Mat>(Mat::Random(n, n)).householderQ();
  return Q * B * Q.transpose();
}

// Whether the real Schur form of A has a 2x2 block, i.e. the case under test.
template <typename Mat>
bool has_complex_pair(const Mat& A) {
  const Mat T = RealSchur<Mat>(A).matrixT();
  return (T.diagonal(-1).array() != typename Mat::Scalar(0)).any();
}

// Real factors with complex conjugate eigenvalue pairs take the real Schur path.
// Up to four factors couple the 2x2 blocks in systems up to 16 wide, with the
// updates across blocks running at every inner factor; five fall back to the
// complex Schur path.
template <typename RealScalar>
void test_ksum_real_schur() {
  using Mat = Matrix<RealScalar, Dynamic, Dynamic>;
  using Vec = Matrix<RealScalar, Dynamic, 1>;
  const Mat A = rotational_nonnormal<Mat>(4, RealScalar(1)), B = rotational_nonnormal<Mat>(3, RealScalar(2));
  const Mat C = rotational_nonnormal<Mat>(2, RealScalar(-0.5)), D = rotational_nonnormal<Mat>(3, RealScalar(1.5));
  const Mat E = rotational_nonnormal<Mat>(2, RealScalar(1));
  VERIFY(has_complex_pair(A) && has_complex_pair(B) && has_complex_pair(C) && has_complex_pair(D));
  check_ksum_solves(A, B);
  check_ksum_solves(B, C);

  const Mat ABC = reference_ksum<RealScalar>(A, reference_ksum<RealScalar>(B, C));
  const Vec b3 = Vec::Random(ABC.rows());
  check_ksum_residual(ABC, Vec(makeKroneckerSum(A, B, C).solve(b3)), b3, 9, A.norm() + B.norm() + C.norm());

  const Mat ABCD = reference_ksum<RealScalar>(A, reference_ksum<RealScalar>(B, reference_ksum<RealScalar>(C, D)));
  const auto K4 = makeKroneckerSum(A, B, C, D);
  const BartelsStewart<std::decay_t<decltype(K4)>> solver4(K4);
  VERIFY(!solver4.isHermitian());
  const Mat B4 = Mat::Random(ABCD.rows(), 2);
  const Mat X4 = solver4.solve(B4);
  for (Index j = 0; j < B4.cols(); ++j)
    check_ksum_residual(ABCD, Vec(X4.col(j)), Vec(B4.col(j)), 12, A.norm() + B.norm() + C.norm() + D.norm());

  const Mat ABCDE = reference_ksum<RealScalar>(
      A, reference_ksum<RealScalar>(B, reference_ksum<RealScalar>(C, reference_ksum<RealScalar>(D, E))));
  const Vec b5 = Vec::Random(ABCDE.rows());
  check_ksum_residual(ABCDE, Vec(makeKroneckerSum(A, B, C, D, E).solve(b5)), b5, 14,
                      A.norm() + B.norm() + C.norm() + D.norm() + E.norm());

  // Nearly singular coupled systems 8 and 16 wide: A (+) (-A^T) (+) delta C has
  // eigenvalue sums delta mu_j, A (+) (-A^T) (+) C (+) (delta I - C^T) sums delta.
  const RealScalar delta = numext::sqrt(NumTraits<RealScalar>::epsilon());
  const Mat At = -A.transpose(), Cd = delta * C, Ct = delta * Mat::Identity(2, 2) - C.transpose();
  const Mat S3 = reference_ksum<RealScalar>(A, reference_ksum<RealScalar>(At, Cd));
  const Vec s3 = Vec::Random(S3.rows());
  check_ksum_residual(S3, Vec(makeKroneckerSum(A, At, Cd).solve(s3)), s3, 10, 2 * A.norm() + Cd.norm());
  const Mat S4 = reference_ksum<RealScalar>(A, reference_ksum<RealScalar>(At, reference_ksum<RealScalar>(C, Ct)));
  const Vec s4 = Vec::Random(S4.rows());
  check_ksum_residual(S4, Vec(makeKroneckerSum(A, At, C, Ct).solve(s4)), s4, 12, 2 * A.norm() + C.norm() + Ct.norm());
}

// The fast path is gated on exact Hermitian symmetry: complex symmetric
// factors, and Hermitian off-diagonals with a non-real diagonal, take the Schur
// path and still solve.
template <typename Scalar>
void test_ksum_hermitian_gate(Index n1, Index n2) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  const Mat M = Mat::Random(n1, n1);
  const Mat Csym = M + M.transpose() + Scalar(4) * Mat::Identity(n1, n1);
  Mat Hdiag = hermitian_spd<Mat>(n1);
  Hdiag(0, 0) += Scalar(0, 1);
  const Mat H2 = hermitian_spd<Mat>(n2);
  VERIFY(Csym != Csym.adjoint());
  check_ksum_solves(Csym, H2);
  check_ksum_solves(Hdiag, H2);
}

// The eigenvalues are the pairwise sums, in Kronecker order.
template <typename Scalar>
void test_ksum_eigenvalues(Index n1, Index n2) {
  using RealScalar = typename NumTraits<Scalar>::Real;
  using Complex = std::complex<RealScalar>;
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using CMat = Matrix<Complex, Dynamic, Dynamic>;
  using CVec = Matrix<Complex, Dynamic, 1>;
  using RealVec = Matrix<RealScalar, Dynamic, 1>;

  // Entry (i1 n2 + i2) n1 + i3 of H1 (+) (H2 (+) H1) is l1[i1] + l2[i2] + l1[i3],
  // with each factor's eigenvalues in ComplexEigenSolver's order.
  const Mat Ar = Mat::Random(n1, n1), Br = Mat::Random(n2, n2);
  const Mat H1 = Ar + Ar.adjoint(), H2 = Br + Br.adjoint();
  auto K = makeKroneckerSum(H1, makeKroneckerSum(H2, H1));
  const CVec l1 = ComplexEigenSolver<CMat>(H1.template cast<Complex>(), false).eigenvalues();
  const CVec l2 = ComplexEigenSolver<CMat>(H2.template cast<Complex>(), false).eigenvalues();
  const CVec lambda = K.eigenvalues();
  for (Index i1 = 0; i1 < n1; ++i1)
    for (Index i2 = 0; i2 < n2; ++i2)
      for (Index i3 = 0; i3 < n1; ++i3) VERIFY_IS_EQUAL(lambda[(i1 * n2 + i2) * n1 + i3], l1[i1] + (l2[i2] + l1[i3]));

  // As a set they are the spectrum of the dense matrix.
  const Mat ref = reference_ksum<Scalar>(H1, reference_ksum<Scalar>(H2, H1));
  RealVec sorted = lambda.real();
  std::sort(sorted.data(), sorted.data() + sorted.size());
  const RealVec expected = SelfAdjointEigenSolver<Mat>(ref).eigenvalues();
  const RealScalar tol = RealScalar(8) * RealScalar(ref.rows()) * NumTraits<RealScalar>::epsilon() * ref.norm();
  VERIFY((sorted - expected).cwiseAbs().maxCoeff() <= tol);

  // A non-normal pair: entry i1 n2 + i2 is the sum, in the factors' order.
  const Mat A = Mat::Random(n1, n1), B = Mat::Random(n2, n2);
  const CVec a = ComplexEigenSolver<CMat>(A.template cast<Complex>(), false).eigenvalues();
  const CVec c = ComplexEigenSolver<CMat>(B.template cast<Complex>(), false).eigenvalues();
  const CVec mu = makeKroneckerSum(A, B).eigenvalues();
  for (Index i1 = 0; i1 < n1; ++i1)
    for (Index i2 = 0; i2 < n2; ++i2) VERIFY_IS_EQUAL(mu[i1 * n2 + i2], a[i1] + c[i2]);

  // A Kronecker-product factor contributes the products of its factors'
  // eigenvalues, in its own Kronecker order: entry (i1 n2 + i2) n1 + i3 of
  // A (+) (B (x) A) is a[i1] + c[i2] a[i3].
  const CVec nu = makeKroneckerSum(A, makeKroneckerOperator(B, A)).eigenvalues();
  CVec expectedNu(n1 * n2 * n1);
  for (Index i1 = 0; i1 < n1; ++i1)
    for (Index i2 = 0; i2 < n2; ++i2)
      for (Index i3 = 0; i3 < n1; ++i3) expectedNu[(i1 * n2 + i2) * n1 + i3] = a[i1] + c[i2] * a[i3];
  VERIFY_IS_APPROX(nu, expectedNu);
}

// The finite-difference use case: an implicit Euler step of the heat equation
// u_t = (d_yy + d_xx) u on an n1 x n2 grid with D = tridiag(-1, 2, -1) the SPD
// negated second difference, (I + tau (Dy (+) Dx)) u = b, against SparseLU of
// the materialized matrix; and matrix-free CG on Dy (+) Dx.
void test_ksum_finite_difference(Index n1, Index n2) {
  using Sparse = SparseMatrix<double>;
  using Vec = VectorXd;
  const double tau = 0.25, eps = NumTraits<double>::epsilon();
  const Sparse Dx = second_difference(n2), Dy = second_difference(n1);
  Sparse Iy(n1, n1);
  Iy.setIdentity();
  auto M = makeKroneckerSum(Sparse(Iy + tau * Dy), Sparse(tau * Dx));
  Sparse Ms;
  Ms = M;
  const BartelsStewart<decltype(M)> step(M);
  VERIFY(step.isHermitian());
  const Vec b = Vec::Random(n1 * n2);
  const Vec u = step.solve(b);
  SparseLU<Sparse> lu(Ms);
  const Vec uLU = lu.solve(b);
  // The spectrum lies in (1, 1 + 8 tau), so cond(M) < 1 + 8 tau and both
  // solutions are within cond * (backward error) of the exact one.
  const double kappa = 1 + 8 * tau, normM = 1 + 8 * tau;
  VERIFY((u - uLU).norm() <= 2 * kappa * 4 * double(n1 + n2) * eps * uLU.norm());
  const Vec u2 = step.solve(u);  // the decompositions are reused
  VERIFY((Ms * u2 - u).norm() <= 4 * double(n1 + n2) * eps * normM * u2.norm());

  auto L = makeKroneckerSum(Dy, Dx);
  ConjugateGradient<decltype(L), Lower | Upper, IdentityPreconditioner> cg(L);
  const Vec x = cg.solve(b);
  VERIFY_IS_EQUAL(cg.info(), Success);
  Sparse Ls;
  Ls = L;
  VERIFY_IS_APPROX((Ls * x).eval(), b);
}

// Fixed-size factors keep the compile-time size.
void test_ksum_fixed() {
  const Matrix3d A = Matrix3d::Random();
  const Matrix2d B = Matrix2d::Random();
  const KroneckerSum<Matrix3d, Matrix2d> K(A, B);
  STATIC_CHECK(int(KroneckerSum<Matrix3d, Matrix2d>::RowsAtCompileTime) == 6);
  const MatrixXd ref = reference_ksum<double>(A, B);
  const Matrix<double, 6, 1> x = Matrix<double, 6, 1>::Random();
  const Matrix<double, 6, 1> y = K * x;
  VERIFY_IS_APPROX(VectorXd(y), VectorXd(ref * x));
  const Matrix<double, 6, 6> Kd = K;
  VERIFY_IS_APPROX(MatrixXd(Kd), ref);
}

EIGEN_DECLARE_TEST(structured_kronecker_sum) {
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1((test_ksum_product<double>(3, 4)));
    CALL_SUBTEST_1((test_ksum_product<double>(1, 1)));
    CALL_SUBTEST_1(test_ksum_fixed());
    CALL_SUBTEST_5((test_ksum_product<float>(4, 3)));
    CALL_SUBTEST_6((test_ksum_product<std::complex<double>>(3, 3)));
    CALL_SUBTEST_2((test_ksum_nested<double>(2, 3, 4)));
    CALL_SUBTEST_7((test_ksum_nested<std::complex<double>>(3, 2, 2)));
    CALL_SUBTEST_3((test_ksum_solve<double>(5, 7)));
    CALL_SUBTEST_3((test_ksum_solve<double>(1, 1)));
    CALL_SUBTEST_3((test_ksum_solve_nested<double>(5, 7)));
    CALL_SUBTEST_3((test_ksum_real_schur<double>()));
    CALL_SUBTEST_8((test_ksum_solve<float>(4, 5)));
    CALL_SUBTEST_12((test_ksum_solve_nested<float>(4, 5)));
    CALL_SUBTEST_10((test_ksum_real_schur<float>()));
    CALL_SUBTEST_9((test_ksum_solve<std::complex<double>>(4, 6)));
    CALL_SUBTEST_9((test_ksum_solve_nested<std::complex<double>>(4, 6)));
    CALL_SUBTEST_9((test_ksum_hermitian_gate<std::complex<double>>(4, 3)));
    CALL_SUBTEST_11((test_ksum_hermitian_gate<std::complex<float>>(3, 4)));
    CALL_SUBTEST_4((test_ksum_eigenvalues<double>(3, 4)));
    CALL_SUBTEST_4((test_ksum_eigenvalues<std::complex<float>>(3, 2)));
    CALL_SUBTEST_4(test_ksum_finite_difference(12, 9));
  }
}
