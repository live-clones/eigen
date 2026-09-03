// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Eigen Authors
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// Tests for gpu::LDLT (cusolverDn<t>sytrf / cusolverDnXsytrs / cusolverDn<t>sytri)
// and the one-shot d_A.ldlt().solve(d_B) expression, for float, double,
// complex<float>, complex<double>, Lower and Upper. Complex inputs are
// complex-symmetric (A = A^T), the matrix class these routines factor.

#define EIGEN_USE_GPU
#include "main.h"
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/QR>
#include <unsupported/Eigen/GPU>

#include "./gpu_test_helpers.h"

using namespace Eigen;

template <typename Scalar>
using TestMatrix = Matrix<Scalar, Dynamic, Dynamic>;

// gpu::LDLT makes complex users spell out that sytrf is complex-symmetric.
template <typename Scalar>
constexpr int kSymmetricBit = NumTraits<Scalar>::IsComplex ? Symmetric : 0;

// Both generators below keep kappa_2(A) <= 3, which the CPU cross-check relies on.
template <typename Scalar>
constexpr typename NumTraits<Scalar>::Real kConditionBound = 3;

// Every residual bound below is 10 * n * eps: the normwise backward error of a
// Bunch-Kaufman solve is O(n u) with a modest constant for matrices without
// element growth (Higham, Accuracy and Stability of Numerical Algorithms, 2nd
// ed., section 11.1), and the Frobenius norms in backward_error() only loosen it.

// A = D + E with D = diag(+n, -n, +n, ...) and E = M + M^T, M random in [-1, 1]:
// indefinite, and for the sizes used here ||E||_2 ~ 1.6 sqrt(n) < n/2, so
// sigma_min(A) > n/2 (real or complex-symmetric). |a_kk| dominates its column,
// so every Bunch-Kaufman step is a 1x1 pivot without interchange.
template <typename MatrixType>
MatrixType make_symmetric_indefinite(Index n) {
  using Scalar = typename MatrixType::Scalar;
  const MatrixType M = MatrixType::Random(n, n);
  MatrixType A = M + M.transpose();
  for (Index i = 0; i < n; ++i) A(i, i) += (i % 2 == 0) ? Scalar(n) : Scalar(-n);
  return A;
}

// A = J + delta (M + M^T), J the anti-identity (eigenvalues +-1), delta =
// 1/(8 sqrt(n)) so the perturbation's 2-norm stays below ~0.25. The O(delta)
// diagonal loses to the unit entry in the mirrored row, so Bunch-Kaufman takes
// a 2x2 pivot with a far interchange (ipiv < 0) at every step: the path the
// widened negative pivots and the 2x2 blocks in sytrs/sytri depend on.
template <typename MatrixType>
MatrixType make_antidiagonal_indefinite(Index n) {
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;
  const MatrixType M = MatrixType::Random(n, n);
  const RealScalar delta = RealScalar(1) / (RealScalar(8) * numext::sqrt(RealScalar(n)));
  MatrixType A = delta * (M + M.transpose());
  for (Index i = 0; i < n; ++i) A(i, n - 1 - i) += Scalar(1);
  return A;
}

// Either generator above, for the tests that run against both.
template <typename Scalar>
using MatrixGenerator = TestMatrix<Scalar> (*)(Index);

// Normalized by ||A|| ||X|| rather than ||B|| to be condition-number agnostic.
template <typename MatrixType>
typename NumTraits<typename MatrixType::Scalar>::Real backward_error(const MatrixType& A, const MatrixType& X,
                                                                     const MatrixType& B) {
  return (A * X - B).norm() / (A.norm() * X.norm());
}

// Mirror the UpLo triangle onto the other one with a plain transpose, not an
// adjoint: for complex Scalar the matrices here are complex-symmetric.
template <int UpLo, typename MatrixType>
MatrixType symmetric_from_triangle(const MatrixType& t) {
  constexpr int Tri = UpLo & (Lower | Upper);
  MatrixType full = t.template triangularView<Tri>();
  const MatrixType mirrored = full.transpose();
  full.template triangularView<Tri == Lower ? StrictlyUpper : StrictlyLower>() = mirrored;
  return full;
}

// The test matrices are genuinely indefinite: Cholesky rejects them and the
// spectrum has both signs.
void test_matrices_are_indefinite() {
  for (const MatrixXd& A : {make_symmetric_indefinite<MatrixXd>(64), make_antidiagonal_indefinite<MatrixXd>(64)}) {
    VERIFY_IS_EQUAL(LLT<MatrixXd>(A).info(), NumericalIssue);
    const VectorXd ev = SelfAdjointEigenSolver<MatrixXd>(A, EigenvaluesOnly).eigenvalues();
    VERIFY(ev.minCoeff() < 0 && ev.maxCoeff() > 0);
    VERIFY(ev.cwiseAbs().maxCoeff() < kConditionBound<double> * ev.cwiseAbs().minCoeff());
  }
}

// ---- Factorization + solve (host path) --------------------------------------

template <typename Scalar, int UpLo>
void test_sytrf_sytrs(Index n, Index nrhs, MatrixGenerator<Scalar> make) {
  using MatrixType = TestMatrix<Scalar>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  const MatrixType A = make(n);
  const MatrixType B = MatrixType::Random(n, nrhs);
  const RealScalar tol = RealScalar(10) * RealScalar(n) * NumTraits<Scalar>::epsilon();

  gpu::LDLT<Scalar, UpLo> ldlt(A);
  VERIFY_IS_EQUAL(ldlt.info(), Success);
  VERIFY_IS_EQUAL(ldlt.rows(), n);
  VERIFY_IS_EQUAL(ldlt.cols(), n);

  const MatrixType X = ldlt.solve(B);
  VERIFY(backward_error(A, X, B) < tol);

  // Two backward-stable solutions differ by at most kappa * (eta_gpu + eta_cpu) * ||X||.
  const MatrixType X_ref = PartialPivLU<MatrixType>(A).solve(B);
  VERIFY((X - X_ref).norm() < kConditionBound<Scalar> * RealScalar(2) * tol * X_ref.norm());
}

// Q diag(+-10^(-6 i/(n-1))) Q^T: kappa_2 = 1e6 while staying nonsingular, so
// only the backward error is meaningful here. With Q unitary, Q^T conj(Q) = I,
// so the singular values are the |d_i| for complex Scalar as well.
template <typename Scalar, int UpLo>
void test_ill_conditioned(Index n, Index nrhs) {
  using MatrixType = TestMatrix<Scalar>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  Matrix<Scalar, Dynamic, 1> d(n);
  for (Index i = 0; i < n; ++i) {
    const RealScalar magnitude = numext::pow(RealScalar(10), RealScalar(-6) * RealScalar(i) / RealScalar(n - 1));
    d(i) = (i % 3 == 0) ? Scalar(-magnitude) : Scalar(magnitude);
  }
  const MatrixType Q = HouseholderQR<MatrixType>(MatrixType::Random(n, n)).householderQ();
  const MatrixType A = Q * d.asDiagonal() * Q.transpose();
  const MatrixType B = MatrixType::Random(n, nrhs);

  gpu::LDLT<Scalar, UpLo> ldlt(A);
  VERIFY_IS_EQUAL(ldlt.info(), Success);
  const MatrixType X = ldlt.solve(B);
  VERIFY(backward_error(A, X, B) < RealScalar(10) * RealScalar(n) * NumTraits<Scalar>::epsilon());
}

// Only the UpLo triangle is read: NaN in the other one must not reach the
// solution, on the cached and the one-shot path.
template <typename Scalar, int UpLo>
void test_unreferenced_triangle(Index n) {
  using MatrixType = TestMatrix<Scalar>;
  using RealScalar = typename NumTraits<Scalar>::Real;
  constexpr int Tri = UpLo & (Lower | Upper);

  const MatrixType A = make_antidiagonal_indefinite<MatrixType>(n);
  const MatrixType B = MatrixType::Random(n, 3);
  const RealScalar tol = RealScalar(10) * RealScalar(n) * NumTraits<Scalar>::epsilon();
  MatrixType A_masked = A;
  A_masked.template triangularView<Tri == Lower ? StrictlyUpper : StrictlyLower>().setConstant(
      Scalar(NumTraits<RealScalar>::quiet_NaN()));

  gpu::LDLT<Scalar, UpLo> ldlt(A_masked);
  VERIFY_IS_EQUAL(ldlt.info(), Success);
  VERIFY(backward_error(A, ldlt.solve(B), B) < tol);

  auto d_A = gpu::DeviceMatrix<Scalar>::fromHost(A_masked);
  auto d_B = gpu::DeviceMatrix<Scalar>::fromHost(B);
  gpu::DeviceMatrix<Scalar> d_X = d_A.template ldlt<UpLo>().solve(d_B);
  VERIFY(backward_error(A, d_X.toHost(), B) < tol);
}

// Multiple solves reuse the device-resident factor.
template <typename Scalar, int UpLo>
void test_multiple_solves(Index n) {
  using MatrixType = TestMatrix<Scalar>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  const MatrixType A = make_symmetric_indefinite<MatrixType>(n);
  const RealScalar tol = RealScalar(10) * RealScalar(n) * NumTraits<Scalar>::epsilon();
  gpu::LDLT<Scalar, UpLo> ldlt(A);
  VERIFY_IS_EQUAL(ldlt.info(), Success);

  for (int k = 0; k < 5; ++k) {
    const MatrixType B = MatrixType::Random(n, 3);
    const MatrixType X = ldlt.solve(B);
    VERIFY(backward_error(A, X, B) < tol);
  }
}

// ---- Empty sizes ----------------------------------------------------------------

template <typename Scalar, int UpLo>
void test_empty() {
  using MatrixType = TestMatrix<Scalar>;

  gpu::LDLT<Scalar, UpLo> ldlt_empty(MatrixType(0, 0));
  VERIFY_IS_EQUAL(ldlt_empty.info(), Success);
  VERIFY_IS_EQUAL(ldlt_empty.solve(MatrixType(0, 3)).cols(), 3);
  VERIFY_IS_EQUAL(ldlt_empty.solve(gpu::DeviceMatrix<Scalar>(0, 2)).cols(), 2);
  VERIFY(ldlt_empty.inverse().empty());

  const MatrixType A = make_symmetric_indefinite<MatrixType>(8);
  gpu::LDLT<Scalar, UpLo> ldlt(A);
  VERIFY_IS_EQUAL(ldlt.info(), Success);
  const MatrixType X = ldlt.solve(MatrixType(8, 0));
  VERIFY_IS_EQUAL(X.rows(), 8);
  VERIFY_IS_EQUAL(X.cols(), 0);
  gpu::DeviceMatrix<Scalar> d_X = ldlt.solve(gpu::DeviceMatrix<Scalar>(8, 0));
  VERIFY_IS_EQUAL(d_X.rows(), 8);
  VERIFY_IS_EQUAL(d_X.cols(), 0);
}

// ---- Singular matrix detection ----------------------------------------------

void test_singular() {
  const MatrixXd A = MatrixXd::Zero(8, 8);
  gpu::LDLT<double> ldlt(A);
  VERIFY_IS_EQUAL(ldlt.info(), NumericalIssue);
}

// solve(DeviceMatrix) must not silently return garbage when the factorization
// failed: it must sync the info word and assert just like solve(MatrixBase).
void test_singular_device_solve_asserts() {
  const MatrixXd A = MatrixXd::Zero(8, 8);
  const MatrixXd B = MatrixXd::Random(8, 4);
  gpu::LDLT<double> ldlt(A);
  VERIFY_IS_EQUAL(ldlt.info(), NumericalIssue);
  auto d_B = gpu::DeviceMatrix<double>::fromHost(B);
  VERIFY_RAISES_ASSERT(ldlt.solve(d_B));
  VERIFY_RAISES_ASSERT(ldlt.inverse());
}

// ---- DeviceMatrix-native API --------------------------------------------------

template <typename Scalar, int UpLo>
void test_device_matrix_solve(Index n, Index nrhs) {
  using MatrixType = TestMatrix<Scalar>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  const MatrixType A = make_symmetric_indefinite<MatrixType>(n);
  const MatrixType B = MatrixType::Random(n, nrhs);

  auto d_A = gpu::DeviceMatrix<Scalar>::fromHost(A);
  auto d_B = gpu::DeviceMatrix<Scalar>::fromHost(B);

  gpu::LDLT<Scalar, UpLo> ldlt;
  ldlt.compute(d_A);
  VERIFY_IS_EQUAL(ldlt.info(), Success);

  gpu::DeviceMatrix<Scalar> d_X = ldlt.solve(d_B);
  const MatrixType X = d_X.toHost();
  VERIFY(backward_error(A, X, B) < RealScalar(10) * RealScalar(n) * NumTraits<Scalar>::epsilon());
}

template <typename Scalar, int UpLo>
void test_device_matrix_move_compute(Index n) {
  using MatrixType = TestMatrix<Scalar>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  const MatrixType A = make_symmetric_indefinite<MatrixType>(n);
  const MatrixType B = MatrixType::Random(n, 1);

  auto d_A = gpu::DeviceMatrix<Scalar>::fromHost(A);
  gpu::LDLT<Scalar, UpLo> ldlt;
  ldlt.compute(std::move(d_A));
  VERIFY_IS_EQUAL(ldlt.info(), Success);
  VERIFY(d_A.empty());

  const MatrixType X = ldlt.solve(B);
  VERIFY(backward_error(A, X, B) < RealScalar(10) * RealScalar(n) * NumTraits<Scalar>::epsilon());
}

// compute → solve → solve again with the result as RHS → toHost; the final
// toHost() is the only sync point.
template <typename Scalar, int UpLo>
void test_chaining(Index n) {
  using MatrixType = TestMatrix<Scalar>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  const MatrixType A = make_antidiagonal_indefinite<MatrixType>(n);
  const MatrixType B = MatrixType::Random(n, 3);
  const RealScalar tol = RealScalar(10) * RealScalar(n) * NumTraits<Scalar>::epsilon();

  auto d_A = gpu::DeviceMatrix<Scalar>::fromHost(A);
  auto d_B = gpu::DeviceMatrix<Scalar>::fromHost(B);

  gpu::LDLT<Scalar, UpLo> ldlt;
  ldlt.compute(d_A);
  VERIFY_IS_EQUAL(ldlt.info(), Success);

  gpu::DeviceMatrix<Scalar> d_X = ldlt.solve(d_B);
  gpu::DeviceMatrix<Scalar> d_Y = ldlt.solve(d_X);
  const MatrixType X = d_X.toHost();
  const MatrixType Y = d_Y.toHost();

  VERIFY(backward_error(A, X, B) < tol);
  VERIFY(backward_error(A, Y, X) < tol);
}

// ---- Context binding + in-place (rvalue) solve --------------------------------

template <typename Scalar, int UpLo>
void test_context_bound_solver(Index n, Index nrhs) {
  using MatrixType = TestMatrix<Scalar>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  const MatrixType A = make_symmetric_indefinite<MatrixType>(n);
  const MatrixType B = MatrixType::Random(n, nrhs);
  const RealScalar tol = RealScalar(10) * RealScalar(n) * NumTraits<Scalar>::epsilon();

  gpu::Context ctx;
  auto d_A = gpu::DeviceMatrix<Scalar>::fromHost(A, ctx.stream());
  gpu::LDLT<Scalar, UpLo> ldlt(ctx, d_A);
  VERIFY_IS_EQUAL(ldlt.info(), Success);
  VERIFY(ldlt.stream() == ctx.stream());

  auto d_B = gpu::DeviceMatrix<Scalar>::fromHost(B, ctx.stream());
  gpu::DeviceMatrix<Scalar> d_X = ldlt.solve(d_B);
  VERIFY(backward_error(A, d_X.toHost(), B) < tol);

  // In-place rvalue solve: consumes the RHS, no copy/allocation.
  gpu::DeviceMatrix<Scalar> d_X2 = ldlt.solve(std::move(d_B));
  VERIFY(backward_error(A, d_X2.toHost(), B) < tol);
  VERIFY(d_B.data() == nullptr);
}

// ---- Non-plain host input ---------------------------------------------------

// compute() binds plain contiguous column-major input in place through Ref and
// evaluates anything else into a temporary. A symmetric matrix is byte-identical
// in row-major storage, so that layout cannot detect a mistake; a block whose
// outerStride() differs from rows() can.
template <typename Scalar, int UpLo>
void test_non_plain_input(Index n) {
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  const MatrixType A = make_symmetric_indefinite<MatrixType>(n);
  const MatrixType B = MatrixType::Random(n, 3);
  const RealScalar tol = RealScalar(10) * RealScalar(n) * NumTraits<Scalar>::epsilon();

  MatrixType padded = MatrixType::Random(n + 3, n + 5);
  padded.block(2, 1, n, n) = A;
  gpu::LDLT<Scalar, UpLo> ldlt_block(padded.block(2, 1, n, n));
  VERIFY_IS_EQUAL(ldlt_block.info(), Success);
  VERIFY(backward_error(A, ldlt_block.solve(B), B) < tol);

  // Unevaluated expression; 2*A is still symmetric indefinite.
  const MatrixType A2 = Scalar(2) * A;
  gpu::LDLT<Scalar, UpLo> ldlt_expr(Scalar(2) * A);
  VERIFY_IS_EQUAL(ldlt_expr.info(), Success);
  VERIFY(backward_error(A2, ldlt_expr.solve(B), B) < tol);

  gpu::LDLT<Scalar, UpLo> ldlt_array(A.array());
  VERIFY_IS_EQUAL(ldlt_array.info(), Success);
  VERIFY(backward_error(A, ldlt_array.solve(B), B) < tol);

  // Strided right-hand side: solve() binds B through Ref as well.
  MatrixType padded_B = MatrixType::Random(n + 2, B.cols() + 4);
  padded_B.block(1, 3, n, B.cols()) = B;
  gpu::LDLT<Scalar, UpLo> ldlt(A);
  VERIFY_IS_EQUAL(ldlt.info(), Success);
  VERIFY(backward_error(A, ldlt.solve(padded_B.block(1, 3, n, B.cols())), B) < tol);
}

// ---- One-shot expression: d_X = d_A.ldlt().solve(d_B) ------------------------

template <typename Scalar, int UpLo>
void test_solve_expr(Index n, Index nrhs, MatrixGenerator<Scalar> make) {
  using MatrixType = TestMatrix<Scalar>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  const MatrixType A = make(n);
  const MatrixType B = MatrixType::Random(n, nrhs);
  const RealScalar tol = RealScalar(10) * RealScalar(n) * NumTraits<Scalar>::epsilon();

  auto d_A = gpu::DeviceMatrix<Scalar>::fromHost(A);
  auto d_B = gpu::DeviceMatrix<Scalar>::fromHost(B);

  gpu::DeviceMatrix<Scalar> d_X;
  d_X = d_A.template ldlt<UpLo>().solve(d_B);
  VERIFY(backward_error(A, d_X.toHost(), B) < tol);

  // Copy-initialization form, evaluated on the thread-local Context.
  gpu::DeviceMatrix<Scalar> d_Y = d_A.template ldlt<UpLo>().solve(d_B);
  VERIFY(backward_error(A, d_Y.toHost(), B) < tol);

  // Explicit Context; the one-shot scratch is reused across calls.
  gpu::Context ctx;
  gpu::DeviceMatrix<Scalar> d_Z;
  for (int k = 0; k < 3; ++k) {
    d_Z.device(ctx) = d_A.template ldlt<UpLo>().solve(d_B);
    VERIFY(backward_error(A, d_Z.toHost(), B) < tol);
  }
}

// The default triangle is Lower; only real Scalar may leave the mode implicit.
template <typename Scalar>
void test_solve_expr_default_triangle(Index n) {
  using MatrixType = TestMatrix<Scalar>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  const MatrixType A = make_symmetric_indefinite<MatrixType>(n);
  const MatrixType B = MatrixType::Random(n, 2);

  auto d_A = gpu::DeviceMatrix<Scalar>::fromHost(A);
  auto d_B = gpu::DeviceMatrix<Scalar>::fromHost(B);
  gpu::DeviceMatrix<Scalar> d_X = d_A.ldlt().solve(d_B);
  VERIFY(backward_error(A, d_X.toHost(), B) < RealScalar(10) * RealScalar(n) * NumTraits<Scalar>::epsilon());
}

// ---- Inverse (sytri) -----------------------------------------------------------

template <typename Scalar, int UpLo>
void test_sytri(Index n, MatrixGenerator<Scalar> make) {
  using MatrixType = TestMatrix<Scalar>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  const MatrixType A = make(n);
  const RealScalar tol = RealScalar(10) * RealScalar(n) * NumTraits<Scalar>::epsilon();
  gpu::LDLT<Scalar, UpLo> ldlt(A);
  VERIFY_IS_EQUAL(ldlt.info(), Success);

  // sytri fills only the UpLo triangle; mirror it on the host. A and the
  // mirrored inverse are both symmetric, so the left and right residuals agree.
  const MatrixType A_inv = symmetric_from_triangle<UpLo>(ldlt.inverse().toHost());
  const MatrixType I = MatrixType::Identity(n, n);
  VERIFY(backward_error(A, A_inv, I) < tol);

  // inverse() works on a copy: the factorization still serves solves.
  const MatrixType B = MatrixType::Random(n, 2);
  VERIFY(backward_error(A, ldlt.solve(B), B) < tol);
}

// ---- Per-scalar driver ---------------------------------------------------------

template <typename Scalar>
void test_scalar() {
  using MatrixType = TestMatrix<Scalar>;
  constexpr int L = Lower | kSymmetricBit<Scalar>;
  constexpr int U = Upper | kSymmetricBit<Scalar>;
  const MatrixGenerator<Scalar> shifted = make_symmetric_indefinite<MatrixType>;
  const MatrixGenerator<Scalar> antidiagonal = make_antidiagonal_indefinite<MatrixType>;

  CALL_SUBTEST((test_context_bound_solver<Scalar, L>(64, 4)));
  CALL_SUBTEST((test_sytrf_sytrs<Scalar, L>(1, 1, shifted)));
  CALL_SUBTEST((test_sytrf_sytrs<Scalar, L>(64, 1, shifted)));
  CALL_SUBTEST((test_sytrf_sytrs<Scalar, L>(256, 8, shifted)));
  CALL_SUBTEST((test_sytrf_sytrs<Scalar, U>(64, 4, shifted)));
  CALL_SUBTEST((test_sytrf_sytrs<Scalar, U>(256, 1, shifted)));
  CALL_SUBTEST((test_sytrf_sytrs<Scalar, L>(2, 1, antidiagonal)));
  CALL_SUBTEST((test_sytrf_sytrs<Scalar, L>(64, 4, antidiagonal)));
  CALL_SUBTEST((test_sytrf_sytrs<Scalar, L>(257, 3, antidiagonal)));
  CALL_SUBTEST((test_sytrf_sytrs<Scalar, U>(64, 4, antidiagonal)));
  CALL_SUBTEST((test_sytrf_sytrs<Scalar, U>(257, 3, antidiagonal)));
  CALL_SUBTEST((test_ill_conditioned<Scalar, L>(64, 2)));
  CALL_SUBTEST((test_ill_conditioned<Scalar, U>(128, 1)));
  CALL_SUBTEST((test_unreferenced_triangle<Scalar, L>(64)));
  CALL_SUBTEST((test_unreferenced_triangle<Scalar, U>(64)));

  CALL_SUBTEST((test_multiple_solves<Scalar, L>(128)));
  CALL_SUBTEST((test_empty<Scalar, L>()));

  CALL_SUBTEST((test_device_matrix_solve<Scalar, L>(64, 4)));
  CALL_SUBTEST((test_device_matrix_solve<Scalar, U>(128, 1)));
  CALL_SUBTEST((test_device_matrix_move_compute<Scalar, L>(64)));
  CALL_SUBTEST((test_chaining<Scalar, L>(64)));

  CALL_SUBTEST((test_non_plain_input<Scalar, L>(64)));

  CALL_SUBTEST((test_solve_expr<Scalar, L>(64, 4, shifted)));
  CALL_SUBTEST((test_solve_expr<Scalar, U>(128, 1, shifted)));
  CALL_SUBTEST((test_solve_expr<Scalar, L>(64, 4, antidiagonal)));
  CALL_SUBTEST((test_solve_expr<Scalar, U>(63, 2, antidiagonal)));

  CALL_SUBTEST((test_sytri<Scalar, L>(64, shifted)));
  CALL_SUBTEST((test_sytri<Scalar, U>(128, shifted)));
  CALL_SUBTEST((test_sytri<Scalar, L>(64, antidiagonal)));
  CALL_SUBTEST((test_sytri<Scalar, U>(63, antidiagonal)));
}

EIGEN_DECLARE_TEST(gpu_cusolver_ldlt) {
  gpu_test::require_cuda_device();
  // Split by scalar so each part compiles in parallel.
  CALL_SUBTEST_1(test_scalar<float>());
  CALL_SUBTEST_2(test_scalar<double>());
  CALL_SUBTEST_3(test_scalar<std::complex<float>>());
  CALL_SUBTEST_4(test_scalar<std::complex<double>>());
  CALL_SUBTEST_5(test_matrices_are_indefinite());
  CALL_SUBTEST_5(test_solve_expr_default_triangle<float>(64));
  CALL_SUBTEST_5(test_solve_expr_default_triangle<double>(64));
  CALL_SUBTEST_5(test_singular());
  CALL_SUBTEST_5(test_singular_device_solve_asserts());
}
