// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#define TEST_ENABLE_TEMPORARY_TRACKING

#include "main.h"
#include <Eigen/Core>

// The scaled permutation P * D of a permutation and a diagonal matrix. Every closed-form operation below moves
// or multiplies single coefficients, so the comparisons with the dense references are exact.
template <typename Scalar, int Size>
void scaled_permutation(Index size) {
  using PermutationType = PermutationMatrix<Size>;
  using DiagonalType = DiagonalMatrix<Scalar, Size>;
  using ScaledType = ScaledPermutationMatrix<Scalar, Size>;
  using MatrixType = Matrix<Scalar, Size, Size>;
  using VectorType = Matrix<Scalar, Size, 1>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  Matrix<int, Size, 1> indices, indices2;
  randomPermutationVector(indices, size);
  randomPermutationVector(indices2, size);
  const PermutationType perm(indices), perm2(indices2);
  const DiagonalType diag(VectorType::Random(size)), diag2(VectorType::Random(size));
  const MatrixType permDense = perm.toDenseMatrix().template cast<Scalar>();
  const MatrixType perm2Dense = perm2.toDenseMatrix().template cast<Scalar>();
  const MatrixType inverseDense = perm.inverse().toDenseMatrix().template cast<Scalar>();
  const MatrixType diagDense = diag.toDenseMatrix();
  const MatrixType diag2Dense = diag2.toDenseMatrix();
  const Scalar alpha = internal::random<Scalar>();

  // P * D and the three sibling products are scaled permutations, not dense expressions.
  STATIC_CHECK((internal::is_same<decltype(perm * diag), ScaledType>::value));
  STATIC_CHECK((internal::is_same<decltype(diag * perm.inverse()), ScaledType>::value));
  const ScaledType scaled = perm * diag;
  const MatrixType scaledDense = permDense * diagDense;
  VERIFY_IS_EQUAL(scaled.toDenseMatrix(), scaledDense);
  VERIFY_IS_EQUAL(MatrixType(diag * perm), MatrixType(diagDense * permDense));
  VERIFY_IS_EQUAL(MatrixType(perm.inverse() * diag), MatrixType(inverseDense * diagDense));
  VERIFY_IS_EQUAL(MatrixType(perm.transpose() * diag), MatrixType(inverseDense * diagDense));
  VERIFY_IS_EQUAL(MatrixType(diag * perm.inverse()), MatrixType(diagDense * inverseDense));
  VERIFY_IS_EQUAL(MatrixType(indices.asPermutation() * diag.diagonal().asDiagonal()), scaledDense);
  VERIFY_IS_EQUAL(MatrixType(diag * Map<PermutationType>(indices.data(), size)), MatrixType(diagDense * permDense));
  VERIFY_IS_EQUAL(MatrixType(ScaledType(perm)), permDense);
  VERIFY_IS_EQUAL(MatrixType(ScaledType(diag)), diagDense);
  for (Index i = 0; i < size; ++i)
    for (Index j = 0; j < size; ++j) VERIFY_IS_EQUAL(scaled.coeff(i, j), scaledDense(i, j));

  // Assignment to dense.
  MatrixType result = MatrixType::Random(size, size), expected = result;
  result = scaled;
  VERIFY_IS_EQUAL(result, scaledDense);
  result = expected;
  result += scaled;
  expected += scaledDense;
  VERIFY_IS_EQUAL(result, expected);
  result -= scaled;
  expected -= scaledDense;
  VERIFY_IS_EQUAL(result, expected);

  // Products of scales are complex multiplications, whose rounding depends on the kernel: compare approximately.
  // Closed-form algebra: every result is a scaled permutation and costs O(n).
  VERIFY_IS_APPROX(MatrixType(scaled * diag2), MatrixType(scaledDense * diag2Dense));
  VERIFY_IS_APPROX(MatrixType(diag2 * scaled), MatrixType(diag2Dense * scaledDense));
  VERIFY_IS_EQUAL(MatrixType(scaled * perm2), MatrixType(scaledDense * perm2Dense));
  VERIFY_IS_EQUAL(MatrixType(perm2 * scaled), MatrixType(perm2Dense * scaledDense));
  VERIFY_IS_EQUAL(MatrixType(scaled * perm2.inverse()), MatrixType(scaledDense * perm2Dense.transpose()));
  VERIFY_IS_EQUAL(MatrixType(perm2.inverse() * scaled), MatrixType(perm2Dense.transpose() * scaledDense));
  const ScaledType scaled2 = diag2 * perm2;
  VERIFY_IS_APPROX(MatrixType(scaled * scaled2), MatrixType(scaledDense * scaled2.toDenseMatrix()));
  VERIFY_IS_APPROX(MatrixType(alpha * scaled), MatrixType(alpha * scaledDense));
  VERIFY_IS_APPROX(MatrixType(scaled * alpha), MatrixType(scaledDense * alpha));
  VERIFY_IS_EQUAL(MatrixType(-scaled), MatrixType(-scaledDense));
  VERIFY_IS_EQUAL(MatrixType(scaled.transpose()), MatrixType(scaledDense.transpose()));
  VERIFY_IS_EQUAL(MatrixType(scaled.adjoint()), MatrixType(scaledDense.adjoint()));
  STATIC_CHECK((internal::is_same<decltype(perm * diag * perm.inverse()), ScaledType>::value));
  VERIFY_IS_EQUAL(MatrixType(perm * diag * perm.inverse()), MatrixType(permDense * diagDense * inverseDense));
  VERIFY_IS_APPROX(MatrixType(scaled.inverse() * scaled), MatrixType::Identity(size, size));
  VERIFY_IS_APPROX(MatrixType(scaled.inverse()), scaledDense.inverse());
  // The dense determinant underflows for large sizes; the sign of a unit-scale permutation is exact at any size.
  if (size <= 8) VERIFY_IS_APPROX(scaled.determinant(), scaledDense.determinant());
  VERIFY_IS_EQUAL(ScaledType(perm).determinant(), Scalar(RealScalar(perm.determinant())));

  // Products with dense operands scale and permute rows or columns; no n x n intermediate. Complex products round
  // differently in the dense kernels, so those comparisons are approximate.
  const MatrixType a = MatrixType::Random(size, size);
  const VectorType x = VectorType::Random(size);
  VectorType y(size);
  VERIFY_EVALUATION_COUNT(y.noalias() = scaled * x, 0);
  VERIFY_IS_APPROX(y, VectorType(scaledDense * x));
  VERIFY_EVALUATION_COUNT(result.noalias() = scaled * a, 0);
  VERIFY_IS_APPROX(result, MatrixType(scaledDense * a));
  VERIFY_EVALUATION_COUNT(result.noalias() = a * scaled, 0);
  VERIFY_IS_APPROX(result, MatrixType(a * scaledDense));
  VERIFY_IS_APPROX(VectorType(perm * diag * x), VectorType(scaledDense * x));
  VERIFY_IS_APPROX(VectorType(x.transpose() * (diag * perm)).eval(),
                   VectorType((x.transpose() * diagDense * permDense).eval()));
  result = a;
  expected = a;
  result.noalias() += scaled * a;
  expected += scaledDense * a;
  VERIFY_IS_APPROX(result, expected);
  result.noalias() -= a * scaled.inverse();
  expected -= a * scaledDense.inverse();
  VERIFY_IS_APPROX(result, expected);
  result = alpha * (scaled * a);
  VERIFY_IS_APPROX(result, alpha * scaledDense * a);
  result = a * (scaled * a);
  VERIFY_IS_APPROX(result, a * scaledDense * a);
  result = scaled * (a * a);
  VERIFY_IS_APPROX(result, scaledDense * a * a);
  // In-place: the dense permutation kernel detects the aliasing.
  result = a;
  result.noalias() = scaled * result;
  VERIFY_IS_APPROX(result, MatrixType(scaledDense * a));
  result = a;
  result = result * scaled;
  VERIFY_IS_APPROX(result, MatrixType(a * scaledDense));

  // Triangular and self-adjoint operands are densified, as toDenseMatrix() does.
  const MatrixType lower = a.template triangularView<Lower>();
  const MatrixType upperSelfadjoint = a.template selfadjointView<Upper>();
  VERIFY_IS_APPROX(MatrixType(scaled * a.template triangularView<Lower>()), MatrixType(scaledDense * lower));
  VERIFY_IS_APPROX(MatrixType(a.template triangularView<Lower>() * scaled), MatrixType(lower * scaledDense));
  VERIFY_IS_APPROX(MatrixType(scaled * a.template selfadjointView<Upper>()),
                   MatrixType(scaledDense * upperSelfadjoint));
  VERIFY_IS_APPROX(MatrixType(a.template selfadjointView<Upper>() * scaled),
                   MatrixType(upperSelfadjoint * scaledDense));

  // Lazy sums with a dense matrix.
  STATIC_CHECK(
      (internal::is_same<decltype(a + scaled), const CwiseBinaryOp<internal::scalar_sum_op<Scalar, Scalar>,
                                                                   const MatrixType, const ScaledType>>::value));
  VERIFY_IS_EQUAL(MatrixType(a + scaled), MatrixType(a + scaledDense));
  VERIFY_IS_EQUAL(MatrixType(scaled + a), MatrixType(scaledDense + a));
  VERIFY_IS_EQUAL(MatrixType(a - scaled), MatrixType(a - scaledDense));
  VERIFY_IS_EQUAL(MatrixType(scaled - a), MatrixType(scaledDense - a));
  VERIFY_IS_APPROX((a + scaled) * x, (a + scaledDense) * x);
  VERIFY_IS_APPROX((a + scaled).sum(), (a + scaledDense).sum());

  // Storage and setters.
  ScaledType identity(size);
  identity.setIdentity();
  VERIFY_IS_EQUAL(MatrixType(identity), MatrixType::Identity(size, size));
  VERIFY_IS_EQUAL(MatrixType(identity * scaled), scaledDense);
  ScaledType copy = scaled;
  copy.scales() *= alpha;
  VERIFY_IS_APPROX(MatrixType(copy), MatrixType(alpha * scaledDense));
  copy = scaled2;
  VERIFY_IS_EQUAL(MatrixType(copy), scaled2.toDenseMatrix());
}

EIGEN_DECLARE_TEST(scaled_permutation) {
  CALL_SUBTEST_1((scaled_permutation<float, 1>(1)));
  CALL_SUBTEST_1((scaled_permutation<double, 3>(3)));
  CALL_SUBTEST_1((scaled_permutation<std::complex<double>, 4>(4)));
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_2((scaled_permutation<float, Dynamic>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE))));
    CALL_SUBTEST_2((scaled_permutation<double, Dynamic>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE))));
    CALL_SUBTEST_3((scaled_permutation<std::complex<double>, Dynamic>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE))));
  }
}
