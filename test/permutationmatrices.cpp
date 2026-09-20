// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#define TEST_ENABLE_TEMPORARY_TRACKING

#include "main.h"
#include <Eigen/Core>
#include <cstdint>

using namespace std;
template <typename MatrixType>
void permutationmatrices(const MatrixType& m) {
  typedef typename MatrixType::Scalar Scalar;
  enum { Rows = MatrixType::RowsAtCompileTime, Cols = MatrixType::ColsAtCompileTime, Options = MatrixType::Options };
  typedef PermutationMatrix<Rows> LeftPermutationType;
  typedef Transpositions<Rows> LeftTranspositionsType;
  typedef Matrix<int, Rows, 1> LeftPermutationVectorType;
  typedef Map<LeftPermutationType> MapLeftPerm;
  typedef PermutationMatrix<Cols> RightPermutationType;
  typedef Transpositions<Cols> RightTranspositionsType;
  typedef Matrix<int, Cols, 1> RightPermutationVectorType;
  typedef Map<RightPermutationType> MapRightPerm;

  Index rows = m.rows();
  Index cols = m.cols();

  MatrixType m_original = MatrixType::Random(rows, cols);
  LeftPermutationVectorType lv;
  randomPermutationVector(lv, rows);
  LeftPermutationType lp(lv);
  RightPermutationVectorType rv;
  randomPermutationVector(rv, cols);
  RightPermutationType rp(rv);
  LeftTranspositionsType lt(lv);
  RightTranspositionsType rt(rv);
  MatrixType m_permuted = MatrixType::Random(rows, cols);

  VERIFY_EVALUATION_COUNT(m_permuted.noalias() = lp * m_original * rp,
                          1);  // 1 temp for sub expression "lp * m_original"

  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++) VERIFY_IS_APPROX(m_permuted(lv(i), j), m_original(i, rv(j)));

  Matrix<Scalar, Rows, Rows> lm(lp);
  Matrix<Scalar, Cols, Cols> rm(rp);

  VERIFY_IS_APPROX(m_permuted, lm * m_original * rm);

  m_permuted = m_original;
  VERIFY_EVALUATION_COUNT(m_permuted.noalias() = lp * m_permuted * rp, 1);
  VERIFY_IS_APPROX(m_permuted, lm * m_original * rm);

  LeftPermutationType lpi;
  lpi = lp.inverse();
  VERIFY_IS_APPROX(lpi * m_permuted, lp.inverse() * m_permuted);

  VERIFY_IS_APPROX(lp.inverse() * m_permuted * rp.inverse(), m_original);
  VERIFY_IS_APPROX(lv.asPermutation().inverse() * m_permuted * rv.asPermutation().inverse(), m_original);
  VERIFY_IS_APPROX(
      MapLeftPerm(lv.data(), lv.size()).inverse() * m_permuted * MapRightPerm(rv.data(), rv.size()).inverse(),
      m_original);

  VERIFY((lp * lp.inverse()).toDenseMatrix().isIdentity());
  VERIFY((lv.asPermutation() * lv.asPermutation().inverse()).toDenseMatrix().isIdentity());
  VERIFY(
      (MapLeftPerm(lv.data(), lv.size()) * MapLeftPerm(lv.data(), lv.size()).inverse()).toDenseMatrix().isIdentity());

  LeftPermutationVectorType lv2;
  randomPermutationVector(lv2, rows);
  LeftPermutationType lp2(lv2);
  Matrix<Scalar, Rows, Rows> lm2(lp2);
  VERIFY_IS_APPROX((lp * lp2).toDenseMatrix().template cast<Scalar>(), lm * lm2);
  VERIFY_IS_APPROX((lv.asPermutation() * lv2.asPermutation()).toDenseMatrix().template cast<Scalar>(), lm * lm2);
  VERIFY_IS_APPROX(
      (MapLeftPerm(lv.data(), lv.size()) * MapLeftPerm(lv2.data(), lv2.size())).toDenseMatrix().template cast<Scalar>(),
      lm * lm2);

  LeftPermutationType identityp;
  identityp.setIdentity(rows);
  VERIFY_IS_APPROX(m_original, identityp * m_original);

  // check inplace permutations
  m_permuted = m_original;
  VERIFY_EVALUATION_COUNT(m_permuted.noalias() = lp.inverse() * m_permuted, 1);  // 1 temp to allocate the mask
  VERIFY_IS_APPROX(m_permuted, lp.inverse() * m_original);

  m_permuted = m_original;
  VERIFY_EVALUATION_COUNT(m_permuted.noalias() = m_permuted * rp.inverse(), 1);  // 1 temp to allocate the mask
  VERIFY_IS_APPROX(m_permuted, m_original * rp.inverse());

  m_permuted = m_original;
  VERIFY_EVALUATION_COUNT(m_permuted.noalias() = lp * m_permuted, 1);  // 1 temp to allocate the mask
  VERIFY_IS_APPROX(m_permuted, lp * m_original);

  m_permuted = m_original;
  VERIFY_EVALUATION_COUNT(m_permuted.noalias() = m_permuted * rp, 1);  // 1 temp to allocate the mask
  VERIFY_IS_APPROX(m_permuted, m_original * rp);

  if (rows > 1 && cols > 1) {
    lp2 = lp;
    Index i = internal::random<Index>(0, rows - 1);
    Index j;
    do j = internal::random<Index>(0, rows - 1);
    while (j == i);
    lp2.applyTranspositionOnTheLeft(i, j);
    lm = lp;
    lm.row(i).swap(lm.row(j));
    VERIFY_IS_APPROX(lm, lp2.toDenseMatrix().template cast<Scalar>());

    RightPermutationType rp2 = rp;
    i = internal::random<Index>(0, cols - 1);
    do j = internal::random<Index>(0, cols - 1);
    while (j == i);
    rp2.applyTranspositionOnTheRight(i, j);
    rm = rp;
    rm.col(i).swap(rm.col(j));
    VERIFY_IS_APPROX(rm, rp2.toDenseMatrix().template cast<Scalar>());
  }

  {
    // simple compilation check
    Matrix<Scalar, Cols, Cols> A = rp;
    Matrix<Scalar, Cols, Cols> B = rp.transpose();
    VERIFY_IS_APPROX(A, B.transpose());
  }

  m_permuted = m_original;
  lp = lt;
  rp = rt;
  VERIFY_EVALUATION_COUNT(m_permuted = lt * m_permuted * rt, 1);
  VERIFY_IS_APPROX(m_permuted, lp * m_original * rp.transpose());

  VERIFY_IS_APPROX(lt.inverse() * m_permuted * rt.inverse(), m_original);

  // Check inplace transpositions
  m_permuted = m_original;
  VERIFY_IS_APPROX(m_permuted = lt * m_permuted, lp * m_original);
  m_permuted = m_original;
  VERIFY_IS_APPROX(m_permuted = lt.inverse() * m_permuted, lp.inverse() * m_original);
  m_permuted = m_original;
  VERIFY_IS_APPROX(m_permuted = m_permuted * rt, m_original * rt);
  m_permuted = m_original;
  VERIFY_IS_APPROX(m_permuted = m_permuted * rt.inverse(), m_original * rt.inverse());
}

template <typename T>
void bug890() {
  typedef Matrix<T, Dynamic, Dynamic> MatrixType;
  typedef Matrix<T, Dynamic, 1> VectorType;
  typedef Stride<Dynamic, Dynamic> S;
  typedef Map<MatrixType, Aligned, S> MapType;
  typedef PermutationMatrix<Dynamic> Perm;

  VectorType v1(2), v2(2), op(4), rhs(2);
  v1 << 666, 667;
  op << 1, 0, 0, 1;
  rhs << 42, 42;

  Perm P(2);
  P.indices() << 1, 0;

  MapType(v1.data(), 2, 1, S(1, 1)) = P * MapType(rhs.data(), 2, 1, S(1, 1));
  VERIFY_IS_APPROX(v1, (P * rhs).eval());

  MapType(v1.data(), 2, 1, S(1, 1)) = P.inverse() * MapType(rhs.data(), 2, 1, S(1, 1));
  VERIFY_IS_APPROX(v1, (P.inverse() * rhs).eval());
}

void test_aliasing() {
  // Bug #2869.
  auto P = Eigen::PermutationMatrix<4>{Eigen::Vector4i{0, 2, 3, 1}};
  {
    Eigen::Vector<float, 5> z = {0.0f, 1.1f, 2.2f, 3.3f, 4.4f};
    Eigen::Vector<float, 5> expected = z;
    z.tail<4>() = P * z.head<4>();
    expected.tail<4>() = (P * expected.head<4>()).eval();
    VERIFY_IS_APPROX(z, expected);
  }

  {
    // Obfuscate the aliasing in the RHS expression.
    Eigen::Vector4f a = {1.1f, 2.2f, 3.3f, 4.4f};
    Eigen::Vector4f expected = P * (a + Eigen::Vector4f::Zero()).cwiseSqrt();
    a = P * (a + Eigen::Vector4f::Zero()).cwiseSqrt();
    VERIFY_IS_APPROX(a, expected);
  }
}

template <typename PermutationType, typename OtherPermutationType = PermutationType>
void permutation_inverse_product(Index size) {
  PermutationType lhs(size), result(size);
  OtherPermutationType rhs(size);
  randomPermutationVector(lhs.indices(), size);
  randomPermutationVector(rhs.indices(), size);
  const MatrixXi lhsDense = lhs.toDenseMatrix().template cast<int>();
  const MatrixXi rhsDense = rhs.toDenseMatrix().template cast<int>();
  const MatrixXi expected = lhsDense * rhsDense.transpose();
  result = lhs * rhs.inverse();
  VERIFY(result.toDenseMatrix().template cast<int>() == expected);
  result = lhs * rhs.transpose();
  VERIFY(result.toDenseMatrix().template cast<int>() == expected);
  result = lhs.indices().asPermutation() * rhs.indices().asPermutation().inverse();
  VERIFY(result.toDenseMatrix().template cast<int>() == expected);
  result = Map<PermutationType>(lhs.indices().data(), size) *
           Map<OtherPermutationType>(rhs.indices().data(), size).inverse();
  VERIFY(result.toDenseMatrix().template cast<int>() == expected);
  using OtherStorageIndex = typename OtherPermutationType::StorageIndex;
  Matrix<OtherStorageIndex, Dynamic, 1> storage(2 * size);
  Map<Matrix<OtherStorageIndex, Dynamic, 1>, 0, InnerStride<2>> strided(storage.data(), size);
  strided = rhs.indices();
  result = lhs * strided.asPermutation().inverse();
  VERIFY(result.toDenseMatrix().template cast<int>() == expected);
  result = lhs;
  result = result * rhs.inverse();
  VERIFY(result.toDenseMatrix().template cast<int>() == expected);
  result.indices() = rhs.indices().template cast<typename PermutationType::StorageIndex>();
  result = lhs * result.inverse();
  VERIFY(result.toDenseMatrix().template cast<int>() == expected);
  result = lhs;
  result = result * result.inverse();
  VERIFY(result.toDenseMatrix().isIdentity());
  result = lhs.inverse();
  VERIFY(result.toDenseMatrix().template cast<int>() == lhsDense.transpose());
  result = lhs;
  result = result.inverse();
  VERIFY(result.toDenseMatrix().template cast<int>() == lhsDense.transpose());
  const OtherPermutationType inverse = strided.asPermutation().inverse();
  VERIFY(inverse.toDenseMatrix().template cast<int>() == rhsDense.transpose());
}

template <typename PermutationType>
void permutation_left_transposition(Index size) {
  PermutationType original(size);
  randomPermutationVector(original.indices(), size);
  const MatrixXi dense = original.toDenseMatrix().template cast<int>();
  PermutationType result(size);
  MatrixXi expected(size, size);
  for (Index i = 0; i < size; ++i) {
    for (Index j = 0; j < size; ++j) {
      result = original;
      expected = dense;
      expected.row(i).swap(expected.row(j));
      VERIFY(&result.applyTranspositionOnTheLeft(i, j) == &result);
      VERIFY(result.toDenseMatrix().template cast<int>() == expected);
    }
  }
}

template <typename Scalar, int Size>
void permutation_diagonal_products(Index size) {
  using PermutationType = PermutationMatrix<Size>;
  using DiagonalType = DiagonalMatrix<Scalar, Size>;
  using MatrixType = Matrix<Scalar, Size, Size>;
  using VectorType = Matrix<Scalar, Size, 1>;

  Matrix<int, Size, 1> indices;
  randomPermutationVector(indices, size);
  const PermutationType perm(indices);
  const DiagonalType diag(VectorType::Random(size));
  const MatrixType permDense = perm.toDenseMatrix().template cast<Scalar>();
  const MatrixType inverseDense = perm.inverse().toDenseMatrix().template cast<Scalar>();
  const MatrixType diagDense = diag.toDenseMatrix();

  // Every product below scatters exact copies of the diagonal, so the results are exact.
  MatrixType result(size, size), expected(size, size);
  VERIFY_EVALUATION_COUNT(result.noalias() = perm * diag, 0);
  expected = permDense * diagDense;
  VERIFY_IS_EQUAL(result, expected);
  result = diag * perm;
  expected = diagDense * permDense;
  VERIFY_IS_EQUAL(result, expected);
  result = perm.inverse() * diag;
  expected = inverseDense * diagDense;
  VERIFY_IS_EQUAL(result, expected);
  result = perm.transpose() * diag;
  VERIFY_IS_EQUAL(result, expected);
  result = diag * perm.inverse();
  expected = diagDense * inverseDense;
  VERIFY_IS_EQUAL(result, expected);
  result = diag * perm.transpose();
  VERIFY_IS_EQUAL(result, expected);

  // The wrapper and Map forms of both operands.
  result = indices.asPermutation() * diag.diagonal().asDiagonal();
  expected = permDense * diagDense;
  VERIFY_IS_EQUAL(result, expected);
  result = diag.diagonal().asDiagonal() * Map<PermutationType>(indices.data(), size);
  expected = diagDense * permDense;
  VERIFY_IS_EQUAL(result, expected);

  result = MatrixType::Random(size, size);
  expected = result;
  result += perm * diag;
  expected += permDense * diagDense;
  VERIFY_IS_APPROX(result, expected);
  result -= diag * perm.inverse();
  expected -= diagDense * inverseDense;
  VERIFY_IS_APPROX(result, expected);

  // Nested products: the permutation-diagonal factor is materialized like any other inner product.
  const VectorType x = VectorType::Random(size);
  VERIFY_IS_APPROX((perm * diag) * x, perm * (diag * x));
  VERIFY_IS_APPROX(x.transpose() * (diag * perm), (x.transpose() * diag) * perm);
  result = perm * diag * perm.inverse();
  expected = permDense * diagDense * inverseDense;
  VERIFY_IS_APPROX(result, expected);
}

void permutation_inverse_product_temporaries() {
  PermutationMatrix<Dynamic> lhs(17), rhs(17), result(17);
  randomPermutationVector(lhs.indices(), lhs.size());
  randomPermutationVector(rhs.indices(), rhs.size());
  VERIFY_EVALUATION_COUNT(result = lhs * rhs.inverse(), 1);
  const MatrixXi expected = lhs.toDenseMatrix() * rhs.toDenseMatrix().transpose();
  VERIFY(result.toDenseMatrix() == expected);
  PermutationMatrix<Dynamic> mismatched(18);
  mismatched.setIdentity();
  VERIFY_RAISES_ASSERT(lhs * mismatched.inverse());
  PermutationMatrix<Dynamic> empty(0);
  VERIFY_EVALUATION_COUNT(result = empty * empty.inverse(), 0);
  VERIFY_IS_EQUAL(result.size(), 0);
}

EIGEN_DECLARE_TEST(permutationmatrices) {
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(permutationmatrices(Matrix<float, 1, 1>()));
    CALL_SUBTEST_2(permutationmatrices(Matrix3f()));
    CALL_SUBTEST_3(permutationmatrices(Matrix<double, 3, 3, RowMajor>()));
    CALL_SUBTEST_4(permutationmatrices(Matrix4d()));
    CALL_SUBTEST_5(permutationmatrices(Matrix<double, 40, 60>()));
    CALL_SUBTEST_6(permutationmatrices(Matrix<double, Dynamic, Dynamic, RowMajor>(
        internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE))));
    CALL_SUBTEST_7(permutationmatrices(
        MatrixXcf(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE))));
  }
  CALL_SUBTEST_8(permutation_inverse_product_temporaries());
  CALL_SUBTEST_8((permutation_inverse_product<PermutationMatrix<4>>(4)));
  CALL_SUBTEST_8((permutation_inverse_product<PermutationMatrix<Dynamic, 17>>(17)));
  for (Index size : {0, 1, 2, 7, 8, 9, 15, 16, 17}) {
    EIGEN_UNUSED_VARIABLE(size);
    CALL_SUBTEST_8((permutation_inverse_product<PermutationMatrix<Dynamic>>(size)));
    CALL_SUBTEST_8((permutation_inverse_product<PermutationMatrix<Dynamic, Dynamic, std::int8_t>>(size)));
    CALL_SUBTEST_8((permutation_inverse_product<PermutationMatrix<Dynamic, Dynamic, std::uint32_t>>(size)));
    CALL_SUBTEST_8(
        (permutation_inverse_product<PermutationMatrix<Dynamic>, PermutationMatrix<Dynamic, Dynamic, std::int64_t>>(
            size)));
  }
  CALL_SUBTEST_9((permutation_left_transposition<PermutationMatrix<4>>(4)));
  CALL_SUBTEST_9((permutation_left_transposition<PermutationMatrix<Dynamic, 17>>(17)));
  for (Index size : {1, 2, 7, 8, 9, 15, 16, 17, 33}) {
    EIGEN_UNUSED_VARIABLE(size);
    CALL_SUBTEST_9((permutation_left_transposition<PermutationMatrix<Dynamic>>(size)));
    CALL_SUBTEST_9((permutation_left_transposition<PermutationMatrix<Dynamic, Dynamic, std::int8_t>>(size)));
    CALL_SUBTEST_9((permutation_left_transposition<PermutationMatrix<Dynamic, Dynamic, std::uint32_t>>(size)));
    CALL_SUBTEST_9((permutation_left_transposition<PermutationMatrix<Dynamic, Dynamic, std::int64_t>>(size)));
  }
  CALL_SUBTEST_5(bug890<double>());
  CALL_SUBTEST_4(test_aliasing());
  CALL_SUBTEST_10((permutation_diagonal_products<float, 1>(1)));
  CALL_SUBTEST_10((permutation_diagonal_products<double, 3>(3)));
  CALL_SUBTEST_10((permutation_diagonal_products<std::complex<double>, 4>(4)));
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_10((permutation_diagonal_products<float, Dynamic>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE))));
    CALL_SUBTEST_10((permutation_diagonal_products<double, Dynamic>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE))));
    CALL_SUBTEST_10(
        (permutation_diagonal_products<std::complex<double>, Dynamic>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE))));
  }
}
