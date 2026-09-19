// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include <Eigen/Core>

template <typename LhsScalar, typename RhsScalar, int Order>
void outer_product_scalar_types() {
  using Scalar = typename ScalarBinaryOpTraits<LhsScalar, RhsScalar>::ReturnType;
  using Lhs = Matrix<LhsScalar, Dynamic, 1>;
  using Rhs = Matrix<RhsScalar, 1, Dynamic>;
  using Mat = Matrix<Scalar, Dynamic, Dynamic, Order>;
  STATIC_CHECK((internal::product_type<Lhs, Rhs>::value == OuterProduct));
  for (Index rows : {2, 3, 16, 17}) {
    for (Index cols : {2, 3, 16, 17}) {
      const Lhs lhs = Lhs::Constant(rows, LhsScalar(2));
      const Rhs rhs = Rhs::Constant(cols, RhsScalar(3));
      Mat storage = Mat::Constant(2 * rows, 2 * cols, Scalar(7));
      Mat expected = storage;
      Map<Mat, 0, Stride<Dynamic, 2>> dst(storage.data(), rows, cols, Stride<Dynamic, 2>(2 * storage.outerStride(), 2));
      for (int operation = 0; operation < 3; ++operation) {
        storage.setConstant(Scalar(7));
        expected = storage;
        if (operation == 0) dst.noalias() = lhs * rhs;
        if (operation == 1) dst.noalias() += lhs * rhs;
        if (operation == 2) dst.noalias() -= lhs * rhs;
        for (Index j = 0; j < cols; ++j) {
          for (Index i = 0; i < rows; ++i) {
            const Scalar value = lhs(i) * rhs(j);
            expected(2 * i, 2 * j) = operation == 0 ? value : operation == 1 ? Scalar(7) + value : Scalar(7) - value;
          }
        }
        VERIFY_IS_EQUAL(storage, expected);
      }
    }
  }
}

template <typename Real, int Order>
void outer_product_mixed_infinity() {
  using Scalar = std::complex<Real>;
  using Mat = Matrix<Scalar, Dynamic, Dynamic, Order>;
  const Real infinity = NumTraits<Real>::infinity();
  for (Index n : {2, 3, 16, 17}) {
    Matrix<Scalar, Dynamic, 1> lhs = Matrix<Scalar, Dynamic, 1>::Constant(n, Scalar(infinity, Real(2)));
    Matrix<Real, 1, Dynamic> rhs = Matrix<Real, 1, Dynamic>::Constant(n, Real(2));
    for (int operation = 0; operation < 3; ++operation) {
      Mat actual = Mat::Constant(n, n, Scalar(Real(1), Real(1)));
      if (operation == 0) actual.noalias() = lhs * rhs;
      if (operation == 1) actual.noalias() += lhs * rhs;
      if (operation == 2) actual.noalias() -= lhs * rhs;
      const Scalar product = lhs(0) * rhs(0);
      const Scalar expected = operation == 0   ? product
                              : operation == 1 ? Scalar(1, 1) + product
                                               : Scalar(1, 1) - product;
      for (Index j = 0; j < n; ++j) {
        for (Index i = 0; i < n; ++i) {
          VERIFY_IS_EQUAL(actual(i, j).real(), expected.real());
          VERIFY_IS_EQUAL(actual(i, j).imag(), expected.imag());
        }
      }
      Mat reversed = Mat::Constant(n, n, Scalar(Real(1), Real(1)));
      if (operation == 0) reversed.noalias() = rhs.transpose() * lhs.transpose();
      if (operation == 1) reversed.noalias() += rhs.transpose() * lhs.transpose();
      if (operation == 2) reversed.noalias() -= rhs.transpose() * lhs.transpose();
      for (Index j = 0; j < n; ++j) {
        for (Index i = 0; i < n; ++i) {
          VERIFY_IS_EQUAL(reversed(i, j).real(), expected.real());
          VERIFY_IS_EQUAL(reversed(i, j).imag(), expected.imag());
        }
      }
    }
  }
}

template <typename ProductType, typename Mat>
void check_scaled_product(const ProductType& product, const Mat& expected) {
  Mat actual = product;
  VERIFY_IS_EQUAL(actual, expected);
  actual = product + Mat::Zero(expected.rows(), expected.cols());
  VERIFY_IS_EQUAL(actual, expected);
  actual.setZero();
  actual.noalias() += product;
  VERIFY_IS_EQUAL(actual, expected);
  actual.noalias() -= product;
  VERIFY_IS_EQUAL(actual, Mat::Zero(expected.rows(), expected.cols()));
  Mat triangle = Mat::Constant(expected.rows(), expected.cols(), typename Mat::Scalar(7));
  Mat triangleExpected = triangle;
  triangle.template triangularView<Upper>() = product;
  triangleExpected.template triangularView<Upper>() = expected;
  VERIFY_IS_EQUAL(triangle, triangleExpected);
  triangle.template triangularView<Lower>() = product;
  triangleExpected.template triangularView<Lower>() = expected;
  VERIFY_IS_EQUAL(triangle, triangleExpected);
}

template <int Mode, int Order>
void scaled_selfadjoint_diagonal_product() {
  using Scalar = std::complex<double>;
  using Mat = Matrix<Scalar, 3, 3, Order>;
  Mat matrix;
  matrix << Scalar(2), Scalar(1, 2), Scalar(4, 3), Scalar(3, 4), Scalar(5), Scalar(7, 8), Scalar(9, 2), Scalar(1, 4),
      Scalar(3);
  const Scalar alpha(2, 3);
  const Matrix<Scalar, 3, 1> diagonal = Matrix<Scalar, 3, 1>::Constant(Scalar(2, 1));
  Mat left, right;
  for (Index j = 0; j < 3; ++j) {
    for (Index i = 0; i < 3; ++i) {
      const bool stored = Mode == Lower ? i >= j : i <= j;
      const Scalar value = stored ? matrix(i, j) : numext::conj(matrix(j, i));
      left(i, j) = alpha * (diagonal(i) * value);
      right(i, j) = alpha * (value * diagonal(j));
    }
  }
  check_scaled_product((alpha * matrix.template selfadjointView<Mode>()) * diagonal.asDiagonal(), right);
  check_scaled_product(diagonal.asDiagonal() * (alpha * matrix.template selfadjointView<Mode>()), left);
  check_scaled_product(alpha * (matrix.template selfadjointView<Mode>() * diagonal.asDiagonal()), right);
  check_scaled_product(alpha * (diagonal.asDiagonal() * matrix.template selfadjointView<Mode>()), left);
  const Mat conjugated = matrix.conjugate();
  check_scaled_product((alpha * conjugated.conjugate().template selfadjointView<Mode>()) * diagonal.asDiagonal(),
                       right);
}

template <int Mode, int Order>
void scaled_unit_triangular_product() {
  using Mat = Matrix<double, 3, 3, Order>;
  Mat matrix;
  matrix << 9, 2, 3, 4, 9, 6, 7, 8, 9;
  const Mat rhs = Mat::Constant(2);
  const Vector3d diagonal = Vector3d::Constant(2);
  Mat expected = Mat::Zero(), expectedDiagonal = Mat::Zero();
  for (Index j = 0; j < 3; ++j) {
    for (Index i = 0; i < 3; ++i) {
      for (Index k = 0; k < 3; ++k) {
        const bool stored = Mode == UnitLower ? i > k : i < k;
        const double value = i == k ? 1 : stored ? matrix(i, k) : 0;
        expected(i, j) += 3 * value * rhs(k, j);
      }
      const bool stored = Mode == UnitLower ? i > j : i < j;
      expectedDiagonal(i, j) = 3 * (i == j ? 1 : stored ? matrix(i, j) : 0) * diagonal(j);
    }
  }
  STATIC_CHECK((!internal::product_can_fold_scalar<decltype(matrix.template triangularView<Mode>())>::value));
  check_scaled_product(3.0 * (matrix.template triangularView<Mode>() * rhs), expected);
  check_scaled_product(3.0 * (matrix.template triangularView<Mode>() * diagonal.asDiagonal()), expectedDiagonal);
  Mat aliased = matrix;
  aliased = 3.0 * (aliased.template triangularView<Mode>() * rhs);
  VERIFY_IS_EQUAL(aliased, expected);
}

EIGEN_DECLARE_TEST(product_evaluators) {
  CALL_SUBTEST_1((outer_product_scalar_types<double, double, ColMajor>()));
  CALL_SUBTEST_1((outer_product_scalar_types<double, double, RowMajor>()));
  CALL_SUBTEST_1((outer_product_scalar_types<std::complex<double>, double, ColMajor>()));
  CALL_SUBTEST_1((outer_product_scalar_types<std::complex<double>, double, RowMajor>()));
  CALL_SUBTEST_1((outer_product_scalar_types<double, std::complex<double>, ColMajor>()));
  CALL_SUBTEST_1((outer_product_scalar_types<double, std::complex<double>, RowMajor>()));
  CALL_SUBTEST_1((outer_product_mixed_infinity<float, ColMajor>()));
  CALL_SUBTEST_1((outer_product_mixed_infinity<float, RowMajor>()));
  CALL_SUBTEST_1((outer_product_mixed_infinity<double, ColMajor>()));
  CALL_SUBTEST_1((outer_product_mixed_infinity<double, RowMajor>()));
  CALL_SUBTEST_2((scaled_selfadjoint_diagonal_product<Lower, ColMajor>()));
  CALL_SUBTEST_2((scaled_selfadjoint_diagonal_product<Lower, RowMajor>()));
  CALL_SUBTEST_2((scaled_selfadjoint_diagonal_product<Upper, ColMajor>()));
  CALL_SUBTEST_2((scaled_selfadjoint_diagonal_product<Upper, RowMajor>()));
  CALL_SUBTEST_3((scaled_unit_triangular_product<UnitLower, ColMajor>()));
  CALL_SUBTEST_3((scaled_unit_triangular_product<UnitLower, RowMajor>()));
  CALL_SUBTEST_3((scaled_unit_triangular_product<UnitUpper, ColMajor>()));
  CALL_SUBTEST_3((scaled_unit_triangular_product<UnitUpper, RowMajor>()));
}
