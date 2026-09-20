// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include <Eigen/Core>
#include <vector>

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

template <typename LhsScalar, typename RhsScalar, int Order>
void lazy_outer_product_strides() {
  using Scalar = typename ScalarBinaryOpTraits<LhsScalar, RhsScalar>::ReturnType;
  using Real = typename NumTraits<Scalar>::Real;
  using Lhs = Matrix<LhsScalar, Dynamic, 1>;
  using Rhs = Matrix<RhsScalar, 1, Dynamic>;
  using Mat = Matrix<Scalar, Dynamic, Dynamic, Order>;
  const Lhs lhsStorage = Lhs::Random(34);
  const Rhs rhsStorage = Rhs::Random(51);
  const Map<const Lhs, Unaligned, InnerStride<2>> lhs(lhsStorage.data(), 17);
  const Map<const Rhs, Unaligned, InnerStride<3>> rhs(rhsStorage.data(), 17);
  const auto product = lhs.lazyProduct(rhs);
  STATIC_CHECK((internal::evaluator<decltype(product)>::InnerSize == 1));
  const Mat actual = product;
  for (Index j = 0; j < rhs.size(); ++j) {
    for (Index i = 0; i < lhs.size(); ++i) {
      const Scalar expected = lhs(i) * rhs(j);
      const Real bound = Real(8) * NumTraits<Real>::epsilon() * (Real(1) + numext::abs(expected));
      VERIFY(numext::abs(actual(i, j) - expected) <= bound);
      VERIFY(numext::abs(product.coeff(i, j) - expected) <= bound);
    }
  }
  const Matrix<RhsScalar, 1, 1> single = Matrix<RhsScalar, 1, 1>::Constant(RhsScalar(3));
  const auto columnProduct = lhs.lazyProduct(single);
  for (Index i = 0; i < lhs.size(); ++i) {
    const Scalar expected = lhs(i) * single(0);
    const Real bound = Real(8) * NumTraits<Real>::epsilon() * (Real(1) + numext::abs(expected));
    VERIFY(numext::abs(columnProduct.coeff(i) - expected) <= bound);
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

template <typename Real>
void outer_packet_equal_component(Real actual, Real expected) {
  if ((numext::isnan)(expected)) {
    VERIFY((numext::isnan)(actual));
  } else {
    VERIFY(actual == expected);
    if (expected == Real(0)) VERIFY(std::signbit(actual) == std::signbit(expected));
  }
}

template <typename Real>
void outer_packet_equal_value(std::complex<Real> actual, std::complex<Real> expected) {
  outer_packet_equal_component(actual.real(), expected.real());
  outer_packet_equal_component(actual.imag(), expected.imag());
}

template <typename Scalar>
Scalar outer_packet_value_at(Index i) {
  return Scalar(int(i % 11) - 5);
}

template <typename Scalar, bool Complex = NumTraits<Scalar>::IsComplex>
struct outer_packet_sample_value {
  static Scalar get(Index i) { return outer_packet_value_at<Scalar>(i); }
};

template <typename Scalar>
struct outer_packet_sample_value<Scalar, true> {
  using Real = typename NumTraits<Scalar>::Real;
  static Scalar get(Index i) { return Scalar(outer_packet_value_at<Real>(i), outer_packet_value_at<Real>(3 * i + 1)); }
};

template <typename LhsScalar, typename RhsScalar, int Order, int LhsStride, int RhsStride, int DstStride,
          bool ReverseOuter = false>
void outer_packet_check_layout() {
  using Scalar = typename ScalarBinaryOpTraits<LhsScalar, RhsScalar>::ReturnType;
  using Lhs = Matrix<LhsScalar, Dynamic, 1>;
  using Rhs = Matrix<RhsScalar, 1, Dynamic>;
  using Mat = Matrix<Scalar, Dynamic, Dynamic, Order>;
  using LhsMap = Map<const Lhs, Unaligned, InnerStride<LhsStride>>;
  using RhsMap = Map<const Rhs, Unaligned, InnerStride<RhsStride>>;
  using DstMap = Map<Mat, Unaligned, Stride<Dynamic, DstStride>>;
  const Scalar sentinel(7, -3);
  constexpr int packetSize = internal::unpacket_traits<typename internal::packet_traits<Scalar>::type>::size;
  const std::vector<Index> sizes = {0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 4 * packetSize + 1};
  for (Index rows : sizes) {
    for (Index cols : sizes) {
      std::vector<LhsScalar> lhsStorage(1 + rows * LhsStride);
      std::vector<RhsScalar> rhsStorage(1 + cols * RhsStride);
      for (Index i = 0; i < rows; ++i) lhsStorage[1 + i * LhsStride] = outer_packet_sample_value<LhsScalar>::get(i);
      for (Index j = 0; j < cols; ++j) rhsStorage[1 + j * RhsStride] = outer_packet_sample_value<RhsScalar>::get(j + 3);
      const LhsMap lhs(lhsStorage.data() + 1, rows);
      const RhsMap rhs(rhsStorage.data() + 1, cols);
      const Index outerStride = DstStride * (Order == RowMajor ? cols : rows) + 3;
      const Index outerSize = Order == RowMajor ? rows : cols;
      const Index count = 1 + outerStride * outerSize;
      const Index offset = 1 + (ReverseOuter && outerSize > 0 ? (outerSize - 1) * outerStride : 0);
      const Index signedStride = ReverseOuter ? -outerStride : outerStride;
      std::vector<Scalar> storage(count, sentinel), expected(count);
      DstMap dst(storage.data() + offset, rows, cols, Stride<Dynamic, DstStride>(signedStride, DstStride));
      for (int operation = 0; operation < 3; ++operation) {
        std::fill(storage.begin(), storage.end(), sentinel);
        expected = storage;
        if (operation == 0) dst.noalias() = lhs * rhs;
        if (operation == 1) dst.noalias() += lhs * rhs;
        if (operation == 2) dst.noalias() -= lhs * rhs;
        for (Index i = 0; i < rows; ++i) {
          for (Index j = 0; j < cols; ++j) {
            const Scalar product = lhs(i) * rhs(j);
            const Index destination =
                offset + (Order == RowMajor ? i : j) * signedStride + (Order == RowMajor ? j : i) * DstStride;
            expected[destination] = operation == 0 ? product : operation == 1 ? sentinel + product : sentinel - product;
          }
        }
        for (Index i = 0; i < count; ++i) outer_packet_equal_value(storage[i], expected[i]);
      }
    }
  }
}

template <typename Real, int Order>
void outer_packet_check_special() {
  using Scalar = std::complex<Real>;
  using Mat = Matrix<Scalar, Dynamic, Dynamic, Order>;
  const Real infinity = NumTraits<Real>::infinity();
  const Real nan = NumTraits<Real>::quiet_NaN();
  const Real tiny = (std::numeric_limits<Real>::denorm_min)();
  const std::vector<Real> values = {Real(0),  -Real(0),  Real(1), Real(-1), Real(2), Real(-2),
                                    infinity, -infinity, nan,     tiny,     -tiny};
  constexpr int packetSize = internal::unpacket_traits<typename internal::packet_traits<Scalar>::type>::size;
  const Index count = (std::max)(Index(values.size()), Index(4 * packetSize + 1));
  Matrix<Scalar, Dynamic, 1> lhs(count);
  Matrix<Real, 1, Dynamic> rhs(count);
  for (Index shift = 0; shift < Index(values.size()); ++shift) {
    for (Index i = 0; i < lhs.size(); ++i) {
      lhs(i) = Scalar(values[i % values.size()], values[(i + shift) % values.size()]);
      rhs(i) = values[i % values.size()];
    }
    for (int operation = 0; operation < 3; ++operation) {
      Mat actual = Mat::Constant(lhs.size(), rhs.size(), Scalar(-Real(0), Real(1)));
      if (operation == 0) actual.noalias() = lhs * rhs;
      if (operation == 1) actual.noalias() += lhs * rhs;
      if (operation == 2) actual.noalias() -= lhs * rhs;
      Mat reversed = Mat::Constant(rhs.size(), lhs.size(), Scalar(-Real(0), Real(1)));
      if (operation == 0) reversed.noalias() = rhs.transpose() * lhs.transpose();
      if (operation == 1) reversed.noalias() += rhs.transpose() * lhs.transpose();
      if (operation == 2) reversed.noalias() -= rhs.transpose() * lhs.transpose();
      for (Index i = 0; i < lhs.size(); ++i) {
        for (Index j = 0; j < rhs.size(); ++j) {
          const Scalar product = lhs(i) * rhs(j);
          const Scalar start(-Real(0), Real(1));
          const Scalar expected = operation == 0 ? product : operation == 1 ? start + product : start - product;
          outer_packet_equal_value(actual(i, j), expected);
          outer_packet_equal_value(reversed(j, i), expected);
        }
      }
    }
  }
}

template <typename Real, int Order>
void outer_packet_check_type() {
  using Complex = std::complex<Real>;
  outer_packet_check_layout<Complex, Real, Order, 1, 1, 1>();
  outer_packet_check_layout<Real, Complex, Order, 1, 1, 1>();
  outer_packet_check_layout<Complex, Real, Order, 2, 1, 1>();
  outer_packet_check_layout<Real, Complex, Order, 1, 2, 1>();
  outer_packet_check_layout<Complex, Real, Order, 1, 1, 2>();
  outer_packet_check_layout<Real, Complex, Order, 1, 1, 2>();
  outer_packet_check_layout<Complex, Real, Order, 1, 1, 1, true>();
  outer_packet_check_layout<Real, Complex, Order, 1, 1, 1, true>();
  outer_packet_check_special<Real, Order>();
}

template <typename Real, int Order>
void outer_packet_expressions() {
  using Scalar = std::complex<Real>;
  using Mat = Matrix<Scalar, Dynamic, Dynamic, Order>;
  constexpr int packetSize = internal::unpacket_traits<typename internal::packet_traits<Scalar>::type>::size;
  const Index size = 4 * packetSize + 1;
  Matrix<Scalar, Dynamic, 1> lhs(size);
  Matrix<Real, 1, Dynamic> rhs(size);
  for (Index i = 0; i < size; ++i) {
    lhs(i) = Scalar(Real(i % 7) - Real(3), Real(i % 5) - Real(2));
    rhs(i) = Real(i % 9) - Real(4);
  }
  Mat actual = (lhs + lhs) * rhs;
  Mat reversed = rhs.transpose() * (lhs + lhs).transpose();
  for (Index i = 0; i < size; ++i) {
    for (Index j = 0; j < size; ++j) {
      const Scalar expected = (lhs(i) + lhs(i)) * rhs(j);
      outer_packet_equal_value(actual(i, j), expected);
      outer_packet_equal_value(reversed(j, i), expected);
    }
  }
  actual.noalias() += lhs.conjugate() * rhs;
  reversed.noalias() -= rhs.transpose() * lhs.adjoint();
  for (Index i = 0; i < size; ++i) {
    for (Index j = 0; j < size; ++j) {
      const Scalar initial = (lhs(i) + lhs(i)) * rhs(j);
      const Scalar increment = numext::conj(lhs(i)) * rhs(j);
      outer_packet_equal_value(actual(i, j), initial + increment);
      outer_packet_equal_value(reversed(j, i), initial - increment);
    }
  }
}

EIGEN_DECLARE_TEST(product_evaluators) {
  for (int repeat = 0; repeat < g_repeat; ++repeat) {
    CALL_SUBTEST_4((outer_packet_check_type<float, ColMajor>()));
    CALL_SUBTEST_4((outer_packet_expressions<float, ColMajor>()));
    CALL_SUBTEST_4((outer_packet_check_type<float, RowMajor>()));
    CALL_SUBTEST_4((outer_packet_expressions<float, RowMajor>()));
    CALL_SUBTEST_5((outer_packet_check_type<double, ColMajor>()));
    CALL_SUBTEST_5((outer_packet_expressions<double, ColMajor>()));
    CALL_SUBTEST_5((outer_packet_check_type<double, RowMajor>()));
    CALL_SUBTEST_5((outer_packet_expressions<double, RowMajor>()));
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
    CALL_SUBTEST_1((lazy_outer_product_strides<float, std::complex<float>, ColMajor>()));
    CALL_SUBTEST_1((lazy_outer_product_strides<std::complex<float>, float, RowMajor>()));
    CALL_SUBTEST_1((lazy_outer_product_strides<double, std::complex<double>, RowMajor>()));
    CALL_SUBTEST_1((lazy_outer_product_strides<std::complex<double>, double, ColMajor>()));
    CALL_SUBTEST_1((lazy_outer_product_strides<std::complex<double>, std::complex<double>, ColMajor>()));
    CALL_SUBTEST_1((lazy_outer_product_strides<std::complex<double>, std::complex<double>, RowMajor>()));
    CALL_SUBTEST_2((scaled_selfadjoint_diagonal_product<Lower, ColMajor>()));
    CALL_SUBTEST_2((scaled_selfadjoint_diagonal_product<Lower, RowMajor>()));
    CALL_SUBTEST_2((scaled_selfadjoint_diagonal_product<Upper, ColMajor>()));
    CALL_SUBTEST_2((scaled_selfadjoint_diagonal_product<Upper, RowMajor>()));
    CALL_SUBTEST_3((scaled_unit_triangular_product<UnitLower, ColMajor>()));
    CALL_SUBTEST_3((scaled_unit_triangular_product<UnitLower, RowMajor>()));
    CALL_SUBTEST_3((scaled_unit_triangular_product<UnitUpper, ColMajor>()));
    CALL_SUBTEST_3((scaled_unit_triangular_product<UnitUpper, RowMajor>()));
  }
}
