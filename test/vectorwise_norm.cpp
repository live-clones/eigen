// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#define EIGEN_RUNTIME_NO_MALLOC
#define TEST_ENABLE_TEMPORARY_TRACKING
#include "main.h"
#include <Eigen/Core>

template <typename Derived>
void check_vectorwise_norms(const DenseBase<Derived>& input) {
  using Real = typename Derived::RealScalar;
  const Derived& matrix = input.derived();
  const Index rows = matrix.rows();
  const Index cols = matrix.cols();
  Matrix<Real, 1, Dynamic> columnSquares(cols), columnNorms(cols);
  Matrix<Real, Dynamic, 1> rowSquares(rows), rowNorms(rows);

  using ColumnNorm = decltype(matrix.colwise().squaredNorm());
  using RowNorm = decltype(matrix.rowwise().squaredNorm());
  STATIC_CHECK(ColumnNorm::RowsAtCompileTime == 1);
  STATIC_CHECK(int(ColumnNorm::ColsAtCompileTime) == int(Derived::ColsAtCompileTime));
  STATIC_CHECK(int(RowNorm::RowsAtCompileTime) == int(Derived::RowsAtCompileTime));
  STATIC_CHECK(RowNorm::ColsAtCompileTime == 1);
  STATIC_CHECK((internal::is_same<typename internal::traits<ColumnNorm>::XprKind,
                                  typename internal::traits<Derived>::XprKind>::value));

  internal::set_is_malloc_allowed(false);
  columnSquares = matrix.colwise().squaredNorm();
  rowSquares = matrix.rowwise().squaredNorm();
  columnNorms = matrix.colwise().norm();
  rowNorms = matrix.rowwise().norm();
  internal::set_is_malloc_allowed(true);

  for (int direction = 0; direction < 2; ++direction) {
    const Index count = direction == 0 ? cols : rows;
    const Index length = direction == 0 ? rows : cols;
    for (Index k = 0; k < count; ++k) {
      long double reference = 0;
      for (Index i = 0; i < length; ++i) {
        const auto value = direction == 0 ? matrix.coeff(i, k) : matrix.coeff(k, i);
        const long double real = static_cast<long double>(numext::real(value));
        const long double imag = static_cast<long double>(numext::imag(value));
        reference += real * real + imag * imag;
      }
      const Real square = direction == 0 ? columnSquares(k) : rowSquares(k);
      const Real norm = direction == 0 ? columnNorms(k) : rowNorms(k);
      // Two squares and at most two additions per complex coefficient, plus sqrt rounding.
      const long double relativeBound = (4 * length + 4) * static_cast<long double>(NumTraits<Real>::epsilon());
      VERIFY(numext::abs(static_cast<long double>(square) - reference) <= relativeBound * reference);
      VERIFY(numext::abs(static_cast<long double>(norm) - numext::sqrt(reference)) <=
             relativeBound * numext::sqrt(reference));
    }
  }
}

template <typename Scalar, int Order>
void vectorwise_norm_layout() {
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic, Order>;
  using Real = typename NumTraits<Scalar>::Real;
  // Complex inner reductions must reach the shared squaredNorm kernel; the outer real reduction
  // must retain its existing packet path.
  using InnerReduction =
      typename VectorwiseOp<const MatrixType, Order == RowMajor ? Horizontal : Vertical>::SquaredNormReturnType;
  using InnerInput = internal::remove_all_t<decltype(std::declval<InnerReduction>().nestedExpression())>;
  STATIC_CHECK((internal::is_same<typename InnerInput::Scalar, Scalar>::value));
  using OuterReduction =
      typename VectorwiseOp<const MatrixType, Order == RowMajor ? Vertical : Horizontal>::SquaredNormReturnType;
  if (!NumTraits<Scalar>::IsComplex) {
    VERIFY(bool(internal::evaluator<OuterReduction>::Flags & PacketAccessBit) ==
           bool(internal::evaluator<MatrixType>::Flags & PacketAccessBit));
  }

  for (Index rows : {0, 1, 2, 7, 8, 9, 17}) {
    for (Index cols : {0, 1, 3, 7, 8, 9, 19}) {
      MatrixType matrix = MatrixType::Random(rows, cols);
      check_vectorwise_norms(matrix);
      check_vectorwise_norms(matrix.array());
      check_vectorwise_norms(matrix.transpose());
      check_vectorwise_norms(matrix + matrix);
    }
  }

  Matrix<Scalar, 3, 5, Order> fixed = Matrix<Scalar, 3, 5, Order>::Random();
  check_vectorwise_norms(fixed);
  Matrix<Scalar, Dynamic, Dynamic, Order, 9, 13> bounded(7, 11);
  bounded.setRandom();
  check_vectorwise_norms(bounded);

  MatrixType storage = MatrixType::Random(23, 29);
  check_vectorwise_norms(storage.block(1, 2, 17, 19));
  for (Index innerStride : {1, 2, 3}) {
    const Index innerSize = Order == RowMajor ? 19 : 17;
    const Index outerSize = Order == RowMajor ? 17 : 19;
    const Index outerStride = innerSize * innerStride + 5;
    Matrix<Scalar, Dynamic, 1> buffer = Matrix<Scalar, Dynamic, 1>::Random(outerSize * outerStride + 1);
    Map<const MatrixType, Unaligned, Stride<Dynamic, Dynamic>> strided(
        buffer.data() + 1, 17, 19, Stride<Dynamic, Dynamic>(outerStride, innerStride));
    check_vectorwise_norms(strided);
  }

  MatrixType matrix = MatrixType::Random(17, 19);
  const auto delayed = (matrix + matrix).colwise().squaredNorm();
  Matrix<Real, 1, Dynamic> actual = delayed;
  VERIFY_IS_APPROX(actual, (matrix + matrix).eval().colwise().squaredNorm());
  MatrixType product = matrix.adjoint() * matrix;
  Matrix<Real, 1, Dynamic> productNorms(19);
  // The existing real path materializes the product and its squared coefficients.
  const int expectedTemporaries = NumTraits<Scalar>::IsComplex ? 1 : 2;
  VERIFY_EVALUATION_COUNT(productNorms = (matrix.adjoint() * matrix).colwise().squaredNorm(), expectedTemporaries);
  VERIFY_IS_APPROX(productNorms, product.colwise().squaredNorm());

  matrix.setZero();
  for (Index i = 0; i < matrix.size(); ++i) {
    matrix.data()[i] = Scalar(-Real(0));
  }
  const Matrix<Real, 1, Dynamic> zeros = matrix.colwise().squaredNorm();
  for (Index i = 0; i < zeros.size(); ++i) {
    VERIFY_IS_EQUAL(zeros(i), Real(0));
    VERIFY(!numext::signbit(zeros(i)));
  }
  matrix(3, 5) = Scalar(NumTraits<Real>::infinity());
  VERIFY((numext::isinf)(matrix.colwise().norm()(5)));
  VERIFY((numext::isinf)(matrix.rowwise().squaredNorm()(3)));
  matrix(3, 5) = Scalar(NumTraits<Real>::quiet_NaN());
  VERIFY((numext::isnan)(matrix.colwise().squaredNorm()(5)));
  VERIFY((numext::isnan)(matrix.rowwise().norm()(3)));
}

EIGEN_DECLARE_TEST(vectorwise_norm) {
  CALL_SUBTEST_1((vectorwise_norm_layout<float, ColMajor>()));
  CALL_SUBTEST_2((vectorwise_norm_layout<float, RowMajor>()));
  CALL_SUBTEST_3((vectorwise_norm_layout<double, ColMajor>()));
  CALL_SUBTEST_4((vectorwise_norm_layout<double, RowMajor>()));
  CALL_SUBTEST_5((vectorwise_norm_layout<std::complex<float>, ColMajor>()));
  CALL_SUBTEST_6((vectorwise_norm_layout<std::complex<float>, RowMajor>()));
  CALL_SUBTEST_7((vectorwise_norm_layout<std::complex<double>, ColMajor>()));
  CALL_SUBTEST_8((vectorwise_norm_layout<std::complex<double>, RowMajor>()));
  CALL_SUBTEST_9(([] {
    Matrix<int, 2, 2> integers;
    integers << 3, 0, 4, 5;
    VERIFY_IS_EQUAL(integers.colwise().squaredNorm().eval(), RowVector2i(25, 25));
    VERIFY_IS_EQUAL(integers.transpose().rowwise().squaredNorm().eval(), Vector2i(25, 25));
    Array<bool, 2, 2> flags;
    flags << false, true, true, false;
    VERIFY(flags.colwise().squaredNorm().all());
    VERIFY(flags.rowwise().squaredNorm().all());
    flags.setZero();
    VERIFY(!flags.colwise().squaredNorm().any());
    VERIFY(!flags.rowwise().squaredNorm().any());
  }()));
}
