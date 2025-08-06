// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2025 The Eigen Authors
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

template <typename T>
void test_realview(const T&) {
  using Scalar = typename T::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;

  constexpr Index minRows = T::RowsAtCompileTime == Dynamic ? 1 : T::RowsAtCompileTime;
  constexpr Index maxRows = T::MaxRowsAtCompileTime == Dynamic ? (EIGEN_TEST_MAX_SIZE / 2) : T::MaxRowsAtCompileTime;
  constexpr Index minCols = T::ColsAtCompileTime == Dynamic ? 1 : T::ColsAtCompileTime;
  constexpr Index maxCols = T::MaxColsAtCompileTime == Dynamic ? (EIGEN_TEST_MAX_SIZE / 2) : T::MaxColsAtCompileTime;

  Index rows = internal::random<Index>(minRows, maxRows);
  Index cols = internal::random<Index>(minCols, maxCols);
  Index rowFactor = (NumTraits<Scalar>::IsComplex && !T::IsRowMajor) ? 2 : 1;
  Index colFactor = (NumTraits<Scalar>::IsComplex && T::IsRowMajor) ? 2 : 1;
  Index sizeFactor = NumTraits<Scalar>::IsComplex ? 2 : 1;

  T A(rows, cols), B(rows, cols);

  VERIFY(A.realView().rows() == rowFactor * A.rows());
  VERIFY(A.realView().cols() == colFactor * A.cols());
  VERIFY(A.realView().size() == sizeFactor * A.size());

  RealScalar alpha = internal::random(RealScalar(1), RealScalar(2));
  A.setRandom();
  B = A;

  VERIFY_IS_APPROX(A.matrix().squaredNorm(), A.realView().matrix().squaredNorm());

  for (Index r = 0; r < rows; r++) {
    for (Index c = 0; c < cols; c++) {
      A.coeffRef(r, c) = A.coeff(r, c) * Scalar(alpha);
    }
  }
  B.realView() *= alpha;
  VERIFY_IS_APPROX(A, B);

  alpha = internal::random(RealScalar(1), RealScalar(2));
  A.setRandom();
  B = A;

  for (Index r = 0; r < rows; r++) {
    for (Index c = 0; c < cols; c++) {
      A.coeffRef(r, c) = A.coeff(r, c) / Scalar(alpha);
    }
  }
  B.realView() /= alpha;
  VERIFY_IS_APPROX(A, B);
}

template <typename Scalar, int Rows, int Cols, int MaxRows = Rows, int MaxCols = Cols>
void test_realview_driver() {
  // if Rows == 1, don't test ColMajor as it is not a valid array
  using ColMajorMatrixType = Matrix<Scalar, Rows, Cols, Rows == 1 ? RowMajor : ColMajor, MaxRows, MaxCols>;
  using ColMajorArrayType = Array<Scalar, Rows, Cols, Rows == 1 ? RowMajor : ColMajor, MaxRows, MaxCols>;
  // if Cols == 1, don't test RowMajor as it is not a valid array
  using RowMajorMatrixType = Matrix<Scalar, Rows, Cols, Cols == 1 ? ColMajor : RowMajor, MaxRows, MaxCols>;
  using RowMajorArrayType = Array<Scalar, Rows, Cols, Cols == 1 ? ColMajor : RowMajor, MaxRows, MaxCols>;
  test_realview(ColMajorMatrixType());
  test_realview(ColMajorArrayType());
  test_realview(RowMajorMatrixType());
  test_realview(RowMajorArrayType());
}

template <typename ComplexScalar, int Rows, int Cols, int MaxRows = Rows, int MaxCols = Cols>
void test_realview_driver_complex() {
  using RealScalar = typename NumTraits<ComplexScalar>::Real;
  test_realview_driver<RealScalar, Rows, Cols, MaxRows, MaxCols>();
  test_realview_driver<ComplexScalar, Rows, Cols, MaxRows, MaxCols>();
}

EIGEN_DECLARE_TEST(realview) {
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1((test_realview_driver_complex<std::complex<float>, Dynamic, Dynamic, Dynamic, Dynamic>()));
    CALL_SUBTEST_2((test_realview_driver_complex<std::complex<float>, Dynamic, Dynamic, 17, Dynamic>()));
    CALL_SUBTEST_3((test_realview_driver_complex<std::complex<float>, Dynamic, Dynamic, Dynamic, 19>()));
    CALL_SUBTEST_4((test_realview_driver_complex<std::complex<float>, Dynamic, Dynamic, 17, 19>()));
    CALL_SUBTEST_5((test_realview_driver_complex<std::complex<float>, 17, Dynamic, 17, Dynamic>()));
    CALL_SUBTEST_6((test_realview_driver_complex<std::complex<float>, Dynamic, 19, Dynamic, 19>()));
    CALL_SUBTEST_7((test_realview_driver_complex<std::complex<float>, 17, 19, 17, 19>()));
    CALL_SUBTEST_8((test_realview_driver_complex<std::complex<float>, Dynamic, 1>()));
    CALL_SUBTEST_9((test_realview_driver_complex<std::complex<float>, 1, Dynamic>()));
  }
}
