// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// lu split: real scalar types and fixed-size matrices.

#include "lu_helpers.h"

// Test LU decomposition at blocking and vectorization boundaries.
// PartialPivLU uses blocks of size max(8, min(size/8 rounded to 16, 256)).
// Sizes near these boundaries exercise transitions between full blocks
// and remainder tails, including pivot propagation across block edges.
template <typename Scalar>
void lu_blocking_boundary() {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, Dynamic> MatrixType;

  const Index PS = internal::packet_traits<Scalar>::size;
  const Index sizes[] = {1, 2, 3,  PS - 1, PS, PS + 1, 2 * PS - 1, 2 * PS, 2 * PS + 1, 4 * PS, 4 * PS + 1, 7,
                         8, 9, 15, 16,     17, 31,     32,         33,     63,         64,     65};
  for (Index si = 0; si < Index(sizeof(sizes) / sizeof(sizes[0])); ++si) {
    Index n = sizes[si];
    if (n < 1) continue;

    // Create a diagonally dominant (invertible) matrix.
    MatrixType m = MatrixType::Random(n, n);
    m.diagonal().array() += RealScalar(2 * n);

    // PartialPivLU
    PartialPivLU<MatrixType> plu(m);
    VERIFY_IS_APPROX(m, plu.reconstructedMatrix());
    MatrixType rhs = MatrixType::Random(n, 3);
    MatrixType x = plu.solve(rhs);
    VERIFY_IS_APPROX(m * x, rhs);

    // FullPivLU
    FullPivLU<MatrixType> flu(m);
    VERIFY_IS_APPROX(m, flu.reconstructedMatrix());
    VERIFY(flu.isInvertible());
    x = flu.solve(rhs);
    VERIFY_IS_APPROX(m * x, rhs);
  }

  // Non-square matrices at boundary sizes for FullPivLU.
  const Index rect_sizes[][2] = {{PS, 2 * PS}, {2 * PS, PS}, {15, 33}, {33, 15}, {1, 5}, {5, 1}};
  for (Index si = 0; si < Index(sizeof(rect_sizes) / sizeof(rect_sizes[0])); ++si) {
    Index rows = rect_sizes[si][0];
    Index cols = rect_sizes[si][1];
    MatrixType m = MatrixType::Random(rows, cols);
    FullPivLU<MatrixType> flu(m);
    VERIFY_IS_APPROX(m, flu.reconstructedMatrix());
  }
}

// Test PartialPivLU with RowMajor storage order at blocking boundaries.
template <typename Scalar>
void lu_rowmajor_boundary() {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, Dynamic, RowMajor> RowMatrixType;

  const Index sizes[] = {7, 8, 9, 15, 16, 17, 31, 32, 33};
  for (Index si = 0; si < Index(sizeof(sizes) / sizeof(sizes[0])); ++si) {
    Index n = sizes[si];
    RowMatrixType m = RowMatrixType::Random(n, n);
    m.diagonal().array() += RealScalar(2 * n);
    PartialPivLU<RowMatrixType> plu(m);
    VERIFY_IS_APPROX(m, plu.reconstructedMatrix());
  }
}

TEST(LUTest, Basic) {
  for (int i = 0; i < g_repeat; i++) {
    lu_non_invertible<Matrix3f>();
    lu_invertible<Matrix3f>();
    lu_verify_assert<Matrix3f>();
    lu_partial_piv<Matrix3f>();

    (lu_non_invertible<Matrix<double, 4, 6> >());
    (lu_verify_assert<Matrix<double, 4, 6> >());
    lu_partial_piv<Matrix2d>();
    lu_partial_piv<Matrix4d>();
    (lu_partial_piv<Matrix<double, 6, 6> >());

    lu_non_invertible<MatrixXf>();
    lu_invertible<MatrixXf>();
    lu_verify_assert<MatrixXf>();

    lu_non_invertible<MatrixXd>();
    lu_invertible<MatrixXd>();
    lu_partial_piv<MatrixXd>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE));
    lu_verify_assert<MatrixXd>();

    (lu_non_invertible<Matrix<float, Dynamic, 16> >());

    // Test problem size constructors
    PartialPivLU<MatrixXf>(10);
    FullPivLU<MatrixXf>(10, 20);

    test_2889();
  }

}

// Blocking and vectorization boundary tests (deterministic, outside g_repeat).
TEST(LUTest, BlockingBoundary) {
  lu_blocking_boundary<float>();
  lu_blocking_boundary<double>();
  lu_blocking_boundary<std::complex<float> >();
  lu_blocking_boundary<std::complex<double> >();
}

TEST(LUTest, RowMajorBoundary) {
  lu_rowmajor_boundary<double>();
  lu_rowmajor_boundary<std::complex<float> >();
}
