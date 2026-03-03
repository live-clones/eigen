// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// cholesky split: real scalar types, edge cases, and assertions.
// Complex types are in cholesky_complex.cpp.

#include "cholesky_helpers.h"

// Test Cholesky decomposition at blocking and vectorization boundaries.
// LLT uses blocks of size max(8, min(size/8 rounded to 16, 128)).
// Sizes near these boundaries exercise the transition between full
// blocked and unblocked paths, including triangular solve boundaries.
template <typename Scalar>
void cholesky_blocking_boundary() {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, Dynamic> MatrixType;
  typedef Matrix<Scalar, Dynamic, 1> VectorType;

  const Index PS = internal::packet_traits<Scalar>::size;
  const Index sizes[] = {1, 2, 3,  PS - 1, PS, PS + 1, 2 * PS - 1, 2 * PS, 2 * PS + 1, 4 * PS, 4 * PS + 1, 7,
                         8, 9, 15, 16,     17, 31,     32,         33,     63,         64,     65};
  for (Index si = 0; si < Index(sizeof(sizes) / sizeof(sizes[0])); ++si) {
    Index n = sizes[si];
    if (n < 1) continue;

    // Create a symmetric positive definite matrix: A = R'*R + n*I
    MatrixType R = MatrixType::Random(n, n);
    MatrixType m = R.adjoint() * R;
    m.diagonal().array() += RealScalar(n);

    // LLT
    LLT<MatrixType> llt(m);
    VERIFY(llt.info() == Success);
    VERIFY_IS_APPROX(m, llt.reconstructedMatrix());
    VectorType rhs = VectorType::Random(n);
    VectorType x = llt.solve(rhs);
    VERIFY_IS_APPROX(m * x, rhs);

    // LDLT
    LDLT<MatrixType> ldlt(m);
    VERIFY(ldlt.info() == Success);
    VERIFY_IS_APPROX(m, ldlt.reconstructedMatrix());
    x = ldlt.solve(rhs);
    VERIFY_IS_APPROX(m * x, rhs);
  }
}

// Test Cholesky with RowMajor storage at blocking boundaries.
template <typename Scalar>
void cholesky_rowmajor_boundary() {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, Dynamic, RowMajor> RowMatrixType;
  typedef Matrix<Scalar, Dynamic, 1> VectorType;

  const Index sizes[] = {7, 8, 9, 15, 16, 17, 31, 32, 33};
  for (Index si = 0; si < Index(sizeof(sizes) / sizeof(sizes[0])); ++si) {
    Index n = sizes[si];
    RowMatrixType R = RowMatrixType::Random(n, n);
    RowMatrixType m = R.adjoint() * R;
    m.diagonal().array() += RealScalar(n);

    LLT<RowMatrixType> llt(m);
    VERIFY(llt.info() == Success);
    VERIFY_IS_APPROX(m, llt.reconstructedMatrix());

    LDLT<RowMatrixType> ldlt(m);
    VERIFY(ldlt.info() == Success);
    VERIFY_IS_APPROX(m, ldlt.reconstructedMatrix());
  }
}

TEST(CholeskyTest, Basic) {
  int s = 0;
  for (int i = 0; i < g_repeat; i++) {
    cholesky(Matrix<double, 1, 1>());
    cholesky(Matrix2d());
    cholesky_bug241(Matrix2d());
    cholesky_definiteness(Matrix2d());
    cholesky(Matrix3f());
    cholesky(Matrix4d());

    s = internal::random<int>(1, EIGEN_TEST_MAX_SIZE);
    cholesky(MatrixXd(s, s));
    TEST_SET_BUT_UNUSED_VARIABLE(s);
  }
  // empty matrix, regression test for Bug 785:
  cholesky(MatrixXd(0, 0));

  cholesky_verify_assert<Matrix3f>();
  cholesky_verify_assert<Matrix3d>();
  cholesky_verify_assert<MatrixXf>();
  cholesky_verify_assert<MatrixXd>();

  // Test problem size constructors
  LLT<MatrixXf>(10);
  LDLT<MatrixXf>(10);

  cholesky_faillure_cases<void>();

  TEST_SET_BUT_UNUSED_VARIABLE(nb_temporaries);
}

TEST(CholeskyTest, BlockingBoundary) {
  // Blocking and vectorization boundary tests (deterministic, outside g_repeat).
  cholesky_blocking_boundary<double>();
  cholesky_blocking_boundary<float>();
  cholesky_blocking_boundary<std::complex<double> >();
}

TEST(CholeskyTest, RowMajorBoundary) {
  cholesky_rowmajor_boundary<double>();
  cholesky_rowmajor_boundary<float>();
}
