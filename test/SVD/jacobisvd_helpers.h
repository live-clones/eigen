// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Shared header for split jacobisvd tests.

#ifndef EIGEN_TEST_JACOBISVD_HELPERS_H
#define EIGEN_TEST_JACOBISVD_HELPERS_H

// discard stack allocation as that too bypasses malloc
#define EIGEN_STACK_ALLOCATION_LIMIT 0
#define EIGEN_RUNTIME_NO_MALLOC
#include "main.h"
#include <Eigen/SVD>

#define SVD_DEFAULT(M) JacobiSVD<M>
#define SVD_FOR_MIN_NORM(M) JacobiSVD<M, ColPivHouseholderQRPreconditioner>
#define SVD_STATIC_OPTIONS(M, O) JacobiSVD<M, O>
#include "svd_common.h"

template <typename MatrixType>
void jacobisvd_method() {
  enum { Size = MatrixType::RowsAtCompileTime };
  typedef typename MatrixType::RealScalar RealScalar;
  typedef Matrix<RealScalar, Size, 1> RealVecType;
  MatrixType m = MatrixType::Identity();
  VERIFY_IS_APPROX(m.jacobiSvd().singularValues(), RealVecType::Ones());
  VERIFY_RAISES_ASSERT(m.jacobiSvd().matrixU());
  VERIFY_RAISES_ASSERT(m.jacobiSvd().matrixV());
  VERIFY_IS_APPROX(m.template jacobiSvd<ComputeFullU | ComputeFullV>().solve(m), m);
  VERIFY_IS_APPROX(m.template jacobiSvd<ComputeFullU | ComputeFullV>().transpose().solve(m), m);
  VERIFY_IS_APPROX(m.template jacobiSvd<ComputeFullU | ComputeFullV>().adjoint().solve(m), m);
}

template <typename MatrixType>
void jacobisvd_thin_options(const MatrixType& input = MatrixType()) {
  MatrixType m(input.rows(), input.cols());
  svd_fill_random(m);

  svd_thin_option_checks<MatrixType, 0>(m);
  svd_thin_option_checks<MatrixType, ColPivHouseholderQRPreconditioner>(m);
  svd_thin_option_checks<MatrixType, HouseholderQRPreconditioner>(m);
}

template <typename MatrixType>
void jacobisvd_full_options(const MatrixType& input = MatrixType()) {
  MatrixType m(input.rows(), input.cols());
  svd_fill_random(m);

  svd_option_checks_full_only<MatrixType, 0>(m);
  svd_option_checks_full_only<MatrixType, ColPivHouseholderQRPreconditioner>(m);
  svd_option_checks_full_only<MatrixType, HouseholderQRPreconditioner>(m);
  svd_option_checks_full_only<MatrixType, FullPivHouseholderQRPreconditioner>(
      m);  // FullPiv only used when computing full unitaries
}

template <typename MatrixType>
void jacobisvd_verify_assert(const MatrixType& input = MatrixType()) {
  MatrixType m(input.rows(), input.cols());
  svd_fill_random(m);
  svd_verify_assert<MatrixType, 0>(m);
  svd_verify_assert<MatrixType, ColPivHouseholderQRPreconditioner>(m);
  svd_verify_assert<MatrixType, HouseholderQRPreconditioner>(m);
  svd_verify_assert_full_only<MatrixType, FullPivHouseholderQRPreconditioner>(m);

  svd_verify_constructor_options_assert<JacobiSVD<MatrixType>>(m);
  svd_verify_constructor_options_assert<JacobiSVD<MatrixType, ColPivHouseholderQRPreconditioner>>(m);
  svd_verify_constructor_options_assert<JacobiSVD<MatrixType, HouseholderQRPreconditioner>>(m);
  svd_verify_constructor_options_assert<JacobiSVD<MatrixType, FullPivHouseholderQRPreconditioner>>(m);
}

template <typename MatrixType>
void jacobisvd_verify_inputs(const MatrixType& input = MatrixType()) {
  // check defaults
  typedef JacobiSVD<MatrixType> DefaultSVD;
  MatrixType m(input.rows(), input.cols());
  svd_fill_random(m);
  DefaultSVD defaultSvd(m);
  VERIFY((int)DefaultSVD::QRPreconditioner == (int)ColPivHouseholderQRPreconditioner);
  VERIFY(!defaultSvd.computeU());
  VERIFY(!defaultSvd.computeV());

  // ColPivHouseholderQR is always default in presence of other options.
  VERIFY(((int)JacobiSVD<MatrixType, ComputeThinU>::QRPreconditioner == (int)ColPivHouseholderQRPreconditioner));
  VERIFY(((int)JacobiSVD<MatrixType, ComputeThinV>::QRPreconditioner == (int)ColPivHouseholderQRPreconditioner));
  VERIFY(((int)JacobiSVD<MatrixType, ComputeThinU | ComputeThinV>::QRPreconditioner ==
          (int)ColPivHouseholderQRPreconditioner));
  VERIFY(((int)JacobiSVD<MatrixType, ComputeFullU | ComputeFullV>::QRPreconditioner ==
          (int)ColPivHouseholderQRPreconditioner));
  VERIFY(((int)JacobiSVD<MatrixType, ComputeThinU | ComputeFullV>::QRPreconditioner ==
          (int)ColPivHouseholderQRPreconditioner));
  VERIFY(((int)JacobiSVD<MatrixType, ComputeFullU | ComputeThinV>::QRPreconditioner ==
          (int)ColPivHouseholderQRPreconditioner));
}

template <typename MatrixType>
void svd_triangular_matrix(const MatrixType& input = MatrixType()) {
  MatrixType matrix(input.rows(), input.cols());
  svd_fill_random(matrix);
  // Make sure that we only consider the 'Lower' part of the matrix.
  MatrixType matrix_self_adj = matrix.template selfadjointView<Lower>().toDenseMatrix();

  JacobiSVD<MatrixType, ComputeFullV> svd_triangular(matrix.template selfadjointView<Lower>());
  JacobiSVD<MatrixType, ComputeFullV> svd_full(matrix_self_adj);

  VERIFY_IS_APPROX(svd_triangular.singularValues(), svd_full.singularValues());
}

namespace Foo {
class Bar {
 public:
  Bar() {}
};
inline bool operator<(const Bar&, const Bar&) { return true; }
}  // namespace Foo

inline void msvc_workaround() {
  const Foo::Bar a;
  const Foo::Bar b;
  const Foo::Bar c = std::max EIGEN_NOT_A_MACRO(a, b);
  EIGEN_UNUSED_VARIABLE(c)
}

#endif  // EIGEN_TEST_JACOBISVD_HELPERS_H
