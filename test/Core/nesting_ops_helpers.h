// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Hauke Heibel <hauke.heibel@gmail.com>
// Copyright (C) 2015 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Shared header for split nesting_ops tests.

#ifndef EIGEN_TEST_NESTING_OPS_HELPERS_H
#define EIGEN_TEST_NESTING_OPS_HELPERS_H

#define TEST_ENABLE_TEMPORARY_TRACKING

#include "main.h"

template <int N, typename XprType>
void use_n_times(const XprType& xpr) {
  typename internal::nested_eval<XprType, N>::type mat(xpr);
  typename XprType::PlainObject res(mat.rows(), mat.cols());
  nb_temporaries--;  // remove res
  res.setZero();
  for (int i = 0; i < N; ++i) res += mat;
}

template <int N, typename ReferenceType, typename XprType>
bool verify_eval_type(const XprType&, const ReferenceType&) {
  typedef typename internal::nested_eval<XprType, N>::type EvalType;
  return internal::is_same<internal::remove_all_t<EvalType>, internal::remove_all_t<ReferenceType>>::value;
}

template <typename MatrixType>
void run_nesting_ops_1(const MatrixType& _m) {
  typename internal::nested_eval<MatrixType, 2>::type m(_m);

  // Make really sure that we are in debug mode!
  VERIFY_RAISES_ASSERT(eigen_assert(false));

  // The only intention of these tests is to ensure that this code does
  // not trigger any asserts or segmentation faults... more to come.
  VERIFY_IS_APPROX((m.transpose() * m).diagonal().sum(), (m.transpose() * m).diagonal().sum());
  VERIFY_IS_APPROX((m.transpose() * m).diagonal().array().abs().sum(),
                   (m.transpose() * m).diagonal().array().abs().sum());

  VERIFY_IS_APPROX((m.transpose() * m).array().abs().sum(), (m.transpose() * m).array().abs().sum());
}

template <typename MatrixType>
void run_nesting_ops_2(const MatrixType& _m) {
  typedef typename MatrixType::Scalar Scalar;
  Index rows = _m.rows();
  Index cols = _m.cols();
  MatrixType m1 = MatrixType::Random(rows, cols);
  Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::ColsAtCompileTime, ColMajor> m2;

  if ((MatrixType::SizeAtCompileTime == Dynamic)) {
    VERIFY_EVALUATION_COUNT(use_n_times<1>(m1 + m1 * m1), 1);
    VERIFY_EVALUATION_COUNT(use_n_times<10>(m1 + m1 * m1), 1);

    VERIFY_EVALUATION_COUNT(use_n_times<1>(m1.template triangularView<Lower>().solve(m1.col(0))), 1);
    VERIFY_EVALUATION_COUNT(use_n_times<10>(m1.template triangularView<Lower>().solve(m1.col(0))), 1);

    VERIFY_EVALUATION_COUNT(use_n_times<1>(Scalar(2) * m1.template triangularView<Lower>().solve(m1.col(0))),
                            2);  // FIXME could be one by applying the scaling in-place on the solve result
    VERIFY_EVALUATION_COUNT(use_n_times<1>(m1.col(0) + m1.template triangularView<Lower>().solve(m1.col(0))),
                            2);  // FIXME could be one by adding m1.col() inplace
    VERIFY_EVALUATION_COUNT(use_n_times<10>(m1.col(0) + m1.template triangularView<Lower>().solve(m1.col(0))), 2);
  }

  {
    VERIFY(verify_eval_type<10>(m1, m1));
    if (!NumTraits<Scalar>::IsComplex) {
      VERIFY(verify_eval_type<3>(2 * m1, 2 * m1));
      VERIFY(verify_eval_type<4>(2 * m1, m1));
    } else {
      VERIFY(verify_eval_type<2>(2 * m1, 2 * m1));
      VERIFY(verify_eval_type<3>(2 * m1, m1));
    }
    VERIFY(verify_eval_type<2>(m1 + m1, m1 + m1));
    VERIFY(verify_eval_type<3>(m1 + m1, m1));
    VERIFY(verify_eval_type<1>(m1 * m1.transpose(), m2));
    VERIFY(verify_eval_type<1>(m1 * (m1 + m1).transpose(), m2));
    VERIFY(verify_eval_type<2>(m1 * m1.transpose(), m2));
    VERIFY(verify_eval_type<1>(m1 + m1 * m1, m1));

    VERIFY(verify_eval_type<1>(m1.template triangularView<Lower>().solve(m1), m1));
    VERIFY(verify_eval_type<1>(m1 + m1.template triangularView<Lower>().solve(m1), m1));
  }
}

#endif  // EIGEN_TEST_NESTING_OPS_HELPERS_H
