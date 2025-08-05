// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SELFCWISEBINARYOP_H
#define EIGEN_SELFCWISEBINARYOP_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

// TODO generalize the scalar type of 'other'
namespace internal {
template <typename Derived, bool Complex = NumTraits<typename DenseBase<Derived>::Scalar>::IsComplex>
struct selfcwise_helper;

template <typename Derived>
struct selfcwise_helper<Derived, /*Complex*/ false> {
  using Scalar = typename DenseBase<Derived>::Scalar;
  using ConstantExpr = typename plain_constant_type<Derived, Scalar>::type;
  using MulOp = mul_assign_op<Scalar, Scalar>;
  using DivOp = div_assign_op<Scalar, Scalar>;

  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void run_mul(Derived& derived, const Scalar& other) {
    call_assignment(derived, ConstantExpr(derived.rows(), derived.cols(), other), MulOp());
  }
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void run_div(Derived& derived, const Scalar& other) {
    call_assignment(derived, ConstantExpr(derived.rows(), derived.cols(), other), DivOp());
  }
};
template <typename Derived>
struct selfcwise_helper<Derived, /*Complex*/ true> {
  using RealScalar = typename DenseBase<Derived>::RealScalar;

  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void run_mul(Derived& derived, const RealScalar& other) {
    selfcwise_helper<Derived, false>::run_mul(derived.realView(), other);
  }
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void run_div(Derived& derived, const RealScalar& other) {
    selfcwise_helper<Derived, false>::run_div(derived.realView(), other);
  }
};
}  // namespace internal

template <typename Derived>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Derived& DenseBase<Derived>::operator*=(const Scalar& other) {
  internal::selfcwise_helper<Derived>::run_mul(derived(), other);
  return derived();
}

template <typename Derived>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Derived& DenseBase<Derived>::operator/=(const Scalar& other) {
  internal::selfcwise_helper<Derived>::run_div(derived(), other);
  return derived();
}

}  // end namespace Eigen

#endif  // EIGEN_SELFCWISEBINARYOP_H
