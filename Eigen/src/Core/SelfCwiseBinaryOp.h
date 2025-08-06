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

namespace internal {

template <typename Xpr, bool SpecializeReal = NumTraits<typename traits<Xpr>::Scalar>::IsComplex>
struct self_assign_helper;

template <typename Xpr>
struct self_assign_helper<Xpr, false> {
  using Scalar = typename traits<Xpr>::Scalar;
  using ConstantExpr = typename plain_constant_type<Xpr>::type;
  using MulOp = mul_assign_op<Scalar>;
  using DivOp = div_assign_op<Scalar>;

  static void EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE self_mul_assign(Xpr& xpr, const Scalar& other) {
    call_assignment(xpr, ConstantExpr(xpr.rows(), xpr.cols(), other), MulOp());
  }
  static void EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE self_div_assign(Xpr& xpr, const Scalar& other) {
    call_assignment(xpr, ConstantExpr(xpr.rows(), xpr.cols(), other), DivOp());
  }
  // TODO generalize the scalar type of 'other'
  template <typename RhsScalar>
  static void EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE self_mul_assign(Xpr& xpr, const RhsScalar& other) {
    //EIGEN_STATIC_ASSERT(false, UNSUPPORTED SCALAR TYPE)
    eigen_assert(false && "wtf??");
  }
  template <typename RhsScalar>
  static void EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE self_div_assign(Xpr& xpr, const RhsScalar& other) {
    //EIGEN_STATIC_ASSERT(false, UNSUPPORTED SCALAR TYPE)
    eigen_assert(false && "wtf??");
  }
};

template <typename Xpr>
struct self_assign_helper<Xpr, true> : self_assign_helper<Xpr, false> {
  using Scalar = typename traits<Xpr>::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using RealViewReturnType = typename DenseBase<Xpr>::RealViewReturnType;
  using ConstantExpr = typename plain_constant_type<RealViewReturnType>::type;
  using MulOp = mul_assign_op<RealScalar>;
  using DivOp = div_assign_op<RealScalar>;

  // defer to Base for Scalar and other argument types
  using Base = self_assign_helper<Xpr, false>;
  using Base::self_div_assign;
  using Base::self_mul_assign;

  static void EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE self_mul_assign(Xpr& xpr, const RealScalar& other) {
    RealViewReturnType realView = xpr.realView();
    call_assignment(realView, ConstantExpr(realView.rows(), realView.cols(), other), MulOp());
  }
  static void EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE self_div_assign(Xpr& xpr, const RealScalar& other) {
    RealViewReturnType realView = xpr.realView();
    call_assignment(realView, ConstantExpr(realView.rows(), realView.cols(), other), DivOp());
  }
};
}  // namespace internal

template <typename Derived>
template <typename RhsScalar>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Derived& DenseBase<Derived>::operator*=(const RhsScalar& other) {
  internal::self_assign_helper<Derived>::self_mul_assign(derived(), other);
  return derived();
}

template <typename Derived>
template <typename RhsScalar>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Derived& DenseBase<Derived>::operator/=(const RhsScalar& other) {
  internal::self_assign_helper<Derived>::self_div_assign(derived(), other);
  return derived();
}

}  // end namespace Eigen

#endif  // EIGEN_SELFCWISEBINARYOP_H
