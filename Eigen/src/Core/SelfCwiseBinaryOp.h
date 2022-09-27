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

#include "./InternalHeaderCheck.h"

namespace Eigen { 

// TODO generalize the scalar type of 'other'

template<typename Derived>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Derived& DenseBase<Derived>::operator*=(const Scalar& other)
{
  internal::call_assignment(this->derived(), PlainObject::Constant(rows(),cols(),other), internal::mul_assign_op<Scalar,Scalar>());
  return derived();
}

template<typename Derived>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Derived& ArrayBase<Derived>::operator+=(const Scalar& other)
{
  internal::call_assignment(this->derived(), PlainObject::Constant(rows(),cols(),other), internal::add_assign_op<Scalar,Scalar>());
  return derived();
}

template<typename Derived>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Derived& ArrayBase<Derived>::operator-=(const Scalar& other)
{
  internal::call_assignment(this->derived(), PlainObject::Constant(rows(),cols(),other), internal::sub_assign_op<Scalar,Scalar>());
  return derived();
}
#ifndef EIGEN_LIBDIVIDESUPPORT_MODULE_H
template<typename Derived>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Derived& DenseBase<Derived>::operator/=(const Scalar& other)
{
  internal::call_assignment(this->derived(), PlainObject::Constant(rows(),cols(),other), internal::div_assign_op<Scalar,Scalar>());
  return derived();
}
#else
template<typename Derived, typename DivisorScalar, bool Compatible = IS_LIBDIVIDE_COMPATIBLE(typename Derived::Scalar, DivisorScalar)>
struct quotient_helper
{
	// Derived::Scalar is not a libdivide type and/or DivisorScalar is not representable as one
	typedef typename Derived::Scalar Scalar;
	typedef typename Derived::PlainObject PlainObject;
	static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Derived& run(Derived& lhs, const DivisorScalar& other)
	{
		// preserve current behavior for non-libdivide expressions : implicit conversion of DivisorScalar to Scalar
		// todo: investigate impact of fixing current behavior, the root cause of which is PlainObject::Constant
		internal::call_assignment(lhs.derived(), PlainObject::Constant(lhs.rows(), lhs.cols(), static_cast<Scalar>(other)), internal::div_assign_op<Scalar, Scalar>());
		return lhs.derived();
	}
};
template<typename Derived, typename DivisorScalar>
struct quotient_helper<Derived,DivisorScalar,true>
{
	// Derived::Scalar is a libdivide type and DivisorScalar can be represented as one
	typedef typename Derived::Scalar Scalar;
	typedef internal::scalar_libdivide_op<Scalar, DivisorScalar> LibdivideOp;
	typedef CwiseUnaryOp<LibdivideOp, const Derived> CwiseLibdivideReturnType;
	static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Derived& run(Derived& lhs, const DivisorScalar& other)
	{
		// todo: is it necessary to define assignment ops?
		internal::call_assignment(lhs.derived(), CwiseLibdivideReturnType(lhs.derived(), LibdivideOp(other)));
		return lhs.derived();
	}
};
template<typename Derived>
template<typename DivisorScalar>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Derived& DenseBase<Derived>::operator/=(const DivisorScalar& other)
{
	return quotient_helper<Derived, DivisorScalar>::run(derived(), other);
}
#endif // !EIGEN_LIBDIVIDESUPPORT_H

} // end namespace Eigen

#endif // EIGEN_SELFCWISEBINARYOP_H
