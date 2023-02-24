// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_ALLANDANY_H
#define EIGEN_ALLANDANY_H

#include "./InternalHeaderCheck.h"

namespace Eigen {

/** \returns the number of coefficients which evaluate to true
  *
  * \sa all(), any()
  */
template<typename Derived>
EIGEN_DEVICE_FUNC inline Eigen::Index DenseBase<Derived>::count() const
{
  return derived().template cast<bool>().template cast<Index>().sum();
}

/** \returns true is \c *this contains at least one Not A Number (NaN).
  *
  * \sa allFinite()
  */
template<typename Derived>
EIGEN_DEVICE_FUNC inline bool DenseBase<Derived>::hasNaN() const
{
#if EIGEN_COMP_MSVC || (defined __FAST_MATH__)
  return derived().array().isNaN().any();
#else
  return !((derived().array()==derived().array()).all());
#endif
}

/** \returns true if \c *this contains only finite numbers, i.e., no NaN and no +/-INF values.
  *
  * \sa hasNaN()
  */
template<typename Derived>
EIGEN_DEVICE_FUNC inline bool DenseBase<Derived>::allFinite() const
{
#if EIGEN_COMP_MSVC || (defined __FAST_MATH__)
  return derived().array().isFinite().all();
#else
  return !((derived()-derived()).hasNaN());
#endif
}

} // end namespace Eigen

#endif // EIGEN_ALLANDANY_H
