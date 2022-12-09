// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2016 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_NULLARY_FUNCTORS_H
#define EIGEN_NULLARY_FUNCTORS_H

#include "../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

template<typename Scalar>
struct scalar_constant_op {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE scalar_constant_op(const scalar_constant_op& other) : m_other(other.m_other) { }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE scalar_constant_op(const Scalar& other) : m_other(other) { }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() () const { return m_other; }
  template<typename PacketType>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const PacketType packetOp() const { return internal::pset1<PacketType>(m_other); }
  const Scalar m_other;
};
template<typename Scalar>
struct functor_traits<scalar_constant_op<Scalar> >
{ enum { Cost = 0 /* as the constant value should be loaded in register only once for the whole expression */,
         PacketAccess = packet_traits<Scalar>::Vectorizable, IsRepeatable = true }; };

template<typename Scalar> struct scalar_identity_op {
  template<typename IndexType>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() (IndexType row, IndexType col) const { return row==col ? Scalar(1) : Scalar(0); }
};

template <typename Scalar, bool FastMode>
struct linspaced_op_impl {
  // this specialization is intended (though not required) for situations where (high-low) is a multiple of (num_steps-1)
  // for example, VectorXi::LinSpaced(n,1,n) -> {1, 2 ... n}
  // this specialization does not require vectorized division
  // default for floating point types
  EIGEN_DEVICE_FUNC linspaced_op_impl(const Scalar& low, const Scalar& high, Index num_steps)
      : m_low(low), m_multiplier(num_steps <= Scalar(1) ? Scalar(1) : (high - low) / (num_steps - Scalar(1))) {}

  template <typename IndexType>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator()(IndexType i) const {
    return m_low + (m_multiplier * convert_index<Scalar>(i));
  }

  template <typename Packet, typename IndexType>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(IndexType i) const {
    const Packet plow = pset1<Packet>(m_low);
    const Packet pmult = pset1<Packet>(m_multiplier);
    Packet pi = plset<Packet>(convert_index<Scalar>(i));
    return pmadd(pmult, pi, plow);
  }

  const Scalar m_multiplier;
  const Scalar m_low;
};

template <typename Scalar>
struct linspaced_op_impl<Scalar, false> {
  // this specialization better handles several cases,
  // particularly where Scalar is an integer and (high-low) is NOT a multiple of (num_steps-1)
  // FastMode == false : VectorXi::LinSpaced(10,1,7) -> {1 1 2 3 3 4 5 5 6 7}
  // FastMode == true :  VectorXi::LinSpaced(10,1,7) -> {1 1 1 1 1 1 1 1 1 1}
  // default for integer types
  EIGEN_DEVICE_FUNC linspaced_op_impl(const Scalar& low, const Scalar& high, Index num_steps)
      : m_low(low), m_delta(high - low), m_countm1(num_steps <= Scalar(1) ? Scalar(1) : num_steps - Scalar(1)) {}

  template <typename IndexType>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator()(IndexType i) const {
    return ((m_delta * convert_index<Scalar>(i)) / m_countm1) + m_low;
  }

  template <typename Packet, typename IndexType>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(IndexType i) const {
    const Packet plow = pset1<Packet>(m_low);
    const Packet pdelta = pset1<Packet>(m_delta);
    const Packet pcountm1 = pset1<Packet>(m_countm1);
    Packet pi = plset<Packet>(convert_index<Scalar>(i));
    return padd(pdiv(pmul(pdelta, pi), pcountm1), plow);
  }

  const Scalar m_delta;
  const Scalar m_countm1;
  const Scalar m_low;
};

// ----- Linspace functor ----------------------------------------------------------------

// Forward declaration (we default to random access which does not really give
// us a speed gain when using packet access but it allows to use the functor in
// nested expressions).
template <typename Scalar, bool FastMode>
struct linspaced_op;
template <typename Scalar, bool FastMode>
struct functor_traits<linspaced_op<Scalar, FastMode> > {
    enum {
        Cost = 1,
        PacketAccess =
        packet_traits<Scalar>::HasSetLinear && packet_traits<Scalar>::HasAdd && packet_traits<Scalar>::HasMul,
        IsRepeatable = true
    };
};
template <typename Scalar>
struct functor_traits<linspaced_op<Scalar, false> > {
    enum {
        Cost = 1,
        PacketAccess = packet_traits<Scalar>::HasSetLinear && packet_traits<Scalar>::HasAdd &&
        packet_traits<Scalar>::HasMul && packet_traits<Scalar>::HasDiv,
        IsRepeatable = true
    };
};
template <typename Scalar, bool FastMode> struct linspaced_op
{
    EIGEN_DEVICE_FUNC linspaced_op(const Scalar& low, const Scalar& high, Index num_steps)
        : impl(low, high, num_steps)
    {}

    template<typename IndexType>
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() (IndexType i) const { return impl(i); }

    template<typename Packet, typename IndexType>
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(IndexType i) const { return impl.template packetOp<Packet>(i); }

    // This proxy object handles the actual required temporaries and the different
    // implementations (integer vs. floating point).
    const linspaced_op_impl<Scalar, FastMode> impl;
};

// Linear access is automatically determined from the operator() prototypes available for the given functor.
// If it exposes an operator()(i,j), then we assume the i and j coefficients are required independently
// and linear access is not possible. In all other cases, linear access is enabled.
// Users should not have to deal with this structure.
template<typename Functor> struct functor_has_linear_access { enum { ret = !has_binary_operator<Functor>::value }; };

// For unreliable compilers, let's specialize the has_*ary_operator
// helpers so that at least built-in nullary functors work fine.
#if !( EIGEN_COMP_MSVC || EIGEN_COMP_GNUC || (EIGEN_COMP_ICC>=1600))
template<typename Scalar,typename IndexType>
struct has_nullary_operator<scalar_constant_op<Scalar>,IndexType> { enum { value = 1}; };
template<typename Scalar,typename IndexType>
struct has_unary_operator<scalar_constant_op<Scalar>,IndexType> { enum { value = 0}; };
template<typename Scalar,typename IndexType>
struct has_binary_operator<scalar_constant_op<Scalar>,IndexType> { enum { value = 0}; };

template<typename Scalar,typename IndexType>
struct has_nullary_operator<scalar_identity_op<Scalar>,IndexType> { enum { value = 0}; };
template<typename Scalar,typename IndexType>
struct has_unary_operator<scalar_identity_op<Scalar>,IndexType> { enum { value = 0}; };
template<typename Scalar,typename IndexType>
struct has_binary_operator<scalar_identity_op<Scalar>,IndexType> { enum { value = 1}; };

template<typename Scalar,typename IndexType>
struct has_nullary_operator<linspaced_op<Scalar>,IndexType> { enum { value = 0}; };
template<typename Scalar,typename IndexType>
struct has_unary_operator<linspaced_op<Scalar>,IndexType> { enum { value = 1}; };
template<typename Scalar,typename IndexType>
struct has_binary_operator<linspaced_op<Scalar>,IndexType> { enum { value = 0}; };

template<typename Scalar,typename IndexType>
struct has_nullary_operator<scalar_random_op<Scalar>,IndexType> { enum { value = 1}; };
template<typename Scalar,typename IndexType>
struct has_unary_operator<scalar_random_op<Scalar>,IndexType> { enum { value = 0}; };
template<typename Scalar,typename IndexType>
struct has_binary_operator<scalar_random_op<Scalar>,IndexType> { enum { value = 0}; };
#endif

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_NULLARY_FUNCTORS_H
