// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Eugene Brevdo <ebrevdo@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_TERNARY_FUNCTORS_H
#define EIGEN_TERNARY_FUNCTORS_H

#include "../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

//---------- associative ternary functors ----------

template <typename ConditionScalar, typename ThenScalar, typename ElseScalar>
struct scalar_boolean_select_op {
  static constexpr bool ThenElseAreSame = is_same<ThenScalar, ElseScalar>::value;
  EIGEN_STATIC_ASSERT(ThenElseAreSame, THEN AND ELSE MUST BE SAME TYPE)
  using Scalar = ThenScalar;
  using result_type = Scalar;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar operator()(const ConditionScalar& cond, const ThenScalar& a,
                                                          const ElseScalar& b) const {
    return cond == ConditionScalar(0) ? b : a;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& cond, const Packet& a, const Packet& b) const {
    return pselect(pcmp_eq(cond, pzero(cond)), b, a);
  }
};

template <typename ConditionScalar, typename ThenScalar, typename ElseScalar>
struct functor_traits<scalar_boolean_select_op<ConditionScalar, ThenScalar, ElseScalar>> {
  using Scalar = ThenScalar;
  enum {
    Cost = 1,
    PacketAccess = is_same<ThenScalar, ElseScalar>::value && is_same<ConditionScalar, Scalar>::value &&
                   packet_traits<Scalar>::HasBlend && packet_traits<Scalar>::HasCmp
  };
};

template <typename ConditionScalar, typename ThenScalar, typename ElseScalar>
struct scalar_bitwise_select_op {
  static constexpr bool ThenElseAreSame = is_same<ThenScalar, ElseScalar>::value;
  EIGEN_STATIC_ASSERT(ThenElseAreSame, THEN AND ELSE MUST BE SAME TYPE)
  using Scalar = ThenScalar;
  using result_type = Scalar;
  static constexpr bool CompatibleSizes = sizeof(ConditionScalar) == sizeof(Scalar);
  EIGEN_STATIC_ASSERT(CompatibleSizes, CONDITION THEN AND ELSE MUST BE SAME SIZE)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar operator()(const ConditionScalar& cond, const ThenScalar& a, const ElseScalar& b) const {
    Scalar result;
    const uint8_t* a_bytes = reinterpret_cast<const uint8_t*>(&a);
    const uint8_t* b_bytes = reinterpret_cast<const uint8_t*>(&b);
    const uint8_t* c_bytes = reinterpret_cast<const uint8_t*>(&cond);
    uint8_t* r_bytes = reinterpret_cast<uint8_t*>(&result);
    for (Index i = 0; i < sizeof(Scalar); i++) r_bytes[i] = (a_bytes[i] & c_bytes[i]) | (b_bytes[i] & ~c_bytes[i]);
    return result;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& cond, const Packet& a, const Packet& b) const {
    return pselect(cond, a, b);
  }
};

template <typename ConditionScalar, typename ThenScalar, typename ElseScalar>
struct functor_traits<scalar_bitwise_select_op<ConditionScalar, ThenScalar, ElseScalar>> {
  using Scalar = ThenScalar;
  enum {
    Cost = 1,
    PacketAccess = is_same<ThenScalar, ElseScalar>::value && is_same<ConditionScalar, Scalar>::value &&
    packet_traits<Scalar>::HasBlend
  };
};

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_TERNARY_FUNCTORS_H
