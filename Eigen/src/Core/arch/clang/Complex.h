// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2025 Rasmus Munk Larsen
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_COMPLEX_CLANG_H
#define EIGEN_COMPLEX_CLANG_H

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {
namespace internal {

struct Packet8cf {
  EIGEN_STRONG_INLINE Packet8cf() {}
  EIGEN_STRONG_INLINE explicit Packet8cf(const Packet16f& a) : v(a) {}
  Packet16f v;
};

template <>
struct packet_traits<std::complex<float>> : generic_float_packet_traits {
  using type = Packet8cf;
  using half = Packet8cf;
  enum {
    size = 8,
    HasAbs = 0,
    HasAbs2 = 0,
    HasMin = 0,
    HasMax = 0,
    HasSetLinear = 0
  };
};
template <>
struct unpacket_traits<Packet8cf> : generic_unpacket_traits {
  using type = std::complex<float>;
  using half = Packet8cf;
  using as_real = Packet16f;
  enum {
    size = 16,
  };
};

#define EIGEN_CLANG_COMPLEX_LOAD_STORE(PACKET_TYPE)                                                       \
  template <>                                                                                             \
  EIGEN_STRONG_INLINE PACKET_TYPE ploadu<PACKET_TYPE>(const unpacket_traits<PACKET_TYPE>::type* from) {   \
    return PACKET_TYPE(ploadu<typename unpacket_traits<PACKET_TYPE>::as_real>(&numext::real_ref(*from))); \
  }                                                                                                       \
  template <>                                                                                             \
  EIGEN_STRONG_INLINE PACKET_TYPE pload<PACKET_TYPE>(const unpacket_traits<PACKET_TYPE>::type* from) {    \
    return PACKET_TYPE(pload<typename unpacket_traits<PACKET_TYPE>::as_real>(&numext::real_ref(*from)));  \
  }                                                                                                       \
  template <>                                                                                             \
  EIGEN_STRONG_INLINE void pstoreu<typename unpacket_traits<PACKET_TYPE>::type, PACKET_TYPE>(             \
      typename unpacket_traits<PACKET_TYPE>::type * to, const PACKET_TYPE& from) {                        \
    pstoreu(&numext::real_ref(*to), from.v);                                                              \
  }                                                                                                       \
  template <>                                                                                             \
  EIGEN_STRONG_INLINE void pstore<typename unpacket_traits<PACKET_TYPE>::type, PACKET_TYPE>(              \
      typename unpacket_traits<PACKET_TYPE>::type * to, const PACKET_TYPE& from) {                        \
    pstore(&numext::real_ref(*to), from.v);                                                               \
  }

EIGEN_CLANG_COMPLEX_LOAD_STORE(Packet8cf);
#undef EIGEN_CLANG_COMPLEX_LOAD_STORE

template <>
EIGEN_STRONG_INLINE Packet8cf pset1<Packet8cf>(const std::complex<float>& from) {
  Packet8cf a;
  a.v[0] = std::real(from);
  a.v[1] = std::imag(from);
  return Packet8cf(__builtin_shufflevector(a.v, a.v, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1));
}

// ----------- Unary ops ------------------
#define DELEGATE_UNARY_TO_REAL_OP(PACKET_TYPE, OP)                \
  template <>                                                                  \
  EIGEN_STRONG_INLINE PACKET_TYPE OP<PACKET_TYPE>(const PACKET_TYPE& a) { \
    return PACKET_TYPE(OP(a.v));                                          \
  }

#define EIGEN_CLANG_COMPLEX_UNARY_CWISE_OPS(PACKET_TYPE)                     \
  DELEGATE_UNARY_TO_REAL_OP(PACKET_TYPE, pnegate)                            \
  DELEGATE_UNARY_TO_REAL_OP(PACKET_TYPE, ptrue)

EIGEN_CLANG_COMPLEX_UNARY_CWISE_OPS(Packet8cf);

#undef DELEGATE_UNARY_TO_REAL_OP
#undef EIGEN_CLANG_COMPLEX_UNARY_CWISE_OPS


// ----------- Binary ops ------------------
#define DELEGATE_BINARY_TO_REAL_OP(PACKET_TYPE, OP)                                             \
  template <>                                                                                   \
  EIGEN_STRONG_INLINE PACKET_TYPE OP<PACKET_TYPE>(const PACKET_TYPE& a, const PACKET_TYPE& b) { \
    return PACKET_TYPE(OP(a.v, b.v));                                                           \
  }

#define EIGEN_CLANG_COMPLEX_BINARY_CWISE_OPS(PACKET_TYPE) \
  DELEGATE_BINARY_TO_REAL_OP(PACKET_TYPE, padd)                  \
  DELEGATE_BINARY_TO_REAL_OP(PACKET_TYPE, psub)                  \
  DELEGATE_BINARY_TO_REAL_OP(PACKET_TYPE, pcmp_eq)               \
  DELEGATE_BINARY_TO_REAL_OP(PACKET_TYPE, pand)                  \
  DELEGATE_BINARY_TO_REAL_OP(PACKET_TYPE, por)                   \
  DELEGATE_BINARY_TO_REAL_OP(PACKET_TYPE, pxor)                  \
  DELEGATE_BINARY_TO_REAL_OP(PACKET_TYPE, pandnot)

EIGEN_CLANG_COMPLEX_BINARY_CWISE_OPS(Packet8cf);

#undef DELEGATE_BINARY_TO_REAL_OP
#undef EIGEN_CLANG_COMPLEX_BINARY_CWISE_OPS

#define EIGEN_CLANG_COMPLEX_TERNARY_CWISE_OPS(PACKET_TYPE)

#define EIGEN_CLANG_COMPLEX_MOAR_UNARY_CWISE_OPS(PACKET_TYPE)                \
  template <>                                                                \
  EIGEN_STRONG_INLINE PACKET_TYPE pconj<PACKET_TYPE>(const PACKET_TYPE& a) { \
    using Scalar = unpacket_traits<PACKET_TYPE>::type;                       \
    using RealScalar = NumTraits<Scalar>::Real;                              \
    const Scalar v(RealScalar(0), -RealScalar(0));                           \
    return pxor(a, pset1<PACKET_TYPE>(v));                                   \
  }

EIGEN_CLANG_COMPLEX_MOAR_UNARY_CWISE_OPS(Packet8cf);

#undef EIGEN_CLANG_COMPLEX_MOAR_UNARY_CWISE_OPS

}  // end namespace internal
}  // end namespace Eigen

#endif  // EIGEN_COMPLEX_CLANG_H
