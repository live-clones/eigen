// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2025 Rasmus Munk Larsen
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_REDUCTIONS_CLANG_H
#define EIGEN_REDUCTIONS_CLANG_H

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {
namespace internal {

// --- Reductions ---
#if EIGEN_HAS_BUILTIN(__builtin_reduce_min) && EIGEN_HAS_BUILTIN(__builtin_reduce_max) && \
    EIGEN_HAS_BUILTIN(__builtin_reduce_or)
#define EIGEN_CLANG_PACKET_REDUX_MINMAX(PACKET_TYPE)                                        \
  template <>                                                                               \
  EIGEN_STRONG_INLINE unpacket_traits<PACKET_TYPE>::type predux_min(const PACKET_TYPE& a) { \
    return __builtin_reduce_min(a);                                                         \
  }                                                                                         \
  template <>                                                                               \
  EIGEN_STRONG_INLINE unpacket_traits<PACKET_TYPE>::type predux_max(const PACKET_TYPE& a) { \
    return __builtin_reduce_max(a);                                                         \
  }                                                                                         \
  template <>                                                                               \
  EIGEN_STRONG_INLINE bool predux_any(const PACKET_TYPE& a) {                               \
    return __builtin_reduce_or(a != 0) != 0;                                                \
  }

EIGEN_CLANG_PACKET_REDUX_MINMAX(PacketXf)
EIGEN_CLANG_PACKET_REDUX_MINMAX(PacketXd)
EIGEN_CLANG_PACKET_REDUX_MINMAX(PacketXi)
EIGEN_CLANG_PACKET_REDUX_MINMAX(PacketXl)
#undef EIGEN_CLANG_PACKET_REDUX_MINMAX
#endif

#if EIGEN_HAS_BUILTIN(__builtin_reduce_add) && EIGEN_HAS_BUILTIN(__builtin_reduce_mul)
#define EIGEN_CLANG_PACKET_REDUX_INT(PACKET_TYPE)                                                        \
  template <>                                                                                            \
  EIGEN_STRONG_INLINE unpacket_traits<PACKET_TYPE>::type predux<PACKET_TYPE>(const PACKET_TYPE& a) {     \
    return __builtin_reduce_add(a);                                                                      \
  }                                                                                                      \
  template <>                                                                                            \
  EIGEN_STRONG_INLINE unpacket_traits<PACKET_TYPE>::type predux_mul<PACKET_TYPE>(const PACKET_TYPE& a) { \
    return __builtin_reduce_mul(a);                                                                      \
  }

// __builtin_reduce_{mul,add} are only defined for integer types.
EIGEN_CLANG_PACKET_REDUX_INT(PacketXi)
EIGEN_CLANG_PACKET_REDUX_INT(PacketXl)
#undef EIGEN_CLANG_PACKET_REDUX_INT
#endif

#if EIGEN_HAS_BUILTIN(__builtin_shufflevector)
namespace detail {

// Folds `a` in half with `op` until two elements are left and returns them as
// (even, odd). Callers combine those two with the same operation; splitting the
// final step out is what lets the complex reductions read off the accumulated
// real and imaginary parts separately. The halves are shuffled inline rather
// than through lower_half/upper_half because the last fold produces an 8-byte
// vector, which a helper would have to return through the calling convention.
template <int N>
struct halving_reduce {
  template <typename VectorT, typename Op, std::size_t... Is>
  static EIGEN_STRONG_INLINE scalar_pair_t<VectorT> fold(const VectorT& a, Op op, std::index_sequence<Is...>) {
    return halving_reduce<N / 2>::run(
        op(__builtin_shufflevector(a, a, Is...), __builtin_shufflevector(a, a, (sizeof...(Is) + Is)...)), op);
  }

  template <typename VectorT, typename Op>
  static EIGEN_STRONG_INLINE scalar_pair_t<VectorT> run(const VectorT& a, Op op) {
    return fold(a, op, std::make_index_sequence<N / 2>{});
  }
};

template <>
struct halving_reduce<2> {
  template <typename VectorT, typename Op>
  static EIGEN_STRONG_INLINE scalar_pair_t<VectorT> run(const VectorT& a, Op /*op*/) {
    return {a[0], a[1]};
  }
};

template <typename VectorT>
EIGEN_STRONG_INLINE scalar_pair_t<VectorT> reduce_add_pairs(const VectorT& a) {
  return halving_reduce<vector_elements<VectorT>()>::run(a, [](const auto& x, const auto& y) { return x + y; });
}

template <typename VectorT>
EIGEN_STRONG_INLINE scalar_pair_t<VectorT> reduce_mul_pairs(const VectorT& a) {
  return halving_reduce<vector_elements<VectorT>()>::run(a, [](const auto& x, const auto& y) { return x * y; });
}

// Multiplies the two halves of a complex packet until two complex values are
// left, then multiplies those as scalars. Unlike the reductions above this
// cannot work on the real vector, because complex multiplication mixes lanes.
template <int N>
struct complex_predux_mul {
  template <typename RealScalar>
  static EIGEN_STRONG_INLINE std::complex<RealScalar> run(const complex_packet_wrapper<RealScalar, N>& a) {
    using HalfPacket = complex_packet_wrapper<RealScalar, N / 2>;
    return complex_predux_mul<N / 2>::run(pmul<HalfPacket>(HalfPacket(lower_half(a.v)), HalfPacket(upper_half(a.v))));
  }
};

template <>
struct complex_predux_mul<2> {
  template <typename RealScalar>
  static EIGEN_STRONG_INLINE std::complex<RealScalar> run(const complex_packet_wrapper<RealScalar, 2>& a) {
    return a[0] * a[1];
  }
};

template <>
struct complex_predux_mul<1> {
  template <typename RealScalar>
  static EIGEN_STRONG_INLINE std::complex<RealScalar> run(const complex_packet_wrapper<RealScalar, 1>& a) {
    return a[0];
  }
};

}  // namespace detail

// --- predux and predux_mul for float and double ---
// __builtin_reduce_{add,mul} cover the integer packets above but are not
// defined for floating point, so these fold the packet by hand.
#define EIGEN_CLANG_PACKET_REDUX_FLOAT(PACKET_TYPE)                                                      \
  template <>                                                                                            \
  EIGEN_STRONG_INLINE unpacket_traits<PACKET_TYPE>::type predux<PACKET_TYPE>(const PACKET_TYPE& a) {     \
    const auto even_odd = detail::reduce_add_pairs(a);                                                   \
    return even_odd.first + even_odd.second;                                                             \
  }                                                                                                      \
  template <>                                                                                            \
  EIGEN_STRONG_INLINE unpacket_traits<PACKET_TYPE>::type predux_mul<PACKET_TYPE>(const PACKET_TYPE& a) { \
    const auto even_odd = detail::reduce_mul_pairs(a);                                                   \
    return even_odd.first * even_odd.second;                                                             \
  }

EIGEN_CLANG_PACKET_REDUX_FLOAT(PacketXf)
EIGEN_CLANG_PACKET_REDUX_FLOAT(PacketXd)
#undef EIGEN_CLANG_PACKET_REDUX_FLOAT

// --- predux and predux_mul for complex ---
// The real vector of a complex packet interleaves real and imaginary parts, so
// summing it into an (even, odd) pair accumulates the two parts separately.
#define EIGEN_CLANG_COMPLEX_REDUX(PACKET_TYPE, COMPLEX_SIZE)                                             \
  template <>                                                                                            \
  EIGEN_STRONG_INLINE unpacket_traits<PACKET_TYPE>::type predux<PACKET_TYPE>(const PACKET_TYPE& a) {     \
    const auto re_im = detail::reduce_add_pairs(a.v);                                                    \
    return unpacket_traits<PACKET_TYPE>::type(re_im.first, re_im.second);                                \
  }                                                                                                      \
  template <>                                                                                            \
  EIGEN_STRONG_INLINE unpacket_traits<PACKET_TYPE>::type predux_mul<PACKET_TYPE>(const PACKET_TYPE& a) { \
    return detail::complex_predux_mul<COMPLEX_SIZE>::run(a);                                             \
  }

EIGEN_CLANG_COMPLEX_REDUX(PacketXcf, kComplexFloatSize)
EIGEN_CLANG_COMPLEX_REDUX(PacketXcd, kComplexDoubleSize)
#undef EIGEN_CLANG_COMPLEX_REDUX

#endif

}  // end namespace internal
}  // end namespace Eigen

#endif  // EIGEN_REDUCTIONS_CLANG_H
