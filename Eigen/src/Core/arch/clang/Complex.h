// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2025 Rasmus Munk Larsen
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_COMPLEX_CLANG_H
#define EIGEN_COMPLEX_CLANG_H

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {
namespace internal {

template <typename RealScalar, int N>
struct complex_packet_wrapper {
  using RealPacketT = detail::VectorType<RealScalar, 2 * N>;
  complex_packet_wrapper() = default;
  EIGEN_STRONG_INLINE explicit complex_packet_wrapper(const RealPacketT& a) : v(a) {}
  EIGEN_STRONG_INLINE constexpr std::complex<RealScalar> operator[](Index i) const {
    return std::complex<RealScalar>(v[2 * i], v[2 * i + 1]);
  }
  RealPacketT v;
};

// --- Primary complex packet aliases ---
constexpr int kComplexFloatSize = kFloatPacketSize / 2;    // 2, 4, or 8
constexpr int kComplexDoubleSize = kDoublePacketSize / 2;  // 1, 2, or 4
using PacketXcf = complex_packet_wrapper<float, kComplexFloatSize>;
using PacketXcd = complex_packet_wrapper<double, kComplexDoubleSize>;

// Sub-packet types needed for reductions at larger sizes.
// When PacketXcf IS already a given size, we skip the alias to avoid duplicates.
#if EIGEN_GENERIC_VECTOR_SIZE_BYTES >= 32
using Packet2cf = complex_packet_wrapper<float, 2>;
#endif
#if EIGEN_GENERIC_VECTOR_SIZE_BYTES >= 64
using Packet4cf = complex_packet_wrapper<float, 4>;
using Packet2cd = complex_packet_wrapper<double, 2>;
#endif

struct generic_complex_packet_traits : default_packet_traits {
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    HasAdd = 1,
    HasSub = 1,
    HasMul = 1,
    HasDiv = 1,
    HasNegate = 1,
    HasAbs = 0,
    HasAbs2 = 0,
    HasMin = 0,
    HasMax = 0,
    HasArg = 0,
    HasSetLinear = 0,
    HasConj = 1,
    // Math functions
    HasLog = 1,
    HasExp = 1,
    HasSqrt = 1,
  };
};

template <>
struct packet_traits<std::complex<float>> : generic_complex_packet_traits {
  using type = PacketXcf;
  using half = PacketXcf;
  enum {
    size = kComplexFloatSize,
  };
};

template <>
struct unpacket_traits<PacketXcf> : generic_unpacket_traits {
  using type = std::complex<float>;
  using half = PacketXcf;
  using as_real = PacketXf;
  enum {
    size = kComplexFloatSize,
  };
};

template <>
struct packet_traits<std::complex<double>> : generic_complex_packet_traits {
  using type = PacketXcd;
  using half = PacketXcd;
  enum {
    size = kComplexDoubleSize,
  };
};

template <>
struct unpacket_traits<PacketXcd> : generic_unpacket_traits {
  using type = std::complex<double>;
  using half = PacketXcd;
  using as_real = PacketXd;
  enum {
    size = kComplexDoubleSize,
  };
};

// ------------ Load and store ops ----------
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

EIGEN_CLANG_COMPLEX_LOAD_STORE(PacketXcf);
EIGEN_CLANG_COMPLEX_LOAD_STORE(PacketXcd);
#undef EIGEN_CLANG_COMPLEX_LOAD_STORE

namespace detail {

// Index sequence over the real components -- two per complex value -- of a
// complex packet. Index Is names component Is % 2 of complex value Is / 2.
template <typename ComplexPacket>
using complex_real_indices = vector_indices<typename ComplexPacket::RealPacketT>;

template <typename ComplexPacket, std::size_t... Is>
EIGEN_STRONG_INLINE ComplexPacket complex_pset1_impl(const typename unpacket_traits<ComplexPacket>::type& from,
                                                     std::index_sequence<Is...>) {
  using RealPacket = typename ComplexPacket::RealPacketT;
  using RealScalar = scalar_type_of_vector_t<RealPacket>;
  const RealScalar re = numext::real(from);
  const RealScalar im = numext::imag(from);
  return ComplexPacket(RealPacket{(Is % 2 == 0 ? re : im)...});
}

// Negates the imaginary parts, selecting them from -a.v, whose components
// follow those of a.v in the shuffle.
template <typename ComplexPacket, std::size_t... Is>
EIGEN_STRONG_INLINE ComplexPacket complex_pconj_impl(const ComplexPacket& a, std::index_sequence<Is...>) {
  return ComplexPacket(__builtin_shufflevector(a.v, -a.v, (Is % 2 == 0 ? Is : sizeof...(Is) + Is)...));
}

// {re, im} -> {im, re}.
template <typename ComplexPacket, std::size_t... Is>
EIGEN_STRONG_INLINE ComplexPacket complex_pcplxflip_impl(const ComplexPacket& a, std::index_sequence<Is...>) {
  return ComplexPacket(__builtin_shufflevector(a.v, a.v, (2 * (Is / 2) + (1 - Is % 2))...));
}

// {re, im} -> {re, re}.
template <typename ComplexPacket, std::size_t... Is>
EIGEN_STRONG_INLINE ComplexPacket complex_pdupreal_impl(const ComplexPacket& a, std::index_sequence<Is...>) {
  return ComplexPacket(__builtin_shufflevector(a.v, a.v, (2 * (Is / 2))...));
}

// {re, im} -> {im, im}.
template <typename ComplexPacket, std::size_t... Is>
EIGEN_STRONG_INLINE ComplexPacket complex_pdupimag_impl(const ComplexPacket& a, std::index_sequence<Is...>) {
  return ComplexPacket(__builtin_shufflevector(a.v, a.v, (2 * (Is / 2) + 1)...));
}

// Loads each complex value Repeat times in a row: Repeat == 2 implements
// ploaddup and Repeat == 4 implements ploadquad.
template <std::size_t Repeat, typename ComplexPacket, std::size_t... Is>
EIGEN_STRONG_INLINE ComplexPacket complex_loadrepeat_impl(const typename unpacket_traits<ComplexPacket>::type* from,
                                                          std::index_sequence<Is...>) {
  using RealPacket = typename ComplexPacket::RealPacketT;
  return ComplexPacket(
      RealPacket{(Is % 2 == 0 ? numext::real(from[Is / (2 * Repeat)]) : numext::imag(from[Is / (2 * Repeat)]))...});
}

// Reverses the complex values, keeping each real/imaginary pair together.
template <typename ComplexPacket, std::size_t... Is>
EIGEN_STRONG_INLINE ComplexPacket complex_preverse_impl(const ComplexPacket& a, std::index_sequence<Is...>) {
  constexpr std::size_t kLastValue = sizeof...(Is) - 2;
  return ComplexPacket(__builtin_shufflevector(a.v, a.v, (kLastValue - 2 * (Is / 2) + Is % 2)...));
}

}  // namespace detail

// --- pset1 for complex ---
#define EIGEN_CLANG_COMPLEX_SET1(PACKET_TYPE)                                                          \
  template <>                                                                                          \
  EIGEN_STRONG_INLINE PACKET_TYPE pset1<PACKET_TYPE>(const unpacket_traits<PACKET_TYPE>::type& from) { \
    return detail::complex_pset1_impl<PACKET_TYPE>(from, detail::complex_real_indices<PACKET_TYPE>{}); \
  }

EIGEN_CLANG_COMPLEX_SET1(PacketXcf)
EIGEN_CLANG_COMPLEX_SET1(PacketXcd)
#undef EIGEN_CLANG_COMPLEX_SET1

// ----------- Unary ops ------------------
#define DELEGATE_UNARY_TO_REAL_OP(PACKET_TYPE, OP)                        \
  template <>                                                             \
  EIGEN_STRONG_INLINE PACKET_TYPE OP<PACKET_TYPE>(const PACKET_TYPE& a) { \
    return PACKET_TYPE(OP(a.v));                                          \
  }

#define EIGEN_CLANG_COMPLEX_UNARY_CWISE_OPS(PACKET_TYPE)                                             \
  DELEGATE_UNARY_TO_REAL_OP(PACKET_TYPE, pnegate)                                                    \
  DELEGATE_UNARY_TO_REAL_OP(PACKET_TYPE, pzero)                                                      \
  template <>                                                                                        \
  EIGEN_STRONG_INLINE unpacket_traits<PACKET_TYPE>::type pfirst<PACKET_TYPE>(const PACKET_TYPE& a) { \
    return a[0];                                                                                     \
  }                                                                                                  \
  EIGEN_INSTANTIATE_COMPLEX_MATH_FUNCS(PACKET_TYPE)

EIGEN_CLANG_COMPLEX_UNARY_CWISE_OPS(PacketXcf);
EIGEN_CLANG_COMPLEX_UNARY_CWISE_OPS(PacketXcd);

#undef DELEGATE_UNARY_TO_REAL_OP
#undef EIGEN_CLANG_COMPLEX_UNARY_CWISE_OPS

// --- Operations that rearrange the real and imaginary lanes ---
#define EIGEN_CLANG_COMPLEX_LANE_OPS(PACKET_TYPE)                                          \
  template <>                                                                              \
  EIGEN_STRONG_INLINE PACKET_TYPE pconj<PACKET_TYPE>(const PACKET_TYPE& a) {               \
    return detail::complex_pconj_impl(a, detail::complex_real_indices<PACKET_TYPE>{});     \
  }                                                                                        \
  template <>                                                                              \
  EIGEN_STRONG_INLINE PACKET_TYPE pcplxflip<PACKET_TYPE>(const PACKET_TYPE& a) {           \
    return detail::complex_pcplxflip_impl(a, detail::complex_real_indices<PACKET_TYPE>{}); \
  }                                                                                        \
  template <>                                                                              \
  EIGEN_STRONG_INLINE PACKET_TYPE pdupreal<PACKET_TYPE>(const PACKET_TYPE& a) {            \
    return detail::complex_pdupreal_impl(a, detail::complex_real_indices<PACKET_TYPE>{});  \
  }                                                                                        \
  template <>                                                                              \
  EIGEN_STRONG_INLINE PACKET_TYPE pdupimag<PACKET_TYPE>(const PACKET_TYPE& a) {            \
    return detail::complex_pdupimag_impl(a, detail::complex_real_indices<PACKET_TYPE>{});  \
  }

EIGEN_CLANG_COMPLEX_LANE_OPS(PacketXcf)
EIGEN_CLANG_COMPLEX_LANE_OPS(PacketXcd)

// Sub-packet specializations needed for reductions.
#if EIGEN_GENERIC_VECTOR_SIZE_BYTES >= 32
EIGEN_CLANG_COMPLEX_LANE_OPS(Packet2cf)
#endif
#if EIGEN_GENERIC_VECTOR_SIZE_BYTES >= 64
EIGEN_CLANG_COMPLEX_LANE_OPS(Packet4cf)
EIGEN_CLANG_COMPLEX_LANE_OPS(Packet2cd)
#endif
#undef EIGEN_CLANG_COMPLEX_LANE_OPS

// --- ploaddup and ploadquad ---
#define EIGEN_CLANG_COMPLEX_LOAD_REPEAT(PACKET_TYPE)                                                           \
  template <>                                                                                                  \
  EIGEN_STRONG_INLINE PACKET_TYPE ploaddup<PACKET_TYPE>(const unpacket_traits<PACKET_TYPE>::type* from) {      \
    return detail::complex_loadrepeat_impl<2, PACKET_TYPE>(from, detail::complex_real_indices<PACKET_TYPE>{}); \
  }                                                                                                            \
  template <>                                                                                                  \
  EIGEN_STRONG_INLINE PACKET_TYPE ploadquad<PACKET_TYPE>(const unpacket_traits<PACKET_TYPE>::type* from) {     \
    return detail::complex_loadrepeat_impl<4, PACKET_TYPE>(from, detail::complex_real_indices<PACKET_TYPE>{}); \
  }

EIGEN_CLANG_COMPLEX_LOAD_REPEAT(PacketXcf)
EIGEN_CLANG_COMPLEX_LOAD_REPEAT(PacketXcd)
#undef EIGEN_CLANG_COMPLEX_LOAD_REPEAT

// --- preverse ---
template <>
EIGEN_STRONG_INLINE PacketXcf preverse<PacketXcf>(const PacketXcf& a) {
  return detail::complex_preverse_impl(a, detail::complex_real_indices<PacketXcf>{});
}
template <>
EIGEN_STRONG_INLINE PacketXcd preverse<PacketXcd>(const PacketXcd& a) {
  return detail::complex_preverse_impl(a, detail::complex_real_indices<PacketXcd>{});
}

// ----------- Binary ops ------------------
#define DELEGATE_BINARY_TO_REAL_OP(PACKET_TYPE, OP)                                             \
  template <>                                                                                   \
  EIGEN_STRONG_INLINE PACKET_TYPE OP<PACKET_TYPE>(const PACKET_TYPE& a, const PACKET_TYPE& b) { \
    return PACKET_TYPE(OP(a.v, b.v));                                                           \
  }

#define EIGEN_CLANG_COMPLEX_BINARY_CWISE_OPS(PACKET_TYPE)                                            \
  DELEGATE_BINARY_TO_REAL_OP(PACKET_TYPE, psub)                                                      \
  DELEGATE_BINARY_TO_REAL_OP(PACKET_TYPE, pand)                                                      \
  DELEGATE_BINARY_TO_REAL_OP(PACKET_TYPE, por)                                                       \
  DELEGATE_BINARY_TO_REAL_OP(PACKET_TYPE, pxor)                                                      \
  DELEGATE_BINARY_TO_REAL_OP(PACKET_TYPE, pandnot)                                                   \
  template <>                                                                                        \
  EIGEN_STRONG_INLINE PACKET_TYPE pdiv<PACKET_TYPE>(const PACKET_TYPE& a, const PACKET_TYPE& b) {    \
    return pdiv_complex(a, b);                                                                       \
  }                                                                                                  \
  template <>                                                                                        \
  EIGEN_STRONG_INLINE PACKET_TYPE pcmp_eq<PACKET_TYPE>(const PACKET_TYPE& a, const PACKET_TYPE& b) { \
    const PACKET_TYPE t = PACKET_TYPE(pcmp_eq(a.v, b.v));                                            \
    return PACKET_TYPE(pand(pdupreal(t).v, pdupimag(t).v));                                          \
  }

EIGEN_CLANG_COMPLEX_BINARY_CWISE_OPS(PacketXcf);
EIGEN_CLANG_COMPLEX_BINARY_CWISE_OPS(PacketXcd);

// Binary ops that are needed on sub-packets for predux and predux_mul.
#define EIGEN_CLANG_COMPLEX_REDUCER_BINARY_CWISE_OPS(PACKET_TYPE)                                 \
  DELEGATE_BINARY_TO_REAL_OP(PACKET_TYPE, padd)                                                   \
  template <>                                                                                     \
  EIGEN_STRONG_INLINE PACKET_TYPE pmul<PACKET_TYPE>(const PACKET_TYPE& a, const PACKET_TYPE& b) { \
    return pmul_complex(a, b);                                                                    \
  }

EIGEN_CLANG_COMPLEX_REDUCER_BINARY_CWISE_OPS(PacketXcf);
#if EIGEN_GENERIC_VECTOR_SIZE_BYTES >= 32
EIGEN_CLANG_COMPLEX_REDUCER_BINARY_CWISE_OPS(Packet2cf);
#endif
#if EIGEN_GENERIC_VECTOR_SIZE_BYTES >= 64
EIGEN_CLANG_COMPLEX_REDUCER_BINARY_CWISE_OPS(Packet4cf);
#endif
EIGEN_CLANG_COMPLEX_REDUCER_BINARY_CWISE_OPS(PacketXcd);
#if EIGEN_GENERIC_VECTOR_SIZE_BYTES >= 64
EIGEN_CLANG_COMPLEX_REDUCER_BINARY_CWISE_OPS(Packet2cd);
#endif

#define EIGEN_CLANG_PACKET_SCATTER_GATHER(PACKET_TYPE)                                                               \
  template <>                                                                                                        \
  EIGEN_STRONG_INLINE void pscatter(unpacket_traits<PACKET_TYPE>::type* to, const PACKET_TYPE& from, Index stride) { \
    constexpr int size = unpacket_traits<PACKET_TYPE>::size;                                                         \
    for (int i = 0; i < size; ++i) {                                                                                 \
      to[i * stride] = from[i];                                                                                      \
    }                                                                                                                \
  }                                                                                                                  \
  template <>                                                                                                        \
  EIGEN_STRONG_INLINE PACKET_TYPE pgather<typename unpacket_traits<PACKET_TYPE>::type, PACKET_TYPE>(                 \
      const unpacket_traits<PACKET_TYPE>::type* from, Index stride) {                                                \
    constexpr int size = unpacket_traits<PACKET_TYPE>::size;                                                         \
    PACKET_TYPE result;                                                                                              \
    for (int i = 0; i < size; ++i) {                                                                                 \
      const unpacket_traits<PACKET_TYPE>::type from_i = from[i * stride];                                            \
      result.v[2 * i] = numext::real(from_i);                                                                        \
      result.v[2 * i + 1] = numext::imag(from_i);                                                                    \
    }                                                                                                                \
    return result;                                                                                                   \
  }

EIGEN_CLANG_PACKET_SCATTER_GATHER(PacketXcf);
EIGEN_CLANG_PACKET_SCATTER_GATHER(PacketXcd);
#undef EIGEN_CLANG_PACKET_SCATTER_GATHER

#undef DELEGATE_BINARY_TO_REAL_OP
#undef EIGEN_CLANG_COMPLEX_BINARY_CWISE_OPS
#undef EIGEN_CLANG_COMPLEX_REDUCER_BINARY_CWISE_OPS

// ------------ ternary ops -------------
template <>
EIGEN_STRONG_INLINE PacketXcf pselect<PacketXcf>(const PacketXcf& mask, const PacketXcf& a, const PacketXcf& b) {
  return PacketXcf(reinterpret_cast<PacketXf>(
      pselect(reinterpret_cast<PacketXd>(mask.v), reinterpret_cast<PacketXd>(a.v), reinterpret_cast<PacketXd>(b.v))));
}

// --- zip_in_place for complex ---
namespace detail {

// Complex packets interleave whole complex values, so their real and imaginary
// components move together.
template <>
EIGEN_ALWAYS_INLINE void zip_in_place<PacketXcf>(PacketXcf& p1, PacketXcf& p2) {
  zip_in_place_impl<2>(p1.v, p2.v, complex_real_indices<PacketXcf>{});
}

#if EIGEN_GENERIC_VECTOR_SIZE_BYTES >= 32
// PacketXcd holds a single complex value at 16 bytes, so there is nothing to interleave.
template <>
EIGEN_ALWAYS_INLINE void zip_in_place<PacketXcd>(PacketXcd& p1, PacketXcd& p2) {
  zip_in_place_impl<2>(p1.v, p2.v, complex_real_indices<PacketXcd>{});
}
#endif

}  // namespace detail

// --- ptranspose for complex ---
// PacketXcf: valid block sizes depend on kComplexFloatSize.
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void ptranspose(PacketBlock<PacketXcf, 2>& kernel) {
  detail::ptranspose_impl(kernel);
}
#if EIGEN_GENERIC_VECTOR_SIZE_BYTES >= 32
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void ptranspose(PacketBlock<PacketXcf, 4>& kernel) {
  detail::ptranspose_impl(kernel);
}
#endif
#if EIGEN_GENERIC_VECTOR_SIZE_BYTES >= 64
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void ptranspose(PacketBlock<PacketXcf, 8>& kernel) {
  detail::ptranspose_impl(kernel);
}
#endif

// PacketXcd: valid block sizes depend on kComplexDoubleSize.
#if EIGEN_GENERIC_VECTOR_SIZE_BYTES >= 32
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void ptranspose(PacketBlock<PacketXcd, 2>& kernel) {
  detail::ptranspose_impl(kernel);
}
#endif
#if EIGEN_GENERIC_VECTOR_SIZE_BYTES >= 64
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void ptranspose(PacketBlock<PacketXcd, 4>& kernel) {
  detail::ptranspose_impl(kernel);
}
#endif

EIGEN_MAKE_CONJ_HELPER_CPLX_REAL(PacketXcf, PacketXf)
EIGEN_MAKE_CONJ_HELPER_CPLX_REAL(PacketXcd, PacketXd)

}  // end namespace internal
}  // end namespace Eigen

#endif  // EIGEN_COMPLEX_CLANG_H
