// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2024 Kseniya Zaytseva <kseniya.zaytseva@syntacore.com>
// Copyright (C) 2025 Chip Kerchner <ckerchner@tenstorrent.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_PACKET_MATH_RVV10_H
#define EIGEN_PACKET_MATH_RVV10_H

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {
namespace internal {
#ifndef EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD
#define EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD 8
#endif

#ifndef EIGEN_HAS_SINGLE_INSTRUCTION_MADD
#define EIGEN_HAS_SINGLE_INSTRUCTION_MADD
#endif

#define EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS 32

template <typename Scalar, std::size_t VectorLength, std::size_t VectorLMul>
struct rvv_packet_size_selector {
  enum { size = VectorLength * VectorLMul / (sizeof(Scalar) * CHAR_BIT) };
};

template <std::size_t VectorLength, std::size_t VectorLMul>
struct rvv_packet_alignment_selector {
  enum {
    alignment =
        (VectorLength * VectorLMul) >= 1024
            ? Aligned128
            : ((VectorLength * VectorLMul) >= 512 ? Aligned64
                                                  : ((VectorLength * VectorLMul) >= 256 ? Aligned32 : Aligned16))
  };
};

typedef vbool64_t PacketMask64;
typedef vbool32_t PacketMask32;
typedef vbool16_t PacketMask16;
typedef vbool8_t PacketMask8;
typedef vbool4_t PacketMask4;

template <typename Packet>
struct rvv_half_packet {
  typedef Packet half;
};

template <typename Packet>
using rvv_half_packet_t = typename rvv_half_packet<Packet>::type;

template <typename Scalar>
struct rvv_default_packet_traits : default_packet_traits {
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = rvv_packet_size_selector<Scalar, EIGEN_RISCV64_RVV_VL, EIGEN_RISCV64_DEFAULT_LMUL>::size,

    HasAdd = 1,
    HasSub = 1,
    HasShift = 1,
    HasMul = 1,
    HasNegate = 1,
    HasAbs = 1,
    HasArg = 0,
    HasAbs2 = 1,
    HasMin = 1,
    HasMax = 1,
    HasConj = 1,
    HasSetLinear = 0,
    HasBlend = 0,
    HasReduxp = 0
  };
};

template <typename Scalar>
struct rvv_default_float_packet_traits : rvv_default_packet_traits<Scalar> {
  enum {
    HasCmp = 1,
    HasDiv = 1,
    HasRound = 1,
    HasMul = 1,

    HasSin = EIGEN_FAST_MATH,
    HasCos = EIGEN_FAST_MATH,
    HasTan = EIGEN_FAST_MATH,
    HasLog = 1,
    HasExp = 1,
    HasSqrt = 1,
    HasTanh = EIGEN_FAST_MATH,
    HasErf = EIGEN_FAST_MATH
  };
};

template <typename Scalar, typename Packet, std::size_t VectorLMul>
struct rvv_default_unpacket_traits {
  typedef Scalar type;
  typedef rvv_half_packet_t<Packet> half;
  typedef numext::uint8_t mask_t;
  enum {
    size = rvv_packet_size_selector<Scalar, EIGEN_RISCV64_RVV_VL, VectorLMul>::size,
    alignment = rvv_packet_alignment_selector<EIGEN_RISCV64_RVV_VL, VectorLMul>::alignment,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};

/********************************* int32 **************************************/
typedef eigen_packet_wrapper<vint32m1_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL))), 0> Packet1Xi;
typedef eigen_packet_wrapper<vuint32m1_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL))), 1> Packet1Xu;

typedef eigen_packet_wrapper<vint32m2_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 2))), 2> Packet2Xi;
typedef eigen_packet_wrapper<vuint32m2_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 2))), 3> Packet2Xu;

typedef eigen_packet_wrapper<vint32m4_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 4))), 4> Packet4Xi;
typedef eigen_packet_wrapper<vuint32m4_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 4))), 5> Packet4Xu;

template <>
struct rvv_half_packet<Packet2Xi> : rvv_half_packet<Packet1Xi> {};
template <>
struct rvv_half_packet<Packet2Xu> : rvv_half_packet<Packet1Xu> {};
template <>
struct rvv_half_packet<Packet4Xi> : rvv_half_packet<Packet2Xi> {};
template <>
struct rvv_half_packet<Packet4Xu> : rvv_half_packet<Packet2Xu> {};

#if EIGEN_RISCV64_DEFAULT_LMUL == 1
typedef Packet1Xi PacketXi;
typedef Packet1Xu PacketXu;
#elif EIGEN_RISCV64_DEFAULT_LMUL == 2
typedef Packet2Xi PacketXi;
typedef Packet2Xu PacketXu;
#elif EIGEN_RISCV64_DEFAULT_LMUL == 4
typedef Packet4Xi PacketXi;
typedef Packet4Xu PacketXu;
#endif

template <>
struct packet_traits<numext::int32_t> : rvv_default_packet_traits<numext::int32_t> {
  typedef PacketXi type;
  typedef rvv_half_packet_t<type> half;
};

template <>
struct unpacket_traits<Packet1Xi> : rvv_default_unpacket_traits<numext::int32_t, Packet1Xi, 1> {};
template <>
struct unpacket_traits<Packet2Xi> : rvv_default_unpacket_traits<numext::int32_t, Packet2Xi, 2> {};
template <>
struct unpacket_traits<Packet4Xi> : rvv_default_unpacket_traits<numext::int32_t, Packet4Xi, 4> {};

/********************************* float32 ************************************/

typedef eigen_packet_wrapper<vfloat32m1_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL))), 6> Packet1Xf;
typedef eigen_packet_wrapper<vfloat32m2_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 2))), 7>
    Packet2Xf;
typedef eigen_packet_wrapper<vfloat32m4_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 4))), 8>
    Packet4Xf;

#if EIGEN_RISCV64_DEFAULT_LMUL == 1
typedef Packet1Xf PacketXf;
#elif EIGEN_RISCV64_DEFAULT_LMUL == 2
typedef Packet2Xf PacketXf;
#elif EIGEN_RISCV64_DEFAULT_LMUL == 4
typedef Packet4Xf PacketXf;
#endif

template <>
struct rvv_half_packet<Packet2Xf> : rvv_half_packet<Packet1Xf> {};
template <>
struct rvv_half_packet<Packet4Xf> : rvv_half_packet<Packet2Xf> {};

template <>
struct packet_traits<float> : rvv_default_float_packet_traits<float> {
  typedef PacketXf type;
  typedef rvv_half_packet_t<type> half;
};

template <>
struct unpacket_traits<Packet1Xf> : rvv_default_unpacket_traits<float, Packet1Xf, 1> {
  typedef Packet1Xi integer_packet;
  typedef PacketMask32 packet_mask;
};

template <>
struct unpacket_traits<Packet2Xf> : rvv_default_unpacket_traits<float, Packet2Xf, 2> {
  typedef Packet2Xi integer_packet;
  typedef PacketMask16 packet_mask;
};

template <>
struct unpacket_traits<Packet4Xf> : rvv_default_unpacket_traits<float, Packet4Xf, 4> {
  typedef Packet4Xi integer_packet;
  typedef PacketMask8 packet_mask;
};

/********************************* int64 **************************************/

typedef eigen_packet_wrapper<vint64m1_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL))), 9> Packet1Xl;
typedef eigen_packet_wrapper<vuint64m1_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL))), 10> Packet1Xul;

typedef eigen_packet_wrapper<vint64m2_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 2))), 11> Packet2Xl;
typedef eigen_packet_wrapper<vuint64m2_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 2))), 12>
    Packet2Xul;

typedef eigen_packet_wrapper<vint64m4_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 4))), 13> Packet4Xl;
typedef eigen_packet_wrapper<vuint64m4_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 4))), 14>
    Packet4Xul;

#if EIGEN_RISCV64_DEFAULT_LMUL == 1
typedef Packet1Xl PacketXl;
typedef Packet1Xul PacketXul;
#elif EIGEN_RISCV64_DEFAULT_LMUL == 2
typedef Packet2Xl PacketXl;
typedef Packet2Xul PacketXul;
#elif EIGEN_RISCV64_DEFAULT_LMUL == 4
typedef Packet4Xl PacketXl;
typedef Packet4Xul PacketXul;
#endif

template <>
struct rvv_half_packet<Packet2Xl> : rvv_half_packet<Packet1Xl> {};
template <>
struct rvv_half_packet<Packet4Xl> : rvv_half_packet<Packet2Xl> {};

template <>
struct packet_traits<numext::int64_t> : rvv_default_packet_traits<numext::int64_t> {
  typedef PacketXl type;
  typedef rvv_half_packet_t<type> half;
};

template <>
struct unpacket_traits<Packet1Xl> : rvv_default_unpacket_traits<numext::int64_t, Packet1Xl, 1> {};

template <>
struct unpacket_traits<Packet2Xl> : rvv_default_unpacket_traits<numext::int64_t, Packet2Xl, 2> {};

template <>
struct unpacket_traits<Packet4Xl> : rvv_default_unpacket_traits<numext::int64_t, Packet4Xl, 4> {};

/********************************* double ************************************/

typedef eigen_packet_wrapper<vfloat64m1_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL))), 15> Packet1Xd;
typedef eigen_packet_wrapper<vfloat64m2_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 2))), 16>
    Packet2Xd;
typedef eigen_packet_wrapper<vfloat64m4_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 4))), 17>
    Packet4Xd;

#if EIGEN_RISCV64_DEFAULT_LMUL == 1
typedef Packet1Xd PacketXd;
#elif EIGEN_RISCV64_DEFAULT_LMUL == 2
typedef Packet2Xd PacketXd;
#elif EIGEN_RISCV64_DEFAULT_LMUL == 4
typedef Packet4Xd PacketXd;
#endif

template <>
struct rvv_half_packet<Packet2Xd> : rvv_half_packet<Packet1Xd> {};
template <>
struct rvv_half_packet<Packet4Xd> : rvv_half_packet<Packet2Xd> {};

template <>
struct packet_traits<double> : rvv_default_float_packet_traits<double> {
  typedef PacketXd type;
  typedef rvv_half_packet_t<type> half;
};

template <>
struct unpacket_traits<Packet1Xd> : rvv_default_unpacket_traits<double, Packet1Xd, 1> {
  typedef Packet1Xl integer_packet;
  typedef PacketMask64 packet_mask;
};

template <>
struct unpacket_traits<Packet2Xd> : rvv_default_unpacket_traits<double, Packet2Xd, 2> {
  typedef Packet2Xl integer_packet;
  typedef PacketMask32 packet_mask;
};

template <>
struct unpacket_traits<Packet4Xd> : rvv_default_unpacket_traits<double, Packet4Xd, 4> {
  typedef Packet4Xl integer_packet;
  typedef PacketMask16 packet_mask;
};

/********************************* short **************************************/

typedef eigen_packet_wrapper<vint16m1_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL))), 18> Packet1Xs;
typedef eigen_packet_wrapper<vuint16m1_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL))), 19> Packet1Xsu;

typedef eigen_packet_wrapper<vint16m2_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 2))), 20> Packet2Xs;
typedef eigen_packet_wrapper<vuint16m2_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 2))), 21>
    Packet2Xsu;

typedef eigen_packet_wrapper<vint16m4_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 4))), 22> Packet4Xs;
typedef eigen_packet_wrapper<vuint16m4_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 4))), 23>
    Packet4Xsu;

#if EIGEN_RISCV64_DEFAULT_LMUL == 1
typedef Packet1Xs PacketXs;
typedef Packet1Xul PacketXsu;
#elif EIGEN_RISCV64_DEFAULT_LMUL == 2
typedef Packet2Xs PacketXs;
typedef Packet2Xul PacketXsu;
#elif EIGEN_RISCV64_DEFAULT_LMUL == 4
typedef Packet4Xs PacketXs;
typedef Packet4Xul PacketXsu;
#endif

template <>
struct rvv_half_packet<Packet2Xs> : rvv_half_packet<Packet1Xs> {};
template <>
struct rvv_half_packet<Packet4Xs> : rvv_half_packet<Packet2Xs> {};

template <>
struct packet_traits<numext::int16_t> : rvv_default_packet_traits<numext::int16_t> {
  typedef PacketXs type;
  typedef rvv_half_packet_t<type> half;
};

template <>
struct unpacket_traits<Packet1Xs> : rvv_default_unpacket_traits<numext::int16_t, Packet1Xs, 1> {};

template <>
struct unpacket_traits<Packet2Xs> : rvv_default_unpacket_traits<numext::int16_t, Packet2Xs, 2> {};

template <>
struct unpacket_traits<Packet4Xs> : rvv_default_unpacket_traits<numext::int16_t, Packet4Xs, 4> {};

/********************************* prefetch **************************************/

template <>
EIGEN_STRONG_INLINE void prefetch<numext::int16_t>(const numext::int16_t* addr) {
#if EIGEN_HAS_BUILTIN(__builtin_prefetch) || EIGEN_COMP_GNUC
  __builtin_prefetch(addr);
#endif
}

template <>
EIGEN_STRONG_INLINE void prefetch<numext::int32_t>(const numext::int32_t* addr) {
#if EIGEN_HAS_BUILTIN(__builtin_prefetch) || EIGEN_COMP_GNUC
  __builtin_prefetch(addr);
#endif
}

template <>
EIGEN_STRONG_INLINE void prefetch<numext::int64_t>(const numext::int64_t* addr) {
#if EIGEN_HAS_BUILTIN(__builtin_prefetch) || EIGEN_COMP_GNUC
  __builtin_prefetch(addr);
#endif
}

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_PACKET_MATH_RVV10_H
