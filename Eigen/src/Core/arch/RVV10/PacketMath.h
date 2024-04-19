// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2024 Kseniya Zaytseva <kseniya.zaytseva@syntacore.com>
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

template <typename Scalar, std::size_t VectorLength>
struct rvv_packet_size_selector {
  enum { size = VectorLength / (sizeof(Scalar) * CHAR_BIT) };
};

template <std::size_t VectorLength>
struct rvv_packet_alignment_selector {
  enum { 
    alignment = VectorLength >= 512 ? Aligned64 : (VectorLength >= 256 ? Aligned32 : Aligned16)
    };
};

typedef vbool32_t PacketMask;

/********************************* int32 **************************************/
typedef vint32m1_t PacketXi __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL)));
typedef vuint32m1_t PacketXu __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL)));

template <>
struct packet_traits<numext::int32_t> : default_packet_traits {
  typedef PacketXi type;
  typedef PacketXi half;  // Half not implemented yet
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = rvv_packet_size_selector<numext::int32_t, EIGEN_RISCV64_RVV_VL>::size,

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

template <>
struct unpacket_traits<PacketXi> {
  typedef numext::int32_t type;
  typedef PacketXi half;  // Half not yet implemented
  enum {
    size = rvv_packet_size_selector<numext::int32_t, EIGEN_RISCV64_RVV_VL>::size,
    alignment = rvv_packet_alignment_selector<EIGEN_RISCV64_RVV_VL>::alignment,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};

template <>
EIGEN_STRONG_INLINE void prefetch<numext::int32_t>(const numext::int32_t* addr) {
#if EIGEN_HAS_BUILTIN(__builtin_prefetch) || EIGEN_COMP_GNUC
  __builtin_prefetch(addr);
#endif
}

template <>
EIGEN_STRONG_INLINE PacketXi pset1<PacketXi>(const numext::int32_t& from) {
  return __riscv_vmv_v_x_i32m1(from, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi plset<PacketXi>(const numext::int32_t& a) {
  PacketXi idx = __riscv_vid_v_i32m1(packet_traits<numext::int32_t>::size);
  return __riscv_vadd_vx_i32m1(idx, a, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pzero<PacketXi>(const PacketXi& /*a*/) {
  return __riscv_vmv_v_x_i32m1(0, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi padd<PacketXi>(const PacketXi& a, const PacketXi& b) {
  return __riscv_vadd_vv_i32m1(a, b, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi psub<PacketXi>(const PacketXi& a, const PacketXi& b) {
  return __riscv_vsub(a, b, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pnegate(const PacketXi& a) {
  return __riscv_vneg(a, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pconj(const PacketXi& a) {
  return a;
}

template <>
EIGEN_STRONG_INLINE PacketXi pmul<PacketXi>(const PacketXi& a, const PacketXi& b) {
  return __riscv_vmul(a, b, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pdiv<PacketXi>(const PacketXi& a, const PacketXi& b) {
  return __riscv_vdiv(a, b, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pmadd(const PacketXi& a, const PacketXi& b, const PacketXi& c) {
  return __riscv_vmadd(a, b, c, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pmsub(const PacketXi& a, const PacketXi& b, const PacketXi& c) {
  return __riscv_vmadd(a, b, pnegate(c), packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pnmadd(const PacketXi& a, const PacketXi& b, const PacketXi& c) {
  return __riscv_vnmsub_vv_i32m1(a, b, c, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pnmsub(const PacketXi& a, const PacketXi& b, const PacketXi& c) {
  return __riscv_vnmsub_vv_i32m1(a, b, pnegate(c), packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pmin<PacketXi>(const PacketXi& a, const PacketXi& b) {
  return __riscv_vmin(a, b, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pmax<PacketXi>(const PacketXi& a, const PacketXi& b) {
  return __riscv_vmax(a, b, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pcmp_le<PacketXi>(const PacketXi& a, const PacketXi& b) {
  PacketMask mask = __riscv_vmsle_vv_i32m1_b32(a, b, packet_traits<numext::int32_t>::size);
  return __riscv_vmerge_vxm_i32m1(pzero(a), 0xffffffff, mask, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pcmp_lt<PacketXi>(const PacketXi& a, const PacketXi& b) {
  PacketMask mask = __riscv_vmslt_vv_i32m1_b32(a, b, packet_traits<numext::int32_t>::size);
  return __riscv_vmerge_vxm_i32m1(pzero(a), 0xffffffff, mask, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pcmp_eq<PacketXi>(const PacketXi& a, const PacketXi& b) {
  PacketMask mask = __riscv_vmseq_vv_i32m1_b32(a, b, packet_traits<numext::int32_t>::size);
  return __riscv_vmerge_vxm_i32m1(pzero(a), 0xffffffff, mask, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi ptrue<PacketXi>(const PacketXi& /*a*/) {
  return __riscv_vmv_v_x_i32m1(0xffffffffu, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pand<PacketXi>(const PacketXi& a, const PacketXi& b) {
  return __riscv_vand_vv_i32m1(a, b, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi por<PacketXi>(const PacketXi& a, const PacketXi& b) {
  return __riscv_vor_vv_i32m1(a, b, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pxor<PacketXi>(const PacketXi& a, const PacketXi& b) {
  return __riscv_vxor_vv_i32m1(a, b, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pandnot<PacketXi>(const PacketXi& a, const PacketXi& b) {
  return __riscv_vand_vv_i32m1(a, __riscv_vnot_v_i32m1(b, packet_traits<numext::int32_t>::size), packet_traits<numext::int32_t>::size);
}

template <int N>
EIGEN_STRONG_INLINE PacketXi parithmetic_shift_right(PacketXi a) {
  return  __riscv_vsra_vx_i32m1(a, N, packet_traits<numext::int32_t>::size);
}

template <int N>
EIGEN_STRONG_INLINE PacketXi plogical_shift_right(PacketXi a) {
  return __riscv_vreinterpret_i32m1(__riscv_vsrl_vx_u32m1(__riscv_vreinterpret_u32m1(a), N, packet_traits<numext::int32_t>::size));
}

template <int N>
EIGEN_STRONG_INLINE PacketXi plogical_shift_left(PacketXi a) {
  return __riscv_vsll_vx_i32m1(a, N, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pload<PacketXi>(const numext::int32_t* from) {
  EIGEN_DEBUG_ALIGNED_LOAD return __riscv_vle32_v_i32m1(from, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi ploadu<PacketXi>(const numext::int32_t* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD return __riscv_vle32_v_i32m1(from, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi ploaddup<PacketXi>(const numext::int32_t* from) {
  PacketXu idx = __riscv_vid_v_u32m1(packet_traits<numext::int32_t>::size);
  idx = __riscv_vsll_vx_u32m1(__riscv_vand_vx_u32m1(idx, 0xfffffffeu, packet_traits<numext::int32_t>::size), 1, packet_traits<numext::int32_t>::size);
  // idx = 0 0 sizeof(int32_t) sizeof(int32_t) 2*sizeof(int32_t) 2*sizeof(int32_t) ...
  return __riscv_vloxei32_v_i32m1(from, idx, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi ploadquad<PacketXi>(const numext::int32_t* from) {
  PacketXu idx = __riscv_vid_v_u32m1(packet_traits<numext::int32_t>::size);
  idx = __riscv_vand_vx_u32m1(idx, 0xfffffffcu, packet_traits<numext::int32_t>::size);
  return __riscv_vloxei32_v_i32m1(from, idx, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE void pstore<numext::int32_t>(numext::int32_t* to, const PacketXi& from) {
  EIGEN_DEBUG_ALIGNED_STORE __riscv_vse32_v_i32m1(to, from, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE void pstoreu<numext::int32_t>(numext::int32_t* to, const PacketXi& from) {
  EIGEN_DEBUG_UNALIGNED_STORE __riscv_vse32_v_i32m1(to, from, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_DEVICE_FUNC inline PacketXi pgather<numext::int32_t, PacketXi>(const numext::int32_t* from, Index stride) {
  return __riscv_vlse32_v_i32m1(from, stride * sizeof(numext::int32_t), packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_DEVICE_FUNC inline void pscatter<numext::int32_t, PacketXi>(numext::int32_t* to, const PacketXi& from,
                                                                  Index stride) {
  __riscv_vsse32(to, stride * sizeof(numext::int32_t), from, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE numext::int32_t pfirst<PacketXi>(const PacketXi& a) {
  return __riscv_vmv_x_s_i32m1_i32(a);
}

template <>
EIGEN_STRONG_INLINE PacketXi preverse(const PacketXi& a) {
  PacketXu idx = __riscv_vrsub_vx_u32m1(__riscv_vid_v_u32m1(packet_traits<numext::int32_t>::size), packet_traits<numext::int32_t>::size-1, packet_traits<numext::int32_t>::size);
  return __riscv_vrgather_vv_i32m1(a, idx, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pabs(const PacketXi& a) {
  PacketXi mask = __riscv_vsra_vx_i32m1(a, 31, packet_traits<numext::int32_t>::size);
  return __riscv_vsub_vv_i32m1(__riscv_vxor_vv_i32m1(a, mask, packet_traits<numext::int32_t>::size), mask, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE numext::int32_t predux<PacketXi>(const PacketXi& a) {
  PacketXi vzero = __riscv_vmv_v_x_i32m1(0, packet_traits<numext::int32_t>::size);
  return __riscv_vmv_x(__riscv_vredsum_vs_i32m1_i32m1(a, vzero, packet_traits<numext::int32_t>::size));
}

template <>
EIGEN_STRONG_INLINE numext::int32_t predux_mul<PacketXi>(const PacketXi& a) {
  // Multiply the vector by its reverse
  PacketXi prod = __riscv_vmul_vv_i32m1(preverse(a), a, packet_traits<numext::int32_t>::size);
  PacketXi half_prod;

  if (EIGEN_RISCV64_RVV_VL >= 1024) {
    half_prod =  __riscv_vslidedown_vx_i32m1(prod, 8, packet_traits<numext::int32_t>::size);
    prod = __riscv_vmul_vv_i32m1(prod, half_prod, packet_traits<numext::int32_t>::size);
  }
  if (EIGEN_RISCV64_RVV_VL >= 512) {
    half_prod =  __riscv_vslidedown_vx_i32m1(prod, 4, packet_traits<numext::int32_t>::size);
    prod = __riscv_vmul_vv_i32m1(prod, half_prod, packet_traits<numext::int32_t>::size);
  }
  if (EIGEN_RISCV64_RVV_VL >= 256) {
    half_prod =  __riscv_vslidedown_vx_i32m1(prod, 2, packet_traits<numext::int32_t>::size);
    prod = __riscv_vmul_vv_i32m1(prod, half_prod, packet_traits<numext::int32_t>::size);
  }
  // Last reduction
  half_prod =  __riscv_vslidedown_vx_i32m1(prod, 1, packet_traits<numext::int32_t>::size);
  prod = __riscv_vmul_vv_i32m1(prod, half_prod, packet_traits<numext::int32_t>::size);

  // The reduction is done to the first element.
  return pfirst(prod);
}

template <>
EIGEN_STRONG_INLINE numext::int32_t predux_min<PacketXi>(const PacketXi& a) {
  PacketXi vmax = __riscv_vmv_v_x_i32m1((std::numeric_limits<numext::int32_t>::max)(), packet_traits<numext::int32_t>::size);
  return __riscv_vmv_x(__riscv_vredmin_vs_i32m1_i32m1(a, vmax, packet_traits<numext::int32_t>::size));
}

template <>
EIGEN_STRONG_INLINE numext::int32_t predux_max<PacketXi>(const PacketXi& a) {
  PacketXi vmin = __riscv_vmv_v_x_i32m1((std::numeric_limits<numext::int32_t>::min)(), packet_traits<numext::int32_t>::size);
  return __riscv_vmv_x(__riscv_vredmax_vs_i32m1_i32m1(a, vmin, packet_traits<numext::int32_t>::size));
}

template <int N>
EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<PacketXi, N>& kernel) {
  numext::int32_t buffer[packet_traits<numext::int32_t>::size * N] = {0};
  int i = 0;

  for (i = 0; i < N; i++) {
    __riscv_vsse32(&buffer[i], N * sizeof(numext::int32_t), kernel.packet[i], packet_traits<numext::int32_t>::size);
  }
  for (i = 0; i < N; i++) {
    kernel.packet[i] = __riscv_vle32_v_i32m1(&buffer[i * packet_traits<numext::int32_t>::size], packet_traits<numext::int32_t>::size);
  }
}

/********************************* float32 ************************************/

typedef vfloat32m1_t PacketXf __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL)));

template <>
struct packet_traits<float> : default_packet_traits {
  typedef PacketXf type;
  typedef PacketXf half;

  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = rvv_packet_size_selector<float, EIGEN_RISCV64_RVV_VL>::size,

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
    HasReduxp = 0,

    HasCmp = 1,
    HasDiv = 1,
    HasFloor = 1,
    HasRint = 1,

    HasSin = EIGEN_FAST_MATH,
    HasCos = EIGEN_FAST_MATH,
    HasLog = 1,
    HasExp = 1,
    HasSqrt = 1,
    HasTanh = EIGEN_FAST_MATH,
    HasErf = EIGEN_FAST_MATH
  };
};

template <>
struct unpacket_traits<PacketXf> {
  typedef float type;
  typedef PacketXf half;  // Half not yet implemented
  typedef PacketXi integer_packet;

  enum {
    size = rvv_packet_size_selector<float, EIGEN_RISCV64_RVV_VL>::size,
    alignment = rvv_packet_alignment_selector<EIGEN_RISCV64_RVV_VL>::alignment,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};

template <>
EIGEN_STRONG_INLINE PacketXf ptrue<PacketXf>(const PacketXf& /*a*/) {
  return __riscv_vreinterpret_f32m1(__riscv_vmv_v_x_u32m1(0xffffffffu, packet_traits<float>::size));
}

template <>
EIGEN_STRONG_INLINE PacketXf pzero<PacketXf>(const PacketXf& /*a*/) {
  return __riscv_vfmv_v_f_f32m1(0.0f, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pabs(const PacketXf& a) {
  return __riscv_vfabs_v_f32m1(a, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pset1<PacketXf>(const float& from) {
  return __riscv_vfmv_v_f_f32m1(from, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pset1frombits<PacketXf>(numext::uint32_t from) {
  return __riscv_vreinterpret_f32m1(__riscv_vmv_v_x_u32m1(from, packet_traits<float>::size));
}

template <>
EIGEN_STRONG_INLINE PacketXf plset<PacketXf>(const float& a) {
  PacketXf idx = __riscv_vfcvt_f_x_v_f32m1(__riscv_vid_v_i32m1(packet_traits<numext::int32_t>::size), packet_traits<float>::size);
  return __riscv_vfadd_vf_f32m1(idx, a, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf padd<PacketXf>(const PacketXf& a, const PacketXf& b) {
  return __riscv_vfadd_vv_f32m1(a, b, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf psub<PacketXf>(const PacketXf& a, const PacketXf& b) {
  return __riscv_vfsub_vv_f32m1(a, b, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pnegate(const PacketXf& a) {
  return __riscv_vfneg_v_f32m1(a, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pconj(const PacketXf& a) {
  return a;
}

template <>
EIGEN_STRONG_INLINE PacketXf pmul<PacketXf>(const PacketXf& a, const PacketXf& b) {
  return __riscv_vfmul_vv_f32m1(a, b, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pdiv<PacketXf>(const PacketXf& a, const PacketXf& b) {
  return __riscv_vfdiv_vv_f32m1(a, b, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pmadd(const PacketXf& a, const PacketXf& b, const PacketXf& c) {
  return __riscv_vfmadd_vv_f32m1(a, b, c, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pmsub(const PacketXf& a, const PacketXf& b, const PacketXf& c) {
  return __riscv_vfmsub_vv_f32m1(a, b, c, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pnmadd(const PacketXf& a, const PacketXf& b, const PacketXf& c) {
  return __riscv_vfnmsub_vv_f32m1(a, b, c, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pnmsub(const PacketXf& a, const PacketXf& b, const PacketXf& c) {
  return __riscv_vfnmadd_vv_f32m1(a, b, c, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pmin<PacketXf>(const PacketXf& a, const PacketXf& b) {
  PacketXf nans = __riscv_vfmv_v_f_f32m1((std::numeric_limits<float>::quiet_NaN)(), packet_traits<float>::size);
  PacketMask mask = __riscv_vmfeq_vv_f32m1_b32(a, a, packet_traits<float>::size);
  PacketMask mask2 = __riscv_vmfeq_vv_f32m1_b32(b, b, packet_traits<float>::size);
  mask =  __riscv_vmand_mm_b32(mask, mask2, packet_traits<float>::size);

  return __riscv_vfmin_vv_f32m1_tum(mask, nans, a, b, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pmin<PropagateNaN, PacketXf>(const PacketXf& a, const PacketXf& b) {
  return pmin<PacketXf>(a, b);
}

template <>
EIGEN_STRONG_INLINE PacketXf pmin<PropagateNumbers, PacketXf>(const PacketXf& a, const PacketXf& b) {
  return __riscv_vfmin_vv_f32m1(a, b, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pmax<PacketXf>(const PacketXf& a, const PacketXf& b) {
  PacketXf nans = __riscv_vfmv_v_f_f32m1((std::numeric_limits<float>::quiet_NaN)(), packet_traits<float>::size);
  PacketMask mask = __riscv_vmfeq_vv_f32m1_b32(a, a, packet_traits<float>::size);
  PacketMask mask2 = __riscv_vmfeq_vv_f32m1_b32(b, b, packet_traits<float>::size);
  mask =  __riscv_vmand_mm_b32(mask, mask2, packet_traits<float>::size);

  return __riscv_vfmax_vv_f32m1_tum(mask, nans, a, b, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pmax<PropagateNaN, PacketXf>(const PacketXf& a, const PacketXf& b) {
  return pmax<PacketXf>(a, b);
}

template <>
EIGEN_STRONG_INLINE PacketXf pmax<PropagateNumbers, PacketXf>(const PacketXf& a, const PacketXf& b) {
  return __riscv_vfmax_vv_f32m1(a, b, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pcmp_le<PacketXf>(const PacketXf& a, const PacketXf& b) {
  PacketMask mask = __riscv_vmfle_vv_f32m1_b32(a, b, packet_traits<float>::size);
  return __riscv_vmerge_vvm_f32m1(pzero<PacketXf>(a), ptrue<PacketXf>(a), mask, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pcmp_lt<PacketXf>(const PacketXf& a, const PacketXf& b) {
  PacketMask mask = __riscv_vmflt_vv_f32m1_b32(a, b, packet_traits<float>::size);
  return __riscv_vmerge_vvm_f32m1(pzero<PacketXf>(a), ptrue<PacketXf>(a), mask, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pcmp_eq<PacketXf>(const PacketXf& a, const PacketXf& b) {
  PacketMask mask = __riscv_vmfeq_vv_f32m1_b32(a, b, packet_traits<float>::size);
  return __riscv_vmerge_vvm_f32m1(pzero<PacketXf>(a), ptrue<PacketXf>(a), mask, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pcmp_lt_or_nan<PacketXf>(const PacketXf& a, const PacketXf& b) {
  PacketMask mask = __riscv_vmfge_vv_f32m1_b32(a, b, packet_traits<float>::size);
  return __riscv_vfmerge_vfm_f32m1(ptrue<PacketXf>(a), 0.0f, mask, packet_traits<float>::size);
}

// Logical Operations are not supported for float, so reinterpret casts
template <>
EIGEN_STRONG_INLINE PacketXf pand<PacketXf>(const PacketXf& a, const PacketXf& b) {
  return  __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vv_u32m1(__riscv_vreinterpret_v_f32m1_u32m1(a), __riscv_vreinterpret_v_f32m1_u32m1(b), packet_traits<float>::size));
}

template <>
EIGEN_STRONG_INLINE PacketXf por<PacketXf>(const PacketXf& a, const PacketXf& b) {
  return __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vor_vv_u32m1(__riscv_vreinterpret_v_f32m1_u32m1(a), __riscv_vreinterpret_v_f32m1_u32m1(b), packet_traits<float>::size));
}

template <>
EIGEN_STRONG_INLINE PacketXf pxor<PacketXf>(const PacketXf& a, const PacketXf& b) {
  return __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vxor_vv_u32m1(__riscv_vreinterpret_v_f32m1_u32m1(a), __riscv_vreinterpret_v_f32m1_u32m1(b), packet_traits<float>::size));
}

template <>
EIGEN_STRONG_INLINE PacketXf pandnot<PacketXf>(const PacketXf& a, const PacketXf& b) {
  return __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vv_u32m1(__riscv_vreinterpret_v_f32m1_u32m1(a), __riscv_vnot_v_u32m1(__riscv_vreinterpret_v_f32m1_u32m1(b), packet_traits<float>::size), packet_traits<float>::size));
}

template <>
EIGEN_STRONG_INLINE PacketXf pload<PacketXf>(const float* from) {
  EIGEN_DEBUG_ALIGNED_LOAD return __riscv_vle32_v_f32m1(from, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf ploadu<PacketXf>(const float* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD return __riscv_vle32_v_f32m1(from, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf ploaddup<PacketXf>(const float* from) {
  PacketXu idx = __riscv_vid_v_u32m1(packet_traits<float>::size);
  idx = __riscv_vsll_vx_u32m1(__riscv_vand_vx_u32m1(idx, 0xfffffffeu, packet_traits<float>::size), 1, packet_traits<float>::size);
  return __riscv_vloxei32_v_f32m1(from, idx, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf ploadquad<PacketXf>(const float* from) {
  PacketXu idx = __riscv_vid_v_u32m1(packet_traits<float>::size);
  idx = __riscv_vand_vx_u32m1(idx, 0xfffffffcu, packet_traits<float>::size);
  return  __riscv_vloxei32_v_f32m1(from, idx, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE void pstore<float>(float* to, const PacketXf& from) {
  EIGEN_DEBUG_ALIGNED_STORE __riscv_vse32_v_f32m1(to, from, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE void pstoreu<float>(float* to, const PacketXf& from) {
  EIGEN_DEBUG_UNALIGNED_STORE __riscv_vse32_v_f32m1(to, from, packet_traits<float>::size);
}

template <>
EIGEN_DEVICE_FUNC inline PacketXf pgather<float, PacketXf>(const float* from, Index stride) {
  return __riscv_vlse32_v_f32m1(from, stride * sizeof(float), packet_traits<float>::size);
}

template <>
EIGEN_DEVICE_FUNC inline void pscatter<float, PacketXf>(float* to, const PacketXf& from, Index stride) {
  __riscv_vsse32(to, stride * sizeof(float), from, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE float pfirst<PacketXf>(const PacketXf& a) {
  return __riscv_vfmv_f_s_f32m1_f32(a);
}

template <>
EIGEN_STRONG_INLINE PacketXf psqrt(const PacketXf& a) {
  return __riscv_vfsqrt_v_f32m1(a, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf print<PacketXf>(const PacketXf& a) {
  // Adds and subtracts signum(a) * 2^23 to force rounding.
  const PacketXf limit = pset1<PacketXf>(static_cast<float>(1 << 23));
  const PacketXf abs_a = pabs(a);
  PacketXf r = padd(abs_a, limit);
  // Don't compile-away addition and subtraction.
  EIGEN_OPTIMIZATION_BARRIER(r);
  r = psub(r, limit);
  // If greater than limit, simply return a.  Otherwise, account for sign.
  r = pselect(pcmp_lt(abs_a, limit), pselect(pcmp_lt(a, pzero(a)), pnegate(r), r), a);
  return r;
}

template <>
EIGEN_STRONG_INLINE PacketXf pfloor<PacketXf>(const PacketXf& a) {
  const PacketXf cst_1 = pset1<PacketXf>(1.0f);
  PacketXf tmp = print<PacketXf>(a);
  // If greater, subtract one.
  PacketXf mask = pcmp_lt(a, tmp);
  mask = pand(mask, cst_1);
  return psub(tmp, mask);
}

template <>
EIGEN_STRONG_INLINE PacketXf preverse(const PacketXf& a) {
  PacketXu idx = __riscv_vrsub_vx_u32m1(__riscv_vid_v_u32m1(packet_traits<float>::size), packet_traits<float>::size-1, packet_traits<float>::size);
  return __riscv_vrgather_vv_f32m1(a, idx, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pfrexp<PacketXf>(const PacketXf& a, PacketXf& exponent) {
  return pfrexp_generic(a, exponent);
}

template <>
EIGEN_STRONG_INLINE float predux<PacketXf>(const PacketXf& a) {
  PacketXf vzero = __riscv_vfmv_v_f_f32m1(0.0, packet_traits<float>::size);
  return __riscv_vfmv_f(__riscv_vfredusum_vs_f32m1_f32m1(a, vzero, packet_traits<float>::size));
}

template <>
EIGEN_STRONG_INLINE float predux_mul<PacketXf>(const PacketXf& a) {
  // Multiply the vector by its reverse
  PacketXf prod = __riscv_vfmul_vv_f32m1(preverse(a), a, packet_traits<float>::size);
  PacketXf half_prod;

  if (EIGEN_RISCV64_RVV_VL >= 1024) {
    half_prod =  __riscv_vslidedown_vx_f32m1(prod, 8, packet_traits<float>::size);
    prod = __riscv_vfmul_vv_f32m1(prod, half_prod, packet_traits<float>::size);
  }
  if (EIGEN_RISCV64_RVV_VL >= 512) {
        half_prod =  __riscv_vslidedown_vx_f32m1(prod, 4, packet_traits<float>::size);
    prod = __riscv_vfmul_vv_f32m1(prod, half_prod, packet_traits<float>::size);
  }
  if (EIGEN_RISCV64_RVV_VL >= 256) {
    half_prod =  __riscv_vslidedown_vx_f32m1(prod, 2, packet_traits<float>::size);
    prod = __riscv_vfmul_vv_f32m1(prod, half_prod, packet_traits<float>::size);
  }
  // Last reduction
  half_prod =  __riscv_vslidedown_vx_f32m1(prod, 1, packet_traits<float>::size);
  prod = __riscv_vfmul_vv_f32m1(prod, half_prod, packet_traits<float>::size);

  // The reduction is done to the first element.
  return pfirst(prod);
}

template <>
EIGEN_STRONG_INLINE float predux_min<PacketXf>(const PacketXf& a) {
  PacketXf vmax = __riscv_vfmv_v_f_f32m1((std::numeric_limits<float>::max)(), packet_traits<float>::size);
  return __riscv_vfmv_f(__riscv_vfredmin_vs_f32m1_f32m1(a, vmax, packet_traits<float>::size));
}

template <>
EIGEN_STRONG_INLINE float predux_max<PacketXf>(const PacketXf& a) {
  PacketXf vmin = __riscv_vfmv_v_f_f32m1(-(std::numeric_limits<float>::max)(), packet_traits<float>::size);
  return __riscv_vfmv_f(__riscv_vfredmax_vs_f32m1_f32m1(a, vmin, packet_traits<float>::size));
}

template <int N>
EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<PacketXf, N>& kernel) {
  float buffer[packet_traits<float>::size * N];
  int i = 0;

  for (i = 0; i < N; i++) {
    __riscv_vsse32(&buffer[i], N * sizeof(float), kernel.packet[i], packet_traits<float>::size);
  }

  for (i = 0; i < N; i++) {
    kernel.packet[i] = __riscv_vle32_v_f32m1(&buffer[i * packet_traits<float>::size], packet_traits<float>::size);
  }
}

template <>
EIGEN_STRONG_INLINE PacketXf pldexp<PacketXf>(const PacketXf& a, const PacketXf& exponent) {
  return pldexp_generic(a, exponent);
}

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_PACKET_MATH_RVV10_H
