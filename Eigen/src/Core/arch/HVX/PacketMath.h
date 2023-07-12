// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner (benoit.steiner.goog@gmail.com)
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_PACKET_MATH_HVX_H
#define EIGEN_PACKET_MATH_HVX_H

#ifndef EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS
#define EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS 32
#endif

namespace Eigen {
namespace internal {

// Hexagon compiler uses same HVX_Vector to represent all HVX vector types.
// Wrap different vector type (float32, int32, etc) to different class with
// explicit constructor and casting back-and-force to HVX_Vector.
template <int unique_id>
class HVXPacket {
 public:
  HVXPacket() = default;
  explicit HVXPacket(HVX_Vector v) : m_val(v) {}
  explicit operator HVX_Vector() const { return m_val; }

 private:
  HVX_Vector m_val = Q6_V_vzero();
};

// Generic operations.
template <int unique_id>
EIGEN_STRONG_INLINE void ptranspose(
    PacketBlock<HVXPacket<unique_id>, 4>& kernel) {
  // zip 0,2
  HVX_VectorPair transpose_0_2 = Q6_W_vshuff_VVR(
      HVX_Vector(kernel.packet[2]), HVX_Vector(kernel.packet[0]), -4);
  // zip 1,3
  HVX_VectorPair transpose_1_3 = Q6_W_vshuff_VVR(
      HVX_Vector(kernel.packet[3]), HVX_Vector(kernel.packet[1]), -4);
  // zip 0,1
  HVX_VectorPair transpose_0_1 = Q6_W_vshuff_VVR(
      HEXAGON_HVX_GET_V0(transpose_1_3), HEXAGON_HVX_GET_V0(transpose_0_2), -4);
  // zip 2,3
  HVX_VectorPair transpose_2_3 = Q6_W_vshuff_VVR(
      HEXAGON_HVX_GET_V1(transpose_1_3), HEXAGON_HVX_GET_V1(transpose_0_2), -4);

  kernel.packet[0] = HVXPacket<unique_id>(HEXAGON_HVX_GET_V0(transpose_0_1));
  kernel.packet[1] = HVXPacket<unique_id>(HEXAGON_HVX_GET_V1(transpose_0_1));
  kernel.packet[2] = HVXPacket<unique_id>(HEXAGON_HVX_GET_V0(transpose_2_3));
  kernel.packet[3] = HVXPacket<unique_id>(HEXAGON_HVX_GET_V1(transpose_2_3));
}

#if __HVX_ARCH__ >= 68

typedef HVXPacket<0> Packet32f;   // float32
typedef HVXPacket<1> Packet32qf;  // qfloat32

template <>
struct packet_traits<float> : default_packet_traits {
  typedef Packet32f type;
  typedef Packet32f half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 32,
  };
};

template <>
struct unpacket_traits<Packet32f> {
  typedef float type;
  typedef Packet32f half;
  enum {
    size = 32,
    alignment = Aligned128,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};

// float32 operations.
template <>
EIGEN_STRONG_INLINE Packet32f pset1<Packet32f>(const float& from) {
  union {
    float f;
    int32_t i;
  } u = {.f = from};
  return Packet32f(Q6_V_vsplat_R(u.i));
}

template <>
EIGEN_STRONG_INLINE Packet32f pload<Packet32f>(const float* from) {
  return Packet32f(*reinterpret_cast<const HVX_Vector*>(from));
}
template <>
EIGEN_STRONG_INLINE Packet32f ploadu<Packet32f>(const float* from) {
  return Packet32f(*reinterpret_cast<const HVX_UVector*>(from));
}

template <>
EIGEN_STRONG_INLINE void pstore<float>(float* to, const Packet32f& from) {
  *reinterpret_cast<HVX_Vector*>(to) = HVX_Vector(from);
}
template <>
EIGEN_STRONG_INLINE void pstoreu<float>(float* to, const Packet32f& from) {
  *reinterpret_cast<HVX_UVector*>(to) = HVX_Vector(from);
}

template <>
EIGEN_STRONG_INLINE float predux<Packet32f>(const Packet32f& a) {
  HVX_Vector vsum_4 = Q6_Vqf32_vadd_VsfVsf(
      Q6_V_vror_VR(HVX_Vector(a), 4), HVX_Vector(a));
  HVX_Vector vsum_8 = Q6_Vqf32_vadd_Vqf32Vqf32(
      Q6_V_vror_VR(vsum_4, 8), vsum_4);
  HVX_Vector vsum_16 = Q6_Vqf32_vadd_Vqf32Vqf32(
      Q6_V_vror_VR(vsum_8, 16), vsum_8);
  HVX_Vector vsum_32 = Q6_Vqf32_vadd_Vqf32Vqf32(
      Q6_V_vror_VR(vsum_16, 32), vsum_16);
  HVX_Vector vsum_64 = Q6_Vqf32_vadd_Vqf32Vqf32(
      Q6_V_vror_VR(vsum_32, 64), vsum_32);

  union {
    float f;
    int32_t i;
  } u = {.i = Q6_R_vextract_VR(Q6_Vsf_equals_Vqf32(vsum_64), 0)};
  return u.f;
}

template <>
EIGEN_STRONG_INLINE Packet32f pmul<Packet32f>(const Packet32f& a,
                                              const Packet32f& b) {
  return Packet32f(
      Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(HVX_Vector(a), HVX_Vector(b))));
}

template <>
EIGEN_STRONG_INLINE Packet32f padd<Packet32f>(const Packet32f& a,
                                              const Packet32f& b) {
  return Packet32f(
      Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(HVX_Vector(a), HVX_Vector(b))));
}

// qfloat32 operations.
template <>
EIGEN_STRONG_INLINE Packet32qf pzero<Packet32qf>(const Packet32qf&) {
  return Packet32qf(Q6_V_vzero());
}

template <>
EIGEN_STRONG_INLINE Packet32qf pmul<Packet32qf>(const Packet32qf& a,
                                                const Packet32qf& b) {
  return Packet32qf(Q6_Vqf32_vmpy_Vqf32Vqf32(HVX_Vector(a), HVX_Vector(b)));
}

template <>
EIGEN_STRONG_INLINE Packet32qf padd<Packet32qf>(const Packet32qf& a,
                                                const Packet32qf& b) {
  return Packet32qf(Q6_Vqf32_vadd_Vqf32Vqf32(HVX_Vector(a), HVX_Vector(b)));
}

// Mixed float32 and qfloat32 operations.
EIGEN_STRONG_INLINE Packet32qf pmadd(const Packet32f& a, const Packet32f& b,
                                     const Packet32qf& c) {
  return Packet32qf(Q6_Vqf32_vadd_Vqf32Vqf32(
      Q6_Vqf32_vmpy_VsfVsf(HVX_Vector(a), HVX_Vector(b)), HVX_Vector(c)));
}

EIGEN_STRONG_INLINE Packet32f pmadd(const Packet32qf& a, const Packet32f& b,
                                    const Packet32f& c) {
  return Packet32f(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(
      Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(HVX_Vector(a)), HVX_Vector(b)),
      HVX_Vector(c))));
}

#endif  // __HVX_ARCH__ >= 68

}  // end namespace internal
}  // end namespace Eigen

#endif  // EIGEN_PACKET_MATH_HVX_H
