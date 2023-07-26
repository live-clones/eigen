
#ifndef EIGEN_HVX_PACKET_MATH_H
#define EIGEN_HVX_PACKET_MATH_H

#if defined __HVX__ && (__HVX_LENGTH__ == 128)

#ifndef EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS
#define EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS 32
#endif

namespace Eigen {
namespace internal {

// These defines borrows from qhmath internal header, qhmath_hvx_convert.h
#define vmem(A) *((HVX_Vector*)(A))
#define vmemu(A) *((HVX_UVector*)(A))

// Hexagon compiler uses same HVX_Vector to represent all HVX vector types.
// Wrap different vector type (float32, int32, etc) to different class with
// explicit constructor and casting back-and-force to HVX_Vector.
template <int unique_id>
class HVXPacket {
 public:
  HVXPacket() = default;
  static HVXPacket Create(HVX_Vector v) { return HVXPacket(v); }
  HVX_Vector Get() const { return m_val; }

 private:
  explicit HVXPacket(HVX_Vector v) : m_val(v) {}
  HVX_Vector m_val = Q6_V_vzero();
};

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
    HasSign = 0,
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
  } u;
  u.f = from;
  return Packet32f::Create(Q6_V_vsplat_R(u.i));
}

template <>
EIGEN_STRONG_INLINE Packet32f pload<Packet32f>(const float* from) {
  return Packet32f::Create(vmem(from));
}
template <>
EIGEN_STRONG_INLINE Packet32f ploadu<Packet32f>(const float* from) {
  return Packet32f::Create(vmemu(from));
}

template <>
EIGEN_STRONG_INLINE void pstore<float>(float* to, const Packet32f& from) {
  vmem(to) = from.Get();
}
template <>
EIGEN_STRONG_INLINE void pstoreu<float>(float* to, const Packet32f& from) {
  vmemu(to) = from.Get();
}

template <>
EIGEN_STRONG_INLINE Packet32f pmul<Packet32f>(const Packet32f& a,
                                              const Packet32f& b) {
  return Packet32f::Create(
      Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a.Get(), b.Get())));
}

template <>
EIGEN_STRONG_INLINE Packet32f padd<Packet32f>(const Packet32f& a,
                                              const Packet32f& b) {
  return Packet32f::Create(
      Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a.Get(), b.Get())));
}

template <>
EIGEN_STRONG_INLINE Packet32f psub<Packet32f>(const Packet32f& a,
                                              const Packet32f& b) {
  return Packet32f::Create(
      Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a.Get(), b.Get())));
}

template <>
EIGEN_STRONG_INLINE Packet32f pnegate(const Packet32f& a) {
  return psub(Packet32f::Create(Q6_V_vzero()), a);
}

template <>
EIGEN_STRONG_INLINE Packet32f pcmp_le(const Packet32f& a, const Packet32f& b) {
  HVX_Vector v_true = Q6_Vb_vsplat_R(0xff);
  HVX_VectorPred pred = Q6_Q_vcmp_gt_VsfVsf(a.Get(), b.Get());
  return Packet32f::Create(Q6_V_vmux_QVV(pred, Q6_V_vzero(), v_true));
}

template <>
EIGEN_STRONG_INLINE Packet32f pcmp_eq(const Packet32f& a, const Packet32f& b) {
  HVX_Vector v_true = Q6_Vb_vsplat_R(0xff);
  HVX_VectorPred pred = Q6_Q_vcmp_eq_VwVw(a.Get(), b.Get());
  return Packet32f::Create(Q6_V_vmux_QVV(pred, v_true, Q6_V_vzero()));
}

template <>
EIGEN_STRONG_INLINE Packet32f pcmp_lt(const Packet32f& a, const Packet32f& b) {
  HVX_Vector v_true = Q6_Vb_vsplat_R(0xff);
  HVX_VectorPred pred = Q6_Q_vcmp_gt_VsfVsf(b.Get(), a.Get());
  return Packet32f::Create(Q6_V_vmux_QVV(pred, v_true, Q6_V_vzero()));
}

template <>
EIGEN_STRONG_INLINE Packet32f pcmp_lt_or_nan(const Packet32f& a,
                                             const Packet32f& b) {
  HVX_Vector v_true = Q6_Vb_vsplat_R(0xff);
  HVX_VectorPred pred = Q6_Q_vcmp_gt_VsfVsf(b.Get(), a.Get());
  return Packet32f::Create(Q6_V_vmux_QVV(pred, v_true, Q6_V_vzero()));
}

template <>
EIGEN_STRONG_INLINE Packet32f pabs(const Packet32f& a) {
  HVX_VectorPred pred = Q6_Q_vcmp_gt_VsfVsf(a.Get(), Q6_V_vzero());
  return Packet32f::Create(Q6_V_vmux_QVV(pred, a.Get(), pnegate(a).Get()));
}

template <>
EIGEN_STRONG_INLINE float pfirst(const Packet32f& a) {
  float vsf[32] __attribute__((aligned(128)));
  pstore(vsf, a);
  return vsf[0];
}

EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet32f, 4>& kernel) {
  // zip 0,2
  HVX_VectorPair transpose_0_2 =
      Q6_W_vshuff_VVR(kernel.packet[2].Get(), kernel.packet[0].Get(), -4);
  // zip 1,3
  HVX_VectorPair transpose_1_3 =
      Q6_W_vshuff_VVR(kernel.packet[3].Get(), kernel.packet[1].Get(), -4);
  // zip 0,1
  HVX_VectorPair transpose_0_1 = Q6_W_vshuff_VVR(
      HEXAGON_HVX_GET_V0(transpose_1_3), HEXAGON_HVX_GET_V0(transpose_0_2), -4);
  // zip 2,3
  HVX_VectorPair transpose_2_3 = Q6_W_vshuff_VVR(
      HEXAGON_HVX_GET_V1(transpose_1_3), HEXAGON_HVX_GET_V1(transpose_0_2), -4);

  kernel.packet[0] = Packet32f::Create(HEXAGON_HVX_GET_V0(transpose_0_1));
  kernel.packet[1] = Packet32f::Create(HEXAGON_HVX_GET_V1(transpose_0_1));
  kernel.packet[2] = Packet32f::Create(HEXAGON_HVX_GET_V0(transpose_2_3));
  kernel.packet[3] = Packet32f::Create(HEXAGON_HVX_GET_V1(transpose_2_3));
}

EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet32f, 32>& kernel) {
  // Shuffle the 32-bit lanes.
  HVX_VectorPair VD1_0 =
      Q6_W_vshuff_VVR(kernel.packet[1].Get(), kernel.packet[0].Get(), -4);
  HVX_VectorPair VD3_2 =
      Q6_W_vshuff_VVR(kernel.packet[3].Get(), kernel.packet[2].Get(), -4);
  HVX_VectorPair VD5_4 =
      Q6_W_vshuff_VVR(kernel.packet[5].Get(), kernel.packet[4].Get(), -4);
  HVX_VectorPair VD7_6 =
      Q6_W_vshuff_VVR(kernel.packet[7].Get(), kernel.packet[6].Get(), -4);
  HVX_VectorPair VD9_8 =
      Q6_W_vshuff_VVR(kernel.packet[9].Get(), kernel.packet[8].Get(), -4);
  HVX_VectorPair VD11_10 =
      Q6_W_vshuff_VVR(kernel.packet[11].Get(), kernel.packet[10].Get(), -4);
  HVX_VectorPair VD13_12 =
      Q6_W_vshuff_VVR(kernel.packet[13].Get(), kernel.packet[12].Get(), -4);
  HVX_VectorPair VD15_14 =
      Q6_W_vshuff_VVR(kernel.packet[15].Get(), kernel.packet[14].Get(), -4);
  HVX_VectorPair VD17_16 =
      Q6_W_vshuff_VVR(kernel.packet[17].Get(), kernel.packet[16].Get(), -4);
  HVX_VectorPair VD19_18 =
      Q6_W_vshuff_VVR(kernel.packet[19].Get(), kernel.packet[18].Get(), -4);
  HVX_VectorPair VD21_20 =
      Q6_W_vshuff_VVR(kernel.packet[21].Get(), kernel.packet[20].Get(), -4);
  HVX_VectorPair VD23_22 =
      Q6_W_vshuff_VVR(kernel.packet[23].Get(), kernel.packet[22].Get(), -4);
  HVX_VectorPair VD25_24 =
      Q6_W_vshuff_VVR(kernel.packet[25].Get(), kernel.packet[24].Get(), -4);
  HVX_VectorPair VD27_26 =
      Q6_W_vshuff_VVR(kernel.packet[27].Get(), kernel.packet[26].Get(), -4);
  HVX_VectorPair VD29_28 =
      Q6_W_vshuff_VVR(kernel.packet[29].Get(), kernel.packet[28].Get(), -4);
  HVX_VectorPair VD31_30 =
      Q6_W_vshuff_VVR(kernel.packet[31].Get(), kernel.packet[30].Get(), -4);

  // Shuffle the 64-bit lanes
  HVX_VectorPair VS1_0 =
      Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(VD3_2), HEXAGON_HVX_GET_V0(VD1_0), -8);
  HVX_VectorPair VS3_2 =
      Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(VD3_2), HEXAGON_HVX_GET_V1(VD1_0), -8);
  HVX_VectorPair VS5_4 =
      Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(VD7_6), HEXAGON_HVX_GET_V0(VD5_4), -8);
  HVX_VectorPair VS7_6 =
      Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(VD7_6), HEXAGON_HVX_GET_V1(VD5_4), -8);
  HVX_VectorPair VS9_8 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(VD11_10),
                                         HEXAGON_HVX_GET_V0(VD9_8), -8);
  HVX_VectorPair VS11_10 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(VD11_10),
                                           HEXAGON_HVX_GET_V1(VD9_8), -8);
  HVX_VectorPair VS13_12 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(VD15_14),
                                           HEXAGON_HVX_GET_V0(VD13_12), -8);
  HVX_VectorPair VS15_14 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(VD15_14),
                                           HEXAGON_HVX_GET_V1(VD13_12), -8);
  HVX_VectorPair VS17_16 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(VD19_18),
                                           HEXAGON_HVX_GET_V0(VD17_16), -8);
  HVX_VectorPair VS19_18 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(VD19_18),
                                           HEXAGON_HVX_GET_V1(VD17_16), -8);
  HVX_VectorPair VS21_20 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(VD23_22),
                                           HEXAGON_HVX_GET_V0(VD21_20), -8);
  HVX_VectorPair VS23_22 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(VD23_22),
                                           HEXAGON_HVX_GET_V1(VD21_20), -8);
  HVX_VectorPair VS25_24 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(VD27_26),
                                           HEXAGON_HVX_GET_V0(VD25_24), -8);
  HVX_VectorPair VS27_26 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(VD27_26),
                                           HEXAGON_HVX_GET_V1(VD25_24), -8);
  HVX_VectorPair VS29_28 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(VD31_30),
                                           HEXAGON_HVX_GET_V0(VD29_28), -8);
  HVX_VectorPair VS31_30 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(VD31_30),
                                           HEXAGON_HVX_GET_V1(VD29_28), -8);

  // Shuffle the 128-bit lanes
  VD1_0 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(VS5_4), HEXAGON_HVX_GET_V0(VS1_0),
                          -16);
  VD3_2 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(VS5_4), HEXAGON_HVX_GET_V1(VS1_0),
                          -16);
  VD5_4 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(VS7_6), HEXAGON_HVX_GET_V0(VS3_2),
                          -16);
  VD7_6 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(VS7_6), HEXAGON_HVX_GET_V1(VS3_2),
                          -16);
  VD9_8 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(VS13_12),
                          HEXAGON_HVX_GET_V0(VS9_8), -16);
  VD11_10 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(VS13_12),
                            HEXAGON_HVX_GET_V1(VS9_8), -16);
  VD13_12 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(VS15_14),
                            HEXAGON_HVX_GET_V0(VS11_10), -16);
  VD15_14 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(VS15_14),
                            HEXAGON_HVX_GET_V1(VS11_10), -16);
  VD17_16 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(VS21_20),
                            HEXAGON_HVX_GET_V0(VS17_16), -16);
  VD19_18 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(VS21_20),
                            HEXAGON_HVX_GET_V1(VS17_16), -16);
  VD21_20 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(VS23_22),
                            HEXAGON_HVX_GET_V0(VS19_18), -16);
  VD23_22 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(VS23_22),
                            HEXAGON_HVX_GET_V1(VS19_18), -16);
  VD25_24 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(VS29_28),
                            HEXAGON_HVX_GET_V0(VS25_24), -16);
  VD27_26 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(VS29_28),
                            HEXAGON_HVX_GET_V1(VS25_24), -16);
  VD29_28 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(VS31_30),
                            HEXAGON_HVX_GET_V0(VS27_26), -16);
  VD31_30 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(VS31_30),
                            HEXAGON_HVX_GET_V1(VS27_26), -16);

  // Shuffle the 256-bit lanes
  VS1_0 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(VD9_8), HEXAGON_HVX_GET_V0(VD1_0),
                          -32);
  VS3_2 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(VD9_8), HEXAGON_HVX_GET_V1(VD1_0),
                          -32);
  VS5_4 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(VD11_10),
                          HEXAGON_HVX_GET_V0(VD3_2), -32);
  VS7_6 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(VD11_10),
                          HEXAGON_HVX_GET_V1(VD3_2), -32);
  VS9_8 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(VD13_12),
                          HEXAGON_HVX_GET_V0(VD5_4), -32);
  VS11_10 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(VD13_12),
                            HEXAGON_HVX_GET_V1(VD5_4), -32);
  VS13_12 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(VD15_14),
                            HEXAGON_HVX_GET_V0(VD7_6), -32);
  VS15_14 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(VD15_14),
                            HEXAGON_HVX_GET_V1(VD7_6), -32);
  VS17_16 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(VD25_24),
                            HEXAGON_HVX_GET_V0(VD17_16), -32);
  VS19_18 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(VD25_24),
                            HEXAGON_HVX_GET_V1(VD17_16), -32);
  VS21_20 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(VD27_26),
                            HEXAGON_HVX_GET_V0(VD19_18), -32);
  VS23_22 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(VD27_26),
                            HEXAGON_HVX_GET_V1(VD19_18), -32);
  VS25_24 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(VD29_28),
                            HEXAGON_HVX_GET_V0(VD21_20), -32);
  VS27_26 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(VD29_28),
                            HEXAGON_HVX_GET_V1(VD21_20), -32);
  VS29_28 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(VD31_30),
                            HEXAGON_HVX_GET_V0(VD23_22), -32);
  VS31_30 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(VD31_30),
                            HEXAGON_HVX_GET_V1(VD23_22), -32);

  // Shuffle the 512-bit lanes
  VD1_0 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(VS17_16),
                          HEXAGON_HVX_GET_V0(VS1_0), -64);
  VD3_2 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(VS17_16),
                          HEXAGON_HVX_GET_V1(VS1_0), -64);
  VD5_4 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(VS19_18),
                          HEXAGON_HVX_GET_V0(VS3_2), -64);
  VD7_6 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(VS19_18),
                          HEXAGON_HVX_GET_V1(VS3_2), -64);
  VD9_8 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(VS21_20),
                          HEXAGON_HVX_GET_V0(VS5_4), -64);
  VD11_10 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(VS21_20),
                            HEXAGON_HVX_GET_V1(VS5_4), -64);
  VD13_12 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(VS23_22),
                            HEXAGON_HVX_GET_V0(VS7_6), -64);
  VD15_14 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(VS23_22),
                            HEXAGON_HVX_GET_V1(VS7_6), -64);
  VD17_16 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(VS25_24),
                            HEXAGON_HVX_GET_V0(VS9_8), -64);
  VD19_18 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(VS25_24),
                            HEXAGON_HVX_GET_V1(VS9_8), -64);
  VD21_20 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(VS27_26),
                            HEXAGON_HVX_GET_V0(VS11_10), -64);
  VD23_22 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(VS27_26),
                            HEXAGON_HVX_GET_V1(VS11_10), -64);
  VD25_24 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(VS29_28),
                            HEXAGON_HVX_GET_V0(VS13_12), -64);
  VD27_26 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(VS29_28),
                            HEXAGON_HVX_GET_V1(VS13_12), -64);
  VD29_28 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(VS31_30),
                            HEXAGON_HVX_GET_V0(VS15_14), -64);
  VD31_30 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(VS31_30),
                            HEXAGON_HVX_GET_V1(VS15_14), -64);

  kernel.packet[0] = Packet32f::Create(HEXAGON_HVX_GET_V0(VD1_0));
  kernel.packet[1] = Packet32f::Create(HEXAGON_HVX_GET_V1(VD1_0));
  kernel.packet[2] = Packet32f::Create(HEXAGON_HVX_GET_V0(VD3_2));
  kernel.packet[3] = Packet32f::Create(HEXAGON_HVX_GET_V1(VD3_2));
  kernel.packet[4] = Packet32f::Create(HEXAGON_HVX_GET_V0(VD5_4));
  kernel.packet[5] = Packet32f::Create(HEXAGON_HVX_GET_V1(VD5_4));
  kernel.packet[6] = Packet32f::Create(HEXAGON_HVX_GET_V0(VD7_6));
  kernel.packet[7] = Packet32f::Create(HEXAGON_HVX_GET_V1(VD7_6));
  kernel.packet[8] = Packet32f::Create(HEXAGON_HVX_GET_V0(VD9_8));
  kernel.packet[9] = Packet32f::Create(HEXAGON_HVX_GET_V1(VD9_8));
  kernel.packet[10] = Packet32f::Create(HEXAGON_HVX_GET_V0(VD11_10));
  kernel.packet[11] = Packet32f::Create(HEXAGON_HVX_GET_V1(VD11_10));
  kernel.packet[12] = Packet32f::Create(HEXAGON_HVX_GET_V0(VD13_12));
  kernel.packet[13] = Packet32f::Create(HEXAGON_HVX_GET_V1(VD13_12));
  kernel.packet[14] = Packet32f::Create(HEXAGON_HVX_GET_V0(VD15_14));
  kernel.packet[15] = Packet32f::Create(HEXAGON_HVX_GET_V1(VD15_14));
  kernel.packet[16] = Packet32f::Create(HEXAGON_HVX_GET_V0(VD17_16));
  kernel.packet[17] = Packet32f::Create(HEXAGON_HVX_GET_V1(VD17_16));
  kernel.packet[18] = Packet32f::Create(HEXAGON_HVX_GET_V0(VD19_18));
  kernel.packet[19] = Packet32f::Create(HEXAGON_HVX_GET_V1(VD19_18));
  kernel.packet[20] = Packet32f::Create(HEXAGON_HVX_GET_V0(VD21_20));
  kernel.packet[21] = Packet32f::Create(HEXAGON_HVX_GET_V1(VD21_20));
  kernel.packet[22] = Packet32f::Create(HEXAGON_HVX_GET_V0(VD23_22));
  kernel.packet[23] = Packet32f::Create(HEXAGON_HVX_GET_V1(VD23_22));
  kernel.packet[24] = Packet32f::Create(HEXAGON_HVX_GET_V0(VD25_24));
  kernel.packet[25] = Packet32f::Create(HEXAGON_HVX_GET_V1(VD25_24));
  kernel.packet[26] = Packet32f::Create(HEXAGON_HVX_GET_V0(VD27_26));
  kernel.packet[27] = Packet32f::Create(HEXAGON_HVX_GET_V1(VD27_26));
  kernel.packet[28] = Packet32f::Create(HEXAGON_HVX_GET_V0(VD29_28));
  kernel.packet[29] = Packet32f::Create(HEXAGON_HVX_GET_V1(VD29_28));
  kernel.packet[30] = Packet32f::Create(HEXAGON_HVX_GET_V0(VD31_30));
  kernel.packet[31] = Packet32f::Create(HEXAGON_HVX_GET_V1(VD31_30));
}

template <>
EIGEN_STRONG_INLINE float predux<Packet32f>(const Packet32f& a) {
  HVX_Vector vsum_4 = Q6_Vqf32_vadd_VsfVsf(Q6_V_vror_VR(a.Get(), 4), a.Get());
  HVX_Vector vsum_8 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_vror_VR(vsum_4, 8), vsum_4);
  HVX_Vector vsum_16 =
      Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_vror_VR(vsum_8, 16), vsum_8);
  HVX_Vector vsum_32 =
      Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_vror_VR(vsum_16, 32), vsum_16);
  HVX_Vector vsum_64 =
      Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_vror_VR(vsum_32, 64), vsum_32);
  return pfirst(Packet32f::Create(Q6_Vsf_equals_Vqf32(vsum_64)));
}

template <>
EIGEN_STRONG_INLINE Packet32f ploaddup(const float* from) {
  HVX_Vector load = vmemu(from);
  HVX_VectorPair dup = Q6_W_vshuff_VVR(load, load, -4);
  return Packet32f::Create(HEXAGON_HVX_GET_V0(dup));
}

template <>
EIGEN_STRONG_INLINE Packet32f ploadquad(const float* from) {
  HVX_Vector load = vmemu(from);
  HVX_VectorPair dup = Q6_W_vshuff_VVR(load, load, -4);
  HVX_VectorPair quad =
      Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(dup), HEXAGON_HVX_GET_V0(dup), -8);
  return Packet32f::Create(HEXAGON_HVX_GET_V0(quad));
}

template <>
EIGEN_STRONG_INLINE Packet32f preverse(const Packet32f& a) {
  HVX_Vector delta = Q6_Vb_vsplat_R(0x7c);
  return Packet32f::Create(Q6_V_vdelta_VV(a.Get(), delta));
}

template <>
EIGEN_STRONG_INLINE Packet32f pmin(const Packet32f& a, const Packet32f& b) {
  return Packet32f::Create(Q6_Vsf_vmin_VsfVsf(a.Get(), b.Get()));
}

template <>
EIGEN_STRONG_INLINE Packet32f pmax(const Packet32f& a, const Packet32f& b) {
  return Packet32f::Create(Q6_Vsf_vmax_VsfVsf(a.Get(), b.Get()));
}

template <>
EIGEN_STRONG_INLINE Packet32f pand(const Packet32f& a, const Packet32f& b) {
  return Packet32f::Create(a.Get() & b.Get());
}

template <>
EIGEN_STRONG_INLINE Packet32f por(const Packet32f& a, const Packet32f& b) {
  return Packet32f::Create(a.Get() | b.Get());
}

template <>
EIGEN_STRONG_INLINE Packet32f pxor(const Packet32f& a, const Packet32f& b) {
  return Packet32f::Create(a.Get() ^ b.Get());
}

template <>
EIGEN_STRONG_INLINE Packet32f pnot(const Packet32f& a) {
  return Packet32f::Create(~a.Get());
}

template <>
EIGEN_STRONG_INLINE Packet32f pselect(const Packet32f& mask, const Packet32f& a,
                                      const Packet32f& b) {
  HVX_VectorPred pred = Q6_Q_vcmp_eq_VwVw(mask.Get(), Q6_V_vzero());
  return Packet32f::Create(Q6_V_vmux_QVV(pred, b.Get(), a.Get()));
}

template <typename Op>
EIGEN_STRONG_INLINE float predux_generic(const Packet32f& a, Op op) {
  Packet32f vredux_4 = op(Packet32f::Create(Q6_V_vror_VR(a.Get(), 4)), a);
  Packet32f vredux_8 =
      op(Packet32f::Create(Q6_V_vror_VR(vredux_4.Get(), 8)), vredux_4);
  Packet32f vredux_16 =
      op(Packet32f::Create(Q6_V_vror_VR(vredux_8.Get(), 16)), vredux_8);
  Packet32f vredux_32 =
      op(Packet32f::Create(Q6_V_vror_VR(vredux_16.Get(), 32)), vredux_16);
  Packet32f vredux_64 =
      op(Packet32f::Create(Q6_V_vror_VR(vredux_32.Get(), 64)), vredux_32);
  return pfirst(vredux_64);
}

template <>
EIGEN_STRONG_INLINE float predux_max(const Packet32f& a) {
  return predux_generic(a, pmax<Packet32f>);
}

template <>
EIGEN_STRONG_INLINE float predux_min(const Packet32f& a) {
  return predux_generic(a, pmin<Packet32f>);
}

template <>
EIGEN_STRONG_INLINE bool predux_any(const Packet32f& a) {
  return predux_generic(a, por<Packet32f>) != 0.0f;
}

static const float index_vsf[32] __attribute__((aligned(128))) = {
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};

template <>
EIGEN_STRONG_INLINE Packet32f plset(const float& a) {
  return padd(pload<Packet32f>(index_vsf), pset1<Packet32f>(a));
}

// qfloat32 operations.
template <>
EIGEN_STRONG_INLINE Packet32qf pzero<Packet32qf>(const Packet32qf&) {
  return Packet32qf::Create(Q6_V_vzero());
}

template <>
EIGEN_STRONG_INLINE Packet32qf pmul<Packet32qf>(const Packet32qf& a,
                                                const Packet32qf& b) {
  return Packet32qf::Create(Q6_Vqf32_vmpy_Vqf32Vqf32(a.Get(), b.Get()));
}

template <>
EIGEN_STRONG_INLINE Packet32qf padd<Packet32qf>(const Packet32qf& a,
                                                const Packet32qf& b) {
  return Packet32qf::Create(Q6_Vqf32_vadd_Vqf32Vqf32(a.Get(), b.Get()));
}

// Mixed float32 and qfloat32 operations.
EIGEN_STRONG_INLINE Packet32qf pmadd_f32_to_qf32(const Packet32f& a,
                                                 const Packet32f& b,
                                                 const Packet32qf& c) {
  return Packet32qf::Create(Q6_Vqf32_vadd_Vqf32Vqf32(
      Q6_Vqf32_vmpy_VsfVsf(a.Get(), b.Get()), c.Get()));
}

EIGEN_STRONG_INLINE Packet32f pmadd_qf32_to_f32(const Packet32qf& a,
                                                const Packet32f& b,
                                                const Packet32f& c) {
  return Packet32f::Create(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(
      Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(a.Get()), b.Get()), c.Get())));
}

#endif  // __HVX_ARCH__ >= 68

}  // end namespace internal
}  // end namespace Eigen

#endif  // __HVX__ && (__HVX_LENGTH__ == 128)

#endif  // EIGEN_HVX_PACKET_MATH_H
