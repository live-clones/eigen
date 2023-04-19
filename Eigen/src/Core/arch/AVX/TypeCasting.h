// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_TYPE_CASTING_AVX_H
#define EIGEN_TYPE_CASTING_AVX_H

#include "../../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

#ifndef EIGEN_VECTORIZE_AVX512
template <>
struct type_casting_traits<Eigen::half, float> {
  enum {
    VectorizedCast = 1,
    SrcCoeffRatio = 1,
    TgtCoeffRatio = 1
  };
};


template <>
struct type_casting_traits<float, Eigen::half> {
  enum {
    VectorizedCast = 1,
    SrcCoeffRatio = 1,
    TgtCoeffRatio = 1
  };
};

template <>
struct type_casting_traits<bfloat16, float> {
  enum {
    VectorizedCast = 1,
    SrcCoeffRatio = 1,
    TgtCoeffRatio = 1
  };
};

template <>
struct type_casting_traits<float, bfloat16> {
  enum {
    VectorizedCast = 1,
    SrcCoeffRatio = 1,
    TgtCoeffRatio = 1
  };
};

template <>
struct type_casting_traits<float, bool> {
  enum {
    VectorizedCast = 1,
    SrcCoeffRatio = 2,
    TgtCoeffRatio = 1
  };
};
#endif  // EIGEN_VECTORIZE_AVX512

// float to int
template <>
EIGEN_STRONG_INLINE Packet8i pcast<Packet8f, Packet8i>(const Packet8f& a) {
  return _mm256_cvttps_epi32(a);
}
template <>
struct scalar_cast_op<float, int> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int operator()(float a) const { return static_cast<int>(a); }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet8i packetOp(Packet8f a) const { return pcast<Packet8f, Packet8i>(a); }
};
template <>
struct functor_traits<scalar_cast_op<float, int>> {
  enum { Cost = 1, PacketAccess = true };
};

// int to float
template <>
EIGEN_STRONG_INLINE Packet8f pcast<Packet8i, Packet8f>(const Packet8i& a) {
  return _mm256_cvtepi32_ps(a);
}
template <>
struct scalar_cast_op<int, float> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float operator()(int a) const { return static_cast<float>(a); }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet8f packetOp(Packet8i a) const { return pcast<Packet8i, Packet8f>(a); }
};
template <>
struct functor_traits<scalar_cast_op<int, float>> {
  enum { Cost = 1, PacketAccess = true };
};

// double to float
template <>
EIGEN_STRONG_INLINE Packet8f pcast<Packet4d, Packet8f>(const Packet4d& a, const Packet4d& b) {
  return _mm256_set_m128(_mm256_cvtpd_ps(b), _mm256_cvtpd_ps(a));
}
template <>
struct scalar_cast_op<double, float> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float operator()(double a) const { return static_cast<float>(a); }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet8f packetOp(Packet4d a, Packet4d b) const {
    return pcast<Packet4d, Packet8f>(a, b);
  }
};
template <>
struct functor_traits<scalar_cast_op<double, float>> {
  enum { Cost = 2, PacketAccess = true };
};

// double to int
template <>
EIGEN_STRONG_INLINE Packet8i pcast<Packet4d, Packet8i>(const Packet4d& a, const Packet4d& b) {
  return _mm256_set_m128i(_mm256_cvttpd_epi32(b), _mm256_cvttpd_epi32(a));
}
template <>
struct scalar_cast_op<double, int> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int operator()(double a) const { return static_cast<int>(a); }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet8i packetOp(Packet4d a, Packet4d b) const {
    return pcast<Packet4d, Packet8i>(a, b);
  }
};
template <>
struct functor_traits<scalar_cast_op<double, int>> {
  enum { Cost = 2, PacketAccess = true };
};

// float to bool
template <>
EIGEN_STRONG_INLINE Packet16b pcast<Packet8f, Packet16b>(const Packet8f& a,
                                                         const Packet8f& b) {
  __m256 nonzero_a = _mm256_cmp_ps(a, pzero(a), _CMP_NEQ_UQ);
  __m256 nonzero_b = _mm256_cmp_ps(b, pzero(b), _CMP_NEQ_UQ);
  constexpr char kFF = '\255';
#ifndef EIGEN_VECTORIZE_AVX2
  __m128i shuffle_mask128_a_lo = _mm_set_epi8(kFF, kFF, kFF, kFF, kFF, kFF, kFF, kFF, kFF, kFF, kFF, kFF, 12, 8, 4, 0);
  __m128i shuffle_mask128_a_hi = _mm_set_epi8(kFF, kFF, kFF, kFF, kFF, kFF, kFF, kFF, 12, 8, 4, 0, kFF, kFF, kFF, kFF);
  __m128i shuffle_mask128_b_lo = _mm_set_epi8(kFF, kFF, kFF, kFF, 12, 8, 4, 0, kFF, kFF, kFF, kFF, kFF, kFF, kFF, kFF);
  __m128i shuffle_mask128_b_hi = _mm_set_epi8(12, 8, 4, 0, kFF, kFF, kFF, kFF, kFF, kFF, kFF, kFF, kFF, kFF, kFF, kFF);
  __m128i a_hi = _mm_shuffle_epi8(_mm256_extractf128_si256(_mm256_castps_si256(nonzero_a), 1), shuffle_mask128_a_hi);
  __m128i a_lo = _mm_shuffle_epi8(_mm256_extractf128_si256(_mm256_castps_si256(nonzero_a), 0), shuffle_mask128_a_lo);
  __m128i b_hi = _mm_shuffle_epi8(_mm256_extractf128_si256(_mm256_castps_si256(nonzero_b), 1), shuffle_mask128_b_hi);
  __m128i b_lo = _mm_shuffle_epi8(_mm256_extractf128_si256(_mm256_castps_si256(nonzero_b), 0), shuffle_mask128_b_lo);
  __m128i merged = _mm_or_si128(_mm_or_si128(b_lo, b_hi), _mm_or_si128(a_lo, a_hi));
  return _mm_and_si128(merged, _mm_set1_epi8(1));
 #else
  __m256i a_shuffle_mask = _mm256_set_epi8(kFF, kFF, kFF, kFF, kFF, kFF, kFF, kFF,  12,   8,   4,   0, kFF, kFF, kFF, kFF,
                                           kFF, kFF, kFF, kFF, kFF, kFF, kFF, kFF, kFF, kFF, kFF, kFF,  12,   8,   4,   0);
  __m256i b_shuffle_mask = _mm256_set_epi8( 12,   8,   4,   0, kFF, kFF, kFF, kFF, kFF, kFF, kFF, kFF, kFF, kFF, kFF, kFF,
                                           kFF, kFF, kFF, kFF,  12,   8,   4,   0, kFF, kFF, kFF, kFF, kFF, kFF, kFF, kFF);
  __m256i a_shuff = _mm256_shuffle_epi8(_mm256_castps_si256(nonzero_a), a_shuffle_mask);
  __m256i b_shuff = _mm256_shuffle_epi8(_mm256_castps_si256(nonzero_b), b_shuffle_mask);
  __m256i a_or_b = _mm256_or_si256(a_shuff, b_shuff);
  __m256i merged = _mm256_or_si256(a_or_b, _mm256_castsi128_si256(_mm256_extractf128_si256(a_or_b, 1)));
  return _mm256_castsi256_si128(_mm256_and_si256(merged, _mm256_set1_epi8(1)));
#endif
}
template <>
struct scalar_cast_op<float, bool> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool operator()(float a) const { return static_cast<bool>(a); }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet16b packetOp(Packet8f a, Packet8f b) const {
    return pcast<Packet8f, Packet16b>(a, b);
  }
};
template <>
struct functor_traits<scalar_cast_op<float, bool>> {
  enum { Cost = 4, PacketAccess = true };
};

template<> EIGEN_STRONG_INLINE Packet8i preinterpret<Packet8i,Packet8f>(const Packet8f& a) {
  return _mm256_castps_si256(a);
}

template<> EIGEN_STRONG_INLINE Packet8f preinterpret<Packet8f,Packet8i>(const Packet8i& a) {
  return _mm256_castsi256_ps(a);
}

template <>
EIGEN_STRONG_INLINE Packet8f pcast<Packet8h, Packet8f>(const Packet8h& a) {
  return half2float(a);
}
template <>
struct scalar_cast_op<half, float> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float operator()(const half& a) const { return cast<half, float>(a); }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet8f packetOp(Packet8h a) const { return pcast<Packet8h, Packet8f>(a); }
};
template <>
struct functor_traits<scalar_cast_op<half, float>> {
  enum { Cost = 1, PacketAccess = true };
};

template <>
EIGEN_STRONG_INLINE Packet8f pcast<Packet8bf, Packet8f>(const Packet8bf& a) {
  return Bf16ToF32(a);
}
template <>
struct scalar_cast_op<bfloat16, float> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float operator()(const bfloat16& a) const { return cast<bfloat16, float>(a); }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet8f packetOp(Packet8bf a) const { return pcast<Packet8bf, Packet8f>(a); }
};
template <>
struct functor_traits<scalar_cast_op<bfloat16, float>> {
  enum { Cost = 1, PacketAccess = true };
};

template <>
EIGEN_STRONG_INLINE Packet8h pcast<Packet8f, Packet8h>(const Packet8f& a) {
  return float2half(a);
}
template <>
struct scalar_cast_op<float, half> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half operator()(float a) const { return cast<float, half>(a); }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet8h packetOp(Packet8f a) const { return pcast<Packet8f, Packet8h>(a); }
};
template <>
struct functor_traits<scalar_cast_op<float, half>> {
  enum { Cost = 1, PacketAccess = true };
};

template <>
EIGEN_STRONG_INLINE Packet8bf pcast<Packet8f, Packet8bf>(const Packet8f& a) {
  return F32ToBf16(a);
}
template <>
struct scalar_cast_op<float, bfloat16> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bfloat16 operator()(float a) const { return cast<float, bfloat16>(a); }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet8bf packetOp(Packet8f a) const { return pcast<Packet8f, Packet8bf>(a); }
};
template <>
struct functor_traits<scalar_cast_op<float, bfloat16>> {
  enum { Cost = 1, PacketAccess = true };
};

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_TYPE_CASTING_AVX_H
