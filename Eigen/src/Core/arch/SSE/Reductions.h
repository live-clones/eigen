// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2025 Charlie Schlosser <cs.schlosser@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_REDUCTIONS_SSE_H
#define EIGEN_REDUCTIONS_SSE_H

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

template <typename Op>
EIGEN_STRONG_INLINE int predux_op(const Packet4i& a, Op op) {
  __m128i tmp;
  tmp = op(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)));
  tmp = op(tmp, _mm_shuffle_epi32(tmp, _MM_SHUFFLE(2, 3, 0, 1)));
  return _mm_cvtsi128_si32(tmp);
}

template <typename Op>
EIGEN_STRONG_INLINE uint32_t predux_op(const Packet4ui& a, Op op) {
  __m128i tmp;
  tmp = op(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)));
  tmp = op(tmp, _mm_shuffle_epi32(tmp, _MM_SHUFFLE(2, 3, 0, 1)));
  return static_cast<uint32_t>(_mm_cvtsi128_si32(tmp));
}

template <typename Op>
EIGEN_STRONG_INLINE int64_t predux_op(const Packet2l& a, Op op) {
  __m128i tmp;
  tmp = op(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)));
  return static_cast<int64_t>(_mm_cvtsi128_si64(tmp));
}

template <typename Op>
EIGEN_STRONG_INLINE float predux_op(const Packet4f& a, Op op) {
  __m128 tmp;
#ifdef EIGEN_VECTORIZE_AVX
  tmp = op(a, _mm_permute_ps(a, _MM_SHUFFLE(1, 0, 3, 2)));
  tmp = op(tmp, _mm_permute_ps(tmp, _MM_SHUFFLE(2, 3, 0, 1)));
#else
  tmp = op(a, _mm_shuffle_ps(a, a, _MM_SHUFFLE(1, 0, 3, 2)));
  tmp = op(tmp, _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(2, 3, 0, 1)));
#endif
  return _mm_cvtss_f32(tmp);
}

template <typename Op>
EIGEN_STRONG_INLINE double predux_op(const Packet2d& a, Op op) {
  __m128d tmp;
#ifdef EIGEN_VECTORIZE_AVX
  tmp = op(a, _mm_permute_pd(a, 1));
#else
  tmp = op(a, _mm_shuffle_pd(a, a, 1));
#endif
  return _mm_cvtsd_f64(tmp);
}

/* -- -- -- -- -- -- -- -- -- -- -- -- Packet16b -- -- -- -- -- -- -- -- -- -- -- -- */

template <>
EIGEN_STRONG_INLINE bool predux(const Packet16b& a) {
  Packet4i tmp = _mm_or_si128(a, _mm_unpackhi_epi64(a, a));
  return (pfirst(tmp) != 0) || (pfirst<Packet4i>(_mm_shuffle_epi32(tmp, 1)) != 0);
}

template <>
EIGEN_STRONG_INLINE bool predux_mul(const Packet16b& a) {
  Packet4i tmp = _mm_and_si128(a, _mm_unpackhi_epi64(a, a));
  return ((pfirst<Packet4i>(tmp) == 0x01010101) && (pfirst<Packet4i>(_mm_shuffle_epi32(tmp, 1)) == 0x01010101));
}

/* -- -- -- -- -- -- -- -- -- -- -- -- Packet4i -- -- -- -- -- -- -- -- -- -- -- -- */

template <>
EIGEN_STRONG_INLINE int predux(const Packet4i& a) {
  return predux_op(a, padd<Packet4i>);
}

template <>
EIGEN_STRONG_INLINE int predux_mul(const Packet4i& a) {
  return predux_op(a, pmul<Packet4i>);
}

#ifdef EIGEN_VECTORIZE_SSE4_1
template <>
EIGEN_STRONG_INLINE int predux_min(const Packet4i& a) {
  return predux_op(a, pmin<Packet4i>);
}

template <>
EIGEN_STRONG_INLINE int predux_max(const Packet4i& a) {
  return predux_op(a, pmax<Packet4i>);
}
#endif

template <>
EIGEN_STRONG_INLINE bool predux_any(const Packet4i& x) {
  return _mm_movemask_ps(_mm_castsi128_ps(x)) != 0x0;
}

/* -- -- -- -- -- -- -- -- -- -- -- -- Packet4ui -- -- -- -- -- -- -- -- -- -- -- -- */

template <>
EIGEN_STRONG_INLINE uint32_t predux(const Packet4ui& a) {
  return predux_op(a, padd<Packet4ui>);
}

template <>
EIGEN_STRONG_INLINE uint32_t predux_mul(const Packet4ui& a) {
  return predux_op(a, pmul<Packet4ui>);
}

#ifdef EIGEN_VECTORIZE_SSE4_1
template <>
EIGEN_STRONG_INLINE uint32_t predux_min(const Packet4ui& a) {
  return predux_op(a, pmin<Packet4ui>);
}

template <>
EIGEN_STRONG_INLINE uint32_t predux_max(const Packet4ui& a) {
  return predux_op(a, pmax<Packet4ui>);
}
#endif

template <>
EIGEN_STRONG_INLINE bool predux_any(const Packet4ui& x) {
  return _mm_movemask_ps(_mm_castsi128_ps(x)) != 0x0;
}

/* -- -- -- -- -- -- -- -- -- -- -- -- Packet2l -- -- -- -- -- -- -- -- -- -- -- -- */

template <>
EIGEN_STRONG_INLINE int64_t predux(const Packet2l& a) {
  return predux_op(a, padd<Packet2l>);
}

template <>
EIGEN_STRONG_INLINE bool predux_any(const Packet2l& x) {
  return _mm_movemask_pd(_mm_castsi128_pd(x)) != 0x0;
}

/* -- -- -- -- -- -- -- -- -- -- -- -- Packet4f -- -- -- -- -- -- -- -- -- -- -- -- */

template <>
EIGEN_STRONG_INLINE float predux(const Packet4f& a) {
  return predux_op(a, padd<Packet4f>);
}

template <>
EIGEN_STRONG_INLINE float predux_mul(const Packet4f& a) {
  return predux_op(a, pmul<Packet4f>);
}

template <>
EIGEN_STRONG_INLINE float predux_min(const Packet4f& a) {
  return predux_op(a, pmin<Packet4f>);
}

template <>
EIGEN_STRONG_INLINE float predux_min<PropagateFast>(const Packet4f& a) {
  return predux_op(a, pmin<Packet4f>);
}

template <>
EIGEN_STRONG_INLINE float predux_min<PropagateNumbers>(const Packet4f& a) {
  return predux_op(a, pmin<PropagateNumbers, Packet4f>);
}

template <>
EIGEN_STRONG_INLINE float predux_min<PropagateNaN>(const Packet4f& a) {
  return predux_op(a, pmin<PropagateNaN, Packet4f>);
}

template <>
EIGEN_STRONG_INLINE float predux_max(const Packet4f& a) {
  return predux_op(a, pmax<Packet4f>);
}

template <>
EIGEN_STRONG_INLINE float predux_max<PropagateFast>(const Packet4f& a) {
  return predux_op(a, pmax<Packet4f>);
}

template <>
EIGEN_STRONG_INLINE float predux_max<PropagateNumbers>(const Packet4f& a) {
  return predux_op(a, pmax<PropagateNumbers, Packet4f>);
}

template <>
EIGEN_STRONG_INLINE float predux_max<PropagateNaN>(const Packet4f& a) {
  return predux_op(a, pmax<PropagateNaN, Packet4f>);
}

template <>
EIGEN_STRONG_INLINE bool predux_any(const Packet4f& x) {
  return _mm_movemask_ps(x) != 0x0;
}

/* -- -- -- -- -- -- -- -- -- -- -- -- Packet2d -- -- -- -- -- -- -- -- -- -- -- -- */

template <>
EIGEN_STRONG_INLINE double predux(const Packet2d& a) {
  return predux_op(a, padd<Packet2d>);
}

template <>
EIGEN_STRONG_INLINE double predux_mul(const Packet2d& a) {
  return predux_op(a, pmul<Packet2d>);
}

template <>
EIGEN_STRONG_INLINE double predux_min(const Packet2d& a) {
  return predux_op(a, pmin<Packet2d>);
}

template <>
EIGEN_STRONG_INLINE double predux_min<PropagateFast>(const Packet2d& a) {
  return predux_op(a, pmin<Packet2d>);
}

template <>
EIGEN_STRONG_INLINE double predux_min<PropagateNumbers>(const Packet2d& a) {
  return predux_op(a, pmin<PropagateNumbers, Packet2d>);
}

template <>
EIGEN_STRONG_INLINE double predux_min<PropagateNaN>(const Packet2d& a) {
  return predux_op(a, pmin<PropagateNaN, Packet2d>);
}

template <>
EIGEN_STRONG_INLINE double predux_max(const Packet2d& a) {
  return predux_op(a, pmax<Packet2d>);
}

template <>
EIGEN_STRONG_INLINE double predux_max<PropagateFast>(const Packet2d& a) {
  return predux_op(a, pmax<Packet2d>);
}

template <>
EIGEN_STRONG_INLINE double predux_max<PropagateNumbers>(const Packet2d& a) {
  return predux_op(a, pmax<PropagateNumbers, Packet2d>);
}

template <>
EIGEN_STRONG_INLINE double predux_max<PropagateNaN>(const Packet2d& a) {
  return predux_op(a, pmax<PropagateNaN, Packet2d>);
}

template <>
EIGEN_STRONG_INLINE bool predux_any(const Packet2d& x) {
  return _mm_movemask_pd(x) != 0x0;
}

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_REDUCTIONS_SSE_H
