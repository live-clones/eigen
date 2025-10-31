// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2025 Rasmus Munk Larsen
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_PACKET_MATH_CLANG_H
#define EIGEN_PACKET_MATH_CLANG_H

namespace Eigen {

namespace internal {

template <typename ScalarT, int n>
using VectorType = ScalarT __attribute__((vector_size(n * sizeof(ScalarT))));

// --- vector type helpers ---
template <typename VectorT>
struct ScalarTypeOfVector {
  using type = std::remove_all_extents_t<std::remove_cvref_t<decltype(VectorT()[0])>>;
};

template <typename VectorT>
using scalar_type_of_vector_t = typename ScalarTypeOfVector<VectorT>::type;

template <typename VectorT>
using HalfPacket = VectorType<typename unpacket_traits<VectorT>::type, unpacket_traits<VectorT>::size / 2>;

template <typename VectorT>
using QuarterPacket = VectorType<typename unpacket_traits<VectorT>::type, unpacket_traits<VectorT>::size / 4>;

// load and store helpers.
template <typename VectorT>
EIGEN_STRONG_INLINE VectorT load_vector_unaligned(const scalar_type_of_vector_t<VectorT>* from) {
  VectorT to;
  constexpr int n = __builtin_vectorelements(to);
  for (int i = 0; i < n; ++i) {
    to[i] = from[i];
  }
  return to;
}

template <typename VectorT>
EIGEN_STRONG_INLINE VectorT load_vector_aligned(const scalar_type_of_vector_t<VectorT>* from) {
  return *(VectorT*)from;
}

template <typename VectorT>
EIGEN_STRONG_INLINE void store_vector_unaligned(scalar_type_of_vector_t<VectorT>* to, const VectorT& from) {
  constexpr int n = __builtin_vectorelements(from);
  for (int i = 0; i < n; ++i) {
    *to++ = from[i];
  }
}

template <typename VectorT>
EIGEN_STRONG_INLINE void store_vector_aligned(scalar_type_of_vector_t<VectorT>* to, const VectorT& from) {
  return *(VectorT*)to = from;
}

template <typename VectorT>
EIGEN_STRONG_INLINE VectorT const_vector(const scalar_type_of_vector_t<VectorT>& value) {
  VectorT result;
  constexpr int n = __builtin_vectorelements(result);
  for (int i = 0; i < n; ++i) {
    result[i] = value;
  }
  return result;
}

// --- Packet type Definitions (512 bit) ---
using Packet16f = VectorType<float, 16>;
using Packet8d = VectorType<double, 8>;
using Packet16i = VectorType<int32_t, 16>;
using Packet8l = VectorType<int64_t, 8>;

// --- packet_traits specializations ---
template <>
struct packet_traits<float> : default_packet_traits {
  using type = Packet16f;
  using half = Packet16f;
  enum {
    Vectorizable = 1,
    size = 16,
    AlignedOnScalar = 1,
    MemoryAlignment = 64,
    HasAdd = 1,
    HasSub = 1,
    HasMul = 1,
    HasDiv = 1,
    HasNegate = 1,
    HasAbs = 1,
    HasFloor = 1,
    HasCeil = 1,
    HasRound = 1,
    HasMinMax = 1,
    HasCmp = 1,
    HasBlend = 0,
    HasSet1 = 1,
    HasCast = 1,
    HasBitwise = 1,
    HasRedux = 1,
    HasSign = 1,
    HasArg = 0,
    HasConj = 1,
    // Math functions
    HasReciprocal = 1,
    HasSin = 1,
    HasCos = 1,
    HasACos = 1,
    HasASin = 1,
    HasATan = 1,
    HasATanh = 1,
    HasLog = 1,
    HasLog1p = 1,
    HasExpm1 = 1,
    HasExp = 1,
    HasPow = 1,
    HasNdtri = 1,
    HasBessel = 1,
    HasSqrt = 1,
    HasRsqrt = 1,
    HasCbrt = 1,
    HasTanh = 1,
    HasErf = 1,
    HasErfc = 1
  };
};

template <>
struct packet_traits<double> : default_packet_traits {
  using type = Packet8d;
  using half = Packet8d;
  enum {
    Vectorizable = 1,
    size = 8,
    AlignedOnScalar = 1,
    MemoryAlignment = 64,
    HasAdd = 1,
    HasSub = 1,
    HasMul = 1,
    HasDiv = 1,
    HasNegate = 1,
    HasAbs = 1,
    HasFloor = 1,
    HasCeil = 1,
    HasRound = 1,
    HasMinMax = 1,
    HasCmp = 1,
    HasBlend = 0,
    HasSet1 = 1,
    HasCast = 1,
    HasBitwise = 1,
    HasRedux = 1,
    HasSign = 1,
    HasArg = 0,
    HasConj = 1,
    // Math functions
    HasReciprocal = 1,
    HasSin = 1,
    HasCos = 1,
    HasACos = 0,
    HasASin = 0,
    HasATan = 1,
    HasATanh = 1,
    HasLog = 1,
    HasLog1p = 1,
    HasExpm1 = 1,
    HasExp = 1,
    HasPow = 1,
    HasNdtri = 1,
    HasBessel = 1,
    HasSqrt = 1,
    HasRsqrt = 1,
    HasCbrt = 1,
    HasTanh = 1,
    HasErf = 1,
    HasErfc = 1
  };
};

template <>
struct packet_traits<int32_t> : default_packet_traits {
  using type = Packet16i;
  using half = Packet16i;
  enum {
    Vectorizable = 1,
    size = 16,
    AlignedOnScalar = 1,
    MemoryAlignment = 64,
    HasAdd = 1,
    HasSub = 1,
    HasMul = 1,
    HasNegate = 1,
    HasAbs = 1,
    HasMinMax = 1,
    HasCmp = 0,
    HasBlend = 0,
    HasSet1 = 1,
    HasCast = 1,
    HasBitwise = 1,
    HasRedux = 1,
    // Set remaining to 0
    HasDiv = 0,
    HasFloor = 0,
    HasCeil = 0,
    HasRound = 0,
    HasSqrt = 0,
    HasRsqrt = 0,
    HasReciprocal = 0,
    HasArg = 0,
    HasConj = 0,
    HasExp = 0,
    HasLog = 0,
    HasSin = 0,
    HasCos = 0,
  };
};

template <>
struct packet_traits<int64_t> : default_packet_traits {
  using type = Packet8l;
  using half = Packet8l;
  enum {
    Vectorizable = 1,
    size = 8,
    AlignedOnScalar = 1,
    MemoryAlignment = 64,
    HasAdd = 1,
    HasSub = 1,
    HasMul = 1,
    HasNegate = 1,
    HasAbs = 1,
    HasMinMax = 1,
    HasCmp = 0,
    HasBlend = 0,
    HasSet1 = 1,
    HasCast = 1,
    HasBitwise = 1,
    HasRedux = 1,
    // Set remaining to 0
    HasDiv = 0,
    HasFloor = 0,
    HasCeil = 0,
    HasRound = 0,
    HasSqrt = 0,
    HasRsqrt = 0,
    HasReciprocal = 0,
    HasArg = 0,
    HasConj = 0,
    HasExp = 0,
    HasLog = 0,
    HasSin = 0,
    HasCos = 0,
  };
};

// --- unpacket_traits specializations ---
template <>
struct unpacket_traits<Packet16f> {
  using type = float;
  using half = Packet16f;
  using integer_packet = Packet16i;
  enum {
    size = 16,
    alignment = 64,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};
template <>
struct unpacket_traits<Packet8d> {
  using type = double;
  using half = Packet8d;
  using integer_packet = Packet8l;
  enum { size = 8, alignment = 64, vectorizable = true, masked_load_available = false, masked_store_available = false };
};
template <>
struct unpacket_traits<Packet16i> {
  using type = int32_t;
  using half = Packet16i;
  enum {
    size = 16,
    alignment = 64,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};
template <>
struct unpacket_traits<Packet8l> {
  using type = int64_t;
  using half = Packet8l;
  enum { size = 8, alignment = 64, vectorizable = true, masked_load_available = false, masked_store_available = false };
};

// --- Intrinsic-like specializations ---

// --- Load/Store operations ---

#define EIGEN_CLANG_PACKET_LOAD_STORE_PACKET(PACKET_TYPE, SCALAR_TYPE)                                    \
  template <>                                                                                             \
  EIGEN_STRONG_INLINE PACKET_TYPE ploadu<PACKET_TYPE>(const SCALAR_TYPE* from) {                          \
    return load_vector_unaligned<PACKET_TYPE>(from);                                                      \
  }                                                                                                       \
  template <>                                                                                             \
  EIGEN_STRONG_INLINE PACKET_TYPE pload<PACKET_TYPE>(const SCALAR_TYPE* from) {                           \
    return load_vector_aligned<PACKET_TYPE>(from);                                                        \
  }                                                                                                       \
  template <>                                                                                             \
  EIGEN_STRONG_INLINE void pstoreu<SCALAR_TYPE, PACKET_TYPE>(SCALAR_TYPE * to, const PACKET_TYPE& from) { \
    store_vector_unaligned<PACKET_TYPE>(to, from);                                                        \
  }                                                                                                       \
  template <>                                                                                             \
  EIGEN_STRONG_INLINE void pstore<SCALAR_TYPE, PACKET_TYPE>(SCALAR_TYPE * to, const PACKET_TYPE& from) {  \
    store_vector_aligned<PACKET_TYPE>(to, from);                                                          \
  }

EIGEN_CLANG_PACKET_LOAD_STORE_PACKET(Packet16f, float)
EIGEN_CLANG_PACKET_LOAD_STORE_PACKET(Packet8d, double)
EIGEN_CLANG_PACKET_LOAD_STORE_PACKET(Packet16i, int32_t)
EIGEN_CLANG_PACKET_LOAD_STORE_PACKET(Packet8l, int64_t)
#undef EIGEN_CLANG_PACKET_LOAD_STORE_PACKET

// --- Broadcast operation ---
template <>
EIGEN_STRONG_INLINE Packet16f pset1frombits<Packet16f>(uint32_t from) {
  return const_vector<Packet16f>(std::bit_cast<float>(from));
}

template <>
EIGEN_STRONG_INLINE Packet8d pset1frombits<Packet8d>(uint64_t from) {
  return const_vector<Packet8d>(std::bit_cast<double>(from));
}

#define EIGEN_CLANG_PACKET_SET1(PACKET_TYPE)                                                            \
  template <>                                                                                           \
  EIGEN_STRONG_INLINE PACKET_TYPE pset1<PACKET_TYPE>(const unpacket_traits<PACKET_TYPE>::type& from) {  \
    return const_vector<PACKET_TYPE>(from);                                                             \
  }                                                                                                     \
  template <>                                                                                           \
  EIGEN_STRONG_INLINE unpacket_traits<PACKET_TYPE>::type pfirst<PACKET_TYPE>(const PACKET_TYPE& from) { \
    return from[0];                                                                                     \
  }

EIGEN_CLANG_PACKET_SET1(Packet16f)
EIGEN_CLANG_PACKET_SET1(Packet8d)
EIGEN_CLANG_PACKET_SET1(Packet16i)
EIGEN_CLANG_PACKET_SET1(Packet8l)
#undef EIGEN_CLANG_PACKET_SET1

// --- Arithmetic operations ---
#define EIGEN_CLANG_PACKET_ARITHMETIC(PACKET_TYPE)                             \
  template <>                                                                  \
  EIGEN_STRONG_INLINE PACKET_TYPE pisnan<PACKET_TYPE>(const PACKET_TYPE& a) {  \
    return a != a;                                                             \
  }                                                                            \
  template <>                                                                  \
  EIGEN_STRONG_INLINE PACKET_TYPE pnegate<PACKET_TYPE>(const PACKET_TYPE& a) { \
    return -a;                                                                 \
  }

EIGEN_CLANG_PACKET_ARITHMETIC(Packet16f)
EIGEN_CLANG_PACKET_ARITHMETIC(Packet8d)
EIGEN_CLANG_PACKET_ARITHMETIC(Packet16i)
EIGEN_CLANG_PACKET_ARITHMETIC(Packet8l)

#undef EIGEN_CLANG_PACKET_ARITHMETIC
#undef EIGEN_CLANG_PACKET_ARITHMETIC_FLOAT

// --- Bitwise operations (via casting) ---
// Note: pcast functions are not template specializations, just helpers
EIGEN_STRONG_INLINE Packet16i pcast_float_to_int(const Packet16f& a) { return (Packet16i)a; }
EIGEN_STRONG_INLINE Packet16f pcast_int_to_float(const Packet16i& a) { return (Packet16f)a; }
EIGEN_STRONG_INLINE Packet8l pcast_double_to_int(const Packet8d& a) { return (Packet8l)a; }
EIGEN_STRONG_INLINE Packet8d pcast_int_to_double(const Packet8l& a) { return (Packet8d)a; }

// Bitwise ops for integer packets
#define EIGEN_CLANG_PACKET_BITWISE_INT(PACKET_TYPE)                                                  \
  template <>                                                                                        \
  EIGEN_STRONG_INLINE PACKET_TYPE ptrue<PACKET_TYPE>(const PACKET_TYPE& a) {                         \
    return a == a;                                                                                   \
  }                                                                                                  \
  template <>                                                                                        \
  EIGEN_STRONG_INLINE PACKET_TYPE pand<PACKET_TYPE>(const PACKET_TYPE& a, const PACKET_TYPE& b) {    \
    return a & b;                                                                                    \
  }                                                                                                  \
  template <>                                                                                        \
  EIGEN_STRONG_INLINE PACKET_TYPE por<PACKET_TYPE>(const PACKET_TYPE& a, const PACKET_TYPE& b) {     \
    return a | b;                                                                                    \
  }                                                                                                  \
  template <>                                                                                        \
  EIGEN_STRONG_INLINE PACKET_TYPE pxor<PACKET_TYPE>(const PACKET_TYPE& a, const PACKET_TYPE& b) {    \
    return a ^ b;                                                                                    \
  }                                                                                                  \
  template <>                                                                                        \
  EIGEN_STRONG_INLINE PACKET_TYPE pandnot<PACKET_TYPE>(const PACKET_TYPE& a, const PACKET_TYPE& b) { \
    return a & ~b;                                                                                   \
  }                                                                                                  \
  template <int N>                                                                                   \
  EIGEN_STRONG_INLINE PACKET_TYPE parithmetic_shift_right(const PACKET_TYPE& a) {                    \
    return a >> N;                                                                                   \
  }                                                                                                  \
  template <int N>                                                                                   \
  EIGEN_STRONG_INLINE PACKET_TYPE plogical_shift_right(const PACKET_TYPE& a) {                       \
    return a >> N;                                                                                   \
  }                                                                                                  \
  template <int N>                                                                                   \
  EIGEN_STRONG_INLINE PACKET_TYPE plogical_shift_left(const PACKET_TYPE& a) {                        \
    return a << N;                                                                                   \
  }

EIGEN_CLANG_PACKET_BITWISE_INT(Packet16i)
EIGEN_CLANG_PACKET_BITWISE_INT(Packet8l)
#undef EIGEN_CLANG_PACKET_BITWISE_INT

// Bitwise ops for floating point packets
#define EIGEN_CLANG_PACKET_BITWISE_FLOAT(PACKET_TYPE, CAST_TO_INT, CAST_FROM_INT)                    \
  template <>                                                                                        \
  EIGEN_STRONG_INLINE PACKET_TYPE ptrue<PACKET_TYPE>(const PACKET_TYPE& a) {                         \
    return CAST_FROM_INT(CAST_TO_INT(a) == CAST_TO_INT(a));                                          \
  }                                                                                                  \
  template <>                                                                                        \
  EIGEN_STRONG_INLINE PACKET_TYPE pand<PACKET_TYPE>(const PACKET_TYPE& a, const PACKET_TYPE& b) {    \
    return CAST_FROM_INT(CAST_TO_INT(a) & CAST_TO_INT(b));                                           \
  }                                                                                                  \
  template <>                                                                                        \
  EIGEN_STRONG_INLINE PACKET_TYPE por<PACKET_TYPE>(const PACKET_TYPE& a, const PACKET_TYPE& b) {     \
    return CAST_FROM_INT(CAST_TO_INT(a) | CAST_TO_INT(b));                                           \
  }                                                                                                  \
  template <>                                                                                        \
  EIGEN_STRONG_INLINE PACKET_TYPE pxor<PACKET_TYPE>(const PACKET_TYPE& a, const PACKET_TYPE& b) {    \
    return CAST_FROM_INT(CAST_TO_INT(a) ^ CAST_TO_INT(b));                                           \
  }                                                                                                  \
  template <>                                                                                        \
  EIGEN_STRONG_INLINE PACKET_TYPE pandnot<PACKET_TYPE>(const PACKET_TYPE& a, const PACKET_TYPE& b) { \
    return CAST_FROM_INT(CAST_TO_INT(a) & ~CAST_TO_INT(b));                                          \
  }

EIGEN_CLANG_PACKET_BITWISE_FLOAT(Packet16f, pcast_float_to_int, pcast_int_to_float)
EIGEN_CLANG_PACKET_BITWISE_FLOAT(Packet8d, pcast_double_to_int, pcast_int_to_double)
#undef EIGEN_CLANG_PACKET_BITWISE_FLOAT

// --- Blending operation ---
/* #define EIGEN_CLANG_PACKET_BLEND(PACKET_TYPE, MASK_TYPE) \ */
/*     template<> EIGEN_STRONG_INLINE PACKET_TYPE pblend<PACKET_TYPE>(const
 * PACKET_TYPE& if_true, const PACKET_TYPE& if_false, const MASK_TYPE& mask) {
 * \ */
/*         return mask ? if_true : if_false; \ */
/*     } */

/* EIGEN_CLANG_PACKET_BLEND(Packet16f, Packet16i) */
/* EIGEN_CLANG_PACKET_BLEND(Packet8d,  Packet8l) */
/* #undef EIGEN_CLANG_PACKET_BLEND */

// --- Min/Max operations ---
#define EIGEN_CLANG_PACKET_MINMAX(PACKET_TYPE)                                                    \
  template <>                                                                                     \
  EIGEN_STRONG_INLINE PACKET_TYPE pmin<PACKET_TYPE>(const PACKET_TYPE& a, const PACKET_TYPE& b) { \
    return __builtin_elementwise_min(a, b);                                                       \
  }                                                                                               \
  template <>                                                                                     \
  EIGEN_STRONG_INLINE PACKET_TYPE pmax<PACKET_TYPE>(const PACKET_TYPE& a, const PACKET_TYPE& b) { \
    return __builtin_elementwise_max(a, b);                                                       \
  }                                                                                               \
  template <>                                                                                     \
  EIGEN_STRONG_INLINE PACKET_TYPE pabs<PACKET_TYPE>(const PACKET_TYPE& a) {                       \
    return __builtin_elementwise_abs(a);                                                          \
  }

EIGEN_CLANG_PACKET_MINMAX(Packet16f)
EIGEN_CLANG_PACKET_MINMAX(Packet8d)
EIGEN_CLANG_PACKET_MINMAX(Packet16i)
EIGEN_CLANG_PACKET_MINMAX(Packet8l)
#undef EIGEN_CLANG_PACKET_MINMAX

// --- Math functions (float/double only) ---
#define EIGEN_CLANG_PACKET_MATH_FLOAT(PACKET_TYPE)                            \
  template <>                                                                 \
  EIGEN_STRONG_INLINE PACKET_TYPE pfloor<PACKET_TYPE>(const PACKET_TYPE& a) { \
    return __builtin_elementwise_floor(a);                                    \
  }                                                                           \
  template <>                                                                 \
  EIGEN_STRONG_INLINE PACKET_TYPE pceil<PACKET_TYPE>(const PACKET_TYPE& a) {  \
    return __builtin_elementwise_ceil(a);                                     \
  }                                                                           \
  template <>                                                                 \
  EIGEN_STRONG_INLINE PACKET_TYPE pround<PACKET_TYPE>(const PACKET_TYPE& a) { \
    return __builtin_elementwise_round(a);                                    \
  }                                                                           \
  template <>                                                                 \
  EIGEN_STRONG_INLINE PACKET_TYPE print<PACKET_TYPE>(const PACKET_TYPE& a) {  \
    return __builtin_elementwise_roundeven(a);                                \
  }                                                                           \
  template <>                                                                 \
  EIGEN_STRONG_INLINE PACKET_TYPE ptrunc<PACKET_TYPE>(const PACKET_TYPE& a) { \
    return __builtin_elementwise_trunc(a);                                    \
  }                                                                           \
  template <>                                                                 \
  EIGEN_STRONG_INLINE PACKET_TYPE psqrt<PACKET_TYPE>(const PACKET_TYPE& a) {  \
    return __builtin_elementwise_sqrt(a);                                     \
  }

EIGEN_CLANG_PACKET_MATH_FLOAT(Packet16f)
EIGEN_CLANG_PACKET_MATH_FLOAT(Packet8d)
#undef EIGEN_CLANG_PACKET_MATH_FLOAT

// --- Reductions ---
#define EIGEN_CLANG_PACKET_REDUX(PACKET_TYPE)                                                            \
  template <>                                                                                            \
  EIGEN_STRONG_INLINE unpacket_traits<PACKET_TYPE>::type predux_min<PACKET_TYPE>(const PACKET_TYPE& a) { \
    return __builtin_reduce_min(a);                                                                      \
  }                                                                                                      \
  template <>                                                                                            \
  EIGEN_STRONG_INLINE unpacket_traits<PACKET_TYPE>::type predux_max<PACKET_TYPE>(const PACKET_TYPE& a) { \
    return __builtin_reduce_max(a);                                                                      \
  }
EIGEN_CLANG_PACKET_REDUX(Packet16f)
EIGEN_CLANG_PACKET_REDUX(Packet8d)
EIGEN_CLANG_PACKET_REDUX(Packet16i)
EIGEN_CLANG_PACKET_REDUX(Packet8l)
#undef EIGEN_CLANG_PACKET_REDUX

#define EIGEN_CLANG_PACKET_REDUX_INT(PACKET_TYPE)                                                        \
  template <>                                                                                            \
  EIGEN_STRONG_INLINE unpacket_traits<PACKET_TYPE>::type predux<PACKET_TYPE>(const PACKET_TYPE& a) {     \
    return __builtin_reduce_add(a);                                                                      \
  }                                                                                                      \
  template <>                                                                                            \
  EIGEN_STRONG_INLINE unpacket_traits<PACKET_TYPE>::type predux_mul<PACKET_TYPE>(const PACKET_TYPE& a) { \
    return __builtin_reduce_mul(a);                                                                      \
  }

//     builtin_reduce_{mul,add} are only defined for integer types.
EIGEN_CLANG_PACKET_REDUX_INT(Packet16i)
EIGEN_CLANG_PACKET_REDUX_INT(Packet8l)
#undef EIGEN_CLANG_PACKET_REDUX_INT

template <typename VectorT>
EIGEN_STRONG_INLINE scalar_type_of_vector_t<VectorT> ReduceAdd16(const VectorT& a) {
  auto t1 = __builtin_shufflevector(a, a, 0, 2, 4, 6, 8, 10, 12, 14) +
            __builtin_shufflevector(a, a, 1, 3, 5, 7, 9, 11, 13, 15);
  auto t2 = __builtin_shufflevector(t1, t1, 0, 2, 4, 6) + __builtin_shufflevector(t1, t1, 1, 3, 5, 7);
  auto t3 = __builtin_shufflevector(t2, t2, 0, 2) + __builtin_shufflevector(t2, t2, 1, 3);
  return t3[0] + t3[1];
}

template <typename VectorT>
EIGEN_STRONG_INLINE scalar_type_of_vector_t<VectorT> ReduceAdd8(const VectorT& a) {
  auto t1 = __builtin_shufflevector(a, a, 0, 2, 4, 6) + __builtin_shufflevector(a, a, 1, 3, 5, 7);
  auto t2 = __builtin_shufflevector(t1, t1, 0, 2) + __builtin_shufflevector(t1, t1, 1, 3);
  return t2[0] + t2[1];
}

template <>
EIGEN_STRONG_INLINE float predux<Packet16f>(const Packet16f& a) {
  return ReduceAdd16(a);
}

template <>
EIGEN_STRONG_INLINE double predux<Packet8d>(const Packet8d& a) {
  return ReduceAdd8(a);
}

// template <typename VectorT>
// EIGEN_STRONG_INLINE HalfVectort<VectorT>
// ReduceHalf16(const VectorT& a) {
//   return __builtin_shufflevector(a, a, 0, 1, 2, 3, 4, 5, 6, 7) +
//          __builtin_shufflevector(a, a, 8, 9, 10, 11, 12, 13, 14, 15);
// }

// template <typename VectorT>
// EIGEN_STRONG_INLINE HalfVectort<VectorT>
// ReduceHalf8(const VectorT& a) {
//   return __builtin_shufflevector(a, a, 0, 1, 2, 3) +
//          __builtin_shufflevector(a, a, 4, 5, 6, 7);
// }

// template <>
// EIGEN_STRONG_INLINE HalfPacket<Packet16f> predux<Packet16f>(const Packet16f&
// a) {
//   return ReduceAdd16(a);
// }

// template <>
// EIGEN_STRONG_INLINE double predux<Packet8d>(const Packet8d& a) {
//   return ReduceAdd8(a);
// }

// template <>
// EIGEN_STRONG_INLINE float predux<Packet16f>(const Packet16f& a) {
//   return ReduceAdd16(a);
// }

// template <>
// EIGEN_STRONG_INLINE double predux<Packet8d>(const Packet8d& a) {
//   return ReduceAdd8(a);
// }

// --- Fused Multiply-Add (MADD) ---
#if __has_builtin(__builtin_elementwise_fma)
#define EIGEN_CLANG_PACKET_MADD(PACKET_TYPE)                                                      \
  template <>                                                                                     \
  EIGEN_STRONG_INLINE PACKET_TYPE pmadd<PACKET_TYPE>(const PACKET_TYPE& a, const PACKET_TYPE& b,  \
                                                     const PACKET_TYPE& c) {                      \
    return __builtin_elementwise_fma(a, b, c);                                                    \
  }                                                                                               \
  template <>                                                                                     \
  EIGEN_STRONG_INLINE PACKET_TYPE pmsub<PACKET_TYPE>(const PACKET_TYPE& a, const PACKET_TYPE& b,  \
                                                     const PACKET_TYPE& c) {                      \
    return __builtin_elementwise_fma(a, b, -c);                                                   \
  }                                                                                               \
  template <>                                                                                     \
  EIGEN_STRONG_INLINE PACKET_TYPE pnmadd<PACKET_TYPE>(const PACKET_TYPE& a, const PACKET_TYPE& b, \
                                                      const PACKET_TYPE& c) {                     \
    return __builtin_elementwise_fma(-a, b, c);                                                   \
  }                                                                                               \
  template <>                                                                                     \
  EIGEN_STRONG_INLINE PACKET_TYPE pnmsub<PACKET_TYPE>(const PACKET_TYPE& a, const PACKET_TYPE& b, \
                                                      const PACKET_TYPE& c) {                     \
    return -(__builtin_elementwise_fma(a, b, c));                                                 \
  }
#else
// Fallback if FMA builtin is not available
#define EIGEN_CLANG_PACKET_MADD(PACKET_TYPE)                                                     \
  template <>                                                                                    \
  EIGEN_STRONG_INLINE PACKET_TYPE pmadd<PACKET_TYPE>(const PACKET_TYPE& a, const PACKET_TYPE& b, \
                                                     const PACKET_TYPE& c) {                     \
    return (a * b) + c;                                                                          \
  }
#endif

EIGEN_CLANG_PACKET_MADD(Packet16f)
EIGEN_CLANG_PACKET_MADD(Packet8d)
#undef EIGEN_CLANG_PACKET_MADD

template <typename Packet>
EIGEN_STRONG_INLINE Packet preverse_impl_8(const Packet& a) {
  return __builtin_shufflevector(a, a, 7, 6, 5, 4, 3, 2, 1, 0);
}
template <typename Packet>
EIGEN_STRONG_INLINE Packet preverse_impl_16(const Packet& a) {
  return __builtin_shufflevector(a, a, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
}

#define EIGEN_CLANG_PACKET_REVERSE(PACKET_TYPE, SIZE)                           \
  template <>                                                                   \
  EIGEN_STRONG_INLINE PACKET_TYPE preverse<PACKET_TYPE>(const PACKET_TYPE& a) { \
    return preverse_impl_##SIZE(a);                                             \
  }

EIGEN_CLANG_PACKET_REVERSE(Packet16f, 16)
EIGEN_CLANG_PACKET_REVERSE(Packet8d, 8)
EIGEN_CLANG_PACKET_REVERSE(Packet16i, 16)
EIGEN_CLANG_PACKET_REVERSE(Packet8l, 8)
#undef EIGEN_CLANG_PACKET_REVERSE

#define EIGEN_CLANG_PACKET_SCATTER(PACKET_TYPE)                                                                      \
  template <>                                                                                                        \
  EIGEN_STRONG_INLINE void pscatter(unpacket_traits<PACKET_TYPE>::type* to, const PACKET_TYPE& from, Index stride) { \
    constexpr size_t size = unpacket_traits<PACKET_TYPE>::size;                                                      \
    for (int i = 0; i < size; ++i) {                                                                                 \
      to[i * stride] = from[i];                                                                                      \
    }                                                                                                                \
  }

EIGEN_CLANG_PACKET_SCATTER(Packet16f)
EIGEN_CLANG_PACKET_SCATTER(Packet8d)
EIGEN_CLANG_PACKET_SCATTER(Packet16i)
EIGEN_CLANG_PACKET_SCATTER(Packet8l)
#undef EIGEN_CLANG_PACKET_SCATTER

#define EIGEN_CLANG_PACKET_GATHER(PACKET_TYPE, SCALAR_TYPE)                                                  \
  template <>                                                                                                \
  EIGEN_STRONG_INLINE PACKET_TYPE pgather<SCALAR_TYPE, PACKET_TYPE>(const SCALAR_TYPE* from, Index stride) { \
    constexpr size_t size = unpacket_traits<PACKET_TYPE>::size;                                              \
    SCALAR_TYPE arr[size];                                                                                   \
    for (int i = 0; i < size; ++i) {                                                                         \
      arr[i] = from[i * stride];                                                                             \
    }                                                                                                        \
    return *(PACKET_TYPE*)arr;                                                                               \
  }

EIGEN_CLANG_PACKET_GATHER(Packet16f, float)
EIGEN_CLANG_PACKET_GATHER(Packet8d, double)
EIGEN_CLANG_PACKET_GATHER(Packet16i, int32_t)
EIGEN_CLANG_PACKET_GATHER(Packet8l, int64_t)
#undef EIGEN_CLANG_PACKET_GATHER

#define EIGEN_CLANG_PACKET_SELECT(PACKET_TYPE)                                                        \
  template <>                                                                                         \
  EIGEN_STRONG_INLINE PACKET_TYPE pselect<PACKET_TYPE>(const PACKET_TYPE& mask, const PACKET_TYPE& a, \
                                                       const PACKET_TYPE& b) {                        \
    return __builtin_elementwise_abs(mask) == 0 ? b : a;                                              \
  }

EIGEN_CLANG_PACKET_SELECT(Packet16f)
EIGEN_CLANG_PACKET_SELECT(Packet8d)
EIGEN_CLANG_PACKET_SELECT(Packet16i)
EIGEN_CLANG_PACKET_SELECT(Packet8l)
#undef EIGEN_CLANG_PACKET_SELECT

template <typename Packet>
EIGEN_STRONG_INLINE Packet ploaddup16(const typename unpacket_traits<Packet>::type* from) {
  static_assert((unpacket_traits<Packet>::size) % 2 == 0);
  using HalfPacket = HalfPacket<Packet>;
  HalfPacket a = load_vector_unaligned<HalfPacket>(from);
  return __builtin_shufflevector(a, a, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
}

template <typename Packet>
EIGEN_STRONG_INLINE Packet ploadquad16(const typename unpacket_traits<Packet>::type* from) {
  static_assert((unpacket_traits<Packet>::size) % 4 == 0);
  using QuarterPacket = QuarterPacket<Packet>;
  QuarterPacket a = load_vector_unaligned<QuarterPacket>(from);
  return __builtin_shufflevector(a, a, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);
}

template <typename Packet>
EIGEN_STRONG_INLINE Packet ploaddup8(const typename unpacket_traits<Packet>::type* from) {
  static_assert((unpacket_traits<Packet>::size) % 2 == 0);
  using HalfPacket = HalfPacket<Packet>;
  HalfPacket a = load_vector_unaligned<HalfPacket>(from);
  return __builtin_shufflevector(a, a, 0, 0, 1, 1, 2, 2, 3, 3);
}

template <typename Packet>
EIGEN_STRONG_INLINE Packet ploadquad8(const typename unpacket_traits<Packet>::type* from) {
  static_assert((unpacket_traits<Packet>::size) % 4 == 0);
  using QuarterPacket = QuarterPacket<Packet>;
  QuarterPacket a = load_vector_unaligned<QuarterPacket>(from);
  return __builtin_shufflevector(a, a, 0, 0, 0, 0, 1, 1, 1, 1);
}

template <>
EIGEN_STRONG_INLINE Packet16f ploaddup<Packet16f>(const float* from) {
  return ploaddup16<Packet16f>(from);
}
template <>
EIGEN_STRONG_INLINE Packet8d ploaddup<Packet8d>(const double* from) {
  return ploaddup8<Packet8d>(from);
}
template <>
EIGEN_STRONG_INLINE Packet16i ploaddup<Packet16i>(const int32_t* from) {
  return ploaddup16<Packet16i>(from);
}
template <>
EIGEN_STRONG_INLINE Packet8l ploaddup<Packet8l>(const int64_t* from) {
  return ploaddup8<Packet8l>(from);
}

template <>
EIGEN_STRONG_INLINE Packet16f ploadquad<Packet16f>(const float* from) {
  return ploadquad16<Packet16f>(from);
}
template <>
EIGEN_STRONG_INLINE Packet8d ploadquad<Packet8d>(const double* from) {
  return ploadquad8<Packet8d>(from);
}
template <>
EIGEN_STRONG_INLINE Packet16i ploadquad<Packet16i>(const int32_t* from) {
  return ploadquad16<Packet16i>(from);
}
template <>
EIGEN_STRONG_INLINE Packet8l ploadquad<Packet8l>(const int64_t* from) {
  return ploadquad8<Packet8l>(from);
}

template <>
EIGEN_STRONG_INLINE Packet16f plset<Packet16f>(const float& a) {
  Packet16f x{a + 0.0f, a + 1.0f, a + 2.0f,  a + 3.0f,  a + 4.0f,  a + 5.0f,  a + 6.0f,  a + 7.0f,
              a + 8.0f, a + 9.0f, a + 10.0f, a + 11.0f, a + 12.0f, a + 13.0f, a + 14.0f, a + 15.0f};
  return x;
}
template <>
EIGEN_STRONG_INLINE Packet8d plset<Packet8d>(const double& a) {
  return Packet8d{a + 0.0, a + 1.0, a + 2.0, a + 3.0, a + 4.0, a + 5.0, a + 6.0, a + 7.0};
}
template <>
EIGEN_STRONG_INLINE Packet16i plset<Packet16i>(const int32_t& a) {
  return Packet16i{a + 0, a + 1, a + 2,  a + 3,  a + 4,  a + 5,  a + 6,  a + 7,
                   a + 8, a + 9, a + 10, a + 11, a + 12, a + 13, a + 14, a + 15};
}
template <>
EIGEN_STRONG_INLINE Packet8l plset<Packet8l>(const int64_t& a) {
  return Packet8l{a + 0, a + 1, a + 2, a + 3, a + 4, a + 5, a + 6, a + 7};
}

// Helpers for ptranspose.
namespace detail {

template <typename Packet>
EIGEN_ALWAYS_INLINE void zip_in_place16(Packet& p1, Packet& p2) {
  Packet tmp = __builtin_shufflevector(p1, p2, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
  p2 = __builtin_shufflevector(p1, p2, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
  p1 = tmp;
}

template <typename Packet>
EIGEN_ALWAYS_INLINE void zip_in_place8(Packet& p1, Packet& p2) {
  Packet tmp = __builtin_shufflevector(p1, p2, 0, 8, 1, 9, 2, 10, 3, 11);
  p2 = __builtin_shufflevector(p1, p2, 4, 12, 5, 13, 6, 14, 7, 15);
  p1 = tmp;
}

template <typename Packet>
void zip_in_place(Packet& p1, Packet& p2);

template <>
EIGEN_ALWAYS_INLINE void zip_in_place<Packet16f>(Packet16f& p1, Packet16f& p2) {
  zip_in_place16(p1, p2);
}

template <>
EIGEN_ALWAYS_INLINE void zip_in_place<Packet8d>(Packet8d& p1, Packet8d& p2) {
  zip_in_place8(p1, p2);
}

template <>
EIGEN_ALWAYS_INLINE void zip_in_place<Packet16i>(Packet16i& p1, Packet16i& p2) {
  zip_in_place16(p1, p2);
}

template <>
EIGEN_ALWAYS_INLINE void zip_in_place<Packet8l>(Packet8l& p1, Packet8l& p2) {
  zip_in_place8(p1, p2);
}

template <typename Packet>
EIGEN_ALWAYS_INLINE void ptranspose_impl(PacketBlock<Packet, 2>& kernel) {
  zip_in_place(kernel.packet[0], kernel.packet[1]);
}

template <typename Packet>
EIGEN_ALWAYS_INLINE void ptranspose_impl(PacketBlock<Packet, 4>& kernel) {
  zip_in_place(kernel.packet[0], kernel.packet[2]);
  zip_in_place(kernel.packet[1], kernel.packet[3]);
  zip_in_place(kernel.packet[0], kernel.packet[1]);
  zip_in_place(kernel.packet[2], kernel.packet[3]);
}

template <typename Packet>
EIGEN_ALWAYS_INLINE void ptranspose_impl(PacketBlock<Packet, 8>& kernel) {
  zip_in_place(kernel.packet[0], kernel.packet[4]);
  zip_in_place(kernel.packet[1], kernel.packet[5]);
  zip_in_place(kernel.packet[2], kernel.packet[6]);
  zip_in_place(kernel.packet[3], kernel.packet[7]);

  zip_in_place(kernel.packet[0], kernel.packet[2]);
  zip_in_place(kernel.packet[1], kernel.packet[3]);
  zip_in_place(kernel.packet[4], kernel.packet[6]);
  zip_in_place(kernel.packet[5], kernel.packet[7]);

  zip_in_place(kernel.packet[0], kernel.packet[1]);
  zip_in_place(kernel.packet[2], kernel.packet[3]);
  zip_in_place(kernel.packet[4], kernel.packet[5]);
  zip_in_place(kernel.packet[6], kernel.packet[7]);
}

template <typename Packet>
EIGEN_ALWAYS_INLINE void ptranspose_impl(PacketBlock<Packet, 16>& kernel) {
  EIGEN_UNROLL_LOOP
  for (int i = 0; i < 4; ++i) {
    const int m = (1 << i);
    EIGEN_UNROLL_LOOP
    for (int j = 0; j < m; ++j) {
      const int n = (1 << (3 - i));
      EIGEN_UNROLL_LOOP
      for (int k = 0; k < n; ++k) {
        const int idx = 2 * j * n + k;
        zip_in_place(kernel.packet[idx], kernel.packet[idx + n]);
      }
    }
  }
}

}  // namespace detail

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet16f, 16>& kernel) {
  detail::ptranspose_impl(kernel);
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet16f, 8>& kernel) {
  detail::ptranspose_impl(kernel);
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet16f, 4>& kernel) {
  detail::ptranspose_impl(kernel);
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet16f, 2>& kernel) {
  detail::ptranspose_impl(kernel);
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet8d, 8>& kernel) {
  detail::ptranspose_impl(kernel);
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet8d, 4>& kernel) {
  detail::ptranspose_impl(kernel);
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet8d, 2>& kernel) {
  detail::ptranspose_impl(kernel);
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet16i, 16>& kernel) {
  detail::ptranspose_impl(kernel);
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet16i, 8>& kernel) {
  detail::ptranspose_impl(kernel);
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet16i, 4>& kernel) {
  detail::ptranspose_impl(kernel);
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet16i, 2>& kernel) {
  detail::ptranspose_impl(kernel);
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet8l, 8>& kernel) {
  detail::ptranspose_impl(kernel);
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet8l, 4>& kernel) {
  detail::ptranspose_impl(kernel);
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet8l, 2>& kernel) {
  detail::ptranspose_impl(kernel);
}

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_PACKET_MATH_CLANG_H
