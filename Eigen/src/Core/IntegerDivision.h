// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2025 Charlie Schlosser <cs.schlosser@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_INTDIV_H
#define EIGEN_INTDIV_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

namespace integer_division {

template <typename Scalar, bool Signed = std::is_signed<Scalar>::value>
struct unsigned_abs_impl {
  using UnsignedScalar = std::make_unsigned_t<Scalar>;
  constexpr static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE UnsignedScalar run(const Scalar& a) {
    UnsignedScalar result = static_cast<UnsignedScalar>(a);
    if (a < 0) result = 0u - result;
    return result;
  }
};
template <typename UnsignedScalar>
struct unsigned_abs_impl<UnsignedScalar, false> {
  constexpr static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE UnsignedScalar run(const UnsignedScalar& a) { return a; }
};

// simplifies the arithmetic gymnastics required to appease UBSAN
template <typename Scalar>
constexpr EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE std::make_unsigned_t<Scalar> unsigned_abs(const Scalar& a) {
  return unsigned_abs_impl<Scalar>::run(a);
}

template <typename T>
struct DoubleWordInteger {
  EIGEN_STATIC_ASSERT((std::is_integral<T>::value) && (std::is_unsigned<T>::value),
                      "SCALAR MUST BE A BUILT IN UNSIGNED INTEGER")
  static constexpr int k = CHAR_BIT * sizeof(T);

  constexpr EIGEN_DEVICE_FUNC DoubleWordInteger(T highBits, T lowBits) : hi(highBits), lo(lowBits) {}

  constexpr static EIGEN_DEVICE_FUNC DoubleWordInteger FromSum(T a, T b) {
    // convenient constructor that returns the full sum a + b
    T sum = a + b;
    return DoubleWordInteger(sum < a ? 1 : 0, sum);
  }
  constexpr static EIGEN_DEVICE_FUNC DoubleWordInteger FromProduct(T a, T b) {
    // convenient constructor that returns the full product a * b
    constexpr int kh = k / 2;
    constexpr T kLowMask = T(-1) >> kh;

    T a_h = a >> kh;
    T a_l = a & kLowMask;
    T b_h = b >> kh;
    T b_l = b & kLowMask;

    T ab_hh = a_h * b_h;
    T ab_hl = a_h * b_l;
    T ab_lh = a_l * b_h;
    T ab_ll = a_l * b_l;

    DoubleWordInteger<T> result(ab_hh, ab_ll);
    result += DoubleWordInteger<T>(ab_hl >> kh, ab_hl << kh);
    result += DoubleWordInteger<T>(ab_lh >> kh, ab_lh << kh);

    eigen_assert(result.lo == T(a * b));
    return result;
  }

  constexpr EIGEN_DEVICE_FUNC DoubleWordInteger& operator+=(const DoubleWordInteger& rhs) {
    hi += rhs.hi;
    lo += rhs.lo;
    if (lo < rhs.lo) hi++;
    return *this;
  }
  constexpr EIGEN_DEVICE_FUNC DoubleWordInteger& operator+=(const T& rhs) {
    lo += rhs;
    if (lo < rhs) hi++;
    return *this;
  }
  constexpr EIGEN_DEVICE_FUNC DoubleWordInteger& operator-=(const DoubleWordInteger& rhs) {
    if (lo < rhs.lo) hi--;
    hi -= rhs.hi;
    lo -= rhs.lo;
    return *this;
  }
  constexpr EIGEN_DEVICE_FUNC DoubleWordInteger& operator-=(const T& rhs) {
    if (lo < rhs) hi--;
    lo -= rhs;
    return *this;
  }
  constexpr EIGEN_DEVICE_FUNC DoubleWordInteger& operator>>=(int shift) {
    if (shift >= k) {
      lo = hi << (shift - k);
      hi = 0;
    } else {
      lo >>= shift;
      lo |= hi << (k - shift);
      hi >>= shift;
    }
    return *this;
  }
  constexpr EIGEN_DEVICE_FUNC DoubleWordInteger& operator<<=(int shift) {
    if (shift >= k) {
      hi = lo << (shift - k);
      lo = 0;
    } else {
      hi <<= shift;
      hi |= lo >> (k - shift);
      lo <<= shift;
    }
    return *this;
  }

  constexpr EIGEN_DEVICE_FUNC DoubleWordInteger operator+(const DoubleWordInteger& rhs) const {
    DoubleWordInteger result = *this;
    result += rhs;
    return result;
  }
  constexpr EIGEN_DEVICE_FUNC DoubleWordInteger operator+(const T& rhs) const {
    DoubleWordInteger result = *this;
    result += rhs;
    return result;
  }
  constexpr EIGEN_DEVICE_FUNC DoubleWordInteger operator-(const DoubleWordInteger& rhs) const {
    DoubleWordInteger result = *this;
    result -= rhs;
    return result;
  }
  constexpr EIGEN_DEVICE_FUNC DoubleWordInteger operator-(const T& rhs) const {
    DoubleWordInteger result = *this;
    result -= rhs;
    return result;
  }
  constexpr EIGEN_DEVICE_FUNC DoubleWordInteger operator>>(int shift) const {
    DoubleWordInteger result = *this;
    result >>= shift;
    return result;
  }
  constexpr EIGEN_DEVICE_FUNC DoubleWordInteger operator<<(int shift) const {
    DoubleWordInteger result = *this;
    result <<= shift;
    return result;
  }

  constexpr EIGEN_DEVICE_FUNC bool operator==(const DoubleWordInteger& rhs) const {
    return hi == rhs.hi && lo == rhs.lo;
  }
  constexpr EIGEN_DEVICE_FUNC bool operator<(const DoubleWordInteger& rhs) const {
    return hi != rhs.hi ? hi < rhs.hi : lo < rhs.lo;
  }
  constexpr EIGEN_DEVICE_FUNC bool operator!=(const DoubleWordInteger& rhs) const { return !(*this == rhs); }
  constexpr EIGEN_DEVICE_FUNC bool operator>(const DoubleWordInteger& rhs) const { return rhs < *this; }
  constexpr EIGEN_DEVICE_FUNC bool operator<=(const DoubleWordInteger& rhs) const { return !(*this > rhs); }
  constexpr EIGEN_DEVICE_FUNC bool operator>=(const DoubleWordInteger& rhs) const { return !(*this < rhs); }

  T hi, lo;
};

template <typename T>
constexpr EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T calc_magic_generic(T d, int p) {
  EIGEN_STATIC_ASSERT((std::is_integral<T>::value) && (std::is_unsigned<T>::value),
                      "SCALAR MUST BE A BUILT IN UNSIGNED INTEGER")
  constexpr int k = CHAR_BIT * sizeof(T);

  // calculates ceil(2^(k+p) / d) mod 2^k
  // where p = log2_ceil(d)

  eigen_assert(d != 0 && "Error: Division by zero attempted!");

  // the logic below assumes that d > 1 and p > 0
  // if d == 1, then the magic number is 2^k mod 2^k == 0
  if (d == 1) return 0;

  // magic = 1 + floor(n / d) mod 2^k
  // n = 2^(k+p)-1, which is at least k+1 bits and at most 2k bits
  // p = log2_ceil(d), d <= 2^p
  // 2^k+1 > q >= 2^k
  // subtract 2^k * d, 2^k-1 * d ... and so forth until the high bits of q are depleted

  constexpr T nLowBits = T(-1);
  T nHighBits = nLowBits >> (k - p);

  DoubleWordInteger<T> n(nHighBits, nLowBits);
  DoubleWordInteger<T> q_inc(1, 0);   // the incremental amount to add to q
  DoubleWordInteger<T> qd_inc(d, 0);  // the incremental amount to subtract from n
  DoubleWordInteger<T> q(0, 0);       // the total number of times q is subtracted from n

  // in the worst-case scenario, this loop runs k+1 times
  while (n.hi) {
    if (n >= qd_inc) {
      q += q_inc;
      n -= qd_inc;
    }
    q_inc >>= 1;
    qd_inc >>= 1;
  }
  q += n.lo / d;
  return q.lo + 1;
}
constexpr EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint8_t calc_magic(uint8_t d, int p) {
  uint16_t n = uint16_t(-1) >> (8 - p);
  uint16_t q = 1 + (n / d);
  uint8_t result = static_cast<uint8_t>(q);
  return result;
}
constexpr EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint16_t calc_magic(uint16_t d, int p) {
  uint32_t n = uint32_t(-1) >> (16 - p);
  uint32_t q = 1 + (n / d);
  uint16_t result = static_cast<uint16_t>(q);
  return result;
}
constexpr EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint32_t calc_magic(uint32_t d, int p) {
  uint64_t n = uint64_t(-1) >> (32 - p);
  uint64_t q = 1 + (n / d);
  uint32_t result = static_cast<uint32_t>(q);
  return result;
}
constexpr EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint64_t calc_magic(uint64_t d, int p) {
  return calc_magic_generic(d, p);
}

template <typename T>
constexpr EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T umuluh_generic(T a, T b) {
  return DoubleWordInteger<T>::FromProduct(a, b).hi;
}
constexpr EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint8_t umuluh(uint8_t a, uint8_t b) {
  uint16_t result = (uint16_t(a) * uint16_t(b)) >> 8;
  return static_cast<uint8_t>(result);
}
constexpr EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint16_t umuluh(uint16_t a, uint16_t b) {
  uint32_t result = (uint32_t(a) * uint32_t(b)) >> 16;
  return static_cast<uint16_t>(result);
}
constexpr EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint32_t umuluh(uint32_t a, uint32_t b) {
  uint64_t result = (uint64_t(a) * uint64_t(b)) >> 32;
  return static_cast<uint32_t>(result);
}
// faster but non-constexpr variations of umuluh(uint64_t, uint64_t)
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint64_t umuluh_u64_impl(uint64_t a, uint64_t b) {
#if EIGEN_HAS_BUILTIN_INT128
  __uint128_t v = static_cast<__uint128_t>(a) * static_cast<__uint128_t>(b);
  return static_cast<uint64_t>(v >> 64);
#elif defined(EIGEN_GPU_COMPILE_PHASE)
  return __umul64hi(a, b);
#elif defined(SYCL_DEVICE_ONLY)
  return cl::sycl::mul_hi(a, b);
#elif EIGEN_COMP_MSVC && (EIGEN_ARCH_x86_64 || EIGEN_ARCH_ARM64)
  return __umulh(a, b);
#else
  return umuluh_generic(a, b);
#endif
}
#if EIGEN_HAS_BUILTIN_INT128
constexpr EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint64_t umuluh(uint64_t a, uint64_t b) {
  __uint128_t v = static_cast<__uint128_t>(a) * static_cast<__uint128_t>(b);
  return static_cast<uint64_t>(v >> 64);
}
#elif defined(__cpp_lib_is_constant_evaluated) && (__cpp_lib_is_constant_evaluated >= 201811L)
constexpr EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint64_t umuluh(uint64_t a, uint64_t b) {
  if (std::is_constant_evaluated()) {
    return umuluh_generic(a, b);
  } else {
    return umuluh_u64_impl(a, b);
  }
}
#else
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint64_t umuluh(uint64_t a, uint64_t b) { return umuluh_u64_impl(a, b); }
#endif

template <typename T>
constexpr EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T uintdiv_generic(T a, T magic, int shift) {
  T b = umuluh(a, magic);
  DoubleWordInteger<T> t = DoubleWordInteger<T>::FromSum(b, a) >> shift;
  return t.lo;
}
constexpr EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint8_t uintdiv(uint8_t a, uint8_t magic, int shift) {
  uint16_t b = umuluh(a, magic);
  uint16_t t = (b + a) >> shift;
  return static_cast<uint8_t>(t);
}
constexpr EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint16_t uintdiv(uint16_t a, uint16_t magic, int shift) {
  uint32_t b = umuluh(a, magic);
  uint32_t t = (b + a) >> shift;
  return static_cast<uint16_t>(t);
}
constexpr EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint32_t uintdiv(uint32_t a, uint32_t magic, int shift) {
  uint64_t b = umuluh(a, magic);
  uint64_t t = (b + a) >> shift;
  return static_cast<uint32_t>(t);
}
constexpr EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint64_t uintdiv(uint64_t a, uint64_t magic, int shift) {
#if EIGEN_HAS_BUILTIN_INT128
  __uint128_t b = umuluh(a, magic);
  __uint128_t t = (b + a) >> shift;
  return static_cast<uint64_t>(t);
#else
  return uintdiv_generic(a, magic, shift);
#endif
}

template <typename Scalar, bool Signed = std::is_signed<Scalar>::value>
struct fast_div_op_impl;

template <typename Scalar>
struct fast_div_op_impl<Scalar, false> {
  static constexpr int k = CHAR_BIT * sizeof(Scalar);
  using UnsignedScalar = std::make_unsigned_t<Scalar>;

  template <typename Divisor>
  constexpr EIGEN_DEVICE_FUNC fast_div_op_impl(Divisor d) {
    using UnsignedDivisor = std::make_unsigned_t<Divisor>;
    UnsignedDivisor d_abs = unsigned_abs(d);
    int tz = ctz(d_abs);
    UnsignedDivisor d_odd = d_abs >> tz;
    int p = log2_ceil(d_odd);
    // Intuitively, we want to check if d >= std::numeric_limits<Scalar>::min() and d <=
    // std::numeric_limits<Scalar>::max() However, this excludes edge cases such as int8_t(-128) / uint8_t(128) where
    // 128 is actually outside the range of int8_t. Instead, check if the shift exceeds the bit width of the scalar
    if (tz + p <= k) {
      // d is in range and produces a non-trivial quotient
      magic = calc_magic(static_cast<UnsignedScalar>(d_odd), p);
      shift = tz + p;
    } else {
      // d is out of range and always produces 0
      magic = 0;
      shift = k;
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE UnsignedScalar operator()(const UnsignedScalar& a) const {
    return uintdiv(a, magic, shift);
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& a) const {
    return puintdiv(a, magic, shift);
  }

  UnsignedScalar magic;
  int shift;
};

template <typename Scalar>
struct fast_div_op_impl<Scalar, true> : fast_div_op_impl<Scalar, false> {
  using UnsignedScalar = std::make_unsigned_t<Scalar>;
  using Base = fast_div_op_impl<Scalar, false>;

  // There are two approaches to handling signed integers:
  // 1. Calculate the quotient as abs(n) / abs(d) and flip the sign accordingly.
  // 2. Calculate a signed magic number.
  // Calculating the signed magic number is relatively simple. Substitute the following:
  //   a. int p = log2_floor(d_odd_abs)
  //   b. if p == 0, magic = 1
  //   c. otherwise, use the unsigned implementation
  //   Subsequent calculations are handled in an analogous manner (arithmetic right shifts, etc). If the input is
  //   negative, add one. If the divisor is negative, flip the sign. This approach avoids taking the absolute value of
  //   the input and accounting for the sign.
  // Despite these advantages, benchmarking shows 1. is significantly faster, at least in the vectorized
  // case. Since the vectorized case is still significantly faster than the scalar case, sticking to option 1 is the
  // approach that optimizes most use cases, and requires less work to develop and maintain.

  template <typename Divisor>
  constexpr EIGEN_DEVICE_FUNC fast_div_op_impl(Divisor d) : Base(d), sign(d < 0) {}

  constexpr EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar operator()(const Scalar& a) const {
    bool negative = (a < 0) != sign;
    UnsignedScalar abs_a = unsigned_abs(a);
    UnsignedScalar abs_result = uintdiv(abs_a, Base::magic, Base::shift);
    Scalar result = static_cast<Scalar>(negative ? 0u - abs_result : abs_result);
    return result;
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& a) const {
    return pintdiv(a, Base::magic, Base::shift, sign);
  }

  bool sign;
};

}  // namespace integer_division

template <typename Scalar>
struct fast_div_op : integer_division::fast_div_op_impl<Scalar> {
  EIGEN_STATIC_ASSERT((std::is_integral<Scalar>::value), "THE SCALAR MUST BE A BUILT IN INTEGER")
  using result_type = Scalar;
  using Base = integer_division::fast_div_op_impl<Scalar>;
  template <typename Divisor>
  constexpr EIGEN_DEVICE_FUNC fast_div_op(Divisor d) : Base(d) {
    EIGEN_STATIC_ASSERT((std::is_integral<Divisor>::value), "THE DIVISOR MUST BE A BUILT IN INTEGER")
    eigen_assert(d != 0 && "Error: unable to divide by zero!");
    eigen_assert(((std::is_signed<Scalar>::value) || (d > 0)) &&
                 "Error: unable to divide an unsigned integer by a negative divisor!");
  }
};

template <typename Scalar>
struct functor_traits<fast_div_op<Scalar>> {
  enum {
    PacketAccess = packet_traits<Scalar>::HasFastIntDiv,
    Cost = functor_traits<scalar_product_op<Scalar>>::Cost + 2 * functor_traits<scalar_sum_op<Scalar>>::Cost
  };
};

template <typename Packet>
EIGEN_STRONG_INLINE Packet pintdiv(const Packet& a, std::make_unsigned_t<typename unpacket_traits<Packet>::type> magic,
                                   int shift, bool sign) {
  using UnsignedScalar = std::make_unsigned_t<typename unpacket_traits<Packet>::type>;
  EIGEN_STATIC_ASSERT((find_packet_by_size<UnsignedScalar, unpacket_traits<Packet>::size>::value),
                      NO COMPATIBLE UNSIGNED PACKET)
  using UnsignedPacket = typename find_packet_by_size<UnsignedScalar, unpacket_traits<Packet>::size>::type;

  const Packet cst_divisor_sign = pset1<Packet>(sign ? -1 : 0);
  Packet sign_a = psignbit(a);
  Packet abs_result = (Packet)puintdiv((UnsignedPacket)pabs(a), magic, shift);
  Packet sign_mask = pxor(sign_a, cst_divisor_sign);
  Packet result = psub(pxor(abs_result, sign_mask), sign_mask);
  return result;
}

}  // namespace internal

template <typename Scalar>
struct IntDivider {
  using FastDivOp = internal::fast_div_op<Scalar>;
  template <typename Divisor>
  constexpr explicit EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE IntDivider(Divisor d) : op(d) {}
  constexpr EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar divide(const Scalar& numerator) const { return op(numerator); }
  template <typename Derived>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CwiseUnaryOp<FastDivOp, const Derived> divide(
      const DenseBase<Derived>& xpr) const {
    return CwiseUnaryOp<FastDivOp, const Derived>(xpr.derived(), op);
  }
  FastDivOp op;
};

template <typename Scalar>
constexpr EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar operator/(const Scalar& lhs, const IntDivider<Scalar>& rhs) {
  return rhs.divide(lhs);
}

}  // namespace Eigen

#endif  // EIGEN_INTDIV_H
