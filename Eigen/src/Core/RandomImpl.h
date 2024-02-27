// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_RANDOM_IMPL_H
#define EIGEN_RANDOM_IMPL_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

/****************************************************************************
 * Implementation of random                                               *
 ****************************************************************************/

template <typename Scalar, bool IsComplex, bool IsInteger>
struct random_default_impl {};

template <typename Scalar>
struct random_impl : random_default_impl<Scalar, NumTraits<Scalar>::IsComplex, NumTraits<Scalar>::IsInteger> {};

template <typename Scalar>
struct random_retval {
  typedef Scalar type;
};

template <typename Scalar>
inline EIGEN_MATHFUNC_RETVAL(random, Scalar) random(const Scalar& x, const Scalar& y) {
  return EIGEN_MATHFUNC_IMPL(random, Scalar)::run(x, y);
}

template <typename Scalar>
inline EIGEN_MATHFUNC_RETVAL(random, Scalar) random() {
  return EIGEN_MATHFUNC_IMPL(random, Scalar)::run();
}

template <typename BitsType>
EIGEN_DEVICE_FUNC inline int generic_log_radix_floor(const BitsType& x) {
  if (x == 0) return 0;
  const int digits = NumTraits<BitsType>::digits();
  BitsType test = 1;
  for (int s = 0; s < digits; s++) {
    if (test > x) return s - 1;
    test = test << 1;
  }
  return digits;
}
template <typename BitsType>
EIGEN_DEVICE_FUNC inline int generic_log_radix_ceil(const BitsType& x) {
  const int digits = NumTraits<BitsType>::digits();
  BitsType test = 1;
  for (int s = 0; s < digits; s++) {
    if (test >= x) return s;
    test = test << 1;
  }
  return digits;
}

template <typename BitsType, bool BuiltIn = std::is_integral<BitsType>::value>
struct log_radix_impl {
  static constexpr int kTotalBits = sizeof(BitsType) * CHAR_BIT;
  static EIGEN_DEVICE_FUNC inline int run_ceil(const BitsType& x) {
    const int n = kTotalBits - clz(x);
    bool power_of_two = (x & (x - 1)) == 0;
    return x == 0 ? 0 : power_of_two ? (n - 1) : n;
  }
  static EIGEN_DEVICE_FUNC inline int run_floor(const BitsType& x) {
    const int n = kTotalBits - clz(x);
    return x == 0 ? 0 : n - 1;
  }
};
template <typename BitsType>
struct log_radix_impl<BitsType, false> {
  static EIGEN_DEVICE_FUNC inline int run_floor(const BitsType& x) { return generic_log_radix_floor(x); }
  static EIGEN_DEVICE_FUNC inline int run_ceil(const BitsType& x) { return generic_log_radix_ceil(x); }
};

template <typename BitsType>
int log2_ceil(BitsType x) {
  return log_radix_impl<BitsType>::run_ceil(x);
}

template <typename BitsType>
int log2_floor(BitsType x) {
  return log_radix_impl<BitsType>::run_floor(x);
}

// TODO: replace or provide alternatives to this, e.g. std::random_device
struct eigen_random_device {
  static constexpr int Entropy = meta_floor_log2<(unsigned int)(RAND_MAX) + 1>::value;
  using ReturnType = int;
  static EIGEN_DEVICE_FUNC inline ReturnType run() { return std::rand(); };
};

// implementation to to fill a scalar with numRandomBits beginning from the least significant bit
// this version is intended to operate on unsigned built-in integer types, e.g. uint32_t
template <typename BitsType, bool BuiltIn = std::is_integral<BitsType>::value>
struct random_bits_impl {
  using RandomDevice = eigen_random_device;
  using RandomReturnType = typename RandomDevice::ReturnType;
  static constexpr int kEntropy = RandomDevice::Entropy;
  static constexpr int kTotalBits = sizeof(BitsType) * CHAR_BIT;
  // return a BitsType filled with numRandomBits beginning from the least significant bit
  static EIGEN_DEVICE_FUNC inline BitsType run(int numRandomBits) {
    eigen_assert((numRandomBits >= 0) && (numRandomBits <= kTotalBits));
    const BitsType mask = BitsType(-1) >> ((kTotalBits - numRandomBits) & (kTotalBits - 1));
    BitsType randomBits = BitsType(0);
    for (int shift = 0; shift < numRandomBits; shift += kEntropy) {
      RandomReturnType r = RandomDevice::run();
      randomBits |= static_cast<BitsType>(r) << shift;
    }
    // clear the excess bits
    randomBits &= mask;
    return randomBits;
  }
};

// specialization for non-built-in integer types
// mostly the same as above, but does not assume that the number of digits is known at compile time, nor does it assume
// that the representation is based on two's complement
template <typename BitsType>
struct random_bits_impl<BitsType, false> {
  using RandomDevice = eigen_random_device;
  using RandomReturnType = typename RandomDevice::ReturnType;
  static constexpr int kEntropy = RandomDevice::Entropy;
  // return a BitsType filled with numRandomBits beginning from the least significant bit
  static EIGEN_DEVICE_FUNC inline BitsType run(int numRandomBits) {
    // note: two's complement signed integers do not include the sign bit in 'digits'
    eigen_assert((numRandomBits >= 0) && (numRandomBits <= NumTraits<BitsType>::digits()));
    BitsType randomBits = BitsType(0);
    int shift = 0;
    for (; shift + kEntropy <= numRandomBits; shift += kEntropy) {
      RandomReturnType r = RandomDevice::run();
      randomBits = randomBits | (static_cast<BitsType>(r) << shift);
    }
    // defer to the built-in implementation to mask out the excess bits
    if (shift < numRandomBits) {
      RandomReturnType r = random_bits_impl<RandomReturnType, true>::run(numRandomBits - shift);
      randomBits = randomBits | (static_cast<BitsType>(r) << shift);
    }
    return randomBits;
  }
};

template <typename BitsType>
EIGEN_DEVICE_FUNC inline BitsType getRandomBits(int numRandomBits) {
  return random_bits_impl<BitsType>::run(numRandomBits);
}

// random implementation for a built-in floating point type
template <typename Scalar, bool BuiltIn = std::is_floating_point<Scalar>::value>
struct random_float_impl {
  using BitsType = typename numext::get_integer_by_size<sizeof(Scalar)>::unsigned_type;
  static EIGEN_DEVICE_FUNC inline Scalar run(int numRandomBits) {
    const int mantissaBits = NumTraits<Scalar>::digits() - 1;
    eigen_assert(numRandomBits >= 0 && numRandomBits <= mantissaBits);
    BitsType randomBits = getRandomBits<Scalar>(numRandomBits);
    // if fewer than MantissaBits is requested, shift them to the left
    randomBits <<= (mantissaBits - numRandomBits);
    // randomBits is in the half-open interval [2,4)
    randomBits |= numext::bit_cast<BitsType>(Scalar(2));
    // result is in the half-open interval [-1,1)
    Scalar result = numext::bit_cast<Scalar>(randomBits) - Scalar(3);
    return result;
  }
  static EIGEN_DEVICE_FUNC inline Scalar run() {
    const int mantissa_bits = NumTraits<Scalar>::digits() - 1;
    return run(mantissa_bits);
  }
};
// random implementation for a custom floating point type
template <typename Scalar>
struct random_float_impl<Scalar, false> {
  static EIGEN_DEVICE_FUNC inline Scalar run(int numRandomBits) {
    EIGEN_USING_STD(min);
    const int mantissaBits = (min)(NumTraits<Scalar>::digits(), NumTraits<double>::digits()) - 1;
    EIGEN_ONLY_USED_FOR_DEBUG(mantissaBits);
    eigen_assert(numRandomBits >= 0 && numRandomBits <= mantissaBits);
    Scalar result = static_cast<Scalar>(random_float_impl<double>::run(numRandomBits));
    return result;
  }
  static EIGEN_DEVICE_FUNC inline Scalar run() {
    EIGEN_USING_STD(min);
    const int mantissaBits = (min)(NumTraits<Scalar>::digits(), NumTraits<double>::digits()) - 1;
    return run(mantissaBits);
  }
};

// random implementation where Scalar is long double
// TODO: fix this for PPC
template <bool Specialize = sizeof(long double) == 2 * sizeof(uint64_t) && !EIGEN_ARCH_PPC>
struct random_longdouble_impl {
  static constexpr int Size = sizeof(long double), MantissaBits = NumTraits<long double>::digits() - 1;
  static EIGEN_DEVICE_FUNC inline long double run(int numRandomBits = MantissaBits) {
    eigen_assert(numRandomBits >= 0 && numRandomBits <= MantissaBits);
    const int numLowBits = (std::min)(numRandomBits, 64);
    const int numHighBits = (std::max)(numRandomBits - 64, 0);
    EIGEN_USING_STD(memcpy)
    uint64_t randomBits[2];
    long double result = 2.0L;
    memcpy(&randomBits, &result, Size);
    randomBits[0] |= random_float_impl<uint64_t>::run(numLowBits);
    randomBits[1] |= random_float_impl<uint64_t>::run(numHighBits);
    memcpy(&result, &randomBits, Size);
    result -= 3.0L;
    return result;
  }
};
template <>
struct random_longdouble_impl<false> {
  static constexpr int MantissaBits = NumTraits<double>::digits() - 1;
  static EIGEN_DEVICE_FUNC inline long double run(int numRandomBits = MantissaBits) {
    return static_cast<long double>(random_float_impl<double>::run(numRandomBits));
  }
};
template <>
struct random_float_impl<long double> : random_longdouble_impl<> {};

template <typename Scalar>
struct random_default_impl<Scalar, false, false> {
  static EIGEN_DEVICE_FUNC inline Scalar run(const Scalar& x, const Scalar& y, int numRandomBits) {
    Scalar half_x = Scalar(0.5) * x;
    Scalar half_y = Scalar(0.5) * y;
    Scalar result = (half_x + half_y) + (half_y - half_x) * run(numRandomBits);
    // result is in the half-open interval [x, y) -- provided that x < y
    return result;
  }
  static EIGEN_DEVICE_FUNC inline Scalar run(const Scalar& x, const Scalar& y) {
    const int mantissaBits = NumTraits<Scalar>::digits() - 1;
    return run(x, y, mantissaBits);
  }
  static EIGEN_DEVICE_FUNC inline Scalar run(int numRandomBits) {
    return random_float_impl<Scalar>::run(numRandomBits);
  }
  static EIGEN_DEVICE_FUNC inline Scalar run() { return random_float_impl<Scalar>::run(); }
};

template <typename Scalar, bool IsSigned = NumTraits<Scalar>::IsSigned, bool BuiltIn = std::is_integral<Scalar>::value>
struct random_int_impl;

// random implementation where Scalar is an unsigned integer type, or Scalar is non-negative at runtime
template <typename Scalar, bool BuiltIn>
struct random_int_impl<Scalar, false, BuiltIn> {
  static EIGEN_DEVICE_FUNC inline Scalar run(const Scalar& x, const Scalar& y) {
    if (y <= x) return x;
    const Scalar count = y - x + 1;
    // handle edge case where [x,y] spans the entire range of Scalar
    if (count == 0) return run();
    // calculate the number of random bits needed to fill range
    const int numRandomBits = log2_ceil(count);
    Scalar randomBits;
    do {
      randomBits = getRandomBits<Scalar>(numRandomBits);
      // if the random draw is outside [0, range), try again (rejection sampling)
      // in the worst-case scenario, the probability of rejection is: 1/2 - 1/2^numRandomBits < 50%
    } while (randomBits >= count);
    Scalar result = x + randomBits;
    return result;
  }
  static EIGEN_DEVICE_FUNC inline Scalar run() {
#ifdef EIGEN_MAKING_DOCS
    return run(0, 10);
#else
    return getRandomBits<Scalar>(NumTraits<Scalar>::digits());
#endif
  }
};

// random implementation where Scalar is a built-in signed integer type
template <typename Scalar>
struct random_int_impl<Scalar, true, true> {
  static constexpr int kTotalBits = sizeof(Scalar) * CHAR_BIT;
  using BitsType = typename numext::get_integer_by_size<sizeof(Scalar)>::unsigned_type;
  using RandomUnsignedImpl = random_int_impl<BitsType, false, true>;
  static EIGEN_DEVICE_FUNC inline Scalar run(const Scalar& x, const Scalar& y) {
    if (y <= x) return x;
    // Avoid overflow by representing `range` as an unsigned type
    BitsType range = static_cast<BitsType>(y) - static_cast<BitsType>(x);
    BitsType randomBits = RandomUnsignedImpl::run(BitsType(0), range);
    // Avoid overflow in the case where `x` is negative and there is a large range so
    // `randomBits` would also be negative if cast to `Scalar` first.
    Scalar result = static_cast<Scalar>(static_cast<BitsType>(x) + randomBits);
  }
  static EIGEN_DEVICE_FUNC inline Scalar run() {
#ifdef EIGEN_MAKING_DOCS
    return run(-10, 10);
#else
    return getRandomBits<Scalar>(kTotalBits);
#endif
  }
};

// random implementation where Scalar is a custom signed integer type
template <typename Scalar>
struct random_int_impl<Scalar, true, false> {
  using RandomUnsignedImpl = random_int_impl<Scalar, false, false>;
  static EIGEN_DEVICE_FUNC inline Scalar run(const Scalar& x, const Scalar& y) {
    if (y <= x) return x;
    Scalar range = y - x;
    bool overflow = (range + x) != y;
    // if `range` overflows, generate a random Scalar in the interval [0, MAX]
    // otherwise, generate a random non-negative Scalar in the interval [0, range]
    Scalar randomBits =
        overflow ? getRandomBits<Scalar>(NumTraits<Scalar>::digits()) : RandomUnsignedImpl::run(0, range);
    Scalar result = x + randomBits;
    return result;
  }
  static EIGEN_DEVICE_FUNC inline Scalar run() {
    return run(NumTraits<Scalar>::lowest(), NumTraits<Scalar>::highest());
  }
};

template <typename Scalar>
struct random_default_impl<Scalar, false, true> : random_int_impl<Scalar> {};

template <>
struct random_impl<bool> {
  static EIGEN_DEVICE_FUNC inline bool run(const bool& x, const bool& y) {
    if (y <= x) return x;
    return run();
  }
  static EIGEN_DEVICE_FUNC inline bool run() { return random_float_impl<int>::run(1) ? true : false; }
};

template <typename Scalar>
struct random_default_impl<Scalar, true, false> {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  using Impl = random_impl<RealScalar>;
  static EIGEN_DEVICE_FUNC inline Scalar run(const Scalar& x, const Scalar& y) {
    return Scalar(Impl::run(x.real(), y.real()), Impl::run(x.imag(), y.imag()));
  }
  static EIGEN_DEVICE_FUNC inline Scalar run() { return Scalar(Impl::run(), Impl::run()); }
};

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_RANDOM_IMPL_H
