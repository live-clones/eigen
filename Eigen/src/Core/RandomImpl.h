// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2024 Charles Schlosser <cs.schlosser@gmail.com>
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
  if (x == BitsType(0)) return 0;
  const int digits = NumTraits<BitsType>::digits();
  BitsType test = BitsType(1);
  for (int s = 0; s < digits; s++) {
    if (test > x) return s - 1;
    test = test << 1;
  }
  return digits;
}
template <typename BitsType>
EIGEN_DEVICE_FUNC inline int generic_log_radix_ceil(const BitsType& x) {
  const int digits = NumTraits<BitsType>::digits();
  BitsType test = BitsType(1);
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
int log2_ceil(const BitsType& x) {
  return log_radix_impl<BitsType>::run_ceil(x);
}

template <typename BitsType>
int log2_floor(const BitsType& x) {
  return log_radix_impl<BitsType>::run_floor(x);
}

// TODO: replace or provide alternatives to this, e.g. std::random_device
struct eigen_random_device {
  static constexpr int Entropy = meta_floor_log2<(unsigned int)(RAND_MAX) + 1>::value;
  using ReturnType = int;
  static EIGEN_DEVICE_FUNC inline ReturnType run() { return std::rand(); };
};

// Fill a built-in integer with numRandomBits beginning with the least significant bit
template <typename Scalar, bool BuiltIn = std::is_integral<Scalar>::value>
struct random_bits_impl {
  using BitsType = typename numext::get_integer_by_size<sizeof(Scalar)>::unsigned_type;
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
    return static_cast<Scalar>(randomBits);
  }
};

// Fill a custom integer with numRandomBits beginning with the least significant bit
// mostly the same as above, but does not assume that the number of digits is known at compile time, nor does it assume
// that the representation is two's complement
template <typename BitsType>
struct random_bits_impl<BitsType, false> {
  using RandomDevice = eigen_random_device;
  using RandomReturnType = typename RandomDevice::ReturnType;
  using RandomImpl = random_bits_impl<RandomReturnType, true>;
  static constexpr int kEntropy = RandomDevice::Entropy;
  // return a BitsType filled with numRandomBits beginning from the least significant bit
  static EIGEN_DEVICE_FUNC inline BitsType run(int numRandomBits) {
    // note: 'digits' does not include the sign bit
    eigen_assert((numRandomBits >= 0) && (numRandomBits <= NumTraits<BitsType>::digits()));
    BitsType randomBits = BitsType(0);
    int shift = 0;
    for (; shift + kEntropy <= numRandomBits; shift += kEntropy) {
      RandomReturnType r = RandomDevice::run();
      randomBits = randomBits | (static_cast<BitsType>(r) << shift);
    }
    // defer to the built-in implementation to mask out the excess bits
    if (shift < numRandomBits) {
      RandomReturnType r = RandomImpl::run(numRandomBits - shift);
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
  static constexpr EIGEN_DEVICE_FUNC inline int mantissaBits() {
    const int digits = NumTraits<Scalar>::digits();
    return digits - 1;
  }
  static EIGEN_DEVICE_FUNC inline Scalar run(int numRandomBits) {
    eigen_assert(numRandomBits >= 0 && numRandomBits <= mantissaBits());
    BitsType randomBits = getRandomBits<BitsType>(numRandomBits);
    // if fewer than MantissaBits is requested, shift them to the left
    randomBits <<= (mantissaBits() - numRandomBits);
    // randomBits is in the half-open interval [2,4)
    randomBits |= numext::bit_cast<BitsType>(Scalar(2));
    // result is in the half-open interval [-1,1)
    Scalar result = numext::bit_cast<Scalar>(randomBits) - Scalar(3);
    return result;
  }
};
// random implementation for a custom floating point type
// uses double as the implementation with a mantissa with a size equal to either the target scalar's mantissa or that of
// double, whichever is smaller
template <typename Scalar>
struct random_float_impl<Scalar, false> {
  static EIGEN_DEVICE_FUNC inline int mantissaBits() {
    const int digits = NumTraits<Scalar>::digits();
    constexpr int kDoubleDigits = NumTraits<double>::digits();
    EIGEN_USING_STD(min);
    return (min)(digits, kDoubleDigits) - 1;
  }
  static EIGEN_DEVICE_FUNC inline Scalar run(int numRandomBits) {
    eigen_assert(numRandomBits >= 0 && numRandomBits <= mantissaBits());
    Scalar result = static_cast<Scalar>(random_float_impl<double>::run(numRandomBits));
    return result;
  }
};

// random implementation for long double
// TODO: fix this for PPC
template <bool Specialize = sizeof(long double) == 2 * sizeof(uint64_t) && !EIGEN_ARCH_PPC>
struct random_longdouble_impl {
  static constexpr int Size = sizeof(long double);
  static constexpr EIGEN_DEVICE_FUNC inline int mantissaBits() { return NumTraits<long double>::digits() - 1; }
  static EIGEN_DEVICE_FUNC inline long double run(int numRandomBits) {
    eigen_assert(numRandomBits >= 0 && numRandomBits <= mantissaBits());
    EIGEN_USING_STD(min);
    EIGEN_USING_STD(max);
    EIGEN_USING_STD(memcpy);
    int numLowBits = (min)(numRandomBits, 64);
    int numHighBits = (max)(numRandomBits - 64, 0);
    uint64_t randomBits[2];
    long double result = 2.0L;
    memcpy(&randomBits, &result, Size);
    randomBits[0] |= getRandomBits<uint64_t>(numLowBits);
    randomBits[1] |= getRandomBits<uint64_t>(numHighBits);
    memcpy(&result, &randomBits, Size);
    result -= 3.0L;
    return result;
  }
};
template <>
struct random_longdouble_impl<false> {
  static constexpr EIGEN_DEVICE_FUNC inline int mantissaBits() { return NumTraits<long double>::digits() - 1; }
  static EIGEN_DEVICE_FUNC inline long double run(int numRandomBits) {
    return static_cast<long double>(random_float_impl<double>::run(numRandomBits));
  }
};
template <>
struct random_float_impl<long double> : random_longdouble_impl<> {};

template <typename Scalar>
struct random_default_impl<Scalar, false, false> {
  using Impl = random_float_impl<Scalar>;
  static EIGEN_DEVICE_FUNC inline Scalar run(const Scalar& x, const Scalar& y, int numRandomBits) {
    Scalar half_x = Scalar(0.5) * x;
    Scalar half_y = Scalar(0.5) * y;
    Scalar result = (half_x + half_y) + (half_y - half_x) * run(numRandomBits);
    // result is in the half-open interval [x, y) -- provided that x < y
    return result;
  }
  static EIGEN_DEVICE_FUNC inline Scalar run(const Scalar& x, const Scalar& y) {
    return run(x, y, Impl::mantissaBits());
  }
  static EIGEN_DEVICE_FUNC inline Scalar run(int numRandomBits) { return Impl::run(numRandomBits); }
  static EIGEN_DEVICE_FUNC inline Scalar run() { return Impl::run(Impl::mantissaBits()); }
};

template <typename Scalar, bool IsSigned = NumTraits<Scalar>::IsSigned, bool BuiltIn = std::is_integral<Scalar>::value>
struct random_int_impl;

// random implementation for an unsigned integer type, or an integer that is non-negative at runtime
template <typename Scalar, bool BuiltIn>
struct random_int_impl<Scalar, false, BuiltIn> {
  static EIGEN_DEVICE_FUNC inline Scalar run(const Scalar& x, const Scalar& y) {
    if (y <= x) return x;
    Scalar range = y - x;
    // handle edge case where [x,y] spans the entire range of Scalar
    if (range == NumTraits<Scalar>::highest()) return run();
    Scalar count = range + Scalar(1);
    // calculate the number of random bits needed to fill range
    int numRandomBits = log2_ceil(count);
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

// random implementation for a built-in signed integer type
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
    return result;
  }
  static EIGEN_DEVICE_FUNC inline Scalar run() {
#ifdef EIGEN_MAKING_DOCS
    return run(-10, 10);
#else
    return getRandomBits<Scalar>(kTotalBits);
#endif
  }
};

// random implementation for a custom signed integer type
template <typename Scalar>
struct random_int_impl<Scalar, true, false> {
  using RandomUnsignedImpl = random_int_impl<Scalar, false, false>;
  static EIGEN_DEVICE_FUNC inline Scalar run(const Scalar& x, const Scalar& y) {
    if (y <= x) return x;
    bool overflow = (x < Scalar(0)) && (y > (x + NumTraits<Scalar>::highest()));
    if (overflow) {
      // if the range is greater than `highest`, generate an extra random bit
      // this bit implicitly represents 0 or highest + 1
      bool highBit = getRandomBits<int>(1);
      Scalar offset = x;
      if (highBit) {
        offset = offset + NumTraits<Scalar>::highest();
        offset = offset + Scalar(1);
      }
      Scalar result = Scalar(0);
      do {
        // randomBits is in the interval [0, highest]
        Scalar randomBits = getRandomBits<Scalar>(NumTraits<Scalar>::digits());
        result = offset + randomBits;
        // if highBit is set, add highest + 1
      } while (result < x || result > y);
      return result;
    } else {
      // otherwise, calculate the random number without further complication
      Scalar range = y - x;
      Scalar randomBits = RandomUnsignedImpl::run(Scalar(0), range);
      return x + randomBits;
    }
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
  static EIGEN_DEVICE_FUNC inline bool run() { return getRandomBits<int>(1) ? true : false; }
};

template <typename Scalar>
struct random_default_impl<Scalar, true, false> {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  using Impl = random_impl<RealScalar>;
  static EIGEN_DEVICE_FUNC inline Scalar run(const Scalar& x, const Scalar& y, int numRandomBits) {
    return Scalar(Impl::run(x.real(), y.real(), numRandomBits), Impl::run(x.imag(), y.imag(), numRandomBits));
  }
  static EIGEN_DEVICE_FUNC inline Scalar run(const Scalar& x, const Scalar& y) {
    return Scalar(Impl::run(x.real(), y.real()), Impl::run(x.imag(), y.imag()));
  }
  static EIGEN_DEVICE_FUNC inline Scalar run(int numRandomBits) {
    return Scalar(Impl::run(numRandomBits), Impl::run(numRandomBits));
  }
  static EIGEN_DEVICE_FUNC inline Scalar run() { return Scalar(Impl::run(), Impl::run()); }
};

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_RANDOM_IMPL_H
