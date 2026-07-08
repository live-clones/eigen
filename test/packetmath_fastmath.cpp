// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"

template <typename Scalar, typename Packet>
EIGEN_DONT_INLINE void store_ptrue(Scalar* output) {
  const Packet zero = Eigen::internal::pset1<Packet>(Scalar(0));
  Eigen::internal::pstoreu<Scalar, Packet>(output, Eigen::internal::ptrue(zero));
}

template <typename Scalar>
EIGEN_DONT_INLINE void store_extended_scalar_constants(Scalar* output) {
  output[0] = Eigen::internal::psignmask<Scalar>();
  output[1] = Eigen::internal::pinf<Scalar>();
  output[2] = Eigen::internal::pnan<Scalar>();
}

template <typename Scalar, bool Vectorizable = Eigen::internal::packet_traits<Scalar>::Vectorizable>
struct packetmath_fastmath_runner {
  static void run() {}
};

template <typename Scalar>
struct packetmath_fastmath_runner<Scalar, true> {
  static void run() {
    typedef typename Eigen::internal::packet_traits<Scalar>::type Packet;
    const int packet_size = Eigen::internal::unpacket_traits<Packet>::size;
    Scalar output[packet_size];
    for (int i = 0; i < packet_size; ++i) {
      output[i] = Scalar(0);
    }

    store_ptrue<Scalar, Packet>(output);

    for (int i = 0; i < packet_size; ++i) {
      const unsigned char* lane_bytes = reinterpret_cast<const unsigned char*>(output + i);
      bool has_nonzero_byte = false;
      for (std::size_t j = 0; j < sizeof(Scalar); ++j) {
        has_nonzero_byte = has_nonzero_byte || lane_bytes[j] != 0;
      }
      VERIFY(has_nonzero_byte);
    }
  }
};

template <typename Scalar,
          bool HasIntegerBits =
              !std::is_void<typename Eigen::numext::get_integer_by_size<sizeof(Scalar)>::unsigned_type>::value>
struct extended_scalar_constant_runner {
  static void run() {}
};

template <typename Scalar>
struct extended_scalar_constant_runner<Scalar, false> {
  static void run() {
    Scalar actual[3];
    Scalar expected[3];
    std::memset(static_cast<void*>(actual), 0, sizeof(actual));
    std::memset(static_cast<void*>(expected), 0, sizeof(expected));

    store_extended_scalar_constants(actual);

    // Floating-point classification is optimized away under -ffinite-math-only, so compare object representations
    // against independent opaque conversions. Zeroing first makes extended-precision padding deterministic.
    typedef Eigen::numext::uint32_t Bits;
    static volatile const Bits expected_bits[3] = {0x80000000u, 0x7f800000u, 0x7fc00000u};
    for (int i = 0; i < 3; ++i) {
      const Bits bits = expected_bits[i];
      expected[i] = static_cast<Scalar>(Eigen::numext::bit_cast<float>(bits));
    }

    const unsigned char* actual_bytes = reinterpret_cast<const unsigned char*>(actual);
    const unsigned char* expected_bytes = reinterpret_cast<const unsigned char*>(expected);
    for (std::size_t i = 0; i < sizeof(actual); ++i) {
      VERIFY_IS_EQUAL(actual_bytes[i], expected_bytes[i]);
    }
  }
};

EIGEN_DECLARE_TEST(packetmath_fastmath) {
  CALL_SUBTEST(packetmath_fastmath_runner<float>::run());
  CALL_SUBTEST(packetmath_fastmath_runner<double>::run());
  CALL_SUBTEST(packetmath_fastmath_runner<Eigen::half>::run());
  CALL_SUBTEST(packetmath_fastmath_runner<Eigen::bfloat16>::run());
  CALL_SUBTEST(extended_scalar_constant_runner<long double>::run());
}
