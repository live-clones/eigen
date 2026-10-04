// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Own small TU: GCC 13 -O3 -ffast-math zeroed the trig sign mask only where IPA-CP could specialize it (#3132).
#include "main.h"

template <typename Scalar, int Func, bool Enabled>
struct packet_trig_sign_check {
  static void run() {}
};

template <typename Scalar, int Func>
struct packet_trig_sign_check<Scalar, Func, true> {
  using Packet = typename internal::packet_traits<Scalar>::type;
  static Packet eval(const Packet& p, std::integral_constant<int, 0>) { return internal::psin(p); }
  static Packet eval(const Packet& p, std::integral_constant<int, 1>) { return internal::pcos(p); }
  static Packet eval(const Packet& p, std::integral_constant<int, 2>) { return internal::ptan(p); }

  static void run() {
    constexpr Index kSize = internal::unpacket_traits<Packet>::size;
    const Index n = numext::maxi<Index>(32, kSize);
    std::vector<Scalar> x(n), y(n);
    // +-(0.3 + 0.7 k) covers all four quadrants with sin, cos and tan bounded away from 0.
    for (Index i = 0; i < n; ++i) x[i] = Scalar((i & 1) ? 1 : -1) * (Scalar(0.3) + Scalar(0.7) * Scalar(i % 16));
    // The packet kernels are called directly: through the array API they inline and the defect does not appear.
    for (Index i = 0; i < n; i += kSize) {
      const Packet p = internal::ploadu<Packet>(&x[i]);
      internal::pstoreu(&y[i], eval(p, std::integral_constant<int, Func>()));
    }
    for (Index i = 0; i < n; ++i) {
      const Scalar ref = Func == 0 ? std::sin(x[i]) : Func == 1 ? std::cos(x[i]) : std::tan(x[i]);
      VERIFY_IS_APPROX(y[i], ref);
    }
  }
};

template <typename Scalar>
void check_packet_trig_signs() {
  using Traits = internal::packet_traits<Scalar>;
  packet_trig_sign_check<Scalar, 0, Traits::Vectorizable && Traits::HasSin>::run();
  packet_trig_sign_check<Scalar, 1, Traits::Vectorizable && Traits::HasCos>::run();
  packet_trig_sign_check<Scalar, 2, Traits::Vectorizable && Traits::HasTan>::run();
}

EIGEN_DECLARE_TEST(trig_sign_fastmath) {
  CALL_SUBTEST(check_packet_trig_signs<float>());
  CALL_SUBTEST(check_packet_trig_signs<double>());
}
