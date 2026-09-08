// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <contrib/Eigen/SpecialFunctions>
#include <cmath>
#include <limits>

// mode: 0 = ordinary inputs, 1 = subnormal packets, 2 = one subnormal per 64 coefficients.
template <typename Scalar>
static void BM_Erf(benchmark::State& state) {
  using Bits = typename Eigen::numext::get_integer_by_size<sizeof(Scalar)>::unsigned_type;
  constexpr Bits sign = Bits(1) << (8 * sizeof(Scalar) - 1);
  constexpr Bits minNormal = Bits(1) << (std::numeric_limits<Scalar>::digits - 1);
  const Eigen::Index n = state.range(0);
  const int mode = int(state.range(1));
  Eigen::Array<Scalar, Eigen::Dynamic, 1> input(n), output(n);
  const Eigen::Index packetEnd = n - n % Eigen::internal::packet_traits<Scalar>::size;
  for (Eigen::Index i = 0; i < n; ++i) {
    input(i) = Scalar(double(i % 257 - 128) / 64.0);
    if (i < packetEnd && (mode == 1 || (mode == 2 && i % 64 == 0))) {
      const Bits magnitude = Bits(1 + (i * 37) % (minNormal - 1));
      input(i) = Eigen::numext::bit_cast<Scalar>(Bits(magnitude | (i % 2 ? sign : Bits(0))));
    }
  }
  output = input.erf();
  for (Eigen::Index i = 0; i < n; ++i) {
    const Bits bits = Eigen::numext::bit_cast<Bits>(input(i));
    const Bits magnitude = bits & (sign - 1);
    if (magnitude > 0 && magnitude < minNormal) {
      const int expected = int(std::floor(double(magnitude) * (2.0 / std::sqrt(std::acos(-1.0))) + 0.5));
      const Bits actual = Eigen::numext::bit_cast<Bits>(output(i));
      if ((actual & sign) != (bits & sign) || std::abs(int(actual & (sign - 1)) - expected) > 1) {
        state.SkipWithError("erf lost a subnormal result");
        return;
      }
    } else {
      const double error = std::abs(double(output(i)) - std::erf(double(input(i))));
      if (!(error <= 4 * double(Eigen::NumTraits<Scalar>::epsilon()))) {
        state.SkipWithError("incorrect ordinary erf result");
        return;
      }
    }
  }
  for (auto _ : state) {
    benchmark::DoNotOptimize(input.data());
    output = input.erf();
    benchmark::DoNotOptimize(output.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK_TEMPLATE(BM_Erf, float)->ArgsProduct({{4, 7, 256, 4096}, {0, 1, 2}});
BENCHMARK_TEMPLATE(BM_Erf, Eigen::bfloat16)->ArgsProduct({{4, 7, 256, 4096}, {0, 1, 2}});
