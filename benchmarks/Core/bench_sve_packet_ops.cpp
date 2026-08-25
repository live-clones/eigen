// Benchmarks for the SVE PacketMath implementations of `plset`, `ploaddup`,
// `ploadquad`, `predux_mul`, and `ptranspose`. To compare against a prior
// implementation, build and run this same file against the Eigen checkout
// in question -- it only calls the public `Eigen::internal` packet API, so
// it is source-compatible with whatever PacketMath.h happens to provide.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>

#if defined(EIGEN_VECTORIZE_SVE)

#include <cstdint>
#include <type_traits>

namespace Eigen {
namespace {

using internal::packet_traits;
using internal::PacketBlock;
using internal::pfirst;

template <typename Scalar>
svbool_t all_true() = delete;
template <>
svbool_t all_true<numext::int32_t>() {
  return svptrue_b32();
}
template <>
svbool_t all_true<float>() {
  return svptrue_b32();
}
template <>
svbool_t all_true<double>() {
  return svptrue_b64();
}

template <typename Packet, int N>
__attribute__((noinline)) void call_ptranspose(PacketBlock<Packet, N>& kernel) {
  internal::ptranspose(kernel);
}

// ---- plset ----

template <typename Scalar>
void BM_Plset(benchmark::State& state) {
  using Packet = typename packet_traits<Scalar>::type;
  constexpr int N = packet_traits<Scalar>::size;
  Scalar a = Scalar(7);
  Scalar out[N];

  svst1(all_true<Scalar>(), out, internal::plset<Packet>(a));
  for (int i = 0; i < N; ++i) {
    if (out[i] != static_cast<Scalar>(a + i)) {
      state.SkipWithError("Plset: materialized result does not match scalar reference");
      return;
    }
  }

  for (auto _ : state) {
    asm volatile("" : "+w"(a));
    svst1(all_true<Scalar>(), out, internal::plset<Packet>(a));
    benchmark::DoNotOptimize(out);
  }
}
BENCHMARK(BM_Plset<numext::int32_t>)->Name("Plset_int32");
BENCHMARK(BM_Plset<float>)->Name("Plset_float");
BENCHMARK(BM_Plset<double>)->Name("Plset_double");

// ---- ploaddup ----

template <typename Scalar>
void BM_Ploaddup(benchmark::State& state) {
  using Packet = typename packet_traits<Scalar>::type;
  constexpr int N = packet_traits<Scalar>::size;
  Scalar in[N];
  for (int i = 0; i < N; ++i) in[i] = static_cast<Scalar>(i);
  Scalar out[N];

  svst1(all_true<Scalar>(), out, internal::ploaddup<Packet>(in));
  for (int i = 0; i < N / 2; ++i) {
    if (out[2 * i] != in[i] || out[2 * i + 1] != in[i]) {
      state.SkipWithError("Ploaddup: materialized result does not match scalar reference");
      return;
    }
  }

  for (auto _ : state) {
    svst1(all_true<Scalar>(), out, internal::ploaddup<Packet>(in));
    benchmark::DoNotOptimize(out);
  }
}
BENCHMARK(BM_Ploaddup<numext::int32_t>)->Name("Ploaddup_int32");
BENCHMARK(BM_Ploaddup<float>)->Name("Ploaddup_float");
BENCHMARK(BM_Ploaddup<double>)->Name("Ploaddup_double");

// ---- ploadquad ----

template <typename Scalar>
void BM_Ploadquad(benchmark::State& state) {
  using Packet = typename packet_traits<Scalar>::type;
  constexpr int N = packet_traits<Scalar>::size;
  // `ploadquad` reads one value per group of four lanes; at the smallest
  // vector length that would round down to zero groups, so at least one is
  // read.
  constexpr int kQuarter = (N / 4 > 0) ? N / 4 : 1;
  Scalar in[N];
  for (int i = 0; i < N; ++i) in[i] = static_cast<Scalar>(i);
  Scalar out[N];

  svst1(all_true<Scalar>(), out, internal::ploadquad<Packet>(in));
  for (int j = 0; j < kQuarter; ++j) {
    for (int k = 0; k < 4; ++k) {
      if (out[4 * j + k] != in[j]) {
        state.SkipWithError("Ploadquad: materialized result does not match scalar reference");
        return;
      }
    }
  }

  for (auto _ : state) {
    svst1(all_true<Scalar>(), out, internal::ploadquad<Packet>(in));
    benchmark::DoNotOptimize(out);
  }
}
BENCHMARK(BM_Ploadquad<numext::int32_t>)->Name("Ploadquad_int32");
BENCHMARK(BM_Ploadquad<float>)->Name("Ploadquad_float");
BENCHMARK(BM_Ploadquad<double>)->Name("Ploadquad_double");

// ---- predux_mul ----
// Inputs are chosen so the true product is exactly representable regardless
// of the order the reduction folds lanes together: powers of two multiply
// without rounding for float/double, and multiplying by 1 is exact and
// overflow-free for int32.

template <typename Scalar>
void fill_redux_mul_input(Scalar (&in)[packet_traits<Scalar>::size], Scalar& expected) {
  constexpr int N = packet_traits<Scalar>::size;
  if constexpr (std::is_integral<Scalar>::value) {
    for (int i = 0; i < N; ++i) in[i] = Scalar(1);
    in[0] = Scalar(-3);
    in[N - 1] = Scalar(2);
    expected = Scalar(-6);
  } else {
    expected = Scalar(1);
    for (int i = 0; i < N; ++i) {
      in[i] = static_cast<Scalar>(std::int64_t(1) << i);
      expected *= in[i];
    }
  }
}

template <typename Scalar>
void BM_ReduxMul(benchmark::State& state) {
  using Packet = typename packet_traits<Scalar>::type;
  constexpr int N = packet_traits<Scalar>::size;
  Scalar in[N];
  Scalar expected;
  fill_redux_mul_input<Scalar>(in, expected);
  Packet a = svld1(all_true<Scalar>(), in);

  if (internal::predux_mul<Packet>(a) != expected) {
    state.SkipWithError("ReduxMul: materialized result does not match scalar reference");
    return;
  }

  for (auto _ : state) {
    asm volatile("" : "+w"(a));
    benchmark::DoNotOptimize(internal::predux_mul<Packet>(a));
  }
}
BENCHMARK(BM_ReduxMul<numext::int32_t>)->Name("ReduxMul_int32");
BENCHMARK(BM_ReduxMul<float>)->Name("ReduxMul_float");
BENCHMARK(BM_ReduxMul<double>)->Name("ReduxMul_double");

// ---- ptranspose ----
// Benchmarked at N == the type's packet width, i.e. a full square transpose,
// so the expected result is simply new_packet[i][k] == old_packet[k][i].
// Correctness is checked once on a scratch kernel -- ptranspose applied
// repeatedly toggles between the original and transposed state, so checking
// after the timed loop would depend on the (unpredictable) iteration count.

template <typename Scalar>
void BM_Ptranspose(benchmark::State& state) {
  using Packet = typename packet_traits<Scalar>::type;
  constexpr int N = packet_traits<Scalar>::size;
  Scalar in[N * N];
  for (int i = 0; i < N * N; ++i) in[i] = static_cast<Scalar>(i);

  PacketBlock<Packet, N> check;
  for (int i = 0; i < N; ++i) check.packet[i] = svld1(all_true<Scalar>(), in + i * N);
  call_ptranspose<Packet, N>(check);
  for (int i = 0; i < N; ++i) {
    Scalar row[N];
    svst1(all_true<Scalar>(), row, check.packet[i]);
    for (int k = 0; k < N; ++k) {
      if (row[k] != in[k * N + i]) {
        state.SkipWithError("Ptranspose: materialized result does not match scalar reference");
        return;
      }
    }
  }

  PacketBlock<Packet, N> kernel;
  for (int i = 0; i < N; ++i) kernel.packet[i] = svld1(all_true<Scalar>(), in + i * N);
  for (auto _ : state) {
    call_ptranspose<Packet, N>(kernel);
    benchmark::DoNotOptimize(pfirst<Packet>(kernel.packet[0]));
  }
}
BENCHMARK(BM_Ptranspose<numext::int32_t>)->Name("Ptranspose_int32");
BENCHMARK(BM_Ptranspose<float>)->Name("Ptranspose_float");
BENCHMARK(BM_Ptranspose<double>)->Name("Ptranspose_double");

}  // namespace
}  // namespace Eigen

#endif  // defined(EIGEN_VECTORIZE_SVE)
