// Benchmarks for Array<int64_t>/Array<uint64_t> coefficient-wise ops whose
// NEON (Packet2l/Packet2ul) implementations may be polyfilled: negate,
// abs, multiply, min, max, and comparison-driven select.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <cstdint>

using namespace Eigen;

template <typename Scalar>
static void BM_Negate(benchmark::State& state) {
  const Index n = state.range(0);
  using Arr = Array<Scalar, Dynamic, 1>;
  Arr a = Arr::Random(n);
  Arr b(n);
  for (auto _ : state) {
    b = -a;
    benchmark::DoNotOptimize(b.data());
  }
  state.SetBytesProcessed(state.iterations() * n * sizeof(Scalar) * 2);
}

template <typename Scalar>
static void BM_Abs(benchmark::State& state) {
  const Index n = state.range(0);
  using Arr = Array<Scalar, Dynamic, 1>;
  Arr a = Arr::Random(n);
  Arr b(n);
  for (auto _ : state) {
    b = a.abs();
    benchmark::DoNotOptimize(b.data());
  }
  state.SetBytesProcessed(state.iterations() * n * sizeof(Scalar) * 2);
}

template <typename Scalar>
static void BM_Mul(benchmark::State& state) {
  const Index n = state.range(0);
  using Arr = Array<Scalar, Dynamic, 1>;
  Arr a = Arr::Random(n);
  Arr b = Arr::Random(n);
  Arr c(n);
  for (auto _ : state) {
    c = a * b;
    benchmark::DoNotOptimize(c.data());
  }
  state.SetBytesProcessed(state.iterations() * n * sizeof(Scalar) * 3);
}

template <typename Scalar>
static void BM_Min(benchmark::State& state) {
  const Index n = state.range(0);
  using Arr = Array<Scalar, Dynamic, 1>;
  Arr a = Arr::Random(n);
  Arr b = Arr::Random(n);
  Arr c(n);
  for (auto _ : state) {
    c = a.min(b);
    benchmark::DoNotOptimize(c.data());
  }
  state.SetBytesProcessed(state.iterations() * n * sizeof(Scalar) * 3);
}

template <typename Scalar>
static void BM_Max(benchmark::State& state) {
  const Index n = state.range(0);
  using Arr = Array<Scalar, Dynamic, 1>;
  Arr a = Arr::Random(n);
  Arr b = Arr::Random(n);
  Arr c(n);
  for (auto _ : state) {
    c = a.max(b);
    benchmark::DoNotOptimize(c.data());
  }
  state.SetBytesProcessed(state.iterations() * n * sizeof(Scalar) * 3);
}

// (a OP b).select(a, b): exercises pcmp_{lt,le,eq} feeding pselect, the same
// codepath the optimized pmin/pmax now use internally.
#define BENCH_CWISE_SELECT(NAME, OP)                            \
  template <typename Scalar>                                    \
  static void BM_Select##NAME(benchmark::State& state) {        \
    const Index n = state.range(0);                             \
    using Arr = Array<Scalar, Dynamic, 1>;                       \
    Arr a = Arr::Random(n);                                      \
    Arr b = Arr::Random(n);                                      \
    Arr c(n);                                                    \
    for (auto _ : state) {                                       \
      c = (a OP b).select(a, b);                                 \
      benchmark::DoNotOptimize(c.data());                        \
    }                                                             \
    state.SetBytesProcessed(state.iterations() * n * sizeof(Scalar) * 3); \
  }

BENCH_CWISE_SELECT(Lt, <)
BENCH_CWISE_SELECT(Le, <=)
BENCH_CWISE_SELECT(Eq, ==)

// clang-format off
// Kept small enough that the working set (up to 3 arrays) stays within a
// typical L1D cache, so timings reflect compute cost rather than memory
// bandwidth/latency.
#define INT64_SIZES ->Arg(32)->Arg(64)->Arg(128)->Arg(256)->Arg(512)

BENCHMARK(BM_Negate<int64_t>) INT64_SIZES ->Name("Negate_int64");

BENCHMARK(BM_Abs<int64_t>) INT64_SIZES ->Name("Abs_int64");
BENCHMARK(BM_Abs<uint64_t>) INT64_SIZES ->Name("Abs_uint64");

BENCHMARK(BM_Mul<int64_t>) INT64_SIZES ->Name("Mul_int64");
BENCHMARK(BM_Mul<uint64_t>) INT64_SIZES ->Name("Mul_uint64");

BENCHMARK(BM_Min<int64_t>) INT64_SIZES ->Name("Min_int64");
BENCHMARK(BM_Min<uint64_t>) INT64_SIZES ->Name("Min_uint64");

BENCHMARK(BM_Max<int64_t>) INT64_SIZES ->Name("Max_int64");
BENCHMARK(BM_Max<uint64_t>) INT64_SIZES ->Name("Max_uint64");

BENCHMARK(BM_SelectLt<int64_t>) INT64_SIZES ->Name("SelectLt_int64");
BENCHMARK(BM_SelectLt<uint64_t>) INT64_SIZES ->Name("SelectLt_uint64");

BENCHMARK(BM_SelectLe<int64_t>) INT64_SIZES ->Name("SelectLe_int64");
BENCHMARK(BM_SelectLe<uint64_t>) INT64_SIZES ->Name("SelectLe_uint64");

BENCHMARK(BM_SelectEq<int64_t>) INT64_SIZES ->Name("SelectEq_int64");
BENCHMARK(BM_SelectEq<uint64_t>) INT64_SIZES ->Name("SelectEq_uint64");
// clang-format on
