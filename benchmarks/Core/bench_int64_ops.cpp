// Benchmarks for Array<int64_t>/Array<uint64_t> coefficient-wise ops whose
// NEON (Packet2l/Packet2ul) implementations may be polyfilled: negate,
// abs, multiply, min, max, and comparison-driven select.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <cstdint>

namespace Eigen {
namespace {

template <typename Scalar>
void BM_Negate(benchmark::State& state) {
  const Index n = state.range(0);
  using A = ArrayX<Scalar>;
  A a = A::Random(n);
  A b(n);
  for (auto _ : state) {
    b = -a;
    benchmark::DoNotOptimize(b.data());
  }
  state.SetBytesProcessed(state.iterations() * n * sizeof(Scalar) * 2);
}

template <typename Scalar>
void BM_Abs(benchmark::State& state) {
  const Index n = state.range(0);
  using A = ArrayX<Scalar>;
  A a = A::Random(n);
  A b(n);
  for (auto _ : state) {
    b = a.abs();
    benchmark::DoNotOptimize(b.data());
  }
  state.SetBytesProcessed(state.iterations() * n * sizeof(Scalar) * 2);
}

template <typename Scalar>
void BM_Mul(benchmark::State& state) {
  const Index n = state.range(0);
  using A = ArrayX<Scalar>;
  A a = A::Random(n);
  A b = A::Random(n);
  A c(n);
  for (auto _ : state) {
    c = a * b;
    benchmark::DoNotOptimize(c.data());
  }
  state.SetBytesProcessed(state.iterations() * n * sizeof(Scalar) * 3);
}

template <typename Scalar>
void BM_Min(benchmark::State& state) {
  const Index n = state.range(0);
  using A = ArrayX<Scalar>;
  A a = A::Random(n);
  A b = A::Random(n);
  A c(n);
  for (auto _ : state) {
    c = a.min(b);
    benchmark::DoNotOptimize(c.data());
  }
  state.SetBytesProcessed(state.iterations() * n * sizeof(Scalar) * 3);
}

template <typename Scalar>
void BM_Max(benchmark::State& state) {
  const Index n = state.range(0);
  using A = ArrayX<Scalar>;
  A a = A::Random(n);
  A b = A::Random(n);
  A c(n);
  for (auto _ : state) {
    c = a.max(b);
    benchmark::DoNotOptimize(c.data());
  }
  state.SetBytesProcessed(state.iterations() * n * sizeof(Scalar) * 3);
}

// `(c, d)` must be independent of the predicate inputs `(a, b)`: (a OP b).select(a, b) is
// algebraically `min`/`max` for `Lt`/`Le` and identically `b` for `Eq`, letting an optimizer erase
// the comparison (GCC 13/x86-64 turned `BM_SelectEq<uint64_t>` into a `memcpy`).
#define BENCH_CWISE_SELECT(NAME, OP)                                                                 \
  template <typename Scalar>                                                                         \
  void BM_Select##NAME(benchmark::State& state) {                                                    \
    const Index n = state.range(0);                                                                  \
    using A = ArrayX<Scalar>;                                                                        \
    A a = A::Random(n);                                                                              \
    A b = A::Random(n);                                                                              \
    A c = A::Random(n);                                                                              \
    A d = A::Random(n);                                                                              \
    A out(n);                                                                                        \
    for (auto _ : state) {                                                                           \
      out = (a OP b).select(c, d);                                                                   \
      benchmark::DoNotOptimize(out.data());                                                          \
    }                                                                                                \
    for (Index i = 0; i < n; ++i) {                                                                  \
      const Scalar expected = (a[i] OP b[i]) ? c[i] : d[i];                                          \
      if (out[i] != expected) {                                                                      \
        state.SkipWithError("Select" #NAME ": materialized result does not match scalar reference"); \
        break;                                                                                       \
      }                                                                                              \
    }                                                                                                \
    state.SetBytesProcessed(state.iterations() * n * sizeof(Scalar) * 5);                            \
  }

BENCH_CWISE_SELECT(Lt, <)
BENCH_CWISE_SELECT(Le, <=)
BENCH_CWISE_SELECT(Eq, ==)

// Kept small enough that the working set (up to 5 arrays, for BENCH_CWISE_SELECT)
// stays within a typical L1D cache, so timings reflect compute cost rather than
// memory bandwidth/latency.
#define INT64_SIZES ->Arg(32)->Arg(64)->Arg(128)->Arg(256)->Arg(512)

BENCHMARK(BM_Negate<int64_t>) INT64_SIZES->Name("Negate_int64");

BENCHMARK(BM_Abs<int64_t>) INT64_SIZES->Name("Abs_int64");
BENCHMARK(BM_Abs<uint64_t>) INT64_SIZES->Name("Abs_uint64");

BENCHMARK(BM_Mul<int64_t>) INT64_SIZES->Name("Mul_int64");
BENCHMARK(BM_Mul<uint64_t>) INT64_SIZES->Name("Mul_uint64");

BENCHMARK(BM_Min<int64_t>) INT64_SIZES->Name("Min_int64");
BENCHMARK(BM_Min<uint64_t>) INT64_SIZES->Name("Min_uint64");

BENCHMARK(BM_Max<int64_t>) INT64_SIZES->Name("Max_int64");
BENCHMARK(BM_Max<uint64_t>) INT64_SIZES->Name("Max_uint64");

BENCHMARK(BM_SelectLt<int64_t>) INT64_SIZES->Name("SelectLt_int64");
BENCHMARK(BM_SelectLt<uint64_t>) INT64_SIZES->Name("SelectLt_uint64");

BENCHMARK(BM_SelectLe<int64_t>) INT64_SIZES->Name("SelectLe_int64");
BENCHMARK(BM_SelectLe<uint64_t>) INT64_SIZES->Name("SelectLe_uint64");

BENCHMARK(BM_SelectEq<int64_t>) INT64_SIZES->Name("SelectEq_int64");
BENCHMARK(BM_SelectEq<uint64_t>) INT64_SIZES->Name("SelectEq_uint64");

}  // namespace
}  // namespace Eigen
