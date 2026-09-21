// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// GEMM on the thread pool over the pool size. On a machine whose GEMM kernel
// runs on shared units (the ARM SME backend: one unit per cluster), the rate
// levels off at the unit count and falls beyond it unless the thread count is
// capped; the "uncapped" variant removes the cap (setNbSmeUnits(0)) for the
// comparison and is a no-op on other backends.
#define EIGEN_GEMM_THREADPOOL
#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <Eigen/ThreadPool>

using namespace Eigen;

#ifndef SCALAR
#define SCALAR float
#endif
typedef SCALAR Scalar;
typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

template <typename A, typename B, typename C>
EIGEN_DONT_INLINE void gemm(const A& a, const B& b, C& c) {
  c.noalias() += a * b;
}

template <bool Uncapped>
static void BM_GemmThreads(benchmark::State& state) {
  const int n = static_cast<int>(state.range(0));
  const int threads = static_cast<int>(state.range(1));
  ThreadPool pool(threads);
  setGemmThreadPool(&pool);
#ifdef EIGEN_VECTORIZE_SME
  const int units = nbSmeUnits();
  if (Uncapped) setNbSmeUnits(0);
#endif
  Mat a = Mat::Random(n, n), b = Mat::Random(n, n), c = Mat::Zero(n, n);
  for (auto _ : state) {
    gemm(a, b, c);
    benchmark::DoNotOptimize(c.data());
    benchmark::ClobberMemory();
  }
  state.counters["GFLOPS"] =
      benchmark::Counter(2.0 * n * n * n, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
#ifdef EIGEN_VECTORIZE_SME
  setNbSmeUnits(units);
#endif
}

// clang-format off
BENCHMARK(BM_GemmThreads<false>)->Name("BM_GemmThreads")
    ->ArgsProduct({{1024, 2048}, {1, 2, 4, 8, 12}})->UseRealTime();
BENCHMARK(BM_GemmThreads<true>)->Name("BM_GemmThreads_uncapped")
    ->ArgsProduct({{1024, 2048}, {4, 8, 12}})->UseRealTime();
// clang-format on
