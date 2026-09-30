// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>

using namespace Eigen;

#ifndef SCALAR
#define SCALAR float
#endif

using Scalar = SCALAR;
using Mat = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;

// Both layouts transpose-pack the RHS; RowMajor also transpose-packs the LHS.
template <int LhsOrder>
static void BM_GemmLayout(benchmark::State& state) {
  const Index m = state.range(0), n = state.range(1), k = state.range(2);
  const Matrix<Scalar, Dynamic, Dynamic, LhsOrder> a = Matrix<Scalar, Dynamic, Dynamic, LhsOrder>::Random(m, k);
  const Mat b = Mat::Random(k, n);
  Mat c = Mat::Zero(m, n);
  c.noalias() += a * b;
  // Check sampled coefficients independently, outside the timed loop.
  for (Index i : {Index(0), m / 2, m - 1}) {
    for (Index j : {Index(0), n / 2, n - 1}) {
      Scalar expected = Scalar(0);
      typename NumTraits<Scalar>::Real scale = 0;
      for (Index p = 0; p < k; ++p) {
        expected += a(i, p) * b(p, j);
        scale += numext::abs(a(i, p)) * numext::abs(b(p, j));
      }
      if (!(numext::abs(c(i, j) - expected) <= 8 * k * NumTraits<Scalar>::epsilon() * scale)) {
        state.SkipWithError("GEMM validation failed");
        return;
      }
    }
  }
  c.setZero();
  for (auto _ : state) {
    c.noalias() += a * b;
    benchmark::DoNotOptimize(c.data());
    benchmark::ClobberMemory();
  }
  constexpr double flops = NumTraits<Scalar>::IsComplex ? 8.0 : 2.0;
  state.counters["GFLOPS"] =
      benchmark::Counter(flops * m * n * k, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

// clang-format off
BENCHMARK_TEMPLATE(BM_GemmLayout, RowMajor)
    ->Args({256, 256, 256})->Args({512, 512, 512})
    ->Args({1024, 1024, 1024})->Args({2048, 2048, 2048})
    ->Args({64, 64, 2048})->Args({256, 256, 2048})->Args({2048, 64, 64})
    ->Args({2048, 64, 128})->Args({2048, 64, 256})->Args({2048, 64, 512});
BENCHMARK_TEMPLATE(BM_GemmLayout, ColMajor)
    ->Args({256, 256, 256})->Args({512, 512, 512})
    ->Args({1024, 1024, 1024})->Args({2048, 2048, 2048})
    ->Args({64, 64, 2048})->Args({256, 256, 2048})->Args({2048, 64, 64})
    ->Args({2048, 64, 128})->Args({2048, 64, 256})->Args({2048, 64, 512});
// clang-format on
