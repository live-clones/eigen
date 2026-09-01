// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <Eigen/Cholesky>

using namespace Eigen;

#ifndef SCALAR
#define SCALAR float
#endif

using Scalar = SCALAR;

static void BM_Cholesky_Inverse_Dynamic(benchmark::State& state) {
  int n = state.range(0);
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic>;
  MatrixType a = MatrixType::Random(n, n);
  MatrixType covMat = a * a.adjoint();
  MatrixType inv(n, n);
  for (auto _ : state) {
    // covMat is loop invariant, so it is clobbered to keep the inversion from being hoisted out,
    // and inv is escaped as an lvalue so that the stores into it cannot be elided.
    benchmark::DoNotOptimize(covMat);
    inv = internal::inverse_cholesky(covMat);
    benchmark::DoNotOptimize(inv);
  }
}
BENCHMARK(BM_Cholesky_Inverse_Dynamic)->DenseRange(1, 20, 1);

template <int Size>
static void BM_Cholesky_Inverse_Fixed(benchmark::State& state) {
  using MatrixType = Matrix<Scalar, Size, Size>;
  MatrixType a = MatrixType::Random();
  MatrixType covMat = a * a.adjoint();
  MatrixType inv;
  for (auto _ : state) {
    benchmark::DoNotOptimize(covMat);
    inv = internal::inverse_cholesky(covMat);
    benchmark::DoNotOptimize(inv);
  }
}

template <std::size_t... Is>
void register_fixed_benchmarks(std::index_sequence<Is...>) {
  int dummy[] = {0, (benchmark::RegisterBenchmark(("BM_Cholesky_Inverse_Fixed/" + std::to_string(Is + 1)).c_str(),
                                                  BM_Cholesky_Inverse_Fixed<Is + 1>),
                     0)...};
  (void)dummy;
}

struct RegisterFixed {
  RegisterFixed() { register_fixed_benchmarks(std::make_index_sequence<20>{}); }
} register_fixed_instance;

BENCHMARK_MAIN();
