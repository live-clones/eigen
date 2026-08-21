// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <Eigen/Cholesky>

#include "../bench_common.h"

using namespace Eigen;

#ifndef SCALAR
#define SCALAR float
#endif

typedef SCALAR Scalar;

static void BM_LDLT(benchmark::State& state) {
  int n = state.range(0);
  typedef Matrix<Scalar, Dynamic, Dynamic> MatrixType;
  typedef Matrix<Scalar, Dynamic, Dynamic> SquareMatrixType;
  MatrixType a = MatrixType::Random(n, n);
  SquareMatrixType covMat = a * a.adjoint();
  int r = internal::random<int>(0, n - 1);
  int c = internal::random<int>(0, n - 1);
  Scalar acc = 0;
  for (auto _ : state) {
    LDLT<SquareMatrixType> cholnosqrt(covMat);
    acc += cholnosqrt.matrixL().coeff(r, c);
    benchmark::DoNotOptimize(acc);
  }
}
BENCHMARK(BM_LDLT)->RangeMultiplier(2)->Range(4, 1500);

static void BM_LLT(benchmark::State& state) {
  int n = state.range(0);
  typedef Matrix<Scalar, Dynamic, Dynamic> MatrixType;
  typedef Matrix<Scalar, Dynamic, Dynamic> SquareMatrixType;
  MatrixType a = MatrixType::Random(n, n);
  SquareMatrixType covMat = a * a.adjoint();
  int r = internal::random<int>(0, n - 1);
  int c = internal::random<int>(0, n - 1);
  Scalar acc = 0;
  for (auto _ : state) {
    LLT<SquareMatrixType> chol(covMat);
    acc += chol.matrixL().coeff(r, c);
    benchmark::DoNotOptimize(acc);
  }
  const double cost = eigen_bench::symmetricFactorizationFlops<Scalar>(n);
  state.counters["GFLOPS"] =
      benchmark::Counter(cost, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}
BENCHMARK(BM_LLT)->RangeMultiplier(2)->Range(4, 1500);
