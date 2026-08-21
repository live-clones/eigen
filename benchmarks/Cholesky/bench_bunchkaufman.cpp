// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0
//
// Micro-benchmark for the Bunch-Kaufman factorization of symmetric/Hermitian indefinite matrices,
// compared against the other dense solvers that can handle indefinite systems (LDLT, PartialPivLU)
// and against LLT (definite-only) as a lower bound on achievable performance.

#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <Eigen/Cholesky>

#include "../bench_common.h"
#include <Eigen/LU>

using namespace Eigen;

#ifndef SCALAR
#define SCALAR double
#endif

typedef SCALAR Scalar;
typedef Matrix<Scalar, Dynamic, Dynamic> MatrixType;
typedef Matrix<Scalar, Dynamic, 1> VectorType;

// Half-flops symmetric-factorization cost (multiply + add counted separately), matching bench_cholesky.cpp.
// Flop counts come from benchmarks/bench_common.h so this file and the
// comparison harness scale the same operation identically.

// A symmetric/Hermitian indefinite test matrix.
static MatrixType make_indefinite(int n) {
  MatrixType a = MatrixType::Random(n, n);
  return (a + a.adjoint()).eval();
}

static void BM_BunchKaufman(benchmark::State& state) {
  const int n = state.range(0);
  MatrixType A = make_indefinite(n);
  const int r = internal::random<int>(0, n - 1);
  Scalar acc = 0;
  for (auto _ : state) {
    BunchKaufman<MatrixType> bk(A);
    acc += bk.matrixLDLT().coeff(r, r);
    benchmark::DoNotOptimize(acc);
  }
  state.counters["GFLOPS"] =
      benchmark::Counter(eigen_bench::symmetricFactorizationFlops<Scalar>(n),
                         benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}
BENCHMARK(BM_BunchKaufman)->RangeMultiplier(2)->Range(8, 2048);

static void BM_BunchKaufman_Solve(benchmark::State& state) {
  const int n = state.range(0);
  MatrixType A = make_indefinite(n);
  VectorType b = VectorType::Random(n);
  for (auto _ : state) {
    BunchKaufman<MatrixType> bk(A);
    VectorType x = bk.solve(b);
    benchmark::DoNotOptimize(x.data());
  }
}
BENCHMARK(BM_BunchKaufman_Solve)->RangeMultiplier(2)->Range(8, 2048);

static void BM_LDLT(benchmark::State& state) {
  const int n = state.range(0);
  MatrixType A = make_indefinite(n);
  const int r = internal::random<int>(0, n - 1);
  Scalar acc = 0;
  for (auto _ : state) {
    LDLT<MatrixType> ldlt(A);
    acc += ldlt.matrixLDLT().coeff(r, r);
    benchmark::DoNotOptimize(acc);
  }
  state.counters["GFLOPS"] =
      benchmark::Counter(eigen_bench::symmetricFactorizationFlops<Scalar>(n),
                         benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}
BENCHMARK(BM_LDLT)->RangeMultiplier(2)->Range(8, 2048);

static void BM_PartialPivLU(benchmark::State& state) {
  const int n = state.range(0);
  MatrixType A = make_indefinite(n);
  const int r = internal::random<int>(0, n - 1);
  Scalar acc = 0;
  for (auto _ : state) {
    PartialPivLU<MatrixType> lu(A);
    acc += lu.matrixLU().coeff(r, r);
    benchmark::DoNotOptimize(acc);
  }
  // The LU count, not twice the symmetric one. "Roughly twice" is only true
  // asymptotically: 2 * (n^3/3 + 2n^2) overstates m*n^2 - n^3/3 by 31% at n=8 and
  // 4.6% at n=64, and this benchmark sweeps from n=8, so the small end of the
  // curve was reported materially faster than it is. LU is the one arm here that
  // is not a symmetric factorization, so it gets its own formula.
  state.counters["GFLOPS"] =
      benchmark::Counter(eigen_bench::getrfFlops<Scalar>(n, n), benchmark::Counter::kIsIterationInvariantRate,
                         benchmark::Counter::kIs1000);
}
BENCHMARK(BM_PartialPivLU)->RangeMultiplier(2)->Range(8, 2048);

BENCHMARK_MAIN();
