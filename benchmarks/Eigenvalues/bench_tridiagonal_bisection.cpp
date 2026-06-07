// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Benchmarks the eigenvalues of a random symmetric tridiagonal matrix computed three ways via
// SelfAdjointEigenSolver::computeFromTridiagonal():
//   - BM_qr_*         : the implicit-QR algorithm (default), all eigenvalues;
//   - BM_bisect_*     : Sturm-sequence spectral bisection (#UseBisection), all eigenvalues;
//   - BM_bisect_sub_* : bisection of only the 10% smallest eigenvalues (an index-range subset,
//                       a query QR cannot answer without computing the whole spectrum).
// Suffix _f / _d selects float / double.

#include <benchmark/benchmark.h>
#include <Eigen/Eigenvalues>

using namespace Eigen;

namespace {

enum Mode { kBisectAll, kQrAll, kBisectSubset };

template <typename Scalar>
void run(benchmark::State& state, Mode mode) {
  const Index n = state.range(0);
  Matrix<Scalar, Dynamic, 1> diag = Matrix<Scalar, Dynamic, 1>::Random(n);
  Matrix<Scalar, Dynamic, 1> subdiag = Matrix<Scalar, Dynamic, 1>::Random(n > 1 ? n - 1 : 0);
  SelfAdjointEigenSolver<Matrix<Scalar, Dynamic, Dynamic>> solver(n);

  const int options = EigenvaluesOnly | (mode == kQrAll ? 0 : UseBisection);
  const EigenvalueRange range = mode == kBisectSubset ? EigenvalueRange::indices(0, n / 10) : EigenvalueRange::all();

  for (auto _ : state) {
    solver.computeFromTridiagonal(diag, subdiag, options, range);
    benchmark::DoNotOptimize(solver.eigenvalues().data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * n);
}

void BM_qr_f(benchmark::State& s) { run<float>(s, kQrAll); }
void BM_bisect_f(benchmark::State& s) { run<float>(s, kBisectAll); }
void BM_bisect_sub_f(benchmark::State& s) { run<float>(s, kBisectSubset); }
void BM_qr_d(benchmark::State& s) { run<double>(s, kQrAll); }
void BM_bisect_d(benchmark::State& s) { run<double>(s, kBisectAll); }
void BM_bisect_sub_d(benchmark::State& s) { run<double>(s, kBisectSubset); }

#define EIGEN_BENCH_SIZES ArgsProduct({{16, 32, 64, 128, 256, 512, 1024, 2048, 4096}})

BENCHMARK(BM_qr_f)->EIGEN_BENCH_SIZES;
BENCHMARK(BM_bisect_f)->EIGEN_BENCH_SIZES;
BENCHMARK(BM_bisect_sub_f)->EIGEN_BENCH_SIZES;
BENCHMARK(BM_qr_d)->EIGEN_BENCH_SIZES;
BENCHMARK(BM_bisect_d)->EIGEN_BENCH_SIZES;
BENCHMARK(BM_bisect_sub_d)->EIGEN_BENCH_SIZES;

}  // namespace
