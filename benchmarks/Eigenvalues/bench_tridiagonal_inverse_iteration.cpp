// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Benchmarks the eigenvector stage of the spectral-bisection path: inverse iteration on a real
// symmetric tridiagonal matrix via SelfAdjointEigenSolver::computeEigenvectors(diag, subdiag, evals).
// The eigenvalues are precomputed (bisection) outside the timed region, so only the inverse-iteration
// work (LU factor + back-solves + intra-cluster reorthogonalization) is measured.
//
//   - BM_invit_rand_*    : random tridiagonal, well-separated spectrum (factor/solve bound);
//   - BM_invit_cluster_* : glued-Wilkinson blocks, tight clusters (reorthogonalization bound).
// Suffix _f / _d selects float / double.

#include <benchmark/benchmark.h>
#include <Eigen/Eigenvalues>

using namespace Eigen;

namespace {

enum Kind { kRandom, kClustered };

template <typename Scalar>
void make_matrix(Kind kind, Index n, Matrix<Scalar, Dynamic, 1>& diag, Matrix<Scalar, Dynamic, 1>& sub) {
  diag.resize(n);
  sub.resize(n > 1 ? n - 1 : 0);
  if (kind == kRandom) {
    diag.setRandom();
    sub.setRandom();
  } else {
    // Glued Wilkinson W+_{blk}: blocks laid end to end with a small inter-block "glue" off-diagonal.
    // The large eigenvalues of each block nearly coincide across copies -> clusters at working
    // precision, the stress case for intra-cluster reorthogonalization.
    const Index blk = 15;
    const Index m = (blk - 1) / 2;
    for (Index i = 0; i < n; ++i) diag(i) = Scalar(numext::abs(m - (i % blk)));
    for (Index i = 0; i < n - 1; ++i) sub(i) = ((i + 1) % blk == 0) ? Scalar(1e-3) : Scalar(1);
  }
}

template <typename Scalar>
void run(benchmark::State& state, Kind kind) {
  const Index n = state.range(0);
  Matrix<Scalar, Dynamic, 1> diag, sub;
  make_matrix<Scalar>(kind, n, diag, sub);

  // Precompute the eigenvalues once (not timed).
  SelfAdjointEigenSolver<Matrix<Scalar, Dynamic, Dynamic>> evsolver(n);
  evsolver.computeFromTridiagonal(diag, sub, EigenvaluesOnly | UseBisection);
  const Matrix<Scalar, Dynamic, 1> evals = evsolver.eigenvalues();

  SelfAdjointEigenSolver<Matrix<Scalar, Dynamic, Dynamic>> solver(n);
  for (auto _ : state) {
    solver.computeEigenvectors(diag, sub, evals);
    benchmark::DoNotOptimize(solver.eigenvectors().data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * n);
}

void BM_invit_rand_f(benchmark::State& s) { run<float>(s, kRandom); }
void BM_invit_rand_d(benchmark::State& s) { run<double>(s, kRandom); }
void BM_invit_cluster_f(benchmark::State& s) { run<float>(s, kClustered); }
void BM_invit_cluster_d(benchmark::State& s) { run<double>(s, kClustered); }

#define EIGEN_BENCH_SIZES ArgsProduct({{16, 32, 64, 128, 256, 512, 1024, 2048}})

BENCHMARK(BM_invit_rand_f)->EIGEN_BENCH_SIZES->UseRealTime();
BENCHMARK(BM_invit_rand_d)->EIGEN_BENCH_SIZES->UseRealTime();
BENCHMARK(BM_invit_cluster_f)->EIGEN_BENCH_SIZES->UseRealTime();
BENCHMARK(BM_invit_cluster_d)->EIGEN_BENCH_SIZES->UseRealTime();

}  // namespace
