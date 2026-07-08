// Benchmarks for DPR1EigenSolver, the direct O(n^2) secular-equation
// eigensolver for real symmetric diagonal-plus-rank-one matrices
// D + rho*z*z^T, against the dense O(n^3) SelfAdjointEigenSolver applied to
// the materialized matrix. The dense matrix is formed outside the timed loop,
// so the dense side is not even charged for materialization.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include <unsupported/Eigen/StructuredMatrices>

using namespace Eigen;

typedef Matrix<double, Dynamic, 1> Vec;
typedef Matrix<double, Dynamic, Dynamic> Mat;

// --- full decomposition: eigenvalues + orthogonal eigenvectors ---
static void BM_DPR1EigenSolver(benchmark::State& state) {
  const Index n = state.range(0);
  const Vec d = Vec::Random(n), z = Vec::Random(n);
  const double rho = 2.0;
  for (auto _ : state) {
    DPR1EigenSolver<double> es(d, rho, z);
    benchmark::DoNotOptimize(es.eigenvalues().data());
    benchmark::DoNotOptimize(es.eigenvectors().data());
  }
}
BENCHMARK(BM_DPR1EigenSolver)->Arg(32)->Arg(128)->Arg(512)->Arg(1024);

static void BM_DenseSelfAdjointEigenSolver(benchmark::State& state) {
  const Index n = state.range(0);
  const Vec d = Vec::Random(n), z = Vec::Random(n);
  const double rho = 2.0;
  const Mat dense = Mat(d.asDiagonal()) + rho * z * z.transpose();
  for (auto _ : state) {
    SelfAdjointEigenSolver<Mat> es(dense);
    benchmark::DoNotOptimize(es.eigenvalues().data());
    benchmark::DoNotOptimize(es.eigenvectors().data());
  }
}
BENCHMARK(BM_DenseSelfAdjointEigenSolver)->Arg(32)->Arg(128)->Arg(512)->Arg(1024);

// --- eigenvalues only ---
static void BM_DPR1EigenSolverValuesOnly(benchmark::State& state) {
  const Index n = state.range(0);
  const Vec d = Vec::Random(n), z = Vec::Random(n);
  const double rho = 2.0;
  for (auto _ : state) {
    DPR1EigenSolver<double> es(d, rho, z, EigenvaluesOnly);
    benchmark::DoNotOptimize(es.eigenvalues().data());
  }
}
BENCHMARK(BM_DPR1EigenSolverValuesOnly)->Arg(32)->Arg(128)->Arg(512)->Arg(1024);

static void BM_DenseSelfAdjointEigenSolverValuesOnly(benchmark::State& state) {
  const Index n = state.range(0);
  const Vec d = Vec::Random(n), z = Vec::Random(n);
  const double rho = 2.0;
  const Mat dense = Mat(d.asDiagonal()) + rho * z * z.transpose();
  for (auto _ : state) {
    SelfAdjointEigenSolver<Mat> es(dense, EigenvaluesOnly);
    benchmark::DoNotOptimize(es.eigenvalues().data());
  }
}
BENCHMARK(BM_DenseSelfAdjointEigenSolverValuesOnly)->Arg(32)->Arg(128)->Arg(512)->Arg(1024);
