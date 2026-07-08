// Benchmarks for the Cauchy operator: the O(mn)-flop, O(m+n)-storage direct
// product against its dense equivalents (with and without materializing the
// matrix first), and the O(n^2) GKO pivoted solver (CauchyLU) against the
// O(n^3) dense PartialPivLU of the materialized matrix.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <Eigen/LU>
#include <unsupported/Eigen/StructuredMatrices>

using namespace Eigen;

typedef Matrix<double, Dynamic, 1> Vec;
typedef Matrix<double, Dynamic, Dynamic> Mat;

// Separated node sets (x in [2,3], y in [0,1]) keep all denominators in [1,3],
// the same benign configuration the unit tests use.
static void separatedNodes(Index n, Vec& x, Vec& y) {
  x = (Vec::Random(n) * 0.5).array() + 2.5;
  y = (Vec::Random(n) * 0.5).array() + 0.5;
}

static Mat denseCauchy(const Vec& x, const Vec& y) {
  Mat dense(x.size(), y.size());
  for (Index j = 0; j < y.size(); ++j)
    for (Index i = 0; i < x.size(); ++i) dense(i, j) = 1.0 / (x[i] - y[j]);
  return dense;
}

// --- Matrix-vector product w = C * v ---
static void BM_CauchyProduct(benchmark::State& state) {
  const Index n = state.range(0);
  Vec x, y;
  separatedNodes(n, x, y);
  Cauchy<double> C(x, y);  // O(n) storage; entries generated on the fly
  Vec v = Vec::Random(n), w(n);
  for (auto _ : state) {
    w.noalias() = C * v;
    benchmark::DoNotOptimize(w.data());
  }
}
BENCHMARK(BM_CauchyProduct)->Arg(64)->Arg(256)->Arg(1024);

static void BM_DenseCauchyProduct(benchmark::State& state) {
  // The dense product once the matrix has been materialized (O(n^2) storage).
  const Index n = state.range(0);
  Vec x, y;
  separatedNodes(n, x, y);
  Mat dense = denseCauchy(x, y);
  Vec v = Vec::Random(n), w(n);
  for (auto _ : state) {
    w.noalias() = dense * v;
    benchmark::DoNotOptimize(w.data());
  }
}
BENCHMARK(BM_DenseCauchyProduct)->Arg(64)->Arg(256)->Arg(1024);

static void BM_DenseCauchyMaterializeProduct(benchmark::State& state) {
  // The dense equivalent when the matrix is not kept around: materialize from
  // the node vectors, then multiply.
  const Index n = state.range(0);
  Vec x, y;
  separatedNodes(n, x, y);
  Cauchy<double> C(x, y);
  Vec v = Vec::Random(n), w(n);
  for (auto _ : state) {
    Mat dense = C;
    w.noalias() = dense * v;
    benchmark::DoNotOptimize(w.data());
  }
}
BENCHMARK(BM_DenseCauchyMaterializeProduct)->Arg(64)->Arg(256)->Arg(1024);

// --- Square solve C * u = b from the node vectors ---
static void BM_CauchyLUSolve(benchmark::State& state) {
  const Index n = state.range(0);
  Vec x, y;
  separatedNodes(n, x, y);
  Cauchy<double> C(x, y);
  Vec b = Vec::Random(n), u(n);
  for (auto _ : state) {
    CauchyLU<double> lu(C);  // O(n^2) pivoted factorization (GKO)
    u = lu.solve(b);
    benchmark::DoNotOptimize(u.data());
  }
}
BENCHMARK(BM_CauchyLUSolve)->Arg(64)->Arg(256)->Arg(1024);

static void BM_DensePartialPivLUSolve(benchmark::State& state) {
  const Index n = state.range(0);
  Vec x, y;
  separatedNodes(n, x, y);
  Mat dense = denseCauchy(x, y);
  Vec b = Vec::Random(n), u(n);
  for (auto _ : state) {
    PartialPivLU<Mat> lu(dense);  // O(n^3) pivoted factorization
    u = lu.solve(b);
    benchmark::DoNotOptimize(u.data());
  }
}
BENCHMARK(BM_DensePartialPivLUSolve)->Arg(64)->Arg(256)->Arg(1024);
