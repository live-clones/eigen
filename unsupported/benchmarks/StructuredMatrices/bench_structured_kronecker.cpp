// Benchmarks for the implicit Kronecker operator: applying and solving with
// A (x) B through the vec identity (A (x) B) vec(X) = vec(B X A^T) [Van Loan
// 2000] against materializing the dense Kronecker product. With n x n factors
// the implicit product costs O(n^3) and touches O(n^2) memory, while the dense
// operator costs O(n^4) to apply (plus O(n^4) to form and store); the direct
// solve factors two n x n matrices instead of one n^2 x n^2 matrix.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <Eigen/LU>
#include <unsupported/Eigen/StructuredMatrices>

using namespace Eigen;

typedef Matrix<double, Dynamic, 1> Vec;
typedef Matrix<double, Dynamic, Dynamic> Mat;

// --- Matrix-vector product y = (A (x) B) * x, n x n factors ---
static void BM_KroneckerProductImplicit(benchmark::State& state) {
  const Index n = state.range(0);
  Mat A = Mat::Random(n, n), B = Mat::Random(n, n);
  KroneckerOperator<Mat, Mat> K(A, B);
  Vec x = Vec::Random(n * n), y(n * n);
  for (auto _ : state) {
    y.noalias() = K * x;
    benchmark::DoNotOptimize(y.data());
  }
}
BENCHMARK(BM_KroneckerProductImplicit)->Arg(8)->Arg(16)->Arg(32)->Arg(64);

static void BM_KroneckerProductDense(benchmark::State& state) {
  const Index n = state.range(0);
  Mat A = Mat::Random(n, n), B = Mat::Random(n, n);
  Mat dense = KroneckerOperator<Mat, Mat>(A, B);  // materialized once, outside the loop
  Vec x = Vec::Random(n * n), y(n * n);
  for (auto _ : state) {
    y.noalias() = dense * x;
    benchmark::DoNotOptimize(y.data());
  }
}
BENCHMARK(BM_KroneckerProductDense)->Arg(8)->Arg(16)->Arg(32)->Arg(64);

// Forming the dense Kronecker product: the up-front cost (and O(n^4) storage)
// the implicit operator avoids entirely.
static void BM_KroneckerMaterializeDense(benchmark::State& state) {
  const Index n = state.range(0);
  Mat A = Mat::Random(n, n), B = Mat::Random(n, n);
  KroneckerOperator<Mat, Mat> K(A, B);
  Mat dense(n * n, n * n);
  for (auto _ : state) {
    dense = K;
    benchmark::DoNotOptimize(dense.data());
  }
}
BENCHMARK(BM_KroneckerMaterializeDense)->Arg(8)->Arg(16)->Arg(32)->Arg(64);

// --- Direct solve (A (x) B) x = b, n x n invertible factors ---
// The dense product is materialized once outside the timed loop. Each iteration
// then factorizes two n x n matrices for the implicit operator, versus one
// n^2 x n^2 matrix for the dense baseline.
static void BM_KroneckerSolveImplicit(benchmark::State& state) {
  const Index n = state.range(0);
  Mat A = Mat::Random(n, n) + 2.0 * double(n) * Mat::Identity(n, n);
  Mat B = Mat::Random(n, n) + 2.0 * double(n) * Mat::Identity(n, n);
  KroneckerOperator<Mat, Mat> K(A, B);
  Vec b = Vec::Random(n * n), x(n * n);
  for (auto _ : state) {
    x = K.solve(b);
    benchmark::DoNotOptimize(x.data());
  }
}
BENCHMARK(BM_KroneckerSolveImplicit)->Arg(8)->Arg(16)->Arg(32);

static void BM_KroneckerSolveDense(benchmark::State& state) {
  const Index n = state.range(0);
  Mat A = Mat::Random(n, n) + 2.0 * double(n) * Mat::Identity(n, n);
  Mat B = Mat::Random(n, n) + 2.0 * double(n) * Mat::Identity(n, n);
  Mat dense = KroneckerOperator<Mat, Mat>(A, B);  // materialized once, outside the loop
  Vec b = Vec::Random(n * n), x(n * n);
  for (auto _ : state) {
    x = dense.partialPivLu().solve(b);
    benchmark::DoNotOptimize(x.data());
  }
}
BENCHMARK(BM_KroneckerSolveDense)->Arg(8)->Arg(16)->Arg(32);
