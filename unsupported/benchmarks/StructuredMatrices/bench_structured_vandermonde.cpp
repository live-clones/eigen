// Benchmarks for the Vandermonde operator: the O(mn)-flops / O(m)-storage
// Horner product against the equivalent dense GEMV on the materialized matrix
// (same flop count, O(mn) storage), and the O(n^2) Björck-Pereyra square solve
// against a dense PartialPivLU factor-and-solve (O(n^3)). Both solve variants
// time the full pipeline from the stored representation to the solution; the
// dense one is additionally handed the materialized matrix for free (built
// outside the loop).
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <Eigen/LU>
#include <unsupported/Eigen/StructuredMatrices>

using namespace Eigen;

typedef Matrix<double, Dynamic, 1> Vec;
typedef Matrix<double, Dynamic, Dynamic> Mat;

// Chebyshev nodes in (-1, 1): deterministic, distinct (no Björck-Pereyra
// division by zero, no exact-zero LU pivot) and the well-conditioned real-node
// configuration, matching the accuracy tests.
static Vec chebyshevNodes(Index n) {
  Vec x(n);
  for (Index i = 0; i < n; ++i) x[i] = std::cos(EIGEN_PI * double(2 * i + 1) / double(2 * n));
  return x;
}

// --- Square product y = V * a: Horner rule on the nodes ---
static void BM_VandermondeProduct(benchmark::State& state) {
  const Index n = state.range(0);
  Vec x = chebyshevNodes(n), a = Vec::Random(n), y(n);
  Vandermonde<double> V(x);
  for (auto _ : state) {
    y.noalias() = V * a;
    benchmark::DoNotOptimize(y.data());
  }
}
BENCHMARK(BM_VandermondeProduct)->Arg(64)->Arg(256)->Arg(1024)->Arg(4096);

// --- Square product y = dense * a: GEMV on the pre-materialized matrix ---
static void BM_VandermondeProductDense(benchmark::State& state) {
  const Index n = state.range(0);
  Vec x = chebyshevNodes(n), a = Vec::Random(n), y(n);
  Vandermonde<double> V(x);
  Mat dense = V;  // materialized once, outside the timed loop
  for (auto _ : state) {
    y.noalias() = dense * a;
    benchmark::DoNotOptimize(y.data());
  }
}
BENCHMARK(BM_VandermondeProductDense)->Arg(64)->Arg(256)->Arg(1024)->Arg(4096);

// --- Square solve V * a = f: Björck-Pereyra divided-difference recurrences ---
static void BM_BjorckPereyraSolve(benchmark::State& state) {
  const Index n = state.range(0);
  Vec x = chebyshevNodes(n), f = Vec::Random(n), a(n);
  Vandermonde<double> V(x);
  for (auto _ : state) {
    BjorckPereyra<double> bp(V);  // compute() is the O(n^2) duplicate scan
    a = bp.solve(f);
    benchmark::DoNotOptimize(a.data());
  }
}
BENCHMARK(BM_BjorckPereyraSolve)->Arg(16)->Arg(64)->Arg(256)->Arg(1024);

// --- Square solve dense * a = f: PartialPivLU on the pre-materialized matrix ---
static void BM_VandermondeSolveDense(benchmark::State& state) {
  const Index n = state.range(0);
  Vec x = chebyshevNodes(n), f = Vec::Random(n), a(n);
  Vandermonde<double> V(x);
  Mat dense = V;  // materialized once, outside the timed loop
  for (auto _ : state) {
    PartialPivLU<Mat> lu(dense);  // the O(n^3) elimination
    a = lu.solve(f);
    benchmark::DoNotOptimize(a.data());
  }
}
BENCHMARK(BM_VandermondeSolveDense)->Arg(16)->Arg(64)->Arg(256)->Arg(1024);
