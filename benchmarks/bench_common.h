// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Shared flop-count and counter helpers for the benchmark tree.
//
// Cross-library and cross-benchmark numbers are only comparable when every
// benchmark reports the same counter computed the same way, so the flop
// formulas live here rather than being restated per file. Each helper
// reproduces the value the corresponding benchmark already publishes; see the
// note on trmvFlops for the one place where that means preserving a known
// overcount.

#ifndef EIGEN_BENCHMARKS_BENCH_COMMON_H
#define EIGEN_BENCHMARKS_BENCH_COMMON_H

#include <benchmark/benchmark.h>

#include <Eigen/Core>

namespace eigen_bench {

using Eigen::Index;

// Multiplier turning a REAL-arithmetic flop count into the same count for
// Scalar: 1 for real, 4 for complex.
template <typename Scalar>
constexpr double complexFactor() {
  return Eigen::NumTraits<Scalar>::IsComplex ? 4.0 : 1.0;
}

// Multiplier turning a count of multiply-add PAIRS into scalar flops: 2 for
// real, 8 for complex (4 real multiplies + 4 real adds per complex fma).
template <typename Scalar>
constexpr double flopScale() {
  return 2.0 * complexFactor<Scalar>();
}

// Every helper forms its products in double before applying the scale, so a
// cubic term in a four-digit dimension cannot overflow Index.

// ---- Level 1 -------------------------------------------------------------

template <typename Scalar>
double dotFlops(Index n) {
  return flopScale<Scalar>() * static_cast<double>(n);
}

// ---- Level 2 -------------------------------------------------------------

template <typename Scalar>
double gemvFlops(Index m, Index n) {
  return flopScale<Scalar>() * static_cast<double>(m) * static_cast<double>(n);
}

template <typename Scalar>
double symvFlops(Index n) {
  const double dn = static_cast<double>(n);
  return flopScale<Scalar>() * dn * dn;
}

// The exact count for a triangular matrix-vector product is
// flopScale * n * (n + 1) / 2; Core/bench_trmv.cpp has always reported
// flopScale * n * n, roughly 2x that. The helper preserves the published value
// so that adopting it moves no number; correcting it is a separate change.
template <typename Scalar>
double trmvFlops(Index n) {
  const double dn = static_cast<double>(n);
  return flopScale<Scalar>() * dn * dn;
}

template <typename Scalar>
double syrFlops(Index n) {
  const double dn = static_cast<double>(n);
  return flopScale<Scalar>() * dn * (dn + 1.0) / 2.0;
}

template <typename Scalar>
double syr2Flops(Index n) {
  const double dn = static_cast<double>(n);
  return flopScale<Scalar>() * dn * (dn + 1.0);
}

// ---- Level 3 -------------------------------------------------------------

template <typename Scalar>
double gemmFlops(Index m, Index n, Index k) {
  return flopScale<Scalar>() * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
}

// ---- Factorizations and decompositions -----------------------------------

// Closed form of the summation loop in Cholesky/bench_cholesky.cpp and
// Cholesky/bench_bunchkaufman.cpp, sum_j 2*((n-1-j)*j + (n-1-j) + j):
//   n*(n-1)*(n-2)/3 + 2*n*(n-1)
template <typename Scalar>
double symmetricFactorizationFlops(Index n) {
  const double dn = static_cast<double>(n);
  return complexFactor<Scalar>() * (dn * (dn - 1.0) * (dn - 2.0) / 3.0 + 2.0 * dn * (dn - 1.0));
}

// LAWN 41 (Table I, DGETRF): m*n^2 - n^3/3, i.e. 2*n^3/3 for a square LU.
template <typename Scalar>
double getrfFlops(Index m, Index n) {
  const double dm = static_cast<double>(m);
  const double dn = static_cast<double>(n);
  return complexFactor<Scalar>() * (dm * dn * dn - dn * dn * dn / 3.0);
}

// ---- Counters ------------------------------------------------------------

// The counter name every benchmark in this tree publishes.
inline constexpr const char* kFlopCounterName = "GFLOPS";

// `flops` is the flop count of ONE benchmark iteration. The value Google
// Benchmark writes to JSON is flops per second, not gigaflops: kIs1000 selects
// the base for the console's k/M/G suffix and does not scale the reported
// number. Consumers divide by 1e9 themselves, exactly once.
inline void setFlopRate(benchmark::State& state, double flops) {
  state.counters[kFlopCounterName] =
      benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

}  // namespace eigen_bench

#endif  // EIGEN_BENCHMARKS_BENCH_COMMON_H
