// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0
//
// Benchmarks for small fixed-size A += B*C and A -= B*C.
// These use FMA chains (vfmadd / vfnmadd) when called via .noalias(),
// avoiding the separate multiply + add/subtract that the aliasing path incurs.

#include <benchmark/benchmark.h>
#include <complex>
#include <Eigen/Core>

using namespace Eigen;

// ---------------------------------------------------------------------------
// noalias += : FMA chain (vfmadd throughout, no separate vaddpd)
// ---------------------------------------------------------------------------
template <typename Scalar, int N>
static void BM_NoaliasAddTo(benchmark::State& state) {
  Matrix<Scalar, N, N> A, B, C;
  A.setRandom();
  B.setRandom();
  C.setRandom();
  for (auto _ : state) {
    A.noalias() += B * C;
    benchmark::DoNotOptimize(A.data());
    benchmark::ClobberMemory();
  }
}

// ---------------------------------------------------------------------------
// noalias -= : FMA chain (vfnmadd throughout, no separate vmulpd+vsubpd)
// ---------------------------------------------------------------------------
template <typename Scalar, int N>
static void BM_NoaliasSubTo(benchmark::State& state) {
  Matrix<Scalar, N, N> A, B, C;
  A.setRandom();
  B.setRandom();
  C.setRandom();
  for (auto _ : state) {
    A.noalias() -= B * C;
    benchmark::DoNotOptimize(A.data());
    benchmark::ClobberMemory();
  }
}

// ---------------------------------------------------------------------------
// aliased += (default, A+=B*C): allocates a temporary — kept for comparison
// ---------------------------------------------------------------------------
template <typename Scalar, int N>
static void BM_AliasedAddTo(benchmark::State& state) {
  Matrix<Scalar, N, N> A, B, C;
  A.setRandom();
  B.setRandom();
  C.setRandom();
  for (auto _ : state) {
    A += B * C;
    benchmark::DoNotOptimize(A.data());
    benchmark::ClobberMemory();
  }
}

// ---------------------------------------------------------------------------
// aliased -= (default, A-=B*C): allocates a temporary — kept for comparison
// ---------------------------------------------------------------------------
template <typename Scalar, int N>
static void BM_AliasedSubTo(benchmark::State& state) {
  Matrix<Scalar, N, N> A, B, C;
  A.setRandom();
  B.setRandom();
  C.setRandom();
  for (auto _ : state) {
    A -= B * C;
    benchmark::DoNotOptimize(A.data());
    benchmark::ClobberMemory();
  }
}

// ---------------------------------------------------------------------------
// Row-major variants
// ---------------------------------------------------------------------------
template <typename Scalar, int N>
static void BM_NoaliasAddTo_RowMajor(benchmark::State& state) {
  Matrix<Scalar, N, N, RowMajor> A, B, C;
  A.setRandom();
  B.setRandom();
  C.setRandom();
  for (auto _ : state) {
    A.noalias() += B * C;
    benchmark::DoNotOptimize(A.data());
    benchmark::ClobberMemory();
  }
}

template <typename Scalar, int N>
static void BM_NoaliasSubTo_RowMajor(benchmark::State& state) {
  Matrix<Scalar, N, N, RowMajor> A, B, C;
  A.setRandom();
  B.setRandom();
  C.setRandom();
  for (auto _ : state) {
    A.noalias() -= B * C;
    benchmark::DoNotOptimize(A.data());
    benchmark::ClobberMemory();
  }
}

// 2x2
BENCHMARK_TEMPLATE(BM_NoaliasAddTo,        double, 2)->Name("noalias_add/ColMajor/2x2/double");
BENCHMARK_TEMPLATE(BM_NoaliasSubTo,        double, 2)->Name("noalias_sub/ColMajor/2x2/double");
BENCHMARK_TEMPLATE(BM_AliasedAddTo,        double, 2)->Name("aliased_add/ColMajor/2x2/double");
BENCHMARK_TEMPLATE(BM_AliasedSubTo,        double, 2)->Name("aliased_sub/ColMajor/2x2/double");

// 3x3
BENCHMARK_TEMPLATE(BM_NoaliasAddTo,        double, 3)->Name("noalias_add/ColMajor/3x3/double");
BENCHMARK_TEMPLATE(BM_NoaliasSubTo,        double, 3)->Name("noalias_sub/ColMajor/3x3/double");
BENCHMARK_TEMPLATE(BM_AliasedAddTo,        double, 3)->Name("aliased_add/ColMajor/3x3/double");
BENCHMARK_TEMPLATE(BM_AliasedSubTo,        double, 3)->Name("aliased_sub/ColMajor/3x3/double");

// 4x4
BENCHMARK_TEMPLATE(BM_NoaliasAddTo,        double, 4)->Name("noalias_add/ColMajor/4x4/double");
BENCHMARK_TEMPLATE(BM_NoaliasSubTo,        double, 4)->Name("noalias_sub/ColMajor/4x4/double");
BENCHMARK_TEMPLATE(BM_AliasedAddTo,        double, 4)->Name("aliased_add/ColMajor/4x4/double");
BENCHMARK_TEMPLATE(BM_AliasedSubTo,        double, 4)->Name("aliased_sub/ColMajor/4x4/double");

// 4x4 float
BENCHMARK_TEMPLATE(BM_NoaliasAddTo,        float, 4)->Name("noalias_add/ColMajor/4x4/float");
BENCHMARK_TEMPLATE(BM_NoaliasSubTo,        float, 4)->Name("noalias_sub/ColMajor/4x4/float");
BENCHMARK_TEMPLATE(BM_AliasedAddTo,        float, 4)->Name("aliased_add/ColMajor/4x4/float");
BENCHMARK_TEMPLATE(BM_AliasedSubTo,        float, 4)->Name("aliased_sub/ColMajor/4x4/float");

// 4x4 row-major
BENCHMARK_TEMPLATE(BM_NoaliasAddTo_RowMajor, double, 4)->Name("noalias_add/RowMajor/4x4/double");
BENCHMARK_TEMPLATE(BM_NoaliasSubTo_RowMajor, double, 4)->Name("noalias_sub/RowMajor/4x4/double");

// 8x8 (boundary of CoeffBasedProductMode)
BENCHMARK_TEMPLATE(BM_NoaliasAddTo,        double, 8)->Name("noalias_add/ColMajor/8x8/double");
BENCHMARK_TEMPLATE(BM_NoaliasSubTo,        double, 8)->Name("noalias_sub/ColMajor/8x8/double");
BENCHMARK_TEMPLATE(BM_AliasedAddTo,        double, 8)->Name("aliased_add/ColMajor/8x8/double");
BENCHMARK_TEMPLATE(BM_AliasedSubTo,        double, 8)->Name("aliased_sub/ColMajor/8x8/double");

// ---------------------------------------------------------------------------
// Complex types: std::complex<float> and std::complex<double>
// pmadd/pnmadd on complex packets cover both real and imaginary in one
// instruction (each complex multiply is 2 real FMAs), so the FMA benefit
// carries over to complex arithmetic.
// ---------------------------------------------------------------------------
using cf = std::complex<float>;
using cd = std::complex<double>;

// 2x2 complex<double>
BENCHMARK_TEMPLATE(BM_NoaliasAddTo,  cd, 2)->Name("noalias_add/ColMajor/2x2/cdouble");
BENCHMARK_TEMPLATE(BM_NoaliasSubTo,  cd, 2)->Name("noalias_sub/ColMajor/2x2/cdouble");
BENCHMARK_TEMPLATE(BM_AliasedAddTo,  cd, 2)->Name("aliased_add/ColMajor/2x2/cdouble");
BENCHMARK_TEMPLATE(BM_AliasedSubTo,  cd, 2)->Name("aliased_sub/ColMajor/2x2/cdouble");

// 3x3 complex<double>
BENCHMARK_TEMPLATE(BM_NoaliasAddTo,  cd, 3)->Name("noalias_add/ColMajor/3x3/cdouble");
BENCHMARK_TEMPLATE(BM_NoaliasSubTo,  cd, 3)->Name("noalias_sub/ColMajor/3x3/cdouble");
BENCHMARK_TEMPLATE(BM_AliasedAddTo,  cd, 3)->Name("aliased_add/ColMajor/3x3/cdouble");
BENCHMARK_TEMPLATE(BM_AliasedSubTo,  cd, 3)->Name("aliased_sub/ColMajor/3x3/cdouble");

// 4x4 complex<double>
BENCHMARK_TEMPLATE(BM_NoaliasAddTo,  cd, 4)->Name("noalias_add/ColMajor/4x4/cdouble");
BENCHMARK_TEMPLATE(BM_NoaliasSubTo,  cd, 4)->Name("noalias_sub/ColMajor/4x4/cdouble");
BENCHMARK_TEMPLATE(BM_AliasedAddTo,  cd, 4)->Name("aliased_add/ColMajor/4x4/cdouble");
BENCHMARK_TEMPLATE(BM_AliasedSubTo,  cd, 4)->Name("aliased_sub/ColMajor/4x4/cdouble");

// 4x4 complex<float>
BENCHMARK_TEMPLATE(BM_NoaliasAddTo,  cf, 4)->Name("noalias_add/ColMajor/4x4/cfloat");
BENCHMARK_TEMPLATE(BM_NoaliasSubTo,  cf, 4)->Name("noalias_sub/ColMajor/4x4/cfloat");
BENCHMARK_TEMPLATE(BM_AliasedAddTo,  cf, 4)->Name("aliased_add/ColMajor/4x4/cfloat");
BENCHMARK_TEMPLATE(BM_AliasedSubTo,  cf, 4)->Name("aliased_sub/ColMajor/4x4/cfloat");

// 4x4 complex<double> row-major
BENCHMARK_TEMPLATE(BM_NoaliasAddTo_RowMajor, cd, 4)->Name("noalias_add/RowMajor/4x4/cdouble");
BENCHMARK_TEMPLATE(BM_NoaliasSubTo_RowMajor, cd, 4)->Name("noalias_sub/RowMajor/4x4/cdouble");

BENCHMARK_MAIN();
