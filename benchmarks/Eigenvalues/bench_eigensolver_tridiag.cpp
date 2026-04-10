#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include "../../test/tridiag_test_matrices.h"

using namespace Eigen;

#ifndef SCALAR
#define SCALAR double
#endif

typedef SCALAR Scalar;
typedef Matrix<Scalar, Dynamic, Dynamic> MatrixType;
typedef Matrix<Scalar, Dynamic, 1> VectorType;

// Generator function type — must be a simple two-arg function pointer.
typedef void (*GenFunc)(VectorType&, VectorType&);

// Wrappers for generators with default arguments (can't take their address directly).
void gen_graded(VectorType& d, VectorType& e) { test::tridiag_graded(d, e); }
void gen_kahan(VectorType& d, VectorType& e) { test::tridiag_kahan(d, e); }
void gen_constant(VectorType& d, VectorType& e) { test::tridiag_constant(d, e); }
void gen_geom_diag(VectorType& d, VectorType& e) { test::tridiag_geometric_diagonal(d, e); }
void gen_geom_offdiag(VectorType& d, VectorType& e) { test::tridiag_geometric_offdiag(d, e); }

// Benchmark computeFromTridiagonal on a specific structured matrix.
template <GenFunc gen>
static void BM_Tridiag(benchmark::State& state) {
  int n = state.range(0);
  VectorType diag(n), offdiag(n > 1 ? n - 1 : 0);
  gen(diag, offdiag);
  SelfAdjointEigenSolver<MatrixType> eig(n);
  for (auto _ : state) {
    eig.computeFromTridiagonal(diag, offdiag, ComputeEigenvectors);
    benchmark::DoNotOptimize(eig.eigenvalues().data());
    benchmark::ClobberMemory();
  }
}

template <GenFunc gen>
static void BM_Tridiag_ValsOnly(benchmark::State& state) {
  int n = state.range(0);
  VectorType diag(n), offdiag(n > 1 ? n - 1 : 0);
  gen(diag, offdiag);
  SelfAdjointEigenSolver<MatrixType> eig(n);
  for (auto _ : state) {
    eig.computeFromTridiagonal(diag, offdiag, EigenvaluesOnly);
    benchmark::DoNotOptimize(eig.eigenvalues().data());
    benchmark::ClobberMemory();
  }
}

// Also benchmark full compute() on dense matrices built from tridiagonal.
template <GenFunc gen>
static void BM_Dense(benchmark::State& state) {
  int n = state.range(0);
  VectorType diag(n), offdiag(n > 1 ? n - 1 : 0);
  gen(diag, offdiag);
  // Build dense symmetric tridiagonal.
  MatrixType T = MatrixType::Zero(n, n);
  T.diagonal() = diag;
  if (n > 1) {
    T.template diagonal<1>() = offdiag;
    T.template diagonal<-1>() = offdiag;
  }
  SelfAdjointEigenSolver<MatrixType> eig(n);
  for (auto _ : state) {
    eig.compute(T);
    benchmark::DoNotOptimize(eig.eigenvalues().data());
    benchmark::ClobberMemory();
  }
}

#define SIZES ->RangeMultiplier(2)->Range(4, 256)

// Matrices where deflation matters most:

// Wilkinson: clustered eigenvalue pairs that stress deflation.
BENCHMARK(BM_Tridiag<test::tridiag_wilkinson>) SIZES;
BENCHMARK(BM_Dense<test::tridiag_wilkinson>) SIZES;

// Glued blocks: two identity blocks coupled by eps — should deflate immediately.
BENCHMARK(BM_Tridiag<test::tridiag_glued>) SIZES;
BENCHMARK(BM_Dense<test::tridiag_glued>) SIZES;

// Clustered eigenvalues: d_i = 1 + i*eps.
BENCHMARK(BM_Tridiag<test::tridiag_clustered>) SIZES;

// 1-2-1 Toeplitz: uniform structure, baseline for QR convergence.
BENCHMARK(BM_Tridiag<test::tridiag_1_2_1>) SIZES;
BENCHMARK(BM_Dense<test::tridiag_1_2_1>) SIZES;

// Graded: d_i = 10^(-i), stress deflation on decaying entries.
BENCHMARK(BM_Tridiag<gen_graded>) SIZES;

// Nearly diagonal: offdiag = O(eps), should deflate almost immediately.
BENCHMARK(BM_Tridiag<test::tridiag_nearly_diagonal>) SIZES;

// Clement: known eigenvalues, interesting structure.
BENCHMARK(BM_Tridiag<test::tridiag_clement>) SIZES;

// Arrowhead: decaying offdiagonal.
BENCHMARK(BM_Tridiag<test::tridiag_arrowhead>) SIZES;

// Random dense (for comparison baseline).
static void BM_Random_Dense(benchmark::State& state) {
  int n = state.range(0);
  MatrixType a = MatrixType::Random(n, n);
  MatrixType covMat = a * a.adjoint();
  SelfAdjointEigenSolver<MatrixType> eig(n);
  for (auto _ : state) {
    eig.compute(covMat);
    benchmark::DoNotOptimize(eig.eigenvalues().data());
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_Random_Dense) SIZES;
