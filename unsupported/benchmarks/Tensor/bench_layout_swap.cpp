// Benchmarks for Eigen TensorLayoutSwap.

#include <benchmark/benchmark.h>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;

typedef float Scalar;

static void BM_LayoutSwap_2D(benchmark::State& state) {
  const int M = state.range(0);
  const int N = state.range(1);

  Tensor<Scalar, 2, ColMajor> A(M, N);
  A.setRandom();

  for (auto _ : state) {
    Tensor<Scalar, 2, RowMajor> B = A.swap_layout();
    benchmark::DoNotOptimize(B.data());
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(M) * N * sizeof(Scalar));
}

static void BM_LayoutSwap_3D(benchmark::State& state) {
  const int D0 = state.range(0);
  const int D1 = state.range(1);
  const int D2 = state.range(2);

  Tensor<Scalar, 3, ColMajor> A(D0, D1, D2);
  A.setRandom();

  for (auto _ : state) {
    Tensor<Scalar, 3, RowMajor> B = A.swap_layout();
    benchmark::DoNotOptimize(B.data());
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(D0) * D1 * D2 * sizeof(Scalar));
}

// Composing swap_layout with a coefficient-wise op forces evaluation through
// the executor and exercises any subsequent block consumers.
static void BM_LayoutSwap_Composed(benchmark::State& state) {
  const int M = state.range(0);
  const int N = state.range(1);

  Tensor<Scalar, 2, ColMajor> A(M, N);
  Tensor<Scalar, 2, ColMajor> B(M, N);
  A.setRandom();
  B.setRandom();

  for (auto _ : state) {
    Tensor<Scalar, 2, RowMajor> C = (A + B).swap_layout();
    benchmark::DoNotOptimize(C.data());
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(M) * N * sizeof(Scalar));
}

static void LayoutSwapSizes(::benchmark::Benchmark* b) {
  for (int size : {64, 256, 1024}) {
    b->Args({size, size});
  }
}

static void LayoutSwap3DSizes(::benchmark::Benchmark* b) {
  b->Args({32, 32, 32});
  b->Args({64, 64, 64});
  b->Args({128, 128, 128});
}

BENCHMARK(BM_LayoutSwap_2D)->Apply(LayoutSwapSizes);
BENCHMARK(BM_LayoutSwap_3D)->Apply(LayoutSwap3DSizes);
BENCHMARK(BM_LayoutSwap_Composed)->Apply(LayoutSwapSizes);
