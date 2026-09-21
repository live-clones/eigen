// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>

using namespace Eigen;

static void BM_VecAdd(benchmark::State& state) {
  int size = state.range(0);
  VectorXf a = VectorXf::Random(size);
  VectorXf b = VectorXf::Random(size);
  for (auto _ : state) {
    a = a + b;
    benchmark::DoNotOptimize(a.data());
  }
  state.SetBytesProcessed(state.iterations() * size * sizeof(float) * 3);
}
BENCHMARK(BM_VecAdd)->RangeMultiplier(4)->Range(64, 1 << 20);

static void BM_MatAdd(benchmark::State& state) {
  int n = state.range(0);
  MatrixXf a = MatrixXf::Random(n, n);
  MatrixXf b = MatrixXf::Random(n, n);
  for (auto _ : state) {
    a = a + b;
    benchmark::DoNotOptimize(a.data());
  }
  state.SetBytesProcessed(state.iterations() * n * n * sizeof(float) * 3);
}
BENCHMARK(BM_MatAdd)->RangeMultiplier(2)->Range(8, 512);

template <typename Scalar>
static void BM_Axpy(benchmark::State& state) {
  const Index size = state.range(0);
  using Vec = Matrix<Scalar, Dynamic, 1>;
  Vec x = Vec::Random(size), y = Vec::Random(size);
  Vec actual = y;
  actual += Scalar(0.75) * x;
  for (Index i = 0; i < size; ++i) {
    const long double expected = static_cast<long double>(y[i]) + Scalar(0.75) * static_cast<long double>(x[i]);
    const long double scale =
        numext::abs(static_cast<long double>(y[i])) + Scalar(0.75) * numext::abs(static_cast<long double>(x[i]));
    if (!(numext::abs(static_cast<long double>(actual[i]) - expected) <= 4 * NumTraits<Scalar>::epsilon() * scale)) {
      state.SkipWithError("AXPY differs from the scalar reference");
      return;
    }
  }
  for (auto _ : state) {
    benchmark::ClobberMemory();
    y += Scalar(0.75) * x;
    benchmark::DoNotOptimize(y.data());
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(state.iterations() * size * sizeof(Scalar) * 3);
}
BENCHMARK(BM_Axpy<float>)
    ->RangeMultiplier(4)
    ->Range(64, 1 << 24)
    ->Arg(511)
    ->Arg(512)
    ->Arg(513)
    ->Arg(1023)
    ->Arg(1025)
    ->Arg(524288)
    ->Arg(2097152)
    ->UseRealTime();
BENCHMARK(BM_Axpy<double>)
    ->RangeMultiplier(4)
    ->Range(64, 1 << 24)
    ->Arg(511)
    ->Arg(512)
    ->Arg(513)
    ->Arg(1023)
    ->Arg(1025)
    ->Arg(524288)
    ->Arg(2097152)
    ->UseRealTime();
