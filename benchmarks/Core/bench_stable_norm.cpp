// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Benchmarks ordinary and stable vector normalization across representative sizes and scales.
// The large and small inputs have finite, nonzero mathematical norms, but their squared norms
// overflow and underflow, respectively, when accumulated without scaling.

#include <benchmark/benchmark.h>
#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <vector>

using namespace Eigen;

namespace {

enum InputScale { kOrdinary = 0, kLarge = 1, kSmall = 2 };

const char* inputScaleName(InputScale scale) {
  switch (scale) {
    case kOrdinary:
      return "ordinary";
    case kLarge:
      return "large_finite_norm";
    case kSmall:
      return "small_nonzero_norm";
  }
  return "unknown";
}

template <typename Scalar>
struct coefficient_maker {
  typedef typename NumTraits<Scalar>::Real RealScalar;

  static Scalar run(RealScalar magnitude, Index index) {
    return index % 2 == 0 ? Scalar(magnitude) : Scalar(-magnitude);
  }
};

template <typename RealScalar>
struct coefficient_maker<std::complex<RealScalar>> {
  static std::complex<RealScalar> run(RealScalar magnitude, Index index) {
    const RealScalar real = index % 2 == 0 ? magnitude : -magnitude;
    const RealScalar imag = index % 4 < 2 ? magnitude : -magnitude;
    return std::complex<RealScalar>(real, imag);
  }
};

template <typename Scalar>
Matrix<Scalar, Dynamic, 1> makeInput(Index size, InputScale scale) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, 1> Vector;

  RealScalar magnitude = RealScalar(0.5);
  if (scale == kLarge) {
    // The factor 16 also leaves enough headroom for two-component complex coefficients.
    const double highest = static_cast<double>((std::numeric_limits<RealScalar>::max)());
    magnitude = RealScalar(highest / (16.0 * std::sqrt(static_cast<double>(size))));
  } else if (scale == kSmall) {
    magnitude = (std::numeric_limits<RealScalar>::min)();
  }

  Vector input(size);
  for (Index i = 0; i < size; ++i) input(i) = coefficient_maker<Scalar>::run(magnitude, i);
  return input;
}

template <typename Scalar>
void setNormCounters(benchmark::State& state, Index size) {
  state.SetBytesProcessed(state.iterations() * size * static_cast<int64_t>(sizeof(Scalar)));
}

template <typename Scalar>
void setNormalizeCounters(benchmark::State& state, Index size) {
  // Report one logical read and write per coefficient, independent of the implementation's pass count.
  state.SetBytesProcessed(state.iterations() * size * static_cast<int64_t>(2 * sizeof(Scalar)));
}

Index normalizationBatchSize(Index size) {
  const Index kMaxBatchSize = 128;
  const Index kCoefficientBudget = 65536;
  return (std::max)(Index(1), (std::min)(kMaxBatchSize, kCoefficientBudget / size));
}

template <typename Scalar>
static void BM_Norm(benchmark::State& state) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  const Index size = state.range(0);
  const InputScale scale = static_cast<InputScale>(state.range(1));
  const Matrix<Scalar, Dynamic, 1> input = makeInput<Scalar>(size, scale);
  state.SetLabel(inputScaleName(scale));

  for (auto _ : state) {
    RealScalar result = input.norm();
    benchmark::DoNotOptimize(result);
  }
  setNormCounters<Scalar>(state, size);
}

template <typename Scalar>
static void BM_StableNorm(benchmark::State& state) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  const Index size = state.range(0);
  const InputScale scale = static_cast<InputScale>(state.range(1));
  const Matrix<Scalar, Dynamic, 1> input = makeInput<Scalar>(size, scale);
  state.SetLabel(inputScaleName(scale));

  for (auto _ : state) {
    RealScalar result = input.stableNorm();
    benchmark::DoNotOptimize(result);
  }
  setNormCounters<Scalar>(state, size);
}

template <typename Scalar>
static void BM_BlueNorm(benchmark::State& state) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  const Index size = state.range(0);
  const InputScale scale = static_cast<InputScale>(state.range(1));
  const Matrix<Scalar, Dynamic, 1> input = makeInput<Scalar>(size, scale);
  state.SetLabel(inputScaleName(scale));

  for (auto _ : state) {
    RealScalar result = input.blueNorm();
    benchmark::DoNotOptimize(result);
  }
  setNormCounters<Scalar>(state, size);
}

template <typename Scalar>
static void BM_HypotNorm(benchmark::State& state) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  const Index size = state.range(0);
  const InputScale scale = static_cast<InputScale>(state.range(1));
  const Matrix<Scalar, Dynamic, 1> input = makeInput<Scalar>(size, scale);
  state.SetLabel(inputScaleName(scale));

  for (auto _ : state) {
    RealScalar result = input.hypotNorm();
    benchmark::DoNotOptimize(result);
  }
  setNormCounters<Scalar>(state, size);
}

template <typename Scalar>
static void BM_Normalize(benchmark::State& state) {
  const Index size = state.range(0);
  const InputScale scale = static_cast<InputScale>(state.range(1));
  const Matrix<Scalar, Dynamic, 1> input = makeInput<Scalar>(size, scale);
  const Index batch_size = normalizationBatchSize(size);
  std::vector<Matrix<Scalar, Dynamic, 1>> results(static_cast<std::size_t>(batch_size), input);
  state.SetLabel(inputScaleName(scale));

  while (state.KeepRunningBatch(batch_size)) {
    state.PauseTiming();
    for (Index i = 0; i < batch_size; ++i) results[static_cast<std::size_t>(i)] = input;
    state.ResumeTiming();
    for (Index i = 0; i < batch_size; ++i) {
      results[static_cast<std::size_t>(i)].normalize();
      benchmark::DoNotOptimize(results[static_cast<std::size_t>(i)].data());
    }
    benchmark::ClobberMemory();
  }
  setNormalizeCounters<Scalar>(state, size);
}

template <typename Scalar>
static void BM_StableNormalize(benchmark::State& state) {
  const Index size = state.range(0);
  const InputScale scale = static_cast<InputScale>(state.range(1));
  const Matrix<Scalar, Dynamic, 1> input = makeInput<Scalar>(size, scale);
  const Index batch_size = normalizationBatchSize(size);
  std::vector<Matrix<Scalar, Dynamic, 1>> results(static_cast<std::size_t>(batch_size), input);
  state.SetLabel(inputScaleName(scale));

  while (state.KeepRunningBatch(batch_size)) {
    state.PauseTiming();
    for (Index i = 0; i < batch_size; ++i) results[static_cast<std::size_t>(i)] = input;
    state.ResumeTiming();
    for (Index i = 0; i < batch_size; ++i) {
      results[static_cast<std::size_t>(i)].stableNormalize();
      benchmark::DoNotOptimize(results[static_cast<std::size_t>(i)].data());
    }
    benchmark::ClobberMemory();
  }
  setNormalizeCounters<Scalar>(state, size);
}

// clang-format off
#define NORM_ARGS ->ArgsProduct({{64, 4096, 65536}, {kOrdinary, kLarge, kSmall}})->ArgNames({"size", "scale"})
BENCHMARK(BM_Norm<float>) NORM_ARGS ->Name("Norm_float");
BENCHMARK(BM_Norm<double>) NORM_ARGS ->Name("Norm_double");
BENCHMARK(BM_Norm<half>) NORM_ARGS ->Name("Norm_half");
BENCHMARK(BM_Norm<bfloat16>) NORM_ARGS ->Name("Norm_bfloat16");
BENCHMARK(BM_Norm<std::complex<float>>) NORM_ARGS ->Name("Norm_cfloat");
BENCHMARK(BM_Norm<std::complex<double>>) NORM_ARGS ->Name("Norm_cdouble");
BENCHMARK(BM_StableNorm<float>) NORM_ARGS ->Name("StableNorm_float");
BENCHMARK(BM_StableNorm<double>) NORM_ARGS ->Name("StableNorm_double");
BENCHMARK(BM_StableNorm<half>) NORM_ARGS ->Name("StableNorm_half");
BENCHMARK(BM_StableNorm<bfloat16>) NORM_ARGS ->Name("StableNorm_bfloat16");
BENCHMARK(BM_StableNorm<std::complex<float>>) NORM_ARGS ->Name("StableNorm_cfloat");
BENCHMARK(BM_StableNorm<std::complex<double>>) NORM_ARGS ->Name("StableNorm_cdouble");
BENCHMARK(BM_BlueNorm<float>) NORM_ARGS ->Name("BlueNorm_float");
BENCHMARK(BM_BlueNorm<double>) NORM_ARGS ->Name("BlueNorm_double");
BENCHMARK(BM_BlueNorm<half>) NORM_ARGS ->Name("BlueNorm_half");
BENCHMARK(BM_BlueNorm<bfloat16>) NORM_ARGS ->Name("BlueNorm_bfloat16");
BENCHMARK(BM_BlueNorm<std::complex<float>>) NORM_ARGS ->Name("BlueNorm_cfloat");
BENCHMARK(BM_BlueNorm<std::complex<double>>) NORM_ARGS ->Name("BlueNorm_cdouble");
BENCHMARK(BM_HypotNorm<float>) NORM_ARGS ->Name("HypotNorm_float");
BENCHMARK(BM_HypotNorm<double>) NORM_ARGS ->Name("HypotNorm_double");
BENCHMARK(BM_HypotNorm<half>) NORM_ARGS ->Name("HypotNorm_half");
BENCHMARK(BM_HypotNorm<bfloat16>) NORM_ARGS ->Name("HypotNorm_bfloat16");
BENCHMARK(BM_HypotNorm<std::complex<float>>) NORM_ARGS ->Name("HypotNorm_cfloat");
BENCHMARK(BM_HypotNorm<std::complex<double>>) NORM_ARGS ->Name("HypotNorm_cdouble");
BENCHMARK(BM_Normalize<float>) NORM_ARGS ->Name("Normalize_float");
BENCHMARK(BM_Normalize<double>) NORM_ARGS ->Name("Normalize_double");
BENCHMARK(BM_Normalize<half>) NORM_ARGS ->Name("Normalize_half");
BENCHMARK(BM_Normalize<bfloat16>) NORM_ARGS ->Name("Normalize_bfloat16");
BENCHMARK(BM_Normalize<std::complex<float>>) NORM_ARGS ->Name("Normalize_cfloat");
BENCHMARK(BM_Normalize<std::complex<double>>) NORM_ARGS ->Name("Normalize_cdouble");
BENCHMARK(BM_StableNormalize<float>) NORM_ARGS ->Name("StableNormalize_float");
BENCHMARK(BM_StableNormalize<double>) NORM_ARGS ->Name("StableNormalize_double");
BENCHMARK(BM_StableNormalize<half>) NORM_ARGS ->Name("StableNormalize_half");
BENCHMARK(BM_StableNormalize<bfloat16>) NORM_ARGS ->Name("StableNormalize_bfloat16");
BENCHMARK(BM_StableNormalize<std::complex<float>>) NORM_ARGS ->Name("StableNormalize_cfloat");
BENCHMARK(BM_StableNormalize<std::complex<double>>) NORM_ARGS ->Name("StableNormalize_cdouble");
#undef NORM_ARGS
// clang-format on

}  // namespace
