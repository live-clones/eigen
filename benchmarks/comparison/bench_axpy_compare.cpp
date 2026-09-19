// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Cross-library comparison benchmarks for AXPY, y := y + alpha*x.
//
// Two streams in and one out per element with one multiply-add between them:
// beyond L2 this is a bandwidth measurement like GEMV, and the tiny end
// measures call overhead. Both arms are charged the same 2n.

#include <Eigen/Core>
#include <complex>
#include <string>

#include "benchmarks/bench_common.h"
#include "benchmarks/comparison/bench_compare.h"

#ifdef EIGEN_BENCH_REFERENCE_ARM
// Declared rather than included, so the build needs the reference library but not
// its development headers. The integer width is eigen_bench::BlasInt; see the
// static_assert in bench_compare.h.
extern "C" {
void saxpy_(const eigen_bench::BlasInt* n, const float* alpha, const float* x, const eigen_bench::BlasInt* incx,
            float* y, const eigen_bench::BlasInt* incy);
void daxpy_(const eigen_bench::BlasInt* n, const double* alpha, const double* x, const eigen_bench::BlasInt* incx,
            double* y, const eigen_bench::BlasInt* incy);
void caxpy_(const eigen_bench::BlasInt* n, const std::complex<float>* alpha, const std::complex<float>* x,
            const eigen_bench::BlasInt* incx, std::complex<float>* y, const eigen_bench::BlasInt* incy);
void zaxpy_(const eigen_bench::BlasInt* n, const std::complex<double>* alpha, const std::complex<double>* x,
            const eigen_bench::BlasInt* incx, std::complex<double>* y, const eigen_bench::BlasInt* incy);
}
#endif

using Eigen::Index;
using eigen_bench::BlasInt;
using eigen_bench::ColVector;

// A scale factor that is neither 0 nor 1, the two values a library may
// shortcut, with an imaginary part where the scalar has one.
inline float nontrivialAlpha(float) { return 0.75f; }
inline double nontrivialAlpha(double) { return 0.75; }
inline std::complex<float> nontrivialAlpha(std::complex<float>) { return {0.75f, -0.5f}; }
inline std::complex<double> nontrivialAlpha(std::complex<double>) { return {0.75, -0.5}; }

struct EigenAxpyKernel {
  template <typename Scalar>
  void operator()(Scalar alpha, const ColVector<Scalar>& x, ColVector<Scalar>& y) const {
    y += alpha * x;
  }
};

#ifdef EIGEN_BENCH_REFERENCE_ARM
static void referenceAxpy(BlasInt n, float alpha, const float* x, float* y) {
  const BlasInt inc = 1;
  saxpy_(&n, &alpha, x, &inc, y, &inc);
}

static void referenceAxpy(BlasInt n, double alpha, const double* x, double* y) {
  const BlasInt inc = 1;
  daxpy_(&n, &alpha, x, &inc, y, &inc);
}

static void referenceAxpy(BlasInt n, std::complex<float> alpha, const std::complex<float>* x, std::complex<float>* y) {
  const BlasInt inc = 1;
  caxpy_(&n, &alpha, x, &inc, y, &inc);
}

static void referenceAxpy(BlasInt n, std::complex<double> alpha, const std::complex<double>* x,
                          std::complex<double>* y) {
  const BlasInt inc = 1;
  zaxpy_(&n, &alpha, x, &inc, y, &inc);
}

struct ReferenceAxpyKernel {
  template <typename Scalar>
  void operator()(Scalar alpha, const ColVector<Scalar>& x, ColVector<Scalar>& y) const {
    referenceAxpy(static_cast<BlasInt>(x.size()), alpha, x.data(), y.data());
  }
};
#endif

// Both arms run through this one driver so that allocation, fill, validation,
// the timed region and the counter cannot drift apart between them.
template <typename Scalar, typename Kernel>
static void runAxpy(benchmark::State& state, Kernel kernel) {
  const Index n = static_cast<Index>(state.range(0));

  if (eigen_bench::skipIfDimsExceedBlasInt(state, n)) return;

  // x, y and the expected vector of the check.
  const double operand_bytes = 3.0 * static_cast<double>(sizeof(Scalar)) * static_cast<double>(n);
  if (eigen_bench::skipIfOverMemoryBudget(state, operand_bytes)) return;

  const Scalar alpha = nontrivialAlpha(Scalar());
  const ColVector<Scalar> x = ColVector<Scalar>::Random(n);
  ColVector<Scalar> y = ColVector<Scalar>::Random(n);

  static eigen_bench::ValidatedShapes<Index> validated;
  if (!validated.contains(n)) {
    const ColVector<Scalar> expected = y + alpha * x;
    kernel(alpha, x, y);
    if (!eigen_bench::agreesWithEigen(expected, y, 1)) {
      state.SkipWithError("axpy result disagrees with Eigen at n:" + std::to_string(n));
      return;
    }
    validated.insert(n);
  }

  // y accumulates across iterations and is never reset: a reset would be a
  // second pass over the same memory inside the timed region.
  for (auto _ : state) {
    kernel(alpha, x, y);
    benchmark::DoNotOptimize(y.data());
    benchmark::ClobberMemory();
  }

  eigen_bench::setFlopRate(state, eigen_bench::axpyFlops<Scalar>(n));
}

template <typename Scalar>
static void BM_AxpyEigen(benchmark::State& state) {
  runAxpy<Scalar>(state, EigenAxpyKernel());
}

// Defined only with a vendor linked; REGISTER_COMPARISON_POINT then drops its
// reference-arm argument, so the name is never substituted.
#ifdef EIGEN_BENCH_REFERENCE_ARM
template <typename Scalar>
static void BM_AxpyReference(benchmark::State& state) {
  runAxpy<Scalar>(state, ReferenceAxpyKernel());
}
#endif

// The whole grid: run.py narrows it with --benchmark_filter, so a length absent
// here is out of reach. The streaming lengths carry the operands through L2 and
// L3 into DRAM: 16 Mi doubles are 128 MiB per vector.
// clang-format off
#define AXPY_DIM_NAMES {"n"}

#define AXPY_POINT(...)                                                                                            \
  REGISTER_COMPARISON_POINT(AXPY, f32, float,              BM_AxpyEigen, BM_AxpyReference, AXPY_DIM_NAMES, __VA_ARGS__) \
  REGISTER_COMPARISON_POINT(AXPY, f64, double,             BM_AxpyEigen, BM_AxpyReference, AXPY_DIM_NAMES, __VA_ARGS__) \
  REGISTER_COMPARISON_POINT(AXPY, c32, eigen_bench::c32_t, BM_AxpyEigen, BM_AxpyReference, AXPY_DIM_NAMES, __VA_ARGS__) \
  REGISTER_COMPARISON_POINT(AXPY, c64, eigen_bench::c64_t, BM_AxpyEigen, BM_AxpyReference, AXPY_DIM_NAMES, __VA_ARGS__)

#define AXPY_SIZES(POINT) \
  /* fixed_tiny */ \
  POINT(2) POINT(3) POINT(4) POINT(6) POINT(8) POINT(12) POINT(16) \
  /* small */ \
  POINT(24) POINT(32) POINT(48) POINT(64) POINT(96) POINT(128) \
  /* medium */ \
  POINT(192) POINT(256) POINT(384) POINT(512) POINT(768) POINT(1024) \
  /* large */ \
  POINT(1536) POINT(2048) POINT(3072) POINT(4096) POINT(6144) POINT(8192) POINT(12288) POINT(16384) \
  /* streaming */ \
  POINT(32768) POINT(65536) POINT(131072) POINT(262144) POINT(524288) POINT(1048576) \
  POINT(2097152) POINT(4194304) POINT(8388608) POINT(16777216) \
  /* aliasing */ \
  POINT(100) POINT(1000) POINT(1001) POINT(10000) POINT(100000) POINT(1000000)

AXPY_SIZES(AXPY_POINT)

#undef AXPY_SIZES
#undef AXPY_POINT
#undef AXPY_DIM_NAMES
// clang-format on

EIGEN_BENCH_COMPARISON_MAIN();
