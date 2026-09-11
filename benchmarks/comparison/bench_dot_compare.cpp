// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Cross-library comparison benchmarks for DOT, the inner product x^T y.
//
// Two streams in and a reduction: memory bound beyond L2 like AXPY, with the
// reduction's dependency chain on top at the small end. Real scalars only. The
// Fortran convention for returning a COMPLEX function value is not fixed by the
// platform ABI (gfortran returns it in registers, the f2c convention passes a
// hidden result pointer), and OpenBLAS, oneMKL and the reference BLAS have each
// been built both ways, so a declaration of ?dotc here would be a guess.
//
// sdot itself is subject to a milder form of the same problem: under the f2c
// convention it returns a double. A library built that way fails the check
// below rather than being measured.

#include <Eigen/Core>
#include <string>

#include "benchmarks/bench_common.h"
#include "benchmarks/comparison/bench_compare.h"

#ifdef EIGEN_BENCH_REFERENCE_ARM
// Declared rather than included, so the build needs the reference library but not
// its development headers. The integer width is eigen_bench::BlasInt; see the
// static_assert in bench_compare.h.
extern "C" {
float sdot_(const eigen_bench::BlasInt* n, const float* x, const eigen_bench::BlasInt* incx, const float* y,
            const eigen_bench::BlasInt* incy);
double ddot_(const eigen_bench::BlasInt* n, const double* x, const eigen_bench::BlasInt* incx, const double* y,
             const eigen_bench::BlasInt* incy);
}
#endif

using Eigen::Index;
using eigen_bench::BlasInt;
using eigen_bench::ColVector;

struct EigenDotKernel {
  template <typename Scalar>
  Scalar operator()(const ColVector<Scalar>& x, const ColVector<Scalar>& y) const {
    return x.dot(y);
  }
};

#ifdef EIGEN_BENCH_REFERENCE_ARM
static float referenceDot(BlasInt n, const float* x, const float* y) {
  const BlasInt inc = 1;
  return sdot_(&n, x, &inc, y, &inc);
}

static double referenceDot(BlasInt n, const double* x, const double* y) {
  const BlasInt inc = 1;
  return ddot_(&n, x, &inc, y, &inc);
}

struct ReferenceDotKernel {
  template <typename Scalar>
  Scalar operator()(const ColVector<Scalar>& x, const ColVector<Scalar>& y) const {
    return referenceDot(static_cast<BlasInt>(x.size()), x.data(), y.data());
  }
};
#endif

// Both arms run through this one driver so that allocation, fill, validation,
// the timed region and the counter cannot drift apart between them.
template <typename Scalar, typename Kernel>
static void runDot(benchmark::State& state, Kernel kernel) {
  const Index n = static_cast<Index>(state.range(0));

  if (eigen_bench::skipIfDimsExceedBlasInt(state, n)) return;

  const double operand_bytes = 2.0 * static_cast<double>(sizeof(Scalar)) * static_cast<double>(n);
  if (eigen_bench::skipIfOverMemoryBudget(state, operand_bytes)) return;

  const ColVector<Scalar> x = ColVector<Scalar>::Random(n);
  const ColVector<Scalar> y = ColVector<Scalar>::Random(n);

  static eigen_bench::ValidatedShapes<Index> validated;
  if (!validated.contains(n)) {
    using Cell = Eigen::Matrix<Scalar, 1, 1>;
    const Cell expected = Cell::Constant(x.dot(y));
    const Cell actual = Cell::Constant(kernel(x, y));
    if (!eigen_bench::agreesWithEigen(expected, actual, n)) {
      state.SkipWithError("dot result disagrees with Eigen at n:" + std::to_string(n));
      return;
    }
    validated.insert(n);
  }

  for (auto _ : state) {
    Scalar result = kernel(x, y);
    benchmark::DoNotOptimize(result);
  }

  eigen_bench::setFlopRate(state, eigen_bench::dotFlops<Scalar>(n));
}

template <typename Scalar>
static void BM_DotEigen(benchmark::State& state) {
  runDot<Scalar>(state, EigenDotKernel());
}

// Defined only with a vendor linked; REGISTER_COMPARISON_POINT then drops its
// reference-arm argument, so the name is never substituted.
#ifdef EIGEN_BENCH_REFERENCE_ARM
template <typename Scalar>
static void BM_DotReference(benchmark::State& state) {
  runDot<Scalar>(state, ReferenceDotKernel());
}
#endif

// The whole grid: run.py narrows it with --benchmark_filter, so a length absent
// here is out of reach. The streaming lengths carry the operands through L2 and
// L3 into DRAM: 16 Mi doubles are 128 MiB per vector.
// clang-format off
#define DOT_DIM_NAMES {"n"}

#define DOT_POINT(...)                                                                                      \
  REGISTER_COMPARISON_POINT(DOT, f32, float,  BM_DotEigen, BM_DotReference, DOT_DIM_NAMES, __VA_ARGS__) \
  REGISTER_COMPARISON_POINT(DOT, f64, double, BM_DotEigen, BM_DotReference, DOT_DIM_NAMES, __VA_ARGS__)

#define DOT_SIZES(POINT) \
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

DOT_SIZES(DOT_POINT)

#undef DOT_SIZES
#undef DOT_POINT
#undef DOT_DIM_NAMES
// clang-format on

EIGEN_BENCH_COMPARISON_MAIN();
