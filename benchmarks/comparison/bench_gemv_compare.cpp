// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Cross-library comparison benchmarks for GEMV, the general matrix-vector product.
//
// The second operation in the harness, and the one that demonstrates the shape it
// was built for: this file adds a kernel functor per arm, a driver, and one grid
// macro. ops.toml gains `status = "implemented"` and a `source`; CMake discovers
// the source by glob; run.py, reduce.py, render.py and plots.py are untouched.
//
// GEMV is memory bound above L2, so unlike GEMM it is a bandwidth measurement
// dressed as a flop rate. Both arms are charged the same 2*m*n, so the ratio
// remains meaningful even where the absolute GFLOP/s is far below peak -- but a
// reader of the published page needs the caveat, which is why ops.toml's
// description carries it.

#include <Eigen/Core>
#include <array>
#include <complex>
#include <mutex>
#include <set>
#include <string>

#include "benchmarks/bench_common.h"
#include "benchmarks/comparison/bench_compare.h"

#ifdef EIGEN_BENCH_REFERENCE_ARM
// Fortran BLAS, declared here rather than pulled from a vendor header so that the
// build needs only the library. The integer width follows eigen_bench::BlasInt,
// i.e. Eigen::BlasIndex; getting it wrong silently corrupts every argument, so
// the build must determine it rather than assume it, and bench_compare.h
// static_asserts the vendor table's answer against Eigen's.
extern "C" {
void sgemv_(const char* trans, const eigen_bench::BlasInt* m, const eigen_bench::BlasInt* n, const float* alpha,
            const float* a, const eigen_bench::BlasInt* lda, const float* x, const eigen_bench::BlasInt* incx,
            const float* beta, float* y, const eigen_bench::BlasInt* incy);
void dgemv_(const char* trans, const eigen_bench::BlasInt* m, const eigen_bench::BlasInt* n, const double* alpha,
            const double* a, const eigen_bench::BlasInt* lda, const double* x, const eigen_bench::BlasInt* incx,
            const double* beta, double* y, const eigen_bench::BlasInt* incy);
void cgemv_(const char* trans, const eigen_bench::BlasInt* m, const eigen_bench::BlasInt* n,
            const std::complex<float>* alpha, const std::complex<float>* a, const eigen_bench::BlasInt* lda,
            const std::complex<float>* x, const eigen_bench::BlasInt* incx, const std::complex<float>* beta,
            std::complex<float>* y, const eigen_bench::BlasInt* incy);
void zgemv_(const char* trans, const eigen_bench::BlasInt* m, const eigen_bench::BlasInt* n,
            const std::complex<double>* alpha, const std::complex<double>* a, const eigen_bench::BlasInt* lda,
            const std::complex<double>* x, const eigen_bench::BlasInt* incx, const std::complex<double>* beta,
            std::complex<double>* y, const eigen_bench::BlasInt* incy);
}
#endif

using Eigen::Index;
using eigen_bench::BlasInt;
using eigen_bench::fitsBlasInt;

template <typename Scalar>
using GemvMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

template <typename Scalar>
using GemvVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

// y := y + A*x, the operation ops.toml records as GEMV with alpha = beta = 1 and
// trans = 'N'. The transposed kernel is a different reference call and would get
// its own key (GEMV_T), not a runtime branch here.
struct EigenGemvKernel {
  template <typename Scalar>
  void operator()(const GemvMatrix<Scalar>& a, const GemvVector<Scalar>& x, GemvVector<Scalar>& y) const {
    y.noalias() += a * x;
  }
};

#ifdef EIGEN_BENCH_REFERENCE_ARM
static void referenceGemv(BlasInt m, BlasInt n, const float* a, const float* x, float* y) {
  const char no_trans = 'N';
  const float one = 1.0f;
  const BlasInt inc = 1;
  sgemv_(&no_trans, &m, &n, &one, a, &m, x, &inc, &one, y, &inc);
}

static void referenceGemv(BlasInt m, BlasInt n, const double* a, const double* x, double* y) {
  const char no_trans = 'N';
  const double one = 1.0;
  const BlasInt inc = 1;
  dgemv_(&no_trans, &m, &n, &one, a, &m, x, &inc, &one, y, &inc);
}

static void referenceGemv(BlasInt m, BlasInt n, const std::complex<float>* a, const std::complex<float>* x,
                          std::complex<float>* y) {
  const char no_trans = 'N';
  const std::complex<float> one(1.0f, 0.0f);
  const BlasInt inc = 1;
  cgemv_(&no_trans, &m, &n, &one, a, &m, x, &inc, &one, y, &inc);
}

static void referenceGemv(BlasInt m, BlasInt n, const std::complex<double>* a, const std::complex<double>* x,
                          std::complex<double>* y) {
  const char no_trans = 'N';
  const std::complex<double> one(1.0, 0.0);
  const BlasInt inc = 1;
  zgemv_(&no_trans, &m, &n, &one, a, &m, x, &inc, &one, y, &inc);
}

struct ReferenceGemvKernel {
  template <typename Scalar>
  void operator()(const GemvMatrix<Scalar>& a, const GemvVector<Scalar>& x, GemvVector<Scalar>& y) const {
    // Column-major and contiguous, so lda is the row count.
    referenceGemv(static_cast<BlasInt>(a.rows()), static_cast<BlasInt>(a.cols()), a.data(), x.data(), y.data());
  }
};
#endif

// Both arms run through this one driver so that allocation, fill, validation,
// the timed region and the counter cannot drift apart between them.
template <typename Scalar, typename Kernel>
static void runGemv(benchmark::State& state, Kernel kernel) {
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

  const Index m = static_cast<Index>(state.range(0));
  const Index n = static_cast<Index>(state.range(1));

  if (!fitsBlasInt(m) || !fitsBlasInt(n)) {
    state.SkipWithError("dimension does not fit the reference BLAS integer width");
    return;
  }

  // a (m-by-n) plus the two vectors. Checked before the first allocation, so an
  // over-large point costs one skipped cell rather than the whole run.
  const double operand_bytes = static_cast<double>(sizeof(Scalar)) * (static_cast<double>(m) * static_cast<double>(n) +
                                                                      static_cast<double>(m) + static_cast<double>(n));
  if (eigen_bench::skipIfOverMemoryBudget(state, operand_bytes)) return;

  // Allocation and fill stay per entry on purpose. Hoisting the operands out
  // would leave them resident and warm across entries and change what the timed
  // loop measures; y is additionally accumulated into by the timed loop and
  // never reset, so it has to be re-randomized regardless.
  GemvMatrix<Scalar> a = GemvMatrix<Scalar>::Random(m, n);
  GemvVector<Scalar> x = GemvVector<Scalar>::Random(n);
  GemvVector<Scalar> y = GemvVector<Scalar>::Random(m);

  // Correctness is a deterministic property of (Scalar, kernel, shape), and
  // Google Benchmark enters this body once per repetition plus a few times while
  // it searches for an iteration count -- 13 times at the harness defaults, so
  // re-checking on every entry buys nothing and costs a full untimed GEMV each
  // time. The set is a function-local static of a function template, hence
  // already per (Scalar, Kernel); only the shape has to be keyed. The mutex
  // keeps that true under a threaded runner.
  static std::set<std::array<Index, 2>> validated_shapes;
  static std::mutex validated_shapes_mutex;
  const std::array<Index, 2> shape = {m, n};
  bool shape_is_validated;
  {
    const std::lock_guard<std::mutex> guard(validated_shapes_mutex);
    shape_is_validated = validated_shapes.count(shape) != 0;
  }

  if (!shape_is_validated) {
    // Compared directly against Eigen's own product, unlike the GEMM file. The
    // result here is already an m-vector, so forming the expected value costs
    // one untimed O(mn) matrix-vector product and O(m) memory -- the same order
    // as the operation itself. GEMM needs Freivalds' probabilistic check only
    // because a second m-by-n product would cost an extra factor of k in time
    // and a whole extra matrix in memory; paying that indirection here would
    // weaken the check (a random projection can miss an error) to save nothing.
    const GemvVector<Scalar> expected = y + a * x;
    kernel(a, x, y);

    const RealScalar tolerance =
        RealScalar(64) * Eigen::numext::sqrt(RealScalar(n)) * Eigen::NumTraits<RealScalar>::epsilon();
    const RealScalar magnitude = Eigen::numext::maxi(expected.norm(), y.norm());
    // Negated so that a NaN result fails rather than passes.
    if (!((y - expected).norm() <= tolerance * magnitude)) {
      state.SkipWithError("gemv result disagrees with Eigen at m:" + std::to_string(m) + " n:" + std::to_string(n));
      return;
    }
    // Recorded only after the check passed, so a failure cannot mark the shape
    // good for the entries that follow it.
    const std::lock_guard<std::mutex> guard(validated_shapes_mutex);
    validated_shapes.insert(shape);
  }

  for (auto _ : state) {
    kernel(a, x, y);
    benchmark::DoNotOptimize(y.data());
    benchmark::ClobberMemory();
  }

  eigen_bench::setFlopRate(state, eigen_bench::gemvFlops<Scalar>(m, n));
}

template <typename Scalar>
static void BM_GemvEigen(benchmark::State& state) {
  runGemv<Scalar>(state, EigenGemvKernel());
}

// Defined only with a vendor linked; REGISTER_COMPARISON_POINT then drops its
// reference-arm argument, so the name is never substituted.
#ifdef EIGEN_BENCH_REFERENCE_ARM
template <typename Scalar>
static void BM_GemvReference(benchmark::State& state) {
  runGemv<Scalar>(state, ReferenceGemvKernel());
}
#endif

// The whole matvec2 grid of ops.toml, in the order its default_groups lists it.
// run.py narrows it with --benchmark_filter; registering less here would make a
// group unreachable.
//
// A list macro rather than an arrow chain, because REGISTER_COMPARISON_POINT
// emits the two arms of each shape adjacently and an arrow chain cannot express
// that: see the note above the macro in bench_compare.h.
// clang-format off
#define GEMV_DIM_NAMES {"m", "n"}

#define GEMV_POINT(...)                                                                                            \
  REGISTER_COMPARISON_POINT(GEMV, f32, float,              BM_GemvEigen, BM_GemvReference, GEMV_DIM_NAMES, __VA_ARGS__) \
  REGISTER_COMPARISON_POINT(GEMV, f64, double,             BM_GemvEigen, BM_GemvReference, GEMV_DIM_NAMES, __VA_ARGS__) \
  REGISTER_COMPARISON_POINT(GEMV, c32, eigen_bench::c32_t, BM_GemvEigen, BM_GemvReference, GEMV_DIM_NAMES, __VA_ARGS__) \
  REGISTER_COMPARISON_POINT(GEMV, c64, eigen_bench::c64_t, BM_GemvEigen, BM_GemvReference, GEMV_DIM_NAMES, __VA_ARGS__)

#define GEMV_SIZES(POINT) \
  /* fixed_tiny */ \
  POINT(2,2) POINT(3,3) POINT(4,4) POINT(6,6) POINT(8,8) POINT(12,12) POINT(16,16) \
  /* small */ \
  POINT(24,24) POINT(32,32) POINT(48,48) POINT(64,64) POINT(96,96) POINT(128,128) \
  /* medium */ \
  POINT(192,192) POINT(256,256) POINT(384,384) POINT(512,512) POINT(768,768) POINT(1024,1024) \
  /* large */ \
  POINT(1536,1536) POINT(2048,2048) POINT(3072,3072) POINT(4096,4096) \
  /* xlarge */ \
  POINT(6144,6144) POINT(8192,8192) POINT(12288,12288) POINT(16384,16384) \
  /* aliasing */ \
  POINT(100,100) POINT(200,200) POINT(257,257) POINT(500,500) \
  POINT(1000,1000) POINT(1001,1001) POINT(4097,4097) \
  /* tall_skinny */ \
  POINT(10000,8) POINT(10000,100) POINT(10000,1000) POINT(10000,4000) \
  /* short_fat */ \
  POINT(8,10000) POINT(100,10000) POINT(1000,10000) POINT(4000,10000)

GEMV_SIZES(GEMV_POINT)

#undef GEMV_SIZES
#undef GEMV_POINT
#undef GEMV_DIM_NAMES
// clang-format on

EIGEN_BENCH_COMPARISON_MAIN();
