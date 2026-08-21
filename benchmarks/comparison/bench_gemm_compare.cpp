// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Cross-library comparison benchmarks for GEMM, the general matrix-matrix product.
//
// One operation per source file. Both arms are emitted at each grid point by one
// REGISTER_COMPARISON_POINT line over one shared grid macro; bench_gemv_compare.cpp
// is the same file with a different kernel, driver and grid.
//
// This file supersedes the roles of Core/bench_gemm.cpp's BM_BlasGemm (dead code:
// no target defines HAVE_BLAS) and Tuning/bench_blas_gemm.cpp (float only, smaller
// grid, unguarded <cblas.h>). Both remain in the tree for now.

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
void sgemm_(const char* transa, const char* transb, const eigen_bench::BlasInt* m, const eigen_bench::BlasInt* n,
            const eigen_bench::BlasInt* k, const float* alpha, const float* a, const eigen_bench::BlasInt* lda,
            const float* b, const eigen_bench::BlasInt* ldb, const float* beta, float* c,
            const eigen_bench::BlasInt* ldc);
void dgemm_(const char* transa, const char* transb, const eigen_bench::BlasInt* m, const eigen_bench::BlasInt* n,
            const eigen_bench::BlasInt* k, const double* alpha, const double* a, const eigen_bench::BlasInt* lda,
            const double* b, const eigen_bench::BlasInt* ldb, const double* beta, double* c,
            const eigen_bench::BlasInt* ldc);
void cgemm_(const char* transa, const char* transb, const eigen_bench::BlasInt* m, const eigen_bench::BlasInt* n,
            const eigen_bench::BlasInt* k, const std::complex<float>* alpha, const std::complex<float>* a,
            const eigen_bench::BlasInt* lda, const std::complex<float>* b, const eigen_bench::BlasInt* ldb,
            const std::complex<float>* beta, std::complex<float>* c, const eigen_bench::BlasInt* ldc);
void zgemm_(const char* transa, const char* transb, const eigen_bench::BlasInt* m, const eigen_bench::BlasInt* n,
            const eigen_bench::BlasInt* k, const std::complex<double>* alpha, const std::complex<double>* a,
            const eigen_bench::BlasInt* lda, const std::complex<double>* b, const eigen_bench::BlasInt* ldb,
            const std::complex<double>* beta, std::complex<double>* c, const eigen_bench::BlasInt* ldc);
}
#endif

using Eigen::Index;
using eigen_bench::BlasInt;
using eigen_bench::fitsBlasInt;

template <typename Scalar>
using GemmMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

template <typename Scalar>
using GemmVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

// C := C + A*B, the operation ops.toml records as GEMM with alpha = beta = 1.
struct EigenGemmKernel {
  template <typename Scalar>
  void operator()(const GemmMatrix<Scalar>& a, const GemmMatrix<Scalar>& b, GemmMatrix<Scalar>& c) const {
    c.noalias() += a * b;
  }
};

#ifdef EIGEN_BENCH_REFERENCE_ARM
static void referenceGemm(BlasInt m, BlasInt n, BlasInt k, const float* a, const float* b, float* c) {
  const char no_trans = 'N';
  const float one = 1.0f;
  sgemm_(&no_trans, &no_trans, &m, &n, &k, &one, a, &m, b, &k, &one, c, &m);
}

static void referenceGemm(BlasInt m, BlasInt n, BlasInt k, const double* a, const double* b, double* c) {
  const char no_trans = 'N';
  const double one = 1.0;
  dgemm_(&no_trans, &no_trans, &m, &n, &k, &one, a, &m, b, &k, &one, c, &m);
}

static void referenceGemm(BlasInt m, BlasInt n, BlasInt k, const std::complex<float>* a, const std::complex<float>* b,
                          std::complex<float>* c) {
  const char no_trans = 'N';
  const std::complex<float> one(1.0f, 0.0f);
  cgemm_(&no_trans, &no_trans, &m, &n, &k, &one, a, &m, b, &k, &one, c, &m);
}

static void referenceGemm(BlasInt m, BlasInt n, BlasInt k, const std::complex<double>* a, const std::complex<double>* b,
                          std::complex<double>* c) {
  const char no_trans = 'N';
  const std::complex<double> one(1.0, 0.0);
  zgemm_(&no_trans, &no_trans, &m, &n, &k, &one, a, &m, b, &k, &one, c, &m);
}

struct ReferenceGemmKernel {
  template <typename Scalar>
  void operator()(const GemmMatrix<Scalar>& a, const GemmMatrix<Scalar>& b, GemmMatrix<Scalar>& c) const {
    // Column-major and contiguous, so the leading dimensions are the extents.
    referenceGemm(static_cast<BlasInt>(a.rows()), static_cast<BlasInt>(b.cols()), static_cast<BlasInt>(a.cols()),
                  a.data(), b.data(), c.data());
  }
};
#endif

// Both arms run through this one driver so that allocation, fill, validation,
// the timed region and the counter cannot drift apart between them.
template <typename Scalar, typename Kernel>
static void runGemm(benchmark::State& state, Kernel kernel) {
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

  const Index m = static_cast<Index>(state.range(0));
  const Index n = static_cast<Index>(state.range(1));
  const Index k = static_cast<Index>(state.range(2));

  if (!fitsBlasInt(m) || !fitsBlasInt(n) || !fitsBlasInt(k)) {
    state.SkipWithError("dimension does not fit the reference BLAS integer width");
    return;
  }

  // Allocation and fill stay per entry on purpose. Hoisting the operands out
  // would leave them resident and warm across entries and change what the timed
  // loop measures; c is additionally accumulated into by the timed loop and
  // never reset, so it has to be re-randomized regardless.
  GemmMatrix<Scalar> a = GemmMatrix<Scalar>::Random(m, k);
  GemmMatrix<Scalar> b = GemmMatrix<Scalar>::Random(k, n);
  GemmMatrix<Scalar> c = GemmMatrix<Scalar>::Random(m, n);

  // Correctness is a deterministic property of (Scalar, kernel, shape), and
  // Google Benchmark enters this body once per repetition plus a few times while
  // it searches for an iteration count -- 13 times at the harness defaults.
  // Re-checking on every entry yields no new information and is not free: the
  // check runs a full untimed GEMM, so at the large end of the grid the binary
  // would spend more time validating than measuring. The set is a function-local
  // static of a function template, hence already per (Scalar, Kernel); only the
  // shape has to be keyed. The mutex keeps that true under a threaded runner.
  static std::set<std::array<Index, 3>> validated_shapes;
  static std::mutex validated_shapes_mutex;
  const std::array<Index, 3> shape = {m, n, k};
  bool shape_is_validated;
  {
    const std::lock_guard<std::mutex> guard(validated_shapes_mutex);
    shape_is_validated = validated_shapes.count(shape) != 0;
  }

  if (!shape_is_validated) {
    // Freivalds' check (R. Freivalds, "Probabilistic machines can use less
    // running time", IFIP 1977): comparing (C + A*B)x against Cx + A*(B*x)
    // verifies the kernel's whole result against Eigen's own products in
    // O(mn + nk + mk) work and O(m + n + k) extra memory, instead of forming a
    // second m-by-n product. A wrong kernel survives only if its error is
    // orthogonal to a random x. Outside the timed region, and before it.
    const GemmVector<Scalar> x = GemmVector<Scalar>::Random(n);
    const GemmVector<Scalar> expected = c * x + a * (b * x);
    kernel(a, b, c);
    const GemmVector<Scalar> actual = c * x;

    const RealScalar tolerance =
        RealScalar(64) * Eigen::numext::sqrt(RealScalar(k)) * Eigen::NumTraits<RealScalar>::epsilon();
    const RealScalar magnitude = Eigen::numext::maxi(expected.norm(), actual.norm());
    // Negated so that a NaN result fails rather than passes.
    if (!((actual - expected).norm() <= tolerance * magnitude)) {
      state.SkipWithError("gemm result disagrees with Eigen at m:" + std::to_string(m) + " n:" + std::to_string(n) +
                          " k:" + std::to_string(k));
      return;
    }
    // Recorded only after the check passed, so a failure cannot mark the shape
    // good for the entries that follow it.
    const std::lock_guard<std::mutex> guard(validated_shapes_mutex);
    validated_shapes.insert(shape);
  }

  for (auto _ : state) {
    kernel(a, b, c);
    benchmark::DoNotOptimize(c.data());
    benchmark::ClobberMemory();
  }

  eigen_bench::setFlopRate(state, eigen_bench::gemmFlops<Scalar>(m, n, k));
}

template <typename Scalar>
static void BM_GemmEigen(benchmark::State& state) {
  runGemm<Scalar>(state, EigenGemmKernel());
}

// Defined only with a vendor linked; REGISTER_COMPARISON_POINT then drops its
// reference-arm argument, so the name is never substituted.
#ifdef EIGEN_BENCH_REFERENCE_ARM
template <typename Scalar>
static void BM_GemmReference(benchmark::State& state) {
  runGemm<Scalar>(state, ReferenceGemmKernel());
}
#endif

// The whole square3 grid of ops.toml, in the order its default_groups lists it.
// run.py narrows it with --benchmark_filter; registering less here would make a
// group unreachable.
//
// A list macro rather than an arrow chain, because REGISTER_COMPARISON_POINT
// emits the two arms of each shape adjacently and an arrow chain cannot express
// that: see the note above the macro in bench_compare.h.
// clang-format off
#define GEMM_DIM_NAMES {"m", "n", "k"}

#define GEMM_POINT(...)                                                                                            \
  REGISTER_COMPARISON_POINT(GEMM, f32, float,              BM_GemmEigen, BM_GemmReference, GEMM_DIM_NAMES, __VA_ARGS__) \
  REGISTER_COMPARISON_POINT(GEMM, f64, double,             BM_GemmEigen, BM_GemmReference, GEMM_DIM_NAMES, __VA_ARGS__) \
  REGISTER_COMPARISON_POINT(GEMM, c32, eigen_bench::c32_t, BM_GemmEigen, BM_GemmReference, GEMM_DIM_NAMES, __VA_ARGS__) \
  REGISTER_COMPARISON_POINT(GEMM, c64, eigen_bench::c64_t, BM_GemmEigen, BM_GemmReference, GEMM_DIM_NAMES, __VA_ARGS__)

#define GEMM_SIZES(POINT) \
  /* fixed_tiny */ \
  POINT(2,2,2) POINT(3,3,3) POINT(4,4,4) POINT(6,6,6) POINT(8,8,8) \
  POINT(12,12,12) POINT(16,16,16) \
  /* small */ \
  POINT(24,24,24) POINT(32,32,32) POINT(48,48,48) POINT(64,64,64) \
  POINT(96,96,96) POINT(128,128,128) \
  /* medium */ \
  POINT(192,192,192) POINT(256,256,256) POINT(384,384,384) POINT(512,512,512) \
  POINT(768,768,768) POINT(1024,1024,1024) \
  /* large */ \
  POINT(1536,1536,1536) POINT(2048,2048,2048) POINT(3072,3072,3072) POINT(4096,4096,4096) \
  /* legacy_square_sweep */ \
  POINT(160,160,160) POINT(224,224,224) POINT(288,288,288) POINT(320,320,320) POINT(448,448,448) \
  /* aliasing */ \
  POINT(100,100,100) POINT(200,200,200) POINT(257,257,257) POINT(500,500,500) \
  POINT(1000,1000,1000) POINT(1001,1001,1001) POINT(4097,4097,4097) \
  /* nonsquare */ \
  POINT(64,64,1024) POINT(1024,64,64) POINT(64,1024,64) POINT(256,256,1024) POINT(1024,256,256) \
  /* tall_skinny_cache */ \
  POINT(4096,96,96) POINT(4096,128,128) POINT(4096,144,144) POINT(4096,160,160) \
  POINT(4096,176,176) POINT(8192,128,128) \
  /* tall_skinny_extreme */ \
  POINT(10000,8,8) POINT(10000,100,100) POINT(10000,1000,1000) POINT(10000,4000,4000)

GEMM_SIZES(GEMM_POINT)

#undef GEMM_SIZES
#undef GEMM_POINT
#undef GEMM_DIM_NAMES
// clang-format on

EIGEN_BENCH_COMPARISON_MAIN();
