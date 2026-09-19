// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Cross-library comparison benchmarks for TRSM, the triangular solve with a
// matrix right-hand side: X := A^{-1} B with A lower triangular, on the left.
//
// The operation is destructive, so the timed body must present fresh data on
// every iteration: both arms copy B into the destination and solve in place
// there, the copy a caller pays to keep B.

#include <Eigen/Core>
#include <array>
#include <complex>
#include <string>

#include "benchmarks/bench_common.h"
#include "benchmarks/comparison/bench_compare.h"

#ifdef EIGEN_BENCH_REFERENCE_ARM
// Declared rather than included, so the build needs the reference library but not
// its development headers. The integer width is eigen_bench::BlasInt; see the
// static_assert in bench_compare.h.
extern "C" {
void strsm_(const char* side, const char* uplo, const char* transa, const char* diag, const eigen_bench::BlasInt* m,
            const eigen_bench::BlasInt* n, const float* alpha, const float* a, const eigen_bench::BlasInt* lda,
            float* b, const eigen_bench::BlasInt* ldb) EIGEN_BENCH_FORTRAN_SYMBOL(strsm);
void dtrsm_(const char* side, const char* uplo, const char* transa, const char* diag, const eigen_bench::BlasInt* m,
            const eigen_bench::BlasInt* n, const double* alpha, const double* a, const eigen_bench::BlasInt* lda,
            double* b, const eigen_bench::BlasInt* ldb) EIGEN_BENCH_FORTRAN_SYMBOL(dtrsm);
void ctrsm_(const char* side, const char* uplo, const char* transa, const char* diag, const eigen_bench::BlasInt* m,
            const eigen_bench::BlasInt* n, const std::complex<float>* alpha, const std::complex<float>* a,
            const eigen_bench::BlasInt* lda, std::complex<float>* b, const eigen_bench::BlasInt* ldb)
    EIGEN_BENCH_FORTRAN_SYMBOL(ctrsm);
void ztrsm_(const char* side, const char* uplo, const char* transa, const char* diag, const eigen_bench::BlasInt* m,
            const eigen_bench::BlasInt* n, const std::complex<double>* alpha, const std::complex<double>* a,
            const eigen_bench::BlasInt* lda, std::complex<double>* b, const eigen_bench::BlasInt* ldb)
    EIGEN_BENCH_FORTRAN_SYMBOL(ztrsm);
}
#endif

using Eigen::Index;
using eigen_bench::BlasInt;
using eigen_bench::ColMatrix;
using eigen_bench::ColVector;

// X := A^{-1} B with side = 'L', uplo = 'L', trans = 'N', diag = 'N', alpha = 1.
struct EigenTrsmKernel {
  template <typename Scalar>
  void operator()(const ColMatrix<Scalar>& a, const ColMatrix<Scalar>& b, ColMatrix<Scalar>& out) const {
    out = b;
    a.template triangularView<Eigen::Lower>().solveInPlace(out);
  }
};

#ifdef EIGEN_BENCH_REFERENCE_ARM
static void referenceTrsm(BlasInt m, BlasInt n, const float* a, float* b) {
  const char left = 'L', lower = 'L', no_trans = 'N', non_unit = 'N';
  const float one = 1.0f;
  strsm_(&left, &lower, &no_trans, &non_unit, &m, &n, &one, a, &m, b, &m);
}

static void referenceTrsm(BlasInt m, BlasInt n, const double* a, double* b) {
  const char left = 'L', lower = 'L', no_trans = 'N', non_unit = 'N';
  const double one = 1.0;
  dtrsm_(&left, &lower, &no_trans, &non_unit, &m, &n, &one, a, &m, b, &m);
}

static void referenceTrsm(BlasInt m, BlasInt n, const std::complex<float>* a, std::complex<float>* b) {
  const char left = 'L', lower = 'L', no_trans = 'N', non_unit = 'N';
  const std::complex<float> one(1.0f, 0.0f);
  ctrsm_(&left, &lower, &no_trans, &non_unit, &m, &n, &one, a, &m, b, &m);
}

static void referenceTrsm(BlasInt m, BlasInt n, const std::complex<double>* a, std::complex<double>* b) {
  const char left = 'L', lower = 'L', no_trans = 'N', non_unit = 'N';
  const std::complex<double> one(1.0, 0.0);
  ztrsm_(&left, &lower, &no_trans, &non_unit, &m, &n, &one, a, &m, b, &m);
}

struct ReferenceTrsmKernel {
  template <typename Scalar>
  void operator()(const ColMatrix<Scalar>& a, const ColMatrix<Scalar>& b, ColMatrix<Scalar>& out) const {
    // The same copy the Eigen arm makes, deliberately.
    out = b;
    referenceTrsm(static_cast<BlasInt>(a.rows()), static_cast<BlasInt>(b.cols()), a.data(), out.data());
  }
};
#endif

// Both arms run through this one driver so that allocation, fill, validation,
// the timed region and the counter cannot drift apart between them.
template <typename Scalar, typename Kernel>
static void runTrsm(benchmark::State& state, Kernel kernel) {
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

  const Index m = static_cast<Index>(state.range(0));
  const Index n = static_cast<Index>(state.range(1));

  if (eigen_bench::skipIfDimsExceedBlasInt(state, m, n)) return;

  // a (m-by-m), b and out (m-by-n).
  const double operand_bytes =
      static_cast<double>(sizeof(Scalar)) *
      (static_cast<double>(m) * static_cast<double>(m) + 2.0 * static_cast<double>(m) * static_cast<double>(n));
  if (eigen_bench::skipIfOverMemoryBudget(state, operand_bytes)) return;

  // Only the lower triangle is read, so the upper triangle keeps the noise
  // Random() put there, as a caller's matrix may. A diagonal of 2m dominates a
  // row sum bounded by m-1, so every solve is stable and the check tests the
  // kernel rather than the operand.
  ColMatrix<Scalar> a = ColMatrix<Scalar>::Random(m, m);
  a.diagonal().array() += RealScalar(2 * m);
  const ColMatrix<Scalar> b = ColMatrix<Scalar>::Random(m, n);
  ColMatrix<Scalar> out(m, n);

  static eigen_bench::ValidatedShapes<std::array<Index, 2>> validated;
  const std::array<Index, 2> shape = {m, n};
  if (!validated.contains(shape)) {
    // A*(X*y) against B*y for a random y: O(m*n + m^2) instead of the O(m^2*n)
    // product A*X, and it checks the whole of X.
    kernel(a, b, out);
    const ColVector<Scalar> y = ColVector<Scalar>::Random(n);
    const ColVector<Scalar> expected = b * y;
    const ColVector<Scalar> actual = a.template triangularView<Eigen::Lower>() * (out * y).eval();

    if (!eigen_bench::agreesWithEigen(expected, actual, m)) {
      state.SkipWithError("trsm result disagrees with Eigen at m:" + std::to_string(m) + " n:" + std::to_string(n));
      return;
    }
    validated.insert(shape);
  }

  for (auto _ : state) {
    kernel(a, b, out);
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }

  eigen_bench::setFlopRate(state, eigen_bench::trsmFlops<Scalar>(m, n));
}

template <typename Scalar>
static void BM_TrsmEigen(benchmark::State& state) {
  runTrsm<Scalar>(state, EigenTrsmKernel());
}

// Defined only with a vendor linked; REGISTER_COMPARISON_POINT then drops its
// reference-arm argument, so the name is never substituted.
#ifdef EIGEN_BENCH_REFERENCE_ARM
template <typename Scalar>
static void BM_TrsmReference(benchmark::State& state) {
  runTrsm<Scalar>(state, ReferenceTrsmKernel());
}
#endif

// The whole grid: run.py narrows it with --benchmark_filter, so a shape absent
// here is out of reach. m is the order of the triangle, n the number of
// right-hand sides.
// clang-format off
#define TRSM_DIM_NAMES {"m", "n"}

#define TRSM_POINT(...)                                                                                            \
  REGISTER_COMPARISON_POINT(TRSM, f32, float,              BM_TrsmEigen, BM_TrsmReference, TRSM_DIM_NAMES, __VA_ARGS__) \
  REGISTER_COMPARISON_POINT(TRSM, f64, double,             BM_TrsmEigen, BM_TrsmReference, TRSM_DIM_NAMES, __VA_ARGS__) \
  REGISTER_COMPARISON_POINT(TRSM, c32, eigen_bench::c32_t, BM_TrsmEigen, BM_TrsmReference, TRSM_DIM_NAMES, __VA_ARGS__) \
  REGISTER_COMPARISON_POINT(TRSM, c64, eigen_bench::c64_t, BM_TrsmEigen, BM_TrsmReference, TRSM_DIM_NAMES, __VA_ARGS__)

#define TRSM_SIZES(POINT) \
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
  /* few right-hand sides against a large triangle */ \
  POINT(1024,8) POINT(1024,64) POINT(1024,256) \
  POINT(4096,8) POINT(4096,64) POINT(4096,256) POINT(4096,1024) \
  /* many right-hand sides against a small triangle */ \
  POINT(64,4096) POINT(256,4096) POINT(1024,4096)

TRSM_SIZES(TRSM_POINT)

#undef TRSM_SIZES
#undef TRSM_POINT
#undef TRSM_DIM_NAMES
// clang-format on

EIGEN_BENCH_COMPARISON_MAIN();
