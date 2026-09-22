// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Cross-library comparison benchmarks for SYRK (HERK for a complex scalar), the
// symmetric rank-k update of one triangle: C := C + A*A^H with A n-by-k.

#include <Eigen/Core>
#include <array>
#include <complex>
#include <string>

#include "benchmarks/bench_common.h"
#include "benchmarks/comparison/bench_compare.h"

#ifdef EIGEN_BENCH_REFERENCE_ARM
// Declared rather than included, so the build needs the reference library but not
// its development headers. The integer width is eigen_bench::BlasInt; see the
// static_assert in bench_compare.h. The Hermitian update takes REAL alpha and beta.
extern "C" {
void ssyrk_(const char* uplo, const char* trans, const eigen_bench::BlasInt* n, const eigen_bench::BlasInt* k,
            const float* alpha, const float* a, const eigen_bench::BlasInt* lda, const float* beta, float* c,
            const eigen_bench::BlasInt* ldc) EIGEN_BENCH_FORTRAN_SYMBOL(ssyrk);
void dsyrk_(const char* uplo, const char* trans, const eigen_bench::BlasInt* n, const eigen_bench::BlasInt* k,
            const double* alpha, const double* a, const eigen_bench::BlasInt* lda, const double* beta, double* c,
            const eigen_bench::BlasInt* ldc) EIGEN_BENCH_FORTRAN_SYMBOL(dsyrk);
void cherk_(const char* uplo, const char* trans, const eigen_bench::BlasInt* n, const eigen_bench::BlasInt* k,
            const float* alpha, const std::complex<float>* a, const eigen_bench::BlasInt* lda, const float* beta,
            std::complex<float>* c, const eigen_bench::BlasInt* ldc) EIGEN_BENCH_FORTRAN_SYMBOL(cherk);
void zherk_(const char* uplo, const char* trans, const eigen_bench::BlasInt* n, const eigen_bench::BlasInt* k,
            const double* alpha, const std::complex<double>* a, const eigen_bench::BlasInt* lda, const double* beta,
            std::complex<double>* c, const eigen_bench::BlasInt* ldc) EIGEN_BENCH_FORTRAN_SYMBOL(zherk);
}
#endif

using Eigen::Index;
using eigen_bench::BlasInt;
using eigen_bench::ColMatrix;
using eigen_bench::ColVector;

// C := C + A*A^H on the lower triangle, i.e. uplo = 'L', trans = 'N', alpha =
// beta = 1.
struct EigenSyrkKernel {
  template <typename Scalar>
  void operator()(const ColMatrix<Scalar>& a, ColMatrix<Scalar>& c) const {
    c.template selfadjointView<Eigen::Lower>().rankUpdate(a, Scalar(1));
  }
};

#ifdef EIGEN_BENCH_REFERENCE_ARM
static void referenceSyrk(BlasInt n, BlasInt k, const float* a, float* c) {
  const char lower = 'L', no_trans = 'N';
  const float one = 1.0f;
  ssyrk_(&lower, &no_trans, &n, &k, &one, a, &n, &one, c, &n);
}

static void referenceSyrk(BlasInt n, BlasInt k, const double* a, double* c) {
  const char lower = 'L', no_trans = 'N';
  const double one = 1.0;
  dsyrk_(&lower, &no_trans, &n, &k, &one, a, &n, &one, c, &n);
}

static void referenceSyrk(BlasInt n, BlasInt k, const std::complex<float>* a, std::complex<float>* c) {
  const char lower = 'L', no_trans = 'N';
  const float one = 1.0f;
  cherk_(&lower, &no_trans, &n, &k, &one, a, &n, &one, c, &n);
}

static void referenceSyrk(BlasInt n, BlasInt k, const std::complex<double>* a, std::complex<double>* c) {
  const char lower = 'L', no_trans = 'N';
  const double one = 1.0;
  zherk_(&lower, &no_trans, &n, &k, &one, a, &n, &one, c, &n);
}

struct ReferenceSyrkKernel {
  template <typename Scalar>
  void operator()(const ColMatrix<Scalar>& a, ColMatrix<Scalar>& c) const {
    referenceSyrk(static_cast<BlasInt>(a.rows()), static_cast<BlasInt>(a.cols()), a.data(), c.data());
  }
};
#endif

// Both arms run through this one driver so that allocation, fill, validation,
// the timed region and the counter cannot drift apart between them.
template <typename Scalar, typename Kernel>
static void runSyrk(benchmark::State& state, Kernel kernel) {
  const Index n = static_cast<Index>(state.range(0));
  const Index k = static_cast<Index>(state.range(1));

  if (eigen_bench::skipIfDimsExceedBlasInt(state, n, k)) return;

  // a (n-by-k) and c (n-by-n).
  const double operand_bytes = static_cast<double>(sizeof(Scalar)) * (static_cast<double>(n) * static_cast<double>(k) +
                                                                      static_cast<double>(n) * static_cast<double>(n));
  if (eigen_bench::skipIfOverMemoryBudget(state, operand_bytes)) return;

  const ColMatrix<Scalar> a = ColMatrix<Scalar>::Random(n, k);
  ColMatrix<Scalar> c = ColMatrix<Scalar>::Random(n, n);
  // ?herk reads the diagonal of C as real and writes it back real, where Eigen
  // adds the (real) update to whatever is there. A real diagonal to begin with
  // keeps the two arms computing the same thing; a no-op for a real scalar.
  c.diagonal() = c.diagonal().real().template cast<Scalar>();

  static eigen_bench::ValidatedShapes<std::array<Index, 2>> validated;
  const std::array<Index, 2> shape = {n, k};
  if (!validated.contains(shape)) {
    // Freivalds' check on the stored triangle: (C + A*A^H)x against
    // Cx + A*(A^H x) for a random x, O(n*k + n^2) instead of a second product.
    // Read through the self-adjoint view, so the untouched upper triangle is
    // correctly outside the contract being checked.
    const ColVector<Scalar> x = ColVector<Scalar>::Random(n);
    const ColVector<Scalar> expected = c.template selfadjointView<Eigen::Lower>() * x + a * (a.adjoint() * x).eval();
    kernel(a, c);
    const ColVector<Scalar> actual = c.template selfadjointView<Eigen::Lower>() * x;

    if (!eigen_bench::agreesWithEigen(expected, actual, k)) {
      state.SkipWithError("syrk result disagrees with Eigen at n:" + std::to_string(n) + " k:" + std::to_string(k));
      return;
    }
    validated.insert(shape);
  }

  // c accumulates across iterations and is never reset, as in GEMM.
  for (auto _ : state) {
    kernel(a, c);
    benchmark::DoNotOptimize(c.data());
    benchmark::ClobberMemory();
  }

  eigen_bench::setFlopRate(state, eigen_bench::syrkFlops<Scalar>(n, k));
}

template <typename Scalar>
static void BM_SyrkEigen(benchmark::State& state) {
  runSyrk<Scalar>(state, EigenSyrkKernel());
}

// Defined only with a vendor linked; REGISTER_COMPARISON_POINT then drops its
// reference-arm argument, so the name is never substituted.
#ifdef EIGEN_BENCH_REFERENCE_ARM
template <typename Scalar>
static void BM_SyrkReference(benchmark::State& state) {
  runSyrk<Scalar>(state, ReferenceSyrkKernel());
}
#endif

// The whole grid: run.py narrows it with --benchmark_filter, so a shape absent
// here is out of reach. n is the order of C, k the number of columns of A.
// clang-format off
#define SYRK_DIM_NAMES {"n", "k"}

#define SYRK_POINT(...)                                                                                            \
  REGISTER_COMPARISON_POINT(SYRK, f32, float,              BM_SyrkEigen, BM_SyrkReference, SYRK_DIM_NAMES, __VA_ARGS__) \
  REGISTER_COMPARISON_POINT(SYRK, f64, double,             BM_SyrkEigen, BM_SyrkReference, SYRK_DIM_NAMES, __VA_ARGS__) \
  REGISTER_COMPARISON_POINT(SYRK, c32, eigen_bench::c32_t, BM_SyrkEigen, BM_SyrkReference, SYRK_DIM_NAMES, __VA_ARGS__) \
  REGISTER_COMPARISON_POINT(SYRK, c64, eigen_bench::c64_t, BM_SyrkEigen, BM_SyrkReference, SYRK_DIM_NAMES, __VA_ARGS__)

#define SYRK_SIZES(POINT) \
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
  /* a few columns update a large matrix */ \
  POINT(4096,8) POINT(4096,64) POINT(4096,256) POINT(4096,1024) \
  /* the Gram matrix of a short, wide factor */ \
  POINT(64,4096) POINT(256,4096) POINT(1024,4096)

SYRK_SIZES(SYRK_POINT)

#undef SYRK_SIZES
#undef SYRK_POINT
#undef SYRK_DIM_NAMES
// clang-format on

EIGEN_BENCH_COMPARISON_MAIN();
