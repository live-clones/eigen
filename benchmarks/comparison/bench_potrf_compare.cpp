// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Cross-library comparison benchmarks for POTRF, the Cholesky factorization of a
// symmetric/Hermitian positive definite matrix.
//
// The first operation here whose reference is a LAPACK routine rather than a BLAS
// one. That exercises a path nothing else did: CMake's find_package(LAPACK) and
// the vendor table's PROVIDES declaration decide whether ?potrf resolves at all,
// and run.py turns a missing family into an explicit `reference_routine_absent`
// cell rather than a silent gap.
//
// Unlike GEMM and GEMV this operation is destructive -- it overwrites its input --
// so the timed body must present fresh data on every iteration. Both arms
// therefore copy the operand and factorize the copy, with the destination
// allocated once outside the loop: Eigen's LLT constructor would otherwise
// allocate on every iteration where the reference arm does not, and the
// comparison would be of allocators as much as of kernels.

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <array>
#include <complex>
#include <string>

#include "benchmarks/bench_common.h"
#include "benchmarks/comparison/bench_compare.h"

#ifdef EIGEN_BENCH_REFERENCE_ARM
// Fortran LAPACK, declared here rather than pulled from a vendor header so that
// the build needs only the library. See the note in bench_gemm_compare.cpp on
// the integer width.
extern "C" {
void spotrf_(const char* uplo, const eigen_bench::BlasInt* n, float* a, const eigen_bench::BlasInt* lda,
             eigen_bench::BlasInt* info);
void dpotrf_(const char* uplo, const eigen_bench::BlasInt* n, double* a, const eigen_bench::BlasInt* lda,
             eigen_bench::BlasInt* info);
void cpotrf_(const char* uplo, const eigen_bench::BlasInt* n, std::complex<float>* a, const eigen_bench::BlasInt* lda,
             eigen_bench::BlasInt* info);
void zpotrf_(const char* uplo, const eigen_bench::BlasInt* n, std::complex<double>* a, const eigen_bench::BlasInt* lda,
             eigen_bench::BlasInt* info);
}
#endif

using Eigen::Index;
using eigen_bench::BlasInt;
using eigen_bench::fitsBlasInt;

template <typename Scalar>
using PotrfMatrix = eigen_bench::ColMatrix<Scalar>;

template <typename Scalar>
using PotrfVector = eigen_bench::ColVector<Scalar>;

// A := L*L^H with L lower triangular, the operation ops.toml records as POTRF
// with uplo = 'L'. Both kernels copy the operand into `out` and factorize it
// there in place, leaving L in the lower triangle and saying nothing about the
// upper one, which is exactly LAPACK's contract; a caller that needs the full
// matrix takes a triangularView.
//
// Eigen's in-place decomposition (doc/InplaceDecomposition.dox) is what makes
// the two arms symmetric. The obvious spelling -- an LLT<MatrixType> held across
// iterations and driven by compute(a), then out = llt.matrixLLT() -- charges the
// Eigen arm TWO n-by-n copies where ?potrf pays one, since compute() copies into
// the decomposition's own storage and reading the factor back out copies again.
// At n = 1024 in double that is an extra 8 MB moved per iteration, attributed to
// Eigen and to nothing on the reference side.
//
// One asymmetry remains, and it is deliberate. LLT::compute also evaluates the
// operand's self-adjoint L1 norm (Eigen/src/Cholesky/LLT.h), which ?potrf does
// not: it is what makes LLT::rcond() available afterwards, and a LAPACK user who
// wants the same thing calls ?lange before ?pocon. Measured on Apple M4 it is
// 5-13% of the Eigen arm for real scalars and more at the small end for complex.
// It stays because the row this feeds is labelled with the expression a user
// writes -- ops.toml's eigen_expr, `Eigen::LLT<MatrixType> llt(A)` -- and that
// expression really does cost it. ops.toml's POTRF description carries the
// caveat so a reader of the page sees it too.
struct EigenPotrfKernel {
  template <typename Scalar>
  void operator()(const PotrfMatrix<Scalar>& a, PotrfMatrix<Scalar>& out) const {
    out = a;
    // Holds a Ref, so constructing it allocates nothing; the factorization
    // overwrites `out`.
    Eigen::LLT<Eigen::Ref<PotrfMatrix<Scalar>>> llt(out);
    benchmark::DoNotOptimize(llt.info());
  }
};

#ifdef EIGEN_BENCH_REFERENCE_ARM
static BlasInt referencePotrf(BlasInt n, float* a) {
  const char lower = 'L';
  BlasInt info = 0;
  spotrf_(&lower, &n, a, &n, &info);
  return info;
}

static BlasInt referencePotrf(BlasInt n, double* a) {
  const char lower = 'L';
  BlasInt info = 0;
  dpotrf_(&lower, &n, a, &n, &info);
  return info;
}

static BlasInt referencePotrf(BlasInt n, std::complex<float>* a) {
  const char lower = 'L';
  BlasInt info = 0;
  cpotrf_(&lower, &n, a, &n, &info);
  return info;
}

static BlasInt referencePotrf(BlasInt n, std::complex<double>* a) {
  const char lower = 'L';
  BlasInt info = 0;
  zpotrf_(&lower, &n, a, &n, &info);
  return info;
}

struct ReferencePotrfKernel {
  template <typename Scalar>
  void operator()(const PotrfMatrix<Scalar>& a, PotrfMatrix<Scalar>& out) const {
    // Without the copy the second iteration would factorize an already-factored
    // matrix. It is the same copy the Eigen arm makes, deliberately.
    out = a;
    referencePotrf(static_cast<BlasInt>(a.rows()), out.data());
  }
};
#endif

// Both arms run through this one driver so that allocation, fill, validation,
// the timed region and the counter cannot drift apart between them.
template <typename Scalar, typename Kernel>
static void runPotrf(benchmark::State& state, Kernel kernel) {
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

  const Index n = static_cast<Index>(state.range(0));

  if (eigen_bench::skipIfDimsExceedBlasInt(state, n)) return;

  // Two n-by-n matrices are live at once: the operand and the destination during
  // the timed loop, or the operand and its generator during the fill. Checked
  // before the first allocation, so an over-large point costs one skipped cell,
  // not the whole run.
  const double operand_bytes =
      2.0 * static_cast<double>(sizeof(Scalar)) * static_cast<double>(n) * static_cast<double>(n);
  if (eigen_bench::skipIfOverMemoryBudget(state, operand_bytes)) return;

  // A Hermitian, strictly diagonally dominant operand: positive definite, with a
  // condition number that does not grow with n, so a failure to factorize is a
  // kernel defect rather than a property of the operand.
  //
  // Deliberately not the textbook A*A^H. That is a 2n^3 product -- asymptotically
  // MORE expensive than the n^3/3 factorization it feeds -- and Google Benchmark
  // re-enters this body about 13 times per cell. Measured at n = 4096 it cost
  // 4.5 s of untimed setup against 0.9 s of measured work, and at the top of the
  // xlarge group it would run to half an hour per scalar. This is O(n^2). The
  // generator is scoped so it is freed before the timed loop, which is what keeps
  // the footprint above at 2n^2 rather than 3n^2.
  PotrfMatrix<Scalar> a(n, n);
  {
    const PotrfMatrix<Scalar> noise = PotrfMatrix<Scalar>::Random(n, n);
    // z + conj(z) is real, so this leaves the diagonal real with no second pass.
    a = noise + noise.adjoint();
  }
  // Random() bounds every entry by 1, so an off-diagonal row sum is at most
  // 2(n-1); 4n on the diagonal makes the matrix strictly diagonally dominant.
  a.diagonal().array() += RealScalar(4 * n);

  PotrfMatrix<Scalar> out(n, n);

  static eigen_bench::ValidatedShapes<Index> validated;
  if (!validated.contains(n)) {
    // L*(L^H x) against A*x for a random x, rather than forming L*L^H: the
    // product would cost as much as the factorization it checks and need a
    // second n-by-n matrix, where this is O(n^2) and O(n) extra. Only the lower
    // triangle is read, so the upper triangle -- which LAPACK leaves holding the
    // original operand and Eigen leaves holding whatever LLT stored there -- is
    // correctly not part of the contract being checked.
    kernel(a, out);
    const PotrfVector<Scalar> x = PotrfVector<Scalar>::Random(n);
    const auto lower = out.template triangularView<Eigen::Lower>();
    const PotrfVector<Scalar> actual = lower * (lower.adjoint() * x).eval();
    const PotrfVector<Scalar> expected = a * x;

    // A reference arm that reported a non-zero `info` produces a garbage or
    // untouched triangle, so it fails here too and needs no separate branch.
    if (!eigen_bench::agreesWithEigen(expected, actual, n)) {
      state.SkipWithError("potrf result disagrees with Eigen at n:" + std::to_string(n));
      return;
    }
    validated.insert(n);
  }

  for (auto _ : state) {
    kernel(a, out);
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }

  eigen_bench::setFlopRate(state, eigen_bench::symmetricFactorizationFlops<Scalar>(n));
}

template <typename Scalar>
static void BM_PotrfEigen(benchmark::State& state) {
  runPotrf<Scalar>(state, EigenPotrfKernel());
}

// Defined only with a vendor linked; REGISTER_COMPARISON_POINT then drops its
// reference-arm argument, so the name is never substituted.
#ifdef EIGEN_BENCH_REFERENCE_ARM
template <typename Scalar>
static void BM_PotrfReference(benchmark::State& state) {
  runPotrf<Scalar>(state, ReferencePotrfKernel());
}
#endif

// The whole square1 grid of ops.toml, in the order its default_groups lists it.
// run.py narrows it with --benchmark_filter; registering less here would make a
// group unreachable.
// clang-format off
#define POTRF_DIM_NAMES {"n"}

#define POTRF_POINT(...)                                                                                              \
  REGISTER_COMPARISON_POINT(POTRF, f32, float,              BM_PotrfEigen, BM_PotrfReference, POTRF_DIM_NAMES, __VA_ARGS__) \
  REGISTER_COMPARISON_POINT(POTRF, f64, double,             BM_PotrfEigen, BM_PotrfReference, POTRF_DIM_NAMES, __VA_ARGS__) \
  REGISTER_COMPARISON_POINT(POTRF, c32, eigen_bench::c32_t, BM_PotrfEigen, BM_PotrfReference, POTRF_DIM_NAMES, __VA_ARGS__) \
  REGISTER_COMPARISON_POINT(POTRF, c64, eigen_bench::c64_t, BM_PotrfEigen, BM_PotrfReference, POTRF_DIM_NAMES, __VA_ARGS__)

#define POTRF_SIZES(POINT) \
  /* fixed_tiny */ \
  POINT(2) POINT(3) POINT(4) POINT(6) POINT(8) POINT(12) POINT(16) \
  /* small */ \
  POINT(24) POINT(32) POINT(48) POINT(64) POINT(96) POINT(128) \
  /* medium */ \
  POINT(192) POINT(256) POINT(384) POINT(512) POINT(768) POINT(1024) \
  /* large */ \
  POINT(1536) POINT(2048) POINT(3072) POINT(4096) \
  /* xlarge */ \
  POINT(6144) POINT(8192) POINT(12288) POINT(16384) \
  /* aliasing */ \
  POINT(100) POINT(200) POINT(257) POINT(500) POINT(1000) POINT(1001) POINT(4097)

POTRF_SIZES(POTRF_POINT)

#undef POTRF_SIZES
#undef POTRF_POINT
#undef POTRF_DIM_NAMES
// clang-format on

EIGEN_BENCH_COMPARISON_MAIN();
