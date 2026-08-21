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
#include <mutex>
#include <set>
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
using PotrfMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

template <typename Scalar>
using PotrfVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

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

  if (!fitsBlasInt(n)) {
    state.SkipWithError("dimension does not fit the reference LAPACK integer width");
    return;
  }

  // A*A^H is positive semi-definite; the shift makes it definite with a
  // condition number that does not grow with n, so a failure to factorize is a
  // kernel defect rather than a property of the operand.
  const PotrfMatrix<Scalar> noise = PotrfMatrix<Scalar>::Random(n, n);
  PotrfMatrix<Scalar> a = noise * noise.adjoint();
  a.diagonal().array() += RealScalar(n);

  PotrfMatrix<Scalar> out(n, n);

  // Correctness is a deterministic property of (Scalar, kernel, shape); see the
  // note in bench_gemm_compare.cpp for why it is not re-checked per entry.
  static std::set<Index> validated_orders;
  static std::mutex validated_orders_mutex;
  bool order_is_validated;
  {
    const std::lock_guard<std::mutex> guard(validated_orders_mutex);
    order_is_validated = validated_orders.count(n) != 0;
  }

  if (!order_is_validated) {
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

    const RealScalar tolerance =
        RealScalar(64) * Eigen::numext::sqrt(RealScalar(n)) * Eigen::NumTraits<RealScalar>::epsilon();
    const RealScalar magnitude = Eigen::numext::maxi(expected.norm(), actual.norm());
    // Negated so that a NaN result fails rather than passes. A reference arm
    // that reported a non-zero `info` produces a garbage or untouched triangle,
    // so it fails here too and needs no separate branch.
    if (!((actual - expected).norm() <= tolerance * magnitude)) {
      state.SkipWithError("potrf result disagrees with Eigen at n:" + std::to_string(n));
      return;
    }
    const std::lock_guard<std::mutex> guard(validated_orders_mutex);
    validated_orders.insert(n);
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
  /* aliasing */ \
  POINT(100) POINT(200) POINT(257) POINT(500) POINT(1000) POINT(1001) POINT(4097)

POTRF_SIZES(POTRF_POINT)

#undef POTRF_SIZES
#undef POTRF_POINT
#undef POTRF_DIM_NAMES
// clang-format on

EIGEN_BENCH_COMPARISON_MAIN();
