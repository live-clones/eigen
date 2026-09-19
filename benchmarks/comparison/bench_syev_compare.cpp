// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Cross-library comparison benchmarks for SYEV (HEEV for a complex scalar), the
// eigendecomposition of a symmetric (Hermitian) matrix, eigenvectors included.
//
// The reference here is a LAPACK routine rather than a BLAS one; CMakeLists.txt
// reads the marker below and builds this source Eigen-only when the reference
// library supplies no LAPACK.
// EIGEN_BENCH_REFERENCE_FAMILY: lapack
//
// Both arms reduce to tridiagonal form and run the implicit QR iteration on it,
// the like-for-like pair; ?syevd, divide and conquer, is a different algorithm.
// SelfAdjointEigenSolver has no in-place form: compute() copies the operand
// into its eigenvector storage and works there, so the reference arm pays the
// same one copy explicitly. The solver and LAPACK's workspace are set up once
// per shape, so neither arm allocates inside the timed region.

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <complex>
#include <string>

#include "benchmarks/bench_common.h"
#include "benchmarks/comparison/bench_compare.h"

#ifdef EIGEN_BENCH_REFERENCE_ARM
// Fortran LAPACK, declared here rather than pulled from a vendor header so that
// the build needs only the library. See the note in bench_gemm_compare.cpp on
// the integer width.
extern "C" {
void ssyev_(const char* jobz, const char* uplo, const eigen_bench::BlasInt* n, float* a,
            const eigen_bench::BlasInt* lda, float* w, float* work, const eigen_bench::BlasInt* lwork,
            eigen_bench::BlasInt* info);
void dsyev_(const char* jobz, const char* uplo, const eigen_bench::BlasInt* n, double* a,
            const eigen_bench::BlasInt* lda, double* w, double* work, const eigen_bench::BlasInt* lwork,
            eigen_bench::BlasInt* info);
void cheev_(const char* jobz, const char* uplo, const eigen_bench::BlasInt* n, std::complex<float>* a,
            const eigen_bench::BlasInt* lda, float* w, std::complex<float>* work, const eigen_bench::BlasInt* lwork,
            float* rwork, eigen_bench::BlasInt* info);
void zheev_(const char* jobz, const char* uplo, const eigen_bench::BlasInt* n, std::complex<double>* a,
            const eigen_bench::BlasInt* lda, double* w, std::complex<double>* work, const eigen_bench::BlasInt* lwork,
            double* rwork, eigen_bench::BlasInt* info);
}
#endif

using Eigen::Index;
using eigen_bench::BlasInt;
using eigen_bench::ColMatrix;
using eigen_bench::ColVector;

template <typename Scalar>
using RealVector = ColVector<typename Eigen::NumTraits<Scalar>::Real>;

// Both kernels read the lower triangle (uplo = 'L') and expose the
// eigenvectors as columns of a matrix and the eigenvalues as a real vector.
template <typename Scalar>
class EigenSyevKernel {
 public:
  explicit EigenSyevKernel(Index n) : solver_(n) {}

  void operator()(const ColMatrix<Scalar>& a) {
    solver_.compute(a, Eigen::ComputeEigenvectors);
    benchmark::DoNotOptimize(solver_.info());
  }

  const ColMatrix<Scalar>& eigenvectors() const { return solver_.eigenvectors(); }
  const RealVector<Scalar>& eigenvalues() const { return solver_.eigenvalues(); }

 private:
  Eigen::SelfAdjointEigenSolver<ColMatrix<Scalar>> solver_;
};

#ifdef EIGEN_BENCH_REFERENCE_ARM
// One signature for all four: the real routines have no rwork.
static void referenceSyev(BlasInt n, float* a, float* w, float* work, BlasInt lwork, float*, BlasInt* info) {
  const char vectors = 'V', lower = 'L';
  ssyev_(&vectors, &lower, &n, a, &n, w, work, &lwork, info);
}
static void referenceSyev(BlasInt n, double* a, double* w, double* work, BlasInt lwork, double*, BlasInt* info) {
  const char vectors = 'V', lower = 'L';
  dsyev_(&vectors, &lower, &n, a, &n, w, work, &lwork, info);
}
static void referenceSyev(BlasInt n, std::complex<float>* a, float* w, std::complex<float>* work, BlasInt lwork,
                          float* rwork, BlasInt* info) {
  const char vectors = 'V', lower = 'L';
  cheev_(&vectors, &lower, &n, a, &n, w, work, &lwork, rwork, info);
}
static void referenceSyev(BlasInt n, std::complex<double>* a, double* w, std::complex<double>* work, BlasInt lwork,
                          double* rwork, BlasInt* info) {
  const char vectors = 'V', lower = 'L';
  zheev_(&vectors, &lower, &n, a, &n, w, work, &lwork, rwork, info);
}

template <typename Scalar>
class ReferenceSyevKernel {
 public:
  // ?heev needs max(1, 3n-2) reals of rwork; the real routines ignore it.
  explicit ReferenceSyevKernel(Index n) : out_(n, n), w_(n), rwork_(std::max<Index>(1, 3 * n - 2)) {
    // Workspace query: lwork = -1 returns the optimal size in work[0] and
    // reads nothing else.
    Scalar query = Scalar(0);
    BlasInt info = 0;
    referenceSyev(order(), out_.data(), w_.data(), &query, -1, rwork_.data(), &info);
    work_.resize(std::max<Index>(1, static_cast<Index>(Eigen::numext::real(query))));
  }

  void operator()(const ColMatrix<Scalar>& a) {
    // The one copy the Eigen arm makes inside compute(), deliberately.
    out_ = a;
    BlasInt info = 0;
    referenceSyev(order(), out_.data(), w_.data(), work_.data(), static_cast<BlasInt>(work_.size()), rwork_.data(),
                  &info);
    benchmark::DoNotOptimize(info);
  }

  const ColMatrix<Scalar>& eigenvectors() const { return out_; }
  const RealVector<Scalar>& eigenvalues() const { return w_; }

 private:
  BlasInt order() const { return static_cast<BlasInt>(out_.rows()); }

  ColMatrix<Scalar> out_;
  RealVector<Scalar> w_;
  ColVector<Scalar> work_;
  RealVector<Scalar> rwork_;
};
#endif

// Both arms run through this one driver so that allocation, fill, validation,
// the timed region and the counter cannot drift apart between them.
template <typename Scalar, typename Kernel>
static void runSyev(benchmark::State& state) {
  const Index n = static_cast<Index>(state.range(0));

  if (eigen_bench::skipIfDimsExceedBlasInt(state, n)) return;

  // The operand and the solver's working copy, which the eigenvectors overwrite.
  const double operand_bytes =
      2.0 * static_cast<double>(sizeof(Scalar)) * static_cast<double>(n) * static_cast<double>(n);
  if (eigen_bench::skipIfOverMemoryBudget(state, operand_bytes)) return;

  // A dense Hermitian operand, its generator scoped so it is freed before the
  // timed loop. z + conj(z) is real, so the diagonal is real with no second pass.
  ColMatrix<Scalar> a(n, n);
  {
    const ColMatrix<Scalar> noise = ColMatrix<Scalar>::Random(n, n);
    a = noise + noise.adjoint();
  }
  Kernel kernel(n);

  static eigen_bench::ValidatedShapes<Index> validated;
  if (!validated.contains(n)) {
    // A*(V*x) against V*(w.*x) for a random x: O(n^2), and it does not rely on
    // V being unitary.
    kernel(a);
    const ColVector<Scalar> x = ColVector<Scalar>::Random(n);
    const ColMatrix<Scalar>& v = kernel.eigenvectors();
    const ColVector<Scalar> expected = a * (v * x).eval();
    const ColVector<Scalar> actual = v * kernel.eigenvalues().template cast<Scalar>().cwiseProduct(x);

    if (!eigen_bench::agreesWithEigen(expected, actual, n)) {
      state.SkipWithError("syev result disagrees with Eigen at n:" + std::to_string(n));
      return;
    }
    validated.insert(n);
  }

  for (auto _ : state) {
    kernel(a);
    benchmark::DoNotOptimize(kernel.eigenvectors().data());
    benchmark::ClobberMemory();
  }

  eigen_bench::setFlopRate(state, eigen_bench::syevFlops<Scalar>(n));
}

template <typename Scalar>
static void BM_SyevEigen(benchmark::State& state) {
  runSyev<Scalar, EigenSyevKernel<Scalar>>(state);
}

// Defined only with a vendor linked; REGISTER_COMPARISON_POINT then drops its
// reference-arm argument, so the name is never substituted.
#ifdef EIGEN_BENCH_REFERENCE_ARM
template <typename Scalar>
static void BM_SyevReference(benchmark::State& state) {
  runSyev<Scalar, ReferenceSyevKernel<Scalar>>(state);
}
#endif

// The whole grid: run.py narrows it with --benchmark_filter, so a shape absent
// here is out of reach. The grid stops at 4096: an O(9 n^3) solver at 8192 is
// minutes per call.
// clang-format off
#define SYEV_DIM_NAMES {"n"}

#define SYEV_POINT(...)                                                                                            \
  REGISTER_COMPARISON_POINT(SYEV, f32, float,              BM_SyevEigen, BM_SyevReference, SYEV_DIM_NAMES, __VA_ARGS__) \
  REGISTER_COMPARISON_POINT(SYEV, f64, double,             BM_SyevEigen, BM_SyevReference, SYEV_DIM_NAMES, __VA_ARGS__) \
  REGISTER_COMPARISON_POINT(SYEV, c32, eigen_bench::c32_t, BM_SyevEigen, BM_SyevReference, SYEV_DIM_NAMES, __VA_ARGS__) \
  REGISTER_COMPARISON_POINT(SYEV, c64, eigen_bench::c64_t, BM_SyevEigen, BM_SyevReference, SYEV_DIM_NAMES, __VA_ARGS__)

#define SYEV_SIZES(POINT) \
  /* fixed_tiny */ \
  POINT(2) POINT(3) POINT(4) POINT(6) POINT(8) POINT(12) POINT(16) \
  /* small */ \
  POINT(24) POINT(32) POINT(48) POINT(64) POINT(96) POINT(128) \
  /* medium */ \
  POINT(192) POINT(256) POINT(384) POINT(512) POINT(768) POINT(1024) \
  /* large */ \
  POINT(1536) POINT(2048) POINT(3072) POINT(4096) \
  /* aliasing */ \
  POINT(100) POINT(200) POINT(257) POINT(500) POINT(1000) POINT(1001)

SYEV_SIZES(SYEV_POINT)

#undef SYEV_SIZES
#undef SYEV_POINT
#undef SYEV_DIM_NAMES
// clang-format on

EIGEN_BENCH_COMPARISON_MAIN();
