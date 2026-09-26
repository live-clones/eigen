// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Cross-library comparison benchmarks for GESDD, the singular value
// decomposition by divide and conquer of an m-by-n matrix with m >= n, with
// the thin U (m-by-n) and V (n-by-n).
//
// The reference here is a LAPACK routine rather than a BLAS one; CMakeLists.txt
// reads the marker below and builds this source Eigen-only when the reference
// library supplies no LAPACK.
// EIGEN_BENCH_REFERENCE_FAMILY: lapack
//
// BDCSVD is Eigen's divide-and-conquer SVD, so ?gesdd with jobz = 'S' is its
// counterpart, not ?gesvd. Neither works in place: BDCSVD::compute copies the
// operand, and the reference arm pays the same one copy explicitly. The solver,
// sized once per shape, keeps its workspace across calls, as do LAPACK's work
// arrays.

#include <Eigen/Core>
#include <Eigen/SVD>
#include <algorithm>
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
void sgesdd_(const char* jobz, const eigen_bench::BlasInt* m, const eigen_bench::BlasInt* n, float* a,
             const eigen_bench::BlasInt* lda, float* s, float* u, const eigen_bench::BlasInt* ldu, float* vt,
             const eigen_bench::BlasInt* ldvt, float* work, const eigen_bench::BlasInt* lwork,
             eigen_bench::BlasInt* iwork, eigen_bench::BlasInt* info) EIGEN_BENCH_FORTRAN_SYMBOL(sgesdd);
void dgesdd_(const char* jobz, const eigen_bench::BlasInt* m, const eigen_bench::BlasInt* n, double* a,
             const eigen_bench::BlasInt* lda, double* s, double* u, const eigen_bench::BlasInt* ldu, double* vt,
             const eigen_bench::BlasInt* ldvt, double* work, const eigen_bench::BlasInt* lwork,
             eigen_bench::BlasInt* iwork, eigen_bench::BlasInt* info) EIGEN_BENCH_FORTRAN_SYMBOL(dgesdd);
void cgesdd_(const char* jobz, const eigen_bench::BlasInt* m, const eigen_bench::BlasInt* n, std::complex<float>* a,
             const eigen_bench::BlasInt* lda, float* s, std::complex<float>* u, const eigen_bench::BlasInt* ldu,
             std::complex<float>* vt, const eigen_bench::BlasInt* ldvt, std::complex<float>* work,
             const eigen_bench::BlasInt* lwork, float* rwork, eigen_bench::BlasInt* iwork, eigen_bench::BlasInt* info)
    EIGEN_BENCH_FORTRAN_SYMBOL(cgesdd);
void zgesdd_(const char* jobz, const eigen_bench::BlasInt* m, const eigen_bench::BlasInt* n, std::complex<double>* a,
             const eigen_bench::BlasInt* lda, double* s, std::complex<double>* u, const eigen_bench::BlasInt* ldu,
             std::complex<double>* vt, const eigen_bench::BlasInt* ldvt, std::complex<double>* work,
             const eigen_bench::BlasInt* lwork, double* rwork, eigen_bench::BlasInt* iwork, eigen_bench::BlasInt* info)
    EIGEN_BENCH_FORTRAN_SYMBOL(zgesdd);
}
#endif

using Eigen::Index;
using eigen_bench::BlasInt;
using eigen_bench::ColMatrix;
using eigen_bench::ColVector;

template <typename Scalar>
using RealVector = ColVector<typename Eigen::NumTraits<Scalar>::Real>;

// Both kernels expose U (m-by-r), V (n-by-r, not its adjoint) and the singular
// values, r = min(m, n).
template <typename Scalar>
class EigenGesddKernel {
 public:
  EigenGesddKernel(Index m, Index n) : svd_(m, n) {}

  void operator()(const ColMatrix<Scalar>& a) {
    svd_.compute(a);
    benchmark::DoNotOptimize(svd_.info());
  }

  const ColMatrix<Scalar>& matrixU() const { return svd_.matrixU(); }
  ColMatrix<Scalar> matrixV() const { return svd_.matrixV(); }
  const RealVector<Scalar>& singularValues() const { return svd_.singularValues(); }

 private:
  Eigen::BDCSVD<ColMatrix<Scalar>, Eigen::ComputeThinU | Eigen::ComputeThinV> svd_;
};

#ifdef EIGEN_BENCH_REFERENCE_ARM
// One signature for all four: the real routines have no rwork. jobz = 'S'
// asks for the first min(m, n) columns of U and rows of V^H.
static void referenceGesdd(BlasInt m, BlasInt n, float* a, float* s, float* u, float* vt, float* work, BlasInt lwork,
                           float*, BlasInt* iwork, BlasInt* info) {
  const char thin = 'S';
  const BlasInt r = std::min(m, n);
  sgesdd_(&thin, &m, &n, a, &m, s, u, &m, vt, &r, work, &lwork, iwork, info);
}
static void referenceGesdd(BlasInt m, BlasInt n, double* a, double* s, double* u, double* vt, double* work,
                           BlasInt lwork, double*, BlasInt* iwork, BlasInt* info) {
  const char thin = 'S';
  const BlasInt r = std::min(m, n);
  dgesdd_(&thin, &m, &n, a, &m, s, u, &m, vt, &r, work, &lwork, iwork, info);
}
static void referenceGesdd(BlasInt m, BlasInt n, std::complex<float>* a, float* s, std::complex<float>* u,
                           std::complex<float>* vt, std::complex<float>* work, BlasInt lwork, float* rwork,
                           BlasInt* iwork, BlasInt* info) {
  const char thin = 'S';
  const BlasInt r = std::min(m, n);
  cgesdd_(&thin, &m, &n, a, &m, s, u, &m, vt, &r, work, &lwork, rwork, iwork, info);
}
static void referenceGesdd(BlasInt m, BlasInt n, std::complex<double>* a, double* s, std::complex<double>* u,
                           std::complex<double>* vt, std::complex<double>* work, BlasInt lwork, double* rwork,
                           BlasInt* iwork, BlasInt* info) {
  const char thin = 'S';
  const BlasInt r = std::min(m, n);
  zgesdd_(&thin, &m, &n, a, &m, s, u, &m, vt, &r, work, &lwork, rwork, iwork, info);
}

template <typename Scalar>
class ReferenceGesddKernel {
 public:
  ReferenceGesddKernel(Index m, Index n)
      : out_(m, n),
        s_(std::min(m, n)),
        u_(m, std::min(m, n)),
        vt_(std::min(m, n), n),
        iwork_(8 * std::min(m, n)),
        rwork_(rworkSize(m, n)) {
    // Workspace query: lwork = -1 returns the optimal size in work[0] and
    // reads nothing else.
    Scalar query = Scalar(0);
    BlasInt info = 0;
    referenceGesdd(rows(), cols(), out_.data(), s_.data(), u_.data(), vt_.data(), &query, -1, rwork_.data(),
                   iwork_.data(), &info);
    work_.resize(std::max<Index>(1, static_cast<Index>(Eigen::numext::real(query))));
  }

  void operator()(const ColMatrix<Scalar>& a) {
    // The one copy the Eigen arm makes inside compute(), deliberately.
    out_ = a;
    BlasInt info = 0;
    referenceGesdd(rows(), cols(), out_.data(), s_.data(), u_.data(), vt_.data(), work_.data(),
                   static_cast<BlasInt>(work_.size()), rwork_.data(), iwork_.data(), &info);
    benchmark::DoNotOptimize(info);
  }

  const ColMatrix<Scalar>& matrixU() const { return u_; }
  ColMatrix<Scalar> matrixV() const { return vt_.adjoint(); }
  const RealVector<Scalar>& singularValues() const { return s_; }

 private:
  // ?gesdd's real workspace for jobz = 'S', from its documentation; the real
  // routines ignore it.
  static Index rworkSize(Index m, Index n) {
    const Index r = std::min(m, n);
    const Index big = std::max(m, n);
    return std::max<Index>(1, std::max(5 * r * r + 5 * r, 2 * big * r + 2 * r * r + r));
  }
  BlasInt rows() const { return static_cast<BlasInt>(out_.rows()); }
  BlasInt cols() const { return static_cast<BlasInt>(out_.cols()); }

  ColMatrix<Scalar> out_;
  RealVector<Scalar> s_;
  ColMatrix<Scalar> u_;
  ColMatrix<Scalar> vt_;
  ColVector<Scalar> work_;
  Eigen::Matrix<BlasInt, Eigen::Dynamic, 1> iwork_;
  RealVector<Scalar> rwork_;
};
#endif

// Both arms run through this one driver so that allocation, fill, validation,
// the timed region and the counter cannot drift apart between them.
template <typename Scalar, typename Kernel>
static void runGesdd(benchmark::State& state) {
  const Index m = static_cast<Index>(state.range(0));
  const Index n = static_cast<Index>(state.range(1));
  const Index r = std::min(m, n);

  if (eigen_bench::skipIfDimsExceedBlasInt(state, m, n)) return;

  // The operand, the working copy, U and V^H.
  const double operand_bytes =
      static_cast<double>(sizeof(Scalar)) *
      (2.0 * static_cast<double>(m) * static_cast<double>(n) + static_cast<double>(m) * static_cast<double>(r) +
       static_cast<double>(r) * static_cast<double>(n));
  if (eigen_bench::skipIfOverMemoryBudget(state, operand_bytes)) return;

  const ColMatrix<Scalar> a = ColMatrix<Scalar>::Random(m, n);
  Kernel kernel(m, n);

  static eigen_bench::ValidatedShapes<std::array<Index, 2>> validated;
  const std::array<Index, 2> shape = {m, n};
  if (!validated.contains(shape)) {
    // A*(V*x) against U*(s.*x) for a random x of length min(m, n): O(m*n), and
    // it does not rely on U or V being unitary.
    kernel(a);
    const ColVector<Scalar> x = ColVector<Scalar>::Random(r);
    const ColMatrix<Scalar> v = kernel.matrixV();
    const ColVector<Scalar> expected = a * (v * x).eval();
    const ColVector<Scalar> actual = kernel.matrixU() * kernel.singularValues().template cast<Scalar>().cwiseProduct(x);

    if (!eigen_bench::agreesWithEigen(expected, actual, std::max(m, n))) {
      state.SkipWithError("gesdd result disagrees with Eigen at m:" + std::to_string(m) + " n:" + std::to_string(n));
      return;
    }
    validated.insert(shape);
  }

  for (auto _ : state) {
    kernel(a);
    benchmark::DoNotOptimize(kernel.matrixU().data());
    benchmark::ClobberMemory();
  }

  eigen_bench::setFlopRate(state, eigen_bench::gesvdFlops<Scalar>(std::max(m, n), r));
}

template <typename Scalar>
static void BM_GesddEigen(benchmark::State& state) {
  runGesdd<Scalar, EigenGesddKernel<Scalar>>(state);
}

// Defined only with a vendor linked; REGISTER_COMPARISON_POINT then drops its
// reference-arm argument, so the name is never substituted.
#ifdef EIGEN_BENCH_REFERENCE_ARM
template <typename Scalar>
static void BM_GesddReference(benchmark::State& state) {
  runGesdd<Scalar, ReferenceGesddKernel<Scalar>>(state);
}
#endif

// The whole grid: run.py narrows it with --benchmark_filter, so a shape absent
// here is out of reach. Every shape has m >= n. The square grid stops at 4096:
// an O(22 n^3) solver at 8192 is minutes per call.
// clang-format off
#define GESDD_DIM_NAMES {"m", "n"}

#define GESDD_POINT(...)                                                                                              \
  REGISTER_COMPARISON_POINT(GESDD, f32, float,              BM_GesddEigen, BM_GesddReference, GESDD_DIM_NAMES, __VA_ARGS__) \
  REGISTER_COMPARISON_POINT(GESDD, f64, double,             BM_GesddEigen, BM_GesddReference, GESDD_DIM_NAMES, __VA_ARGS__) \
  REGISTER_COMPARISON_POINT(GESDD, c32, eigen_bench::c32_t, BM_GesddEigen, BM_GesddReference, GESDD_DIM_NAMES, __VA_ARGS__) \
  REGISTER_COMPARISON_POINT(GESDD, c64, eigen_bench::c64_t, BM_GesddEigen, BM_GesddReference, GESDD_DIM_NAMES, __VA_ARGS__)

#define GESDD_SIZES(POINT) \
  /* fixed_tiny */ \
  POINT(2,2) POINT(3,3) POINT(4,4) POINT(6,6) POINT(8,8) POINT(12,12) POINT(16,16) \
  /* small */ \
  POINT(24,24) POINT(32,32) POINT(48,48) POINT(64,64) POINT(96,96) POINT(128,128) \
  /* medium */ \
  POINT(192,192) POINT(256,256) POINT(384,384) POINT(512,512) POINT(768,768) POINT(1024,1024) \
  /* large */ \
  POINT(1536,1536) POINT(2048,2048) POINT(3072,3072) POINT(4096,4096) \
  /* aliasing */ \
  POINT(100,100) POINT(200,200) POINT(257,257) POINT(500,500) POINT(1000,1000) POINT(1001,1001) \
  /* tall and skinny */ \
  POINT(4096,64) POINT(4096,256) POINT(4096,1024) \
  POINT(10000,8) POINT(10000,100) POINT(10000,1000)

GESDD_SIZES(GESDD_POINT)

#undef GESDD_SIZES
#undef GESDD_POINT
#undef GESDD_DIM_NAMES
// clang-format on

EIGEN_BENCH_COMPARISON_MAIN();
