// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Cross-library comparison benchmarks for GEQRF, the Householder QR
// factorization of an m-by-n matrix with m >= n.
//
// The reference here is a LAPACK routine rather than a BLAS one; CMakeLists.txt
// reads the marker below and builds this source Eigen-only when the reference
// library supplies no LAPACK.
// EIGEN_BENCH_REFERENCE_FAMILY: lapack
//
// Destructive, and handled as GETRF is: both arms copy the operand into a
// destination allocated once and factorize the copy, the Eigen arm through the
// in-place HouseholderQR<Ref<MatrixType>> of doc/InplaceDecomposition.dox bound
// to the destination once per shape. LAPACK's tau and work arrays and Eigen's
// coefficient and temporary vectors are all set up once per shape, so neither
// arm allocates inside the timed region.

#include <Eigen/Core>
#include <Eigen/Householder>
#include <Eigen/QR>
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
void sgeqrf_(const eigen_bench::BlasInt* m, const eigen_bench::BlasInt* n, float* a, const eigen_bench::BlasInt* lda,
             float* tau, float* work, const eigen_bench::BlasInt* lwork, eigen_bench::BlasInt* info)
    EIGEN_BENCH_FORTRAN_SYMBOL(sgeqrf);
void dgeqrf_(const eigen_bench::BlasInt* m, const eigen_bench::BlasInt* n, double* a, const eigen_bench::BlasInt* lda,
             double* tau, double* work, const eigen_bench::BlasInt* lwork, eigen_bench::BlasInt* info)
    EIGEN_BENCH_FORTRAN_SYMBOL(dgeqrf);
void cgeqrf_(const eigen_bench::BlasInt* m, const eigen_bench::BlasInt* n, std::complex<float>* a,
             const eigen_bench::BlasInt* lda, std::complex<float>* tau, std::complex<float>* work,
             const eigen_bench::BlasInt* lwork, eigen_bench::BlasInt* info) EIGEN_BENCH_FORTRAN_SYMBOL(cgeqrf);
void zgeqrf_(const eigen_bench::BlasInt* m, const eigen_bench::BlasInt* n, std::complex<double>* a,
             const eigen_bench::BlasInt* lda, std::complex<double>* tau, std::complex<double>* work,
             const eigen_bench::BlasInt* lwork, eigen_bench::BlasInt* info) EIGEN_BENCH_FORTRAN_SYMBOL(zgeqrf);
}
#endif

using Eigen::Index;
using eigen_bench::BlasInt;
using eigen_bench::ColMatrix;
using eigen_bench::ColVector;

// Both kernels leave R in the upper triangle of the destination and the
// Householder vectors below it, and can apply Q to a vector for the check.
template <typename Scalar>
class EigenGeqrfKernel {
 public:
  // Binds to `out`, which must already hold the operand: the constructor
  // factorizes what it is given.
  explicit EigenGeqrfKernel(ColMatrix<Scalar>& out) : qr_(out) {}

  void operator()(const ColMatrix<Scalar>& a) { qr_.compute(a); }

  ColVector<Scalar> applyQ(const ColVector<Scalar>& v) const { return qr_.householderQ() * v; }

 private:
  Eigen::HouseholderQR<Eigen::Ref<ColMatrix<Scalar>>> qr_;
};

#ifdef EIGEN_BENCH_REFERENCE_ARM
static void referenceGeqrf(BlasInt m, BlasInt n, float* a, float* tau, float* work, BlasInt lwork, BlasInt* info) {
  sgeqrf_(&m, &n, a, &m, tau, work, &lwork, info);
}
static void referenceGeqrf(BlasInt m, BlasInt n, double* a, double* tau, double* work, BlasInt lwork, BlasInt* info) {
  dgeqrf_(&m, &n, a, &m, tau, work, &lwork, info);
}
static void referenceGeqrf(BlasInt m, BlasInt n, std::complex<float>* a, std::complex<float>* tau,
                           std::complex<float>* work, BlasInt lwork, BlasInt* info) {
  cgeqrf_(&m, &n, a, &m, tau, work, &lwork, info);
}
static void referenceGeqrf(BlasInt m, BlasInt n, std::complex<double>* a, std::complex<double>* tau,
                           std::complex<double>* work, BlasInt lwork, BlasInt* info) {
  zgeqrf_(&m, &n, a, &m, tau, work, &lwork, info);
}

template <typename Scalar>
class ReferenceGeqrfKernel {
 public:
  explicit ReferenceGeqrfKernel(ColMatrix<Scalar>& out) : out_(out), tau_(out.cols()) {
    // Workspace query: lwork = -1 returns the optimal size in work[0] and
    // reads nothing else.
    Scalar query = Scalar(0);
    BlasInt info = 0;
    referenceGeqrf(rows(), cols(), out_.data(), tau_.data(), &query, -1, &info);
    work_.resize(std::max<Index>(1, static_cast<Index>(Eigen::numext::real(query))));
  }

  void operator()(const ColMatrix<Scalar>& a) {
    // The same copy the Eigen arm makes, deliberately.
    out_ = a;
    BlasInt info = 0;
    referenceGeqrf(rows(), cols(), out_.data(), tau_.data(), work_.data(), static_cast<BlasInt>(work_.size()), &info);
    benchmark::DoNotOptimize(info);
  }

  ColVector<Scalar> applyQ(const ColVector<Scalar>& v) const {
    // ?geqrf's Q is the product of the (I - tau_i v_i v_i^H) with v_i stored
    // below the diagonal, which is what householderSequence builds from that
    // storage.
    return Eigen::householderSequence(out_, tau_) * v;
  }

 private:
  BlasInt rows() const { return static_cast<BlasInt>(out_.rows()); }
  BlasInt cols() const { return static_cast<BlasInt>(out_.cols()); }

  ColMatrix<Scalar>& out_;
  ColVector<Scalar> tau_;
  ColVector<Scalar> work_;
};
#endif

// Both arms run through this one driver so that allocation, fill, validation,
// the timed region and the counter cannot drift apart between them.
template <typename Scalar, typename Kernel>
static void runGeqrf(benchmark::State& state) {
  const Index m = static_cast<Index>(state.range(0));
  const Index n = static_cast<Index>(state.range(1));

  if (eigen_bench::skipIfDimsExceedBlasInt(state, m, n)) return;

  // The operand and the destination.
  const double operand_bytes =
      2.0 * static_cast<double>(sizeof(Scalar)) * static_cast<double>(m) * static_cast<double>(n);
  if (eigen_bench::skipIfOverMemoryBudget(state, operand_bytes)) return;

  const ColMatrix<Scalar> a = ColMatrix<Scalar>::Random(m, n);
  ColMatrix<Scalar> out = a;
  Kernel kernel(out);

  static eigen_bench::ValidatedShapes<std::array<Index, 2>> validated;
  const std::array<Index, 2> shape = {m, n};
  if (!validated.contains(shape)) {
    // Q*(R*x) against A*x for a random x: O(m*n) through the Householder
    // vectors instead of forming Q.
    kernel(a);
    const ColVector<Scalar> x = ColVector<Scalar>::Random(n);
    ColVector<Scalar> rx = ColVector<Scalar>::Zero(m);
    rx.head(n) = out.topRows(n).template triangularView<Eigen::Upper>() * x;
    const ColVector<Scalar> expected = a * x;
    const ColVector<Scalar> actual = kernel.applyQ(rx);

    if (!eigen_bench::agreesWithEigen(expected, actual, m)) {
      state.SkipWithError("geqrf result disagrees with Eigen at m:" + std::to_string(m) + " n:" + std::to_string(n));
      return;
    }
    validated.insert(shape);
  }

  for (auto _ : state) {
    kernel(a);
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }

  eigen_bench::setFlopRate(state, eigen_bench::geqrfFlops<Scalar>(m, n));
}

template <typename Scalar>
static void BM_GeqrfEigen(benchmark::State& state) {
  runGeqrf<Scalar, EigenGeqrfKernel<Scalar>>(state);
}

// Defined only with a vendor linked; REGISTER_COMPARISON_POINT then drops its
// reference-arm argument, so the name is never substituted.
#ifdef EIGEN_BENCH_REFERENCE_ARM
template <typename Scalar>
static void BM_GeqrfReference(benchmark::State& state) {
  runGeqrf<Scalar, ReferenceGeqrfKernel<Scalar>>(state);
}
#endif

// The whole grid: run.py narrows it with --benchmark_filter, so a shape absent
// here is out of reach. Every shape has m >= n, the case the flop count is for.
// clang-format off
#define GEQRF_DIM_NAMES {"m", "n"}

#define GEQRF_POINT(...)                                                                                              \
  REGISTER_COMPARISON_POINT(GEQRF, f32, float,              BM_GeqrfEigen, BM_GeqrfReference, GEQRF_DIM_NAMES, __VA_ARGS__) \
  REGISTER_COMPARISON_POINT(GEQRF, f64, double,             BM_GeqrfEigen, BM_GeqrfReference, GEQRF_DIM_NAMES, __VA_ARGS__) \
  REGISTER_COMPARISON_POINT(GEQRF, c32, eigen_bench::c32_t, BM_GeqrfEigen, BM_GeqrfReference, GEQRF_DIM_NAMES, __VA_ARGS__) \
  REGISTER_COMPARISON_POINT(GEQRF, c64, eigen_bench::c64_t, BM_GeqrfEigen, BM_GeqrfReference, GEQRF_DIM_NAMES, __VA_ARGS__)

#define GEQRF_SIZES(POINT) \
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
  /* tall and skinny */ \
  POINT(4096,8) POINT(4096,64) POINT(4096,256) POINT(4096,1024) \
  POINT(10000,8) POINT(10000,100) POINT(10000,1000) POINT(10000,4000) \
  POINT(100000,8) POINT(100000,64) POINT(100000,256)

GEQRF_SIZES(GEQRF_POINT)

#undef GEQRF_SIZES
#undef GEQRF_POINT
#undef GEQRF_DIM_NAMES
// clang-format on

EIGEN_BENCH_COMPARISON_MAIN();
