// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Cross-library comparison benchmarks for GEEV, the eigendecomposition of a
// general (nonsymmetric) matrix with its right eigenvectors.
//
// The reference here is a LAPACK routine rather than a BLAS one; CMakeLists.txt
// reads the marker below and builds this source Eigen-only when the reference
// library supplies no LAPACK.
// EIGEN_BENCH_REFERENCE_FAMILY: lapack
//
// Hessenberg reduction, QR iteration to the Schur form, then back-substitution
// for the eigenvectors on both sides; ?geev balances the matrix first, which
// EigenSolver does not. Neither works in place: EigenSolver (ComplexEigenSolver
// for a complex scalar) copies the operand, and the reference arm pays the same
// one copy explicitly. The eigenvalues and eigenvectors of a real matrix are
// complex; ?geev packs each conjugate pair into two real columns and
// EigenSolver stores them the same way, so both arms unpack to complex only for
// the check, outside the timed region.

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <complex>
#include <string>
#include <type_traits>

#include "benchmarks/bench_common.h"
#include "benchmarks/comparison/bench_compare.h"

#ifdef EIGEN_BENCH_REFERENCE_ARM
// Fortran LAPACK, declared here rather than pulled from a vendor header so that
// the build needs only the library. See the note in bench_gemm_compare.cpp on
// the integer width.
extern "C" {
void sgeev_(const char* jobvl, const char* jobvr, const eigen_bench::BlasInt* n, float* a,
            const eigen_bench::BlasInt* lda, float* wr, float* wi, float* vl, const eigen_bench::BlasInt* ldvl,
            float* vr, const eigen_bench::BlasInt* ldvr, float* work, const eigen_bench::BlasInt* lwork,
            eigen_bench::BlasInt* info) EIGEN_BENCH_FORTRAN_SYMBOL(sgeev);
void dgeev_(const char* jobvl, const char* jobvr, const eigen_bench::BlasInt* n, double* a,
            const eigen_bench::BlasInt* lda, double* wr, double* wi, double* vl, const eigen_bench::BlasInt* ldvl,
            double* vr, const eigen_bench::BlasInt* ldvr, double* work, const eigen_bench::BlasInt* lwork,
            eigen_bench::BlasInt* info) EIGEN_BENCH_FORTRAN_SYMBOL(dgeev);
void cgeev_(const char* jobvl, const char* jobvr, const eigen_bench::BlasInt* n, std::complex<float>* a,
            const eigen_bench::BlasInt* lda, std::complex<float>* w, std::complex<float>* vl,
            const eigen_bench::BlasInt* ldvl, std::complex<float>* vr, const eigen_bench::BlasInt* ldvr,
            std::complex<float>* work, const eigen_bench::BlasInt* lwork, float* rwork, eigen_bench::BlasInt* info)
    EIGEN_BENCH_FORTRAN_SYMBOL(cgeev);
void zgeev_(const char* jobvl, const char* jobvr, const eigen_bench::BlasInt* n, std::complex<double>* a,
            const eigen_bench::BlasInt* lda, std::complex<double>* w, std::complex<double>* vl,
            const eigen_bench::BlasInt* ldvl, std::complex<double>* vr, const eigen_bench::BlasInt* ldvr,
            std::complex<double>* work, const eigen_bench::BlasInt* lwork, double* rwork, eigen_bench::BlasInt* info)
    EIGEN_BENCH_FORTRAN_SYMBOL(zgeev);
}
#endif

using Eigen::Index;
using eigen_bench::BlasInt;
using eigen_bench::ColMatrix;
using eigen_bench::ColVector;

template <typename Scalar>
using RealScalarOf = typename Eigen::NumTraits<Scalar>::Real;
template <typename Scalar>
using RealVector = ColVector<RealScalarOf<Scalar>>;
template <typename Scalar>
using ComplexScalarOf = std::complex<RealScalarOf<Scalar>>;
template <typename Scalar>
using ComplexMatrix = ColMatrix<ComplexScalarOf<Scalar>>;
template <typename Scalar>
using ComplexVector = ColVector<ComplexScalarOf<Scalar>>;
template <typename Scalar>
using IsComplex = std::integral_constant<bool, Eigen::NumTraits<Scalar>::IsComplex != 0>;

// Both kernels expose the eigenvalues as a complex vector and the right
// eigenvectors as the columns of a complex matrix, for the check only.
template <typename Scalar>
class EigenGeevKernel {
  using Solver = std::conditional_t<IsComplex<Scalar>::value, Eigen::ComplexEigenSolver<ColMatrix<Scalar>>,
                                    Eigen::EigenSolver<ColMatrix<Scalar>>>;

 public:
  explicit EigenGeevKernel(Index n) : solver_(n) {}

  void operator()(const ColMatrix<Scalar>& a) {
    solver_.compute(a, true);
    benchmark::DoNotOptimize(solver_.info());
  }

  ComplexVector<Scalar> eigenvalues() const { return solver_.eigenvalues(); }
  ComplexMatrix<Scalar> eigenvectors() const { return solver_.eigenvectors(); }

 private:
  Solver solver_;
};

#ifdef EIGEN_BENCH_REFERENCE_ARM
// One signature for all four. The real routines return the eigenvalues as
// (wr, wi) and take no rwork; the complex ones return w and ignore wr and wi.
static void referenceGeev(BlasInt n, float* a, float* wr, float* wi, std::complex<float>*, float*, float* vr,
                          float* work, BlasInt lwork, BlasInt* info) {
  const char no_left = 'N', right = 'V';
  const BlasInt one = 1;
  sgeev_(&no_left, &right, &n, a, &n, wr, wi, nullptr, &one, vr, &n, work, &lwork, info);
}
static void referenceGeev(BlasInt n, double* a, double* wr, double* wi, std::complex<double>*, double*, double* vr,
                          double* work, BlasInt lwork, BlasInt* info) {
  const char no_left = 'N', right = 'V';
  const BlasInt one = 1;
  dgeev_(&no_left, &right, &n, a, &n, wr, wi, nullptr, &one, vr, &n, work, &lwork, info);
}
static void referenceGeev(BlasInt n, std::complex<float>* a, float*, float*, std::complex<float>* w, float* rwork,
                          std::complex<float>* vr, std::complex<float>* work, BlasInt lwork, BlasInt* info) {
  const char no_left = 'N', right = 'V';
  const BlasInt one = 1;
  cgeev_(&no_left, &right, &n, a, &n, w, nullptr, &one, vr, &n, work, &lwork, rwork, info);
}
static void referenceGeev(BlasInt n, std::complex<double>* a, double*, double*, std::complex<double>* w, double* rwork,
                          std::complex<double>* vr, std::complex<double>* work, BlasInt lwork, BlasInt* info) {
  const char no_left = 'N', right = 'V';
  const BlasInt one = 1;
  zgeev_(&no_left, &right, &n, a, &n, w, nullptr, &one, vr, &n, work, &lwork, rwork, info);
}

template <typename Scalar>
class ReferenceGeevKernel {
  using Real = RealScalarOf<Scalar>;
  using Complex = ComplexScalarOf<Scalar>;

 public:
  // ?geev for a complex scalar needs 2n reals of rwork; the real routines
  // ignore it.
  explicit ReferenceGeevKernel(Index n) : out_(n, n), vr_(n, n), wr_(n), wi_(n), w_(n), rwork_(2 * n) {
    // Workspace query: lwork = -1 returns the optimal size in work[0] and
    // reads nothing else.
    Scalar query = Scalar(0);
    BlasInt info = 0;
    referenceGeev(order(), out_.data(), wr_.data(), wi_.data(), w_.data(), rwork_.data(), vr_.data(), &query, -1,
                  &info);
    work_.resize(std::max<Index>(1, static_cast<Index>(Eigen::numext::real(query))));
  }

  void operator()(const ColMatrix<Scalar>& a) {
    // The one copy the Eigen arm makes inside compute(), deliberately.
    out_ = a;
    BlasInt info = 0;
    referenceGeev(order(), out_.data(), wr_.data(), wi_.data(), w_.data(), rwork_.data(), vr_.data(), work_.data(),
                  static_cast<BlasInt>(work_.size()), &info);
    benchmark::DoNotOptimize(info);
  }

  ComplexVector<Scalar> eigenvalues() const { return eigenvalues(IsComplex<Scalar>()); }
  ComplexMatrix<Scalar> eigenvectors() const { return eigenvectors(IsComplex<Scalar>()); }

 private:
  // Only the overload for the scalar in hand is ever instantiated.
  ComplexVector<Scalar> eigenvalues(std::true_type) const { return w_; }
  ComplexVector<Scalar> eigenvalues(std::false_type) const {
    ComplexVector<Scalar> values(wr_.size());
    values.real() = wr_;
    values.imag() = wi_;
    return values;
  }
  ComplexMatrix<Scalar> eigenvectors(std::true_type) const { return vr_; }
  ComplexMatrix<Scalar> eigenvectors(std::false_type) const {
    const Index n = vr_.rows();
    ComplexMatrix<Scalar> vectors(n, n);
    for (Index j = 0; j < n; ++j) {
      if (wi_(j) == Real(0)) {
        vectors.col(j) = vr_.col(j).template cast<Complex>();
      } else {
        // A conjugate pair (j, j+1): column j holds the real part of both
        // eigenvectors and column j+1 the imaginary part of the first.
        vectors.col(j).real() = vr_.col(j);
        vectors.col(j).imag() = vr_.col(j + 1);
        vectors.col(j + 1) = vectors.col(j).conjugate();
        ++j;
      }
    }
    return vectors;
  }

  BlasInt order() const { return static_cast<BlasInt>(out_.rows()); }

  ColMatrix<Scalar> out_;
  ColMatrix<Scalar> vr_;
  RealVector<Scalar> wr_;
  RealVector<Scalar> wi_;
  ColVector<Complex> w_;
  ColVector<Scalar> work_;
  RealVector<Scalar> rwork_;
};
#endif

// Both arms run through this one driver so that allocation, fill, validation,
// the timed region and the counter cannot drift apart between them.
template <typename Scalar, typename Kernel>
static void runGeev(benchmark::State& state) {
  const Index n = static_cast<Index>(state.range(0));

  if (eigen_bench::skipIfDimsExceedBlasInt(state, n)) return;

  // The operand, the solver's working copy and its eigenvector matrix, plus
  // the check's four complex n-by-n matrices.
  const double operand_bytes =
      11.0 * static_cast<double>(sizeof(Scalar)) * static_cast<double>(n) * static_cast<double>(n);
  if (eigen_bench::skipIfOverMemoryBudget(state, operand_bytes)) return;

  const ColMatrix<Scalar> a = ColMatrix<Scalar>::Random(n, n);
  Kernel kernel(n);

  static eigen_bench::ValidatedShapes<Index> validated;
  if (!validated.contains(n)) {
    // A*V against V*diag(w), the full residual rather than its product with a
    // random vector: the eigenvector matrix of a nonsymmetric A is not
    // unitary, and a check through V*x would be at the mercy of its
    // conditioning. One complex product, once per shape, outside the timed
    // region.
    kernel(a);
    const ComplexMatrix<Scalar> v = kernel.eigenvectors();
    const ComplexVector<Scalar> w = kernel.eigenvalues();
    const ComplexMatrix<Scalar> expected = a.template cast<ComplexScalarOf<Scalar>>() * v;
    const ComplexMatrix<Scalar> actual = v * w.asDiagonal();

    if (!eigen_bench::agreesWithEigen(expected, actual, n)) {
      state.SkipWithError("geev result disagrees with Eigen at n:" + std::to_string(n));
      return;
    }
    validated.insert(n);
  }

  for (auto _ : state) {
    kernel(a);
    benchmark::ClobberMemory();
  }

  eigen_bench::setFlopRate(state, eigen_bench::geevFlops<Scalar>(n));
}

template <typename Scalar>
static void BM_GeevEigen(benchmark::State& state) {
  runGeev<Scalar, EigenGeevKernel<Scalar>>(state);
}

// Defined only with a vendor linked; REGISTER_COMPARISON_POINT then drops its
// reference-arm argument, so the name is never substituted.
#ifdef EIGEN_BENCH_REFERENCE_ARM
template <typename Scalar>
static void BM_GeevReference(benchmark::State& state) {
  runGeev<Scalar, ReferenceGeevKernel<Scalar>>(state);
}
#endif

// The whole grid: run.py narrows it with --benchmark_filter, so a shape absent
// here is out of reach. The grid stops at 4096: an O(25 n^3) solver at 8192 is
// minutes per call.
// clang-format off
#define GEEV_DIM_NAMES {"n"}

#define GEEV_POINT(...)                                                                                            \
  REGISTER_COMPARISON_POINT(GEEV, f32, float,              BM_GeevEigen, BM_GeevReference, GEEV_DIM_NAMES, __VA_ARGS__) \
  REGISTER_COMPARISON_POINT(GEEV, f64, double,             BM_GeevEigen, BM_GeevReference, GEEV_DIM_NAMES, __VA_ARGS__) \
  REGISTER_COMPARISON_POINT(GEEV, c32, eigen_bench::c32_t, BM_GeevEigen, BM_GeevReference, GEEV_DIM_NAMES, __VA_ARGS__) \
  REGISTER_COMPARISON_POINT(GEEV, c64, eigen_bench::c64_t, BM_GeevEigen, BM_GeevReference, GEEV_DIM_NAMES, __VA_ARGS__)

#define GEEV_SIZES(POINT) \
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

GEEV_SIZES(GEEV_POINT)

#undef GEEV_SIZES
#undef GEEV_POINT
#undef GEEV_DIM_NAMES
// clang-format on

EIGEN_BENCH_COMPARISON_MAIN();
