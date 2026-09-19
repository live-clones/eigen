// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Cross-library comparison benchmarks for GETRF, the LU factorization with
// partial pivoting of a square matrix (PartialPivLU takes no other shape).
//
// The reference here is a LAPACK routine rather than a BLAS one; CMakeLists.txt
// reads the marker below and builds this source Eigen-only when the reference
// library supplies no LAPACK.
// EIGEN_BENCH_REFERENCE_FAMILY: lapack
//
// The operation is destructive, so the timed body must present fresh data on
// every iteration: both arms copy the operand into a destination allocated once
// and factorize the copy. The Eigen arm is the in-place form of
// doc/InplaceDecomposition.dox, a PartialPivLU<Ref<MatrixType>> bound to the
// destination once per shape, whose compute(a) copies a into it and factorizes
// there; so neither arm allocates inside the timed region, LAPACK's pivot vector
// and Eigen's transposition vector both being set up once per shape.

#include <Eigen/Core>
#include <Eigen/LU>
#include <complex>
#include <string>

#include "benchmarks/bench_common.h"
#include "benchmarks/comparison/bench_compare.h"

#ifdef EIGEN_BENCH_REFERENCE_ARM
// Fortran LAPACK, declared here rather than pulled from a vendor header so that
// the build needs only the library. See the note in bench_gemm_compare.cpp on
// the integer width.
extern "C" {
void sgetrf_(const eigen_bench::BlasInt* m, const eigen_bench::BlasInt* n, float* a, const eigen_bench::BlasInt* lda,
             eigen_bench::BlasInt* ipiv, eigen_bench::BlasInt* info) EIGEN_BENCH_FORTRAN_SYMBOL(sgetrf);
void dgetrf_(const eigen_bench::BlasInt* m, const eigen_bench::BlasInt* n, double* a, const eigen_bench::BlasInt* lda,
             eigen_bench::BlasInt* ipiv, eigen_bench::BlasInt* info) EIGEN_BENCH_FORTRAN_SYMBOL(dgetrf);
void cgetrf_(const eigen_bench::BlasInt* m, const eigen_bench::BlasInt* n, std::complex<float>* a,
             const eigen_bench::BlasInt* lda, eigen_bench::BlasInt* ipiv, eigen_bench::BlasInt* info)
    EIGEN_BENCH_FORTRAN_SYMBOL(cgetrf);
void zgetrf_(const eigen_bench::BlasInt* m, const eigen_bench::BlasInt* n, std::complex<double>* a,
             const eigen_bench::BlasInt* lda, eigen_bench::BlasInt* ipiv, eigen_bench::BlasInt* info)
    EIGEN_BENCH_FORTRAN_SYMBOL(zgetrf);
}
#endif

using Eigen::Index;
using eigen_bench::BlasInt;
using eigen_bench::ColMatrix;
using eigen_bench::ColVector;

using Permutation = Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>;

// Both kernels leave L (unit lower) and U (upper) packed in the destination
// and report P such that P*A = L*U, the convention of PartialPivLU::permutationP().
template <typename Scalar>
class EigenGetrfKernel {
 public:
  // Binds to `out`, which must already hold the operand: the constructor
  // factorizes what it is given.
  explicit EigenGetrfKernel(ColMatrix<Scalar>& out) : lu_(out) {}

  void operator()(const ColMatrix<Scalar>& a) { lu_.compute(a); }

  Permutation permutation() const { return lu_.permutationP(); }

 private:
  Eigen::PartialPivLU<Eigen::Ref<ColMatrix<Scalar>>> lu_;
};

#ifdef EIGEN_BENCH_REFERENCE_ARM
static void referenceGetrf(BlasInt n, float* a, BlasInt* ipiv, BlasInt* info) { sgetrf_(&n, &n, a, &n, ipiv, info); }
static void referenceGetrf(BlasInt n, double* a, BlasInt* ipiv, BlasInt* info) { dgetrf_(&n, &n, a, &n, ipiv, info); }
static void referenceGetrf(BlasInt n, std::complex<float>* a, BlasInt* ipiv, BlasInt* info) {
  cgetrf_(&n, &n, a, &n, ipiv, info);
}
static void referenceGetrf(BlasInt n, std::complex<double>* a, BlasInt* ipiv, BlasInt* info) {
  zgetrf_(&n, &n, a, &n, ipiv, info);
}

template <typename Scalar>
class ReferenceGetrfKernel {
 public:
  explicit ReferenceGetrfKernel(ColMatrix<Scalar>& out) : out_(out), ipiv_(out.rows()) {}

  void operator()(const ColMatrix<Scalar>& a) {
    // The same copy the Eigen arm makes, deliberately.
    out_ = a;
    BlasInt info = 0;
    referenceGetrf(static_cast<BlasInt>(a.rows()), out_.data(), ipiv_.data(), &info);
    benchmark::DoNotOptimize(info);
  }

  Permutation permutation() const {
    // ?getrf reports its row interchanges 1-based, to be applied in order; as
    // 0-based transpositions they are exactly what PartialPivLU stores, so the
    // same conversion yields the same P.
    const Eigen::Matrix<int, Eigen::Dynamic, 1> transpositions = (ipiv_.template cast<int>().array() - 1).matrix();
    return Permutation(Eigen::Transpositions<Eigen::Dynamic, Eigen::Dynamic>(transpositions));
  }

 private:
  ColMatrix<Scalar>& out_;
  Eigen::Matrix<BlasInt, Eigen::Dynamic, 1> ipiv_;
};
#endif

// Both arms run through this one driver so that allocation, fill, validation,
// the timed region and the counter cannot drift apart between them.
template <typename Scalar, typename Kernel>
static void runGetrf(benchmark::State& state) {
  const Index n = static_cast<Index>(state.range(0));

  if (eigen_bench::skipIfDimsExceedBlasInt(state, n)) return;

  // The operand and the destination.
  const double operand_bytes =
      2.0 * static_cast<double>(sizeof(Scalar)) * static_cast<double>(n) * static_cast<double>(n);
  if (eigen_bench::skipIfOverMemoryBudget(state, operand_bytes)) return;

  // A dense random operand: partial pivoting keeps the factorization stable
  // for it, and the check below is a residual, which conditioning does not
  // affect.
  const ColMatrix<Scalar> a = ColMatrix<Scalar>::Random(n, n);
  ColMatrix<Scalar> out = a;
  Kernel kernel(out);

  static eigen_bench::ValidatedShapes<Index> validated;
  if (!validated.contains(n)) {
    // P*(A*x) against L*(U*x) for a random x: O(n^2) instead of forming L*U.
    kernel(a);
    const ColVector<Scalar> x = ColVector<Scalar>::Random(n);
    const ColVector<Scalar> expected = kernel.permutation() * (a * x).eval();
    const ColVector<Scalar> actual =
        out.template triangularView<Eigen::UnitLower>() * (out.template triangularView<Eigen::Upper>() * x).eval();

    if (!eigen_bench::agreesWithEigen(expected, actual, n)) {
      state.SkipWithError("getrf result disagrees with Eigen at n:" + std::to_string(n));
      return;
    }
    validated.insert(n);
  }

  for (auto _ : state) {
    kernel(a);
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }

  eigen_bench::setFlopRate(state, eigen_bench::getrfFlops<Scalar>(n, n));
}

template <typename Scalar>
static void BM_GetrfEigen(benchmark::State& state) {
  runGetrf<Scalar, EigenGetrfKernel<Scalar>>(state);
}

// Defined only with a vendor linked; REGISTER_COMPARISON_POINT then drops its
// reference-arm argument, so the name is never substituted.
#ifdef EIGEN_BENCH_REFERENCE_ARM
template <typename Scalar>
static void BM_GetrfReference(benchmark::State& state) {
  runGetrf<Scalar, ReferenceGetrfKernel<Scalar>>(state);
}
#endif

// The whole grid: run.py narrows it with --benchmark_filter, so a shape absent
// here is out of reach.
// clang-format off
#define GETRF_DIM_NAMES {"n"}

#define GETRF_POINT(...)                                                                                              \
  REGISTER_COMPARISON_POINT(GETRF, f32, float,              BM_GetrfEigen, BM_GetrfReference, GETRF_DIM_NAMES, __VA_ARGS__) \
  REGISTER_COMPARISON_POINT(GETRF, f64, double,             BM_GetrfEigen, BM_GetrfReference, GETRF_DIM_NAMES, __VA_ARGS__) \
  REGISTER_COMPARISON_POINT(GETRF, c32, eigen_bench::c32_t, BM_GetrfEigen, BM_GetrfReference, GETRF_DIM_NAMES, __VA_ARGS__) \
  REGISTER_COMPARISON_POINT(GETRF, c64, eigen_bench::c64_t, BM_GetrfEigen, BM_GetrfReference, GETRF_DIM_NAMES, __VA_ARGS__)

#define GETRF_SIZES(POINT) \
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

GETRF_SIZES(GETRF_POINT)

#undef GETRF_SIZES
#undef GETRF_POINT
#undef GETRF_DIM_NAMES
// clang-format on

EIGEN_BENCH_COMPARISON_MAIN();
