// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Eigen Authors
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// GPU Bunch-Kaufman (LDL^T) decomposition of symmetric indefinite matrices
// using cuSOLVER, wrapping cusolverDn<t>sytrf, cusolverDnXsytrs and
// cusolverDn<t>sytri.

#ifndef EIGEN_GPU_LDLT_H
#define EIGEN_GPU_LDLT_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

#include <algorithm>

#include "./DeviceScalarOps.h"
#include "./GpuSolverContext.h"

namespace Eigen {
namespace gpu {
/** \ingroup GPU_Module
 * \class LDLT
 * \brief GPU Bunch-Kaufman (LDL^T) decomposition of a symmetric indefinite matrix via cuSOLVER
 *
 * \tparam Scalar_  Element type: float, double, complex<float>, complex<double>
 * \tparam UpLo_    Triangle used: Lower (default) or Upper. A complex Scalar_ must
 *                  add the Symmetric bit (Lower | Symmetric), see below.
 *
 * Factorizes a symmetric matrix as P^T A P = L D L^T (or U D U^T) with
 * Bunch-Kaufman partial pivoting on the GPU, D being block diagonal with 1x1
 * and 2x2 blocks, and caches the factor and pivots in device memory. Eigen::LDLT's
 * diagonal pivoting is reliable only for (semi)definite matrices; Bunch-Kaufman
 * handles indefinite A with element growth bounded by (2.57)^(n-1) (Higham,
 * Accuracy and Stability of Numerical Algorithms, 2nd ed., section 11.1). Each
 * subsequent solve(B) uploads only B, calls cusolverDnXsytrs, and downloads the
 * result — the factor is not re-transferred.
 *
 * Complex matrices are complex-symmetric (A = A^T), as for LAPACK's
 * csytrf/zsytrf; cuSOLVER has no Hermitian variant (hetrf). Because Eigen::LDLT
 * and gpu::SparseLDLT mean Hermitian at the same position, a complex Scalar_
 * only compiles with the Symmetric mode bit set (gpu::LDLT<std::complex<double>,
 * Lower | Symmetric>). Factor a Hermitian indefinite matrix with gpu::LU.
 *
 * Each LDLT object owns a dedicated CUDA stream and cuSOLVER handle,
 * enabling concurrent factorizations from multiple objects on the same host
 * thread.
 */
template <typename Scalar_, int UpLo_ = Lower>
class LDLT {
 public:
  using Scalar = Scalar_;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using PlainMatrix = Eigen::Matrix<Scalar, Dynamic, Dynamic, ColMajor>;

  static constexpr int UpLo = UpLo_;

  /** Default constructor. Does not factorize; call compute() before solve(). */
  LDLT() = default;

  /** Bind to \p ctx: run on its stream with its cuSOLVER/cuBLAS handles, so
   * solver work chains with other work on the same Context without
   * cross-stream event waits. \p ctx must outlive this object. */
  explicit LDLT(Context& ctx) : solver_ctx_(ctx) {}

  /** Factor A immediately. Equivalent to LDLT ldlt; ldlt.compute(A). */
  template <typename InputType>
  explicit LDLT(const DenseBase<InputType>& A) {
    compute(A);
  }

  /** Factor a device-resident A immediately (D2D copy). */
  explicit LDLT(const DeviceMatrix<Scalar>& d_A) { compute(d_A); }

  /** Factor a device-resident A immediately (adopt, no copy). */
  explicit LDLT(DeviceMatrix<Scalar>&& d_A) { compute(std::move(d_A)); }

  /** Bind to \p ctx and factor A immediately. */
  template <typename InputType>
  LDLT(Context& ctx, const DenseBase<InputType>& A) : solver_ctx_(ctx) {
    compute(A);
  }

  /** Bind to \p ctx and factor a device-resident A (D2D copy). */
  LDLT(Context& ctx, const DeviceMatrix<Scalar>& d_A) : solver_ctx_(ctx) { compute(d_A); }

  ~LDLT() = default;

  // Non-copyable (owns device memory and library handles).
  LDLT(const LDLT&) = delete;
  LDLT& operator=(const LDLT&) = delete;

  // Movable.
  LDLT(LDLT&& o) noexcept
      : solver_ctx_(std::move(o.solver_ctx_)),
        d_factor_(std::move(o.d_factor_)),
        d_ipiv32_(std::move(o.d_ipiv32_)),
        d_ipiv64_(std::move(o.d_ipiv64_)),
        n_(o.n_),
        lda_(o.lda_) {
    o.n_ = 0;
    o.lda_ = 0;
  }

  LDLT& operator=(LDLT&& o) noexcept {
    if (this != &o) {
      solver_ctx_ = std::move(o.solver_ctx_);
      d_factor_ = std::move(o.d_factor_);
      d_ipiv32_ = std::move(o.d_ipiv32_);
      d_ipiv64_ = std::move(o.d_ipiv64_);
      n_ = o.n_;
      lda_ = o.lda_;
      o.n_ = 0;
      o.lda_ = 0;
    }
    return *this;
  }

  /** Compute the Bunch-Kaufman factorization of A (host matrix). The upload is
   * complete on return; factorization remains asynchronous. */
  template <typename InputType>
  LDLT& compute(const DenseBase<InputType>& A) {
    eigen_assert(A.rows() == A.cols());
    if (!begin_compute(A.rows())) return *this;

    // Ref binds column-major direct-access input in place (no host copy);
    // row-major layouts and expressions evaluate into its temporary.
    const Ref<const PlainMatrix> mat(A.derived());
    lda_ = static_cast<int64_t>(mat.rows());
    allocate_factor_storage();
    internal::upload_host_matrix(static_cast<Scalar*>(d_factor_.get()), mat.rows(), mat.data(), mat.outerStride(),
                                 mat.rows(), mat.cols(), solver_ctx_.stream());
    EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(solver_ctx_.stream()));

    factorize();
    return *this;
  }

  /** Compute the Bunch-Kaufman factorization from a device-resident matrix (D2D copy). */
  LDLT& compute(const DeviceMatrix<Scalar>& d_A) {
    eigen_assert(d_A.rows() == d_A.cols());
    if (!begin_compute(d_A.rows())) return *this;

    lda_ = static_cast<int64_t>(d_A.rows());
    d_A.waitReady(solver_ctx_.stream());
    allocate_factor_storage();
    EIGEN_CUDA_RUNTIME_CHECK(
        cudaMemcpyAsync(d_factor_.get(), d_A.data(), factorBytes(), cudaMemcpyDeviceToDevice, solver_ctx_.stream()));

    factorize();
    return *this;
  }

  /** Compute the Bunch-Kaufman factorization from a device matrix (move, no copy). */
  LDLT& compute(DeviceMatrix<Scalar>&& d_A) {
    eigen_assert(d_A.rows() == d_A.cols());
    if (!begin_compute(d_A.rows())) return *this;

    lda_ = static_cast<int64_t>(d_A.rows());
    d_A.waitReady(solver_ctx_.stream());
    d_factor_ = internal::DeviceBuffer::adopt(static_cast<void*>(d_A.release()), factorBytes());

    factorize();
    return *this;
  }

  /** Solve A * X = B using the cached factorization (host → host). */
  template <typename Rhs>
  PlainMatrix solve(const MatrixBase<Rhs>& B) const {
    // Debug builds verify the factorization (info() synchronizes the stream on
    // the first call after compute()); release builds skip both the check and
    // the sync — use info() explicitly when failure must be detected.
    eigen_assert(solver_ctx_.info() == Success && "LDLT::solve called on a failed or uninitialized factorization");
    eigen_assert(B.rows() == n_);
    if (n_ == 0 || B.cols() == 0) return PlainMatrix(n_, B.cols());

    const Ref<const PlainMatrix> rhs(B.derived());
    const int64_t nrhs = static_cast<int64_t>(rhs.cols());
    const int64_t ldb = static_cast<int64_t>(rhs.rows());
    internal::DeviceBuffer d_x(rhsBytes(nrhs, ldb));
    internal::upload_host_matrix(static_cast<Scalar*>(d_x.get()), ldb, rhs.data(), rhs.outerStride(), rhs.rows(),
                                 rhs.cols(), solver_ctx_.stream());
    DeviceMatrix<Scalar> d_X = solve_impl(nrhs, ldb, std::move(d_x));

    PlainMatrix X(n_, B.cols());
    int solve_info = 0;
    EIGEN_CUDA_RUNTIME_CHECK(
        cudaMemcpyAsync(X.data(), d_X.data(), rhsBytes(nrhs, ldb), cudaMemcpyDeviceToHost, solver_ctx_.stream()));
    EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(&solve_info, solver_ctx_.scratch_info(), sizeof(int),
                                             cudaMemcpyDeviceToHost, solver_ctx_.stream()));
    EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(solver_ctx_.stream()));

    eigen_assert(solve_info == 0 && "cusolverDnXsytrs reported an error");
    return X;
  }

  /** Solve A * X = B with device-resident RHS. Asynchronous: returns after
   * enqueuing the solve, except that a solve needing more cuSOLVER workspace
   * than any before it grows the scratch buffer, which synchronizes the stream
   * once. Debug builds verify the factorization status first (one host sync on
   * the first solve after compute()); release builds do not — use info() when
   * failure must be detected. */
  DeviceMatrix<Scalar> solve(const DeviceMatrix<Scalar>& d_B) const {
    eigen_assert(solver_ctx_.info() == Success && "LDLT::solve called on a failed or uninitialized factorization");
    eigen_assert(d_B.rows() == n_);
    if (n_ == 0 || d_B.cols() == 0) return DeviceMatrix<Scalar>(n_, d_B.cols());
    d_B.waitReady(solver_ctx_.stream());
    const int64_t nrhs = static_cast<int64_t>(d_B.cols());
    const int64_t ldb = static_cast<int64_t>(d_B.rows());
    internal::DeviceBuffer d_x(rhsBytes(nrhs, ldb));
    EIGEN_CUDA_RUNTIME_CHECK(
        cudaMemcpyAsync(d_x.get(), d_B.data(), rhsBytes(nrhs, ldb), cudaMemcpyDeviceToDevice, solver_ctx_.stream()));
    return solve_impl(nrhs, ldb, std::move(d_x));
  }

  /** Solve in place: consumes \p d_B and returns it holding the solution —
   * no RHS copy and no allocation (sytrs overwrites its RHS). */
  DeviceMatrix<Scalar> solve(DeviceMatrix<Scalar>&& d_B) const {
    eigen_assert(solver_ctx_.info() == Success && "LDLT::solve called on a failed or uninitialized factorization");
    eigen_assert(d_B.rows() == n_);
    if (n_ == 0 || d_B.cols() == 0) return std::move(d_B);
    d_B.waitReady(solver_ctx_.stream());
    const int64_t nrhs = static_cast<int64_t>(d_B.cols());
    const int64_t ldb = static_cast<int64_t>(d_B.rows());
    internal::DeviceBuffer d_x = internal::DeviceBuffer::adopt(static_cast<void*>(d_B.release()), rhsBytes(nrhs, ldb));
    return solve_impl(nrhs, ldb, std::move(d_x));
  }

  /** Explicit inverse from the cached factorization (cusolverDn<t>sytri).
   * Only the UpLo triangle of the result holds inv(A), as in cuSOLVER; the
   * other triangle is unspecified. Asynchronous under the same workspace-growth
   * caveat as solve(DeviceMatrix); the factorization is left intact for further
   * solves. sytri's own status
   * word is not surfaced: the zero pivot it would report is already visible
   * through info(). */
  DeviceMatrix<Scalar> inverse() const {
    eigen_assert(solver_ctx_.info() == Success && "LDLT::inverse called on a failed or uninitialized factorization");
    constexpr cublasFillMode_t uplo = internal::ldlt_fill_mode<Scalar, UpLo_>::value;

    DeviceMatrix<Scalar> inv(n_, n_);
    if (n_ == 0) return inv;
    const int n = internal::to_blas_int(n_);
    EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(inv.data(), d_factor_.get(), inv.sizeInBytes(), cudaMemcpyDeviceToDevice,
                                             solver_ctx_.stream()));

    int lwork = 0;
    EIGEN_CUSOLVER_CHECK(internal::cusolverDnXsytri_bufferSize(solver_ctx_.cusolverHandle(), uplo, n, inv.data(), n,
                                                               pivots32(), &lwork));
    solver_ctx_.ensure_scratch(static_cast<size_t>(lwork) * sizeof(Scalar));
    EIGEN_CUSOLVER_CHECK(internal::cusolverDnXsytri(solver_ctx_.cusolverHandle(), uplo, n, inv.data(), n, pivots32(),
                                                    static_cast<Scalar*>(solver_ctx_.scratch_workspace()), lwork,
                                                    solver_ctx_.scratch_info()));
    inv.recordReady(solver_ctx_.stream());
    return inv;
  }

  ComputationInfo info() const { return solver_ctx_.info(); }
  Index rows() const { return n_; }
  Index cols() const { return n_; }
  cudaStream_t stream() const { return solver_ctx_.stream(); }

 private:
  mutable internal::GpuSolverContext solver_ctx_;
  internal::DeviceBuffer d_factor_;  // grow-only
  internal::DeviceBuffer d_ipiv32_;  // grow-only; cusolverDn<t>sytrf's 32-bit pivots, also read by sytri
  internal::DeviceBuffer d_ipiv64_;  // grow-only; the same pivots widened for cusolverDnXsytrs
  int64_t n_ = 0;
  int64_t lda_ = 0;

  int* pivots32() const { return static_cast<int*>(d_ipiv32_.get()); }
  const int64_t* pivots64() const { return static_cast<const int64_t*>(d_ipiv64_.get()); }

  bool begin_compute(Index rows) {
    n_ = rows;
    return solver_ctx_.begin_compute(n_ != 0);
  }

  size_t factorBytes() const { return rhsBytes(n_, lda_); }

  static size_t rhsBytes(int64_t cols, int64_t ld) {
    return static_cast<size_t>(ld) * static_cast<size_t>(cols) * sizeof(Scalar);
  }

  void allocate_factor_storage() { internal::ensure_sized(d_factor_, factorBytes()); }

  // Solve in place on `d_x` (which already holds B), then re-wrap as a typed
  // DeviceMatrix carrying shape and a ready event. The release/adopt hop hands
  // ownership of the raw cudaMalloc pointer from the untyped DeviceBuffer to
  // the typed DeviceMatrix without copying.
  DeviceMatrix<Scalar> solve_impl(int64_t nrhs, int64_t ldb, internal::DeviceBuffer&& d_x) const {
    constexpr cudaDataType_t dtype = internal::cusolver_data_type<Scalar>::value;
    constexpr cublasFillMode_t uplo = internal::ldlt_fill_mode<Scalar, UpLo_>::value;

    size_t dev_ws_bytes = 0, host_ws_bytes = 0;
    EIGEN_CUSOLVER_CHECK(cusolverDnXsytrs_bufferSize(solver_ctx_.cusolverHandle(), uplo, n_, nrhs, dtype,
                                                     d_factor_.get(), lda_, pivots64(), dtype, d_x.get(), ldb,
                                                     &dev_ws_bytes, &host_ws_bytes));
    solver_ctx_.ensure_scratch(dev_ws_bytes);
    if (solver_ctx_.h_workspace_.size() < host_ws_bytes) solver_ctx_.h_workspace_.resize(host_ws_bytes);

    EIGEN_CUSOLVER_CHECK(cusolverDnXsytrs(solver_ctx_.cusolverHandle(), uplo, n_, nrhs, dtype, d_factor_.get(), lda_,
                                          pivots64(), dtype, d_x.get(), ldb, solver_ctx_.scratch_workspace(),
                                          dev_ws_bytes, host_ws_bytes > 0 ? solver_ctx_.h_workspace_.data() : nullptr,
                                          host_ws_bytes, solver_ctx_.scratch_info()));

    DeviceMatrix<Scalar> result =
        DeviceMatrix<Scalar>::adopt(static_cast<Scalar*>(d_x.release()), n_, static_cast<Index>(nrhs));
    result.recordReady(solver_ctx_.stream());
    return result;
  }

  void factorize() {
    constexpr cublasFillMode_t uplo = internal::ldlt_fill_mode<Scalar, UpLo_>::value;
    const int n = internal::to_blas_int(n_);
    const int lda = internal::to_blas_int(lda_);
    Scalar* d_factor = static_cast<Scalar*>(d_factor_.get());

    solver_ctx_.mark_pending();

    internal::ensure_sized(d_ipiv32_, static_cast<size_t>(n) * sizeof(int));
    internal::ensure_sized(d_ipiv64_, static_cast<size_t>(n) * sizeof(int64_t));

    int lwork = 0;
    EIGEN_CUSOLVER_CHECK(internal::cusolverDnXsytrf_bufferSize(solver_ctx_.cusolverHandle(), n, d_factor, lda, &lwork));
    // The workspace doubles as widen_pivots' temporary once sytrf has run.
    solver_ctx_.ensure_scratch(
        std::max(static_cast<size_t>(lwork) * sizeof(Scalar), static_cast<size_t>(n) * sizeof(double)));

    EIGEN_CUSOLVER_CHECK(internal::cusolverDnXsytrf(solver_ctx_.cusolverHandle(), uplo, n, d_factor, lda, pivots32(),
                                                    static_cast<Scalar*>(solver_ctx_.scratch_workspace()), lwork,
                                                    solver_ctx_.scratch_info()));
    internal::widen_pivots(pivots32(), static_cast<int64_t*>(d_ipiv64_.get()),
                           static_cast<double*>(solver_ctx_.scratch_workspace()), static_cast<size_t>(n),
                           solver_ctx_.stream());

    solver_ctx_.enqueue_info_copy();
  }
};
}  // namespace gpu
}  // namespace Eigen

#endif  // EIGEN_GPU_LDLT_H
