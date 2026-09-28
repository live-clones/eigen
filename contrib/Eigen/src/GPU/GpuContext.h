// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// Unified GPU execution context: a CUDA stream plus the NVIDIA library handles
// used by gpu::DeviceMatrix operations.

#ifndef EIGEN_GPU_CONTEXT_H
#define EIGEN_GPU_CONTEXT_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

#include "./CuBlasSupport.h"
#include "./CuSolverSupport.h"
#include <cusparse.h>
#include <vector>

namespace Eigen {
namespace gpu {

namespace internal {

// cuSOLVER writes the factorization/solve status words to device memory in
// every build, so d_info must exist even under EIGEN_NO_DEBUG. Only the
// pinned host mirror is debug-only: it feeds the oneshot_check_info assert,
// which release builds compile out.
constexpr size_t kOneShotInfoBytes = 2 * sizeof(int);
#ifdef EIGEN_NO_DEBUG
constexpr size_t kOneShotHostInfoBytes = 0;
#else
constexpr size_t kOneShotHostInfoBytes = kOneShotInfoBytes;
#endif
// Grow-only scratch shared by the one-shot solver expressions
// (d_A.llt().solve(d_B), d_A.lu().solve(d_B)), so repeated one-shot solves on
// a Context perform no per-call device or pinned-host allocations. Holds only
// CUDA-runtime types (no cuSOLVER types) to keep the lazy-linking property of
// Context. Used by the one-shot solve dispatches in DeviceDispatch.h.
struct OneShotSolverScratch {
  explicit OneShotSolverScratch(const StreamHandle& stream) : d_info(kOneShotInfoBytes, stream) {}

  DeviceBuffer d_factor;
  DeviceBuffer d_ipiv;
  DeviceBuffer d_workspace;
  DeviceBuffer d_info;                             // 2 ints: {factorization, solve}
  PinnedHostBuffer h_info{kOneShotHostInfoBytes};  // debug-build info check only
  std::vector<char> h_workspace;
};

// Grows `buf` to at least `needed` bytes, allocated on `stream`. Replacing a
// buffer that queued work still uses is safe: the old one is freed on its home
// stream, ordered after that work, so neither the host nor the stream waits.
// Freeing first lets the allocator reuse the old block for the new one.
inline void ensure_sized(DeviceBuffer& buf, size_t needed, const StreamHandle& stream) {
  if (needed > buf.size()) {
    buf.reset();
    buf = DeviceBuffer(needed, stream);
  }
}

// Selects the Context constructor that defers the cuBLAS handle to its first use.
struct DeferCublas {};
}  // namespace internal

/** \ingroup GPU_Module
 * \class Context
 * \brief Unified GPU execution context: a CUDA stream plus the library handles bound to it.
 *
 * A Context says where GPU work runs: an operation runs on the Context passed
 * to it, on the one its solver, SparseContext, or FFT is bound to, or on
 * threadLocal() when it takes none. DeviceScalar arithmetic and copies are the
 * exception: they run on the first operand's stream(). A solver built without a
 * Context runs on a private one. Multiple contexts run concurrently on
 * independent streams; DeviceMatrix and DeviceScalar order their accesses
 * across contexts automatically.
 *
 * The default constructor creates a non-blocking stream: the module never uses
 * or synchronizes with the legacy default stream. The cuBLAS handle is created
 * with the Context, because creating a library handle synchronizes the device
 * and a first use mid-pipeline would stall every stream. Destroying a Context
 * destroys that handle, and cublasDestroy() synchronizes the device too. The
 * cuSOLVER, cuBLASLt, and cuSPARSE handles are created on first use (cuBLASLt's
 * on the first GEMM), so a translation unit that never touches one of those
 * libraries (the cuFFT test, say) does not link it.
 *
 * A Context is not thread-safe: the library handles are not thread-safe per
 * handle and the lazy initialization above is racy, so use one per thread or
 * synchronize externally.
 */
class Context {
 public:
  /** Create a context with a new non-blocking stream that it owns. The stream
   * stays alive after the Context is destroyed for as long as a DeviceMatrix or
   * DeviceScalar last written on it, or a pending read recorded on it, needs it. */
  Context() : stream_(internal::make_owned_stream()), oneshot_solver_scratch_(stream_) { init_cublas(); }

  /** Run on \p stream without taking ownership. It may be any stream; passing the
   * legacy default stream puts this Context's work there. \p stream must outlive
   * this Context, every DeviceMatrix or DeviceScalar last written on it (their
   * memory is freed stream-ordered on it), and every read enqueued on it that a
   * later write or free of such an object waits for. */
  explicit Context(cudaStream_t stream) : stream_(internal::borrow_stream(stream)), oneshot_solver_scratch_(stream_) {
    init_cublas();
  }

  /** Internal: an owned stream, with the cuBLAS handle created on first use, for
   * the private Context of a component that may never use cuBLAS. */
  explicit Context(internal::DeferCublas) : stream_(internal::make_owned_stream()), oneshot_solver_scratch_(stream_) {}

  ~Context() = default;

  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;
  Context(Context&&) = delete;
  Context& operator=(Context&&) = delete;

  /** Get the thread-local default context.
   *
   * If setThreadLocal() has been called, returns that context.
   * Otherwise lazily creates a new context with a dedicated stream.
   *
   * \note The thread-local instance is destroyed when the thread exits (or at
   * static destruction time for the main thread). On some CUDA driver
   * configurations this may print "CUDA_ERROR_DEINITIALIZED" to stderr if the
   * CUDA context has already been torn down. These errors are harmless and are
   * suppressed in the destructor, but they can produce noise in test output.
   * To avoid this, call cudaDeviceReset() only after all Context instances
   * (including thread-local ones) have been destroyed — or create and own a
   * Context and install it with setThreadLocal(): the lazily-created default
   * is then never constructed, and teardown order is fully under application
   * control. */
  static Context& threadLocal() {
    Context* override = tl_override_ptr();
    if (override) return *override;
    thread_local Context ctx;
    return ctx;
  }

  /** Override the thread-local default context for this thread.
   * The caller retains ownership of \p ctx — it must outlive all uses.
   * Pass nullptr to restore the lazily-created default. */
  static void setThreadLocal(Context* ctx) { tl_override_ptr() = ctx; }

  cudaStream_t stream() const { return stream_.get(); }

  /** Shared handle to stream(); device buffers hold it to keep the stream alive
   * for their stream-ordered free. */
  const internal::StreamHandle& streamHandle() const { return stream_; }

  cublasHandle_t cublasHandle() const {
    if (!cublas_) init_cublas();
    return cublas_.get();
  }

  /** Returns the cuSOLVER handle, creating it on first call. */
  cusolverDnHandle_t cusolverHandle() {
    if (!cusolver_) {
      cusolverDnHandle_t h = nullptr;
      EIGEN_CUSOLVER_CHECK(cusolverDnCreate(&h));
      cusolver_ = LazyCusolverHandle(h, &destroyCusolver);
      EIGEN_CUSOLVER_CHECK(cusolverDnSetStream(h, stream()));
    }
    return cusolver_.get();
  }

  /** cuBLASLt handle (lazy-initialized on first GEMM call). */
  cublasLtHandle_t cublasLtHandle() {
    if (!cublas_lt_) {
      cublasLtHandle_t h = nullptr;
      EIGEN_CUBLAS_CHECK(cublasLtCreate(&h));
      cublas_lt_ = internal::UniqueCublasLtHandle(h);
    }
    return cublas_lt_.get();
  }

  /** Workspace buffer for cublasLtMatmul (grown lazily by cublaslt_gemm).
   * Not thread-safe — all GEMM calls must be on this context's stream. */
  internal::DeviceBuffer& gemmWorkspace() { return gemm_workspace_; }

  /** Plan cache for cublasLtMatmul (caches descriptors and selected algorithm
   * by shape to avoid per-call overhead). Same thread-safety as workspace. */
  internal::CublasLtPlanCache& gemmPlanCache() { return gemm_plan_cache_; }

  /** Grow-only scratch for the one-shot solver expressions
   * (d_A.llt().solve(d_B), d_A.lu().solve(d_B)). Same thread-safety rules as
   * the GEMM workspace: all uses must be on this context's stream. */
  internal::OneShotSolverScratch& oneshotSolverScratch() { return oneshot_solver_scratch_; }

  /** Workspace ceiling passed to the cublasLtMatmul heuristic at plan-creation time.
   * Defaults to internal::kCublasLtMaxWorkspaceBytes (compile-time configurable via
   * EIGEN_CUDA_CUBLASLT_MAX_WORKSPACE_BYTES). */
  std::size_t cublasLtMaxWorkspaceBytes() const { return cublaslt_max_workspace_bytes_; }

  /** Override the workspace ceiling for future plan-cache misses on this context.
   * The cap is consulted at plan-creation time only; pre-existing cached plans
   * keep the cap they were built with. Call gemmPlanCache().clear() to force
   * re-selection under the new cap. */
  void setCublasLtMaxWorkspaceBytes(std::size_t bytes) { cublaslt_max_workspace_bytes_ = bytes; }

  /** cuSPARSE handle, created on first use. */
  cusparseHandle_t cusparseHandle() {
    if (!cusparse_) {
      cusparseHandle_t h = nullptr;
      cusparseStatus_t s1 = cusparseCreate(&h);
      eigen_assert(s1 == CUSPARSE_STATUS_SUCCESS && "cusparseCreate failed");
      EIGEN_UNUSED_VARIABLE(s1);
      cusparse_ = LazyCusparseHandle(h, &destroyCusparse);
      cusparseStatus_t s2 = cusparseSetStream(h, stream());
      eigen_assert(s2 == CUSPARSE_STATUS_SUCCESS && "cusparseSetStream failed");
      EIGEN_UNUSED_VARIABLE(s2);
    }
    return cusparse_.get();
  }

 private:
  static cusolverStatus_t destroyCusolver(cusolverDnHandle_t h) { return cusolverDnDestroy(h); }
  static cusparseStatus_t destroyCusparse(cusparseHandle_t h) { return cusparseDestroy(h); }

  // Function-pointer deleters keep cusolverDnDestroy / cusparseDestroy referenced only by TUs that create handles.
  using LazyCusolverHandle =
      std::unique_ptr<std::remove_pointer_t<cusolverDnHandle_t>, cusolverStatus_t (*)(cusolverDnHandle_t)>;
  using LazyCusparseHandle =
      std::unique_ptr<std::remove_pointer_t<cusparseHandle_t>, cusparseStatus_t (*)(cusparseHandle_t)>;

  // Destroyed in reverse declaration order: the plan cache before the cuBLASLt handle, this Context's stream
  // reference last.
  internal::StreamHandle stream_;
  mutable internal::UniqueCublasHandle cublas_;  // lazy only after the DeferCublas constructor
  LazyCusolverHandle cusolver_{nullptr, nullptr};
  LazyCusparseHandle cusparse_{nullptr, nullptr};
  internal::UniqueCublasLtHandle cublas_lt_;  // lazy
  internal::DeviceBuffer gemm_workspace_;     // lazy
  internal::CublasLtPlanCache gemm_plan_cache_{internal::kCublasLtPlanCacheCapacity};
  internal::OneShotSolverScratch oneshot_solver_scratch_;  // grow-only; allocated on stream_
  std::size_t cublaslt_max_workspace_bytes_ = internal::kCublasLtMaxWorkspaceBytes;

  static Context*& tl_override_ptr() {
    thread_local Context* ptr = nullptr;
    return ptr;
  }

  void init_cublas() const {
    cublasHandle_t h = nullptr;
    EIGEN_CUBLAS_CHECK(cublasCreate(&h));
    cublas_ = internal::UniqueCublasHandle(h);
    EIGEN_CUBLAS_CHECK(cublasSetStream(h, stream()));
  }
};

}  // namespace gpu
}  // namespace Eigen

#endif  // EIGEN_GPU_CONTEXT_H
