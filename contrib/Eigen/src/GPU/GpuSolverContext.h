// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// Shared context for the dense GPU solvers. Each solver holds one by composition
// and delegates handle lifetime and scratch management to it.

#ifndef EIGEN_GPU_SOLVER_CONTEXT_H
#define EIGEN_GPU_SOLVER_CONTEXT_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

#include "./CuSolverSupport.h"
#include "./CuBlasSupport.h"
#include "./DeviceDispatch.h"
#include <vector>

namespace Eigen {
namespace gpu {
namespace internal {

struct GpuSolverContext {
  // A standalone solver runs on a private Context, so that separate solvers run
  // concurrently; a bound one runs on the caller's.
  std::unique_ptr<Context> owned_ctx_;
  Context* ctx_ = nullptr;
  CusolverParams params_;
  DeviceBuffer d_scratch_;
  std::vector<char> h_workspace_;
  ComputationInfo info_ = InvalidInput;
  PinnedHostBuffer pinned_info_{sizeof(int)};  // pinned host memory for async D2H of info word
  bool info_synced_ = true;

  int& info_word() { return *static_cast<int*>(pinned_info_.get()); }
  int info_word() const { return *static_cast<const int*>(pinned_info_.get()); }

  Context& context() const { return *ctx_; }
  cudaStream_t stream() const { return ctx_->stream(); }
  const StreamHandle& streamHandle() const { return ctx_->streamHandle(); }
  cusolverDnHandle_t cusolverHandle() const { return ctx_->cusolverHandle(); }
  cublasHandle_t cublasHandle() const { return ctx_->cublasHandle(); }
  cublasLtHandle_t cublasLtHandle() const { return ctx_->cublasLtHandle(); }
  CublasLtPlanCache& gemmPlanCache() const { return ctx_->gemmPlanCache(); }
  DeviceBuffer& gemmWorkspace() const { return ctx_->gemmWorkspace(); }
  std::size_t cublasLtMaxWorkspaceBytes() const { return ctx_->cublasLtMaxWorkspaceBytes(); }

  GpuSolverContext() : owned_ctx_(new Context()), ctx_(owned_ctx_.get()) { ensure_scratch(0); }

  /** Run on \p ctx's stream with its handles, GEMM plan cache, and GEMM
   * workspace, so solver work chains with the caller's other GPU operations
   * without cross-stream event waits. The Context must outlive this solver context. */
  explicit GpuSolverContext(Context& ctx) : ctx_(&ctx) { ensure_scratch(0); }

  // A pending info copy may still write pinned_info_, whose cudaFreeHost deleter is not stream-ordered.
  ~GpuSolverContext() { wait_for_info_copy(); }

  GpuSolverContext(GpuSolverContext&& o) noexcept
      : owned_ctx_(std::move(o.owned_ctx_)),
        ctx_(o.ctx_),
        params_(std::move(o.params_)),
        d_scratch_(std::move(o.d_scratch_)),
        h_workspace_(std::move(o.h_workspace_)),
        info_(o.info_),
        pinned_info_(std::move(o.pinned_info_)),
        info_synced_(o.info_synced_) {
    o.ctx_ = nullptr;
    o.info_synced_ = true;
  }

  GpuSolverContext& operator=(GpuSolverContext&& o) noexcept {
    if (this != &o) {
      wait_for_info_copy();
      // Scratch before the Context: its free is enqueued on the stream that Context may own.
      d_scratch_ = std::move(o.d_scratch_);
      owned_ctx_ = std::move(o.owned_ctx_);
      ctx_ = o.ctx_;
      params_ = std::move(o.params_);
      h_workspace_ = std::move(o.h_workspace_);
      info_ = o.info_;
      pinned_info_ = std::move(o.pinned_info_);
      info_synced_ = o.info_synced_;
      o.ctx_ = nullptr;
      o.info_synced_ = true;
    }
    return *this;
  }

  GpuSolverContext(const GpuSolverContext&) = delete;
  GpuSolverContext& operator=(const GpuSolverContext&) = delete;

  // Scratch layout: [ workspace (aligned) | info_word (sizeof(int)) ].
  // Workspace size is rounded up to 16 bytes so the info word lands aligned.
  static constexpr size_t kInfoBytes = sizeof(int);
  static constexpr size_t kScratchAlign = 16;

  static size_t scratchBytesFor(size_t workspace_bytes) {
    workspace_bytes = (workspace_bytes + kScratchAlign - 1) & ~(kScratchAlign - 1);
    return workspace_bytes + kInfoBytes;
  }

  // Ensure d_scratch_ holds at least `workspace_bytes` of scratch plus the trailing
  // info word. Grows but never shrinks; see ensure_sized() for why no sync is needed.
  void ensure_scratch(size_t workspace_bytes) {
    ensure_sized(d_scratch_, scratchBytesFor(workspace_bytes), streamHandle());
  }

  void* scratch_workspace() const { return d_scratch_.get(); }

  int* scratch_info() const {
    eigen_assert(d_scratch_ && d_scratch_.size() >= kInfoBytes);
    return reinterpret_cast<int*>(static_cast<char*>(d_scratch_.get()) + d_scratch_.size() - kInfoBytes);
  }

  // Mark a factorization as pending: its info word is not yet available.
  void mark_pending() {
    info_synced_ = false;
    info_ = InvalidInput;
  }

  // Common compute() prologue: reset info state. Returns false for the empty
  // (n == 0) case, which is trivially successful — the caller returns early.
  bool begin_compute(bool nonempty) {
    info_ = InvalidInput;
    if (!nonempty) {
      info_ = Success;
      info_synced_ = true;
      return false;
    }
    return true;
  }

  // Common factorize() epilogue: enqueue the async D2H copy of the info word
  // into pinned host memory. Read later by the lazy sync_info().
  void enqueue_info_copy() {
    EIGEN_CUDA_RUNTIME_CHECK(
        cudaMemcpyAsync(&info_word(), scratch_info(), sizeof(int), cudaMemcpyDeviceToHost, stream()));
  }

  // Synchronize the stream and interpret the info word; no-op once synced.
  void sync_info() {
    if (!info_synced_) {
      EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream()));
      info_ = (info_word() == 0) ? Success : NumericalIssue;
      info_synced_ = true;
    }
  }

  ComputationInfo info() {
    sync_info();
    return info_;
  }

  // Blocking download of solver-owned device data, on the solver's stream.
  void download(void* dst, const void* src, size_t bytes) const {
    if (bytes == 0) return;
    EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream()));
    EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream()));
  }

  void wait_for_info_copy() noexcept {
    if (!info_synced_ && ctx_ && pinned_info_) (void)cudaStreamSynchronize(stream());
  }
};

}  // namespace internal
}  // namespace gpu
}  // namespace Eigen

#endif  // EIGEN_GPU_SOLVER_CONTEXT_H
