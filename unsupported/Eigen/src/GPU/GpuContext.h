// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Unified GPU execution context.
//
// gpu::Context owns a CUDA stream and NVIDIA library handles (cuBLAS,
// cuSOLVER). It is the entry point for all GPU
// operations on gpu::DeviceMatrix.
//
// Usage:
//   gpu::Context ctx;                        // explicit context
//   d_C.device(ctx) = d_A * d_B;            // GEMM on ctx's stream
//
//   d_C = d_A * d_B;                        // thread-local default context
//   gpu::Context& ctx = gpu::Context::threadLocal();

#ifndef EIGEN_GPU_CONTEXT_H
#define EIGEN_GPU_CONTEXT_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

#include "./CuBlasSupport.h"
#include "./CuSolverSupport.h"
#include "./GpuSupport.h"

namespace Eigen {
namespace gpu {

namespace internal {

struct CublasHandleDeleter {
  void operator()(cublasHandle_t h) const {
    if (h) (void)cublasDestroy(h);
  }
};

struct CusolverDnHandleDeleter {
  void operator()(cusolverDnHandle_t h) const {
    if (h) (void)cusolverDnDestroy(h);
  }
};

}  // namespace internal

/** \ingroup GPU_Module
 * \struct CublasHandle
 * \brief RAII smart-pointer wrapper around cublasHandle_t.
 *
 * Default-construction calls cublasCreate; destruction calls cublasDestroy.
 * Move-only. Mirrors the CudaEvent / CudaStream pattern in GpuSupport.h.
 */
struct CublasHandle {
  std::unique_ptr<std::remove_pointer_t<cublasHandle_t>, internal::CublasHandleDeleter> ptr{nullptr};

  CublasHandle() {
    cublasHandle_t raw = nullptr;
    EIGEN_CUBLAS_CHECK(cublasCreate(&raw));
    ptr.reset(raw);
  }

  explicit CublasHandle(cublasHandle_t h) : ptr(h) {}

  CublasHandle(CublasHandle&& o) noexcept : ptr(std::move(o.ptr)) {}
  CublasHandle& operator=(CublasHandle&& o) noexcept {
    if (this != &o) ptr = std::move(o.ptr);
    return *this;
  }

  CublasHandle(const CublasHandle&) = delete;
  CublasHandle& operator=(const CublasHandle&) = delete;

  cublasHandle_t get() { return ptr.get(); }
  cublasHandle_t get() const { return ptr.get(); }
  void reset() { ptr.reset(); }
  explicit operator bool() const { return static_cast<bool>(ptr); }

  cublasStatus_t setStream(cudaStream_t stream) { return cublasSetStream(ptr.get(), stream); }
  cublasStatus_t setStream(const CudaStream& stream) { return cublasSetStream(ptr.get(), stream.get()); }
};

/** \ingroup GPU_Module
 * \struct CusolverDnHandle
 * \brief RAII smart-pointer wrapper around cusolverDnHandle_t.
 *
 * Default-construction calls cusolverDnCreate; destruction calls
 * cusolverDnDestroy. Move-only. Mirrors the CudaEvent / CudaStream pattern.
 */
struct CusolverDnHandle {
  std::unique_ptr<std::remove_pointer_t<cusolverDnHandle_t>, internal::CusolverDnHandleDeleter> ptr{nullptr};

  CusolverDnHandle() {
    cusolverDnHandle_t raw = nullptr;
    EIGEN_CUSOLVER_CHECK(cusolverDnCreate(&raw));
    ptr.reset(raw);
  }

  explicit CusolverDnHandle(cusolverDnHandle_t h) : ptr(h) {}

  CusolverDnHandle(CusolverDnHandle&& o) noexcept : ptr(std::move(o.ptr)) {}
  CusolverDnHandle& operator=(CusolverDnHandle&& o) noexcept {
    if (this != &o) ptr = std::move(o.ptr);
    return *this;
  }

  CusolverDnHandle(const CusolverDnHandle&) = delete;
  CusolverDnHandle& operator=(const CusolverDnHandle&) = delete;

  cusolverDnHandle_t get() { return ptr.get(); }
  cusolverDnHandle_t get() const { return ptr.get(); }
  void reset() { ptr.reset(); }
  explicit operator bool() const { return static_cast<bool>(ptr); }

  cusolverStatus_t setStream(cudaStream_t stream) { return cusolverDnSetStream(ptr.get(), stream); }
  cusolverStatus_t setStream(const CudaStream& stream) { return cusolverDnSetStream(ptr.get(), stream.get()); }
};

/** \ingroup GPU_Module
 * \class Context
 * \brief Unified GPU execution context owning a CUDA stream and library handles.
 *
 * Each Context instance creates a dedicated CUDA stream, a cuBLAS handle,
 * and a cuSOLVER handle, all bound to that stream. Multiple contexts enable
 * concurrent execution on independent streams.
 *
 * A lazily-created thread-local default is available via threadLocal() for
 * simple single-stream usage.
 */
class Context {
 public:
  Context() : stream_(CudaStream::create()) {
    EIGEN_CUBLAS_CHECK(cublas_.setStream(stream_));
    EIGEN_CUSOLVER_CHECK(cusolver_.setStream(stream_));
  }

  // Destructor is implicit: members destroy in reverse declaration order
  // (cusolver_, cublas_, stream_), which is the correct teardown order.

  // Non-copyable, non-movable (owns library handles).
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;
  Context(Context&&) = delete;
  Context& operator=(Context&&) = delete;

  /** Lazily-created thread-local default context.
   *
   * \note The thread-local instance is destroyed when the thread exits (or at
   * static destruction time for the main thread). On some CUDA driver
   * configurations this may print "CUDA_ERROR_DEINITIALIZED" to stderr if the
   * CUDA context has already been torn down. These errors are harmless and are
   * suppressed in the destructor, but they can produce noise in test output.
   * To avoid this, call cudaDeviceReset() only after all Context instances
   * (including thread-local ones) have been destroyed. */
  static Context& threadLocal() {
    thread_local Context ctx;
    return ctx;
  }

  cudaStream_t stream() const { return stream_.get(); }
  cublasHandle_t cublasHandle() const { return cublas_.get(); }
  cusolverDnHandle_t cusolverHandle() const { return cusolver_.get(); }

 private:
  CudaStream stream_;
  CublasHandle cublas_;
  CusolverDnHandle cusolver_;
};

}  // namespace gpu
}  // namespace Eigen

#endif  // EIGEN_GPU_CONTEXT_H
