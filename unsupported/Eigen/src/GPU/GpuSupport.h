// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Generic CUDA runtime support shared across all GPU library integrations
// (cuSOLVER, cuBLAS, cuDSS, etc.):
//   - Error-checking macros
//   - RAII device / pinned-host buffers
//   - RAII wrappers for cudaEvent_t / cudaStream_t
//
// Only depends on <cuda_runtime.h>. No NVIDIA library headers.

#ifndef EIGEN_GPU_SUPPORT_H
#define EIGEN_GPU_SUPPORT_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"
#include "./fwd_decl.h"

#include <cuda_runtime.h>

#include <memory>
#include <type_traits>

namespace Eigen {
namespace gpu {
namespace internal {

// ---- Error-checking macros --------------------------------------------------
// These abort (via eigen_assert) on failure. Not for use in destructors.

#define EIGEN_CUDA_RUNTIME_CHECK(expr)                             \
  do {                                                             \
    cudaError_t _e = (expr);                                       \
    eigen_assert(_e == cudaSuccess && "CUDA runtime call failed"); \
  } while (0)

// ---- Custom deleters for CUDA-allocated memory ------------------------------
// Used with std::unique_ptr to give CUDA allocations RAII semantics with no
// hand-rolled move/dtor boilerplate.

struct CudaFreeDeleter {
  void operator()(void* p) const noexcept {
    if (p) (void)cudaFree(p);
  }
};

struct CudaFreeHostDeleter {
  void operator()(void* p) const noexcept {
    if (p) (void)cudaFreeHost(p);
  }
};

// ---- RAII: device buffer ----------------------------------------------------

class DeviceBuffer {
 public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(size_t bytes) {
    if (bytes > 0) {
      void* p = nullptr;
      EIGEN_CUDA_RUNTIME_CHECK(cudaMalloc(&p, bytes));
      ptr_.reset(p);
    }
  }

  // Construct by taking ownership of a DeviceMatrix's device allocation.
  // Lets code move a typed DeviceMatrix<Scalar> into a type-erased
  // DeviceBuffer without a re-allocation.
  template <typename Scalar_>
  explicit DeviceBuffer(DeviceMatrix<Scalar_>&& mat);

  void* get() const noexcept { return ptr_.get(); }
  void* release() noexcept { return ptr_.release(); }
  explicit operator bool() const noexcept { return static_cast<bool>(ptr_); }

  // Adopt an existing device pointer. Caller relinquishes ownership.
  static DeviceBuffer adopt(void* p) noexcept {
    DeviceBuffer b;
    b.ptr_.reset(p);
    return b;
  }

 private:
  std::unique_ptr<void, CudaFreeDeleter> ptr_;
};

// ---- RAII: pinned host buffer -----------------------------------------------
// For async D2H copies (cudaMemcpyAsync requires pinned host memory for true
// asynchrony and to avoid compute-sanitizer warnings).

class PinnedHostBuffer {
 public:
  PinnedHostBuffer() = default;

  explicit PinnedHostBuffer(size_t bytes) {
    if (bytes > 0) {
      void* p = nullptr;
      EIGEN_CUDA_RUNTIME_CHECK(cudaMallocHost(&p, bytes));
      ptr_.reset(p);
    }
  }

  void* get() const noexcept { return ptr_.get(); }
  explicit operator bool() const noexcept { return static_cast<bool>(ptr_); }

 private:
  std::unique_ptr<void, CudaFreeHostDeleter> ptr_;
};

// ---- Scalar → cudaDataType_t ------------------------------------------------
// Shared by cuBLAS and cuSOLVER. cudaDataType_t is defined in library_types.h
// which is included transitively by cuda_runtime.h.

template <typename Scalar>
struct cuda_data_type;

template <>
struct cuda_data_type<float> {
  static constexpr cudaDataType_t value = CUDA_R_32F;
};
template <>
struct cuda_data_type<double> {
  static constexpr cudaDataType_t value = CUDA_R_64F;
};
template <>
struct cuda_data_type<std::complex<float>> {
  static constexpr cudaDataType_t value = CUDA_C_32F;
};
template <>
struct cuda_data_type<std::complex<double>> {
  static constexpr cudaDataType_t value = CUDA_C_64F;
};

// ---- Deleters for CUDA event / stream RAII wrappers -------------------------

struct CudaEventDeleter {
  void operator()(cudaEvent_t e) const {
    if (e) (void)cudaEventDestroy(e);
  }
};

struct CudaStreamDeleter {
  void operator()(cudaStream_t s) const {
    if (s) (void)cudaStreamDestroy(s);
  }
};

}  // namespace internal

struct CudaStream;

// ---- RAII: CUDA event -------------------------------------------------------
// Move-only. reset() and destruction sync before destroying so work waiting on
// this event does not reference freed memory guarded by the event.

struct CudaEvent {
  std::unique_ptr<std::remove_pointer_t<cudaEvent_t>, internal::CudaEventDeleter> ptr{nullptr};

  // Adopt an existing raw event (may be nullptr).
  explicit CudaEvent(cudaEvent_t event) : ptr(event) {}

  // Create a new event with the given flags (default: cudaEventDefault).
  explicit CudaEvent(unsigned int flags = cudaEventDefault) {
    cudaEvent_t raw = nullptr;
    EIGEN_CUDA_RUNTIME_CHECK(cudaEventCreateWithFlags(&raw, flags));
    ptr.reset(raw);
  }

  CudaEvent(CudaEvent&& o) noexcept : ptr(std::move(o.ptr)) {}
  CudaEvent& operator=(CudaEvent&& o) noexcept {
    if (this != &o) {
      reset();
      ptr = std::move(o.ptr);
    }
    return *this;
  }

  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;

  ~CudaEvent() { reset(); }

  // Lifted std::unique_ptr API.
  cudaEvent_t get() { return ptr.get(); }
  cudaEvent_t get() const { return ptr.get(); }
  explicit operator bool() const { return static_cast<bool>(ptr); }

  void reset() {
    if (ptr) (void)sync();
    ptr.reset();
  }

  // CUDA API.
  cudaError_t sync() const { return cudaEventSynchronize(ptr.get()); }
  cudaError_t query() const { return cudaEventQuery(ptr.get()); }

  void record(cudaStream_t stream) { EIGEN_CUDA_RUNTIME_CHECK(cudaEventRecord(ptr.get(), stream)); }
  void record(const CudaStream& stream);
};

// ---- RAII: CUDA stream ------------------------------------------------------
// Move-only owning wrapper. Default-constructed wraps the null (default) stream
// and does not free it on destruction (deleter is a no-op on null).

struct CudaStream {
  std::unique_ptr<std::remove_pointer_t<cudaStream_t>, internal::CudaStreamDeleter> ptr{nullptr};

  CudaStream() = default;
  explicit CudaStream(cudaStream_t stream) : ptr(stream) {}

  static CudaStream create(unsigned int flags = cudaStreamDefault) {
    cudaStream_t raw = nullptr;
    EIGEN_CUDA_RUNTIME_CHECK(cudaStreamCreateWithFlags(&raw, flags));
    return CudaStream(raw);
  }

  CudaStream(CudaStream&& o) noexcept : ptr(std::move(o.ptr)) {}
  CudaStream& operator=(CudaStream&& o) noexcept {
    if (this != &o) ptr = std::move(o.ptr);
    return *this;
  }

  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;

  // Lifted std::unique_ptr API.
  cudaStream_t get() { return ptr.get(); }
  cudaStream_t get() const { return ptr.get(); }
  void reset() { ptr.reset(); }
  explicit operator bool() const { return static_cast<bool>(ptr); }

  void wait(const CudaEvent& event) { EIGEN_CUDA_RUNTIME_CHECK(cudaStreamWaitEvent(ptr.get(), event.get(), 0)); }
};

inline void CudaEvent::record(const CudaStream& stream) {
  record(stream.get());
}

// ---- Non-owning stream reference --------------------------------------------
// Lets DeviceMatrix track the producer stream without taking ownership.

struct CudaStreamRef {
  cudaStream_t ptr{nullptr};

  CudaStreamRef() = default;
  explicit CudaStreamRef(cudaStream_t stream) : ptr(stream) {}

  cudaStream_t get() { return ptr; }
  cudaStream_t get() const { return ptr; }
  void reset() { ptr = nullptr; }
  void reset(cudaStream_t stream) { ptr = stream; }
  explicit operator bool() const { return ptr != nullptr; }

  void wait(const CudaEvent& event) { EIGEN_CUDA_RUNTIME_CHECK(cudaStreamWaitEvent(ptr, event.get(), 0)); }
};

}  // namespace gpu
}  // namespace Eigen

#endif  // EIGEN_GPU_SUPPORT_H
