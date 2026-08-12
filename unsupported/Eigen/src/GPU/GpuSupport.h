// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// Generic CUDA runtime support shared by all GPU library integrations.
// Depends only on <cuda_runtime.h>; no NVIDIA library headers.

#ifndef EIGEN_GPU_SUPPORT_H
#define EIGEN_GPU_SUPPORT_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

#include <cuda_runtime.h>

#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

namespace Eigen {
namespace gpu {
// Transpose/adjoint flag for BLAS-, solver-, and sparse-style calls. Each
// library's support header maps it to its own enum (cublasOperation_t,
// cusparseOperation_t, ...) via a to_<lib>_op() helper.
enum class GpuOp { NoTrans, Trans, ConjTrans };

namespace internal {
// Aborts via eigen_assert on failure, so it must not be used in destructors.
#define EIGEN_CUDA_RUNTIME_CHECK(expr)                             \
  do {                                                             \
    cudaError_t _e = (expr);                                       \
    eigen_assert(_e == cudaSuccess && "CUDA runtime call failed"); \
  } while (0)

// cuBLAS and the legacy cuSOLVER APIs take dimensions and leading dimensions as
// 32-bit `int`, while Eigen's Index is 64-bit by default and GPU allocations can
// exceed INT_MAX in one dimension. Narrow through this helper at every such call
// site so an out-of-range value asserts instead of silently overflowing.
inline int to_blas_int(int64_t v) {
  eigen_assert(v >= 0 && v <= static_cast<int64_t>((std::numeric_limits<int>::max)()) &&
               "dimension exceeds the int range supported by cuBLAS / cuSOLVER");
  return static_cast<int>(v);
}

// cudaMallocAsync / cudaFreeAsync (CUDA 11.2+) allocate from a stream-ordered
// memory pool: both are cheap enqueues instead of the device-wide
// synchronization performed by cudaMalloc / cudaFree. Stream-bound internal
// buffers use the overloads that take their execution stream. DeviceMatrix,
// whose constructors are not bound to a Context, retains the legacy-stream
// overloads below. Support is detected once per process, from the device
// current at first use.

inline bool device_supports_memory_pools() {
#ifdef EIGEN_GPU_NO_STREAM_ORDERED_ALLOC
  return false;
#else
  static const bool supported = [] {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return false;
    int v = 0;
    if (cudaDeviceGetAttribute(&v, cudaDevAttrMemoryPoolsSupported, device) != cudaSuccess) return false;
    if (v == 0) return false;
    // Keep freed memory in the pool instead of trimming at every stream
    // synchronize — repeated alloc/free cycles (temporaries in loops) then
    // recycle at user-space speed.
    cudaMemPool_t pool = nullptr;
    if (cudaDeviceGetDefaultMemPool(&pool, device) == cudaSuccess) {
      // The attribute value type is cuuint64_t; use a same-size stand-in to
      // avoid requiring the driver-API header.
      unsigned long long threshold = ~0ULL;
      (void)cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
    }
    return true;
  }();
  return supported;
#endif
}

inline void* device_malloc(size_t bytes) {
  void* p = nullptr;
  if (device_supports_memory_pools()) {
    EIGEN_CUDA_RUNTIME_CHECK(cudaMallocAsync(&p, bytes, /*legacy default stream*/ nullptr));
  } else {
    EIGEN_CUDA_RUNTIME_CHECK(cudaMalloc(&p, bytes));
  }
  return p;
}

inline void device_free(void* p) noexcept {
  if (!p) return;
  if (device_supports_memory_pools()) {
    (void)cudaFreeAsync(p, /*legacy default stream*/ nullptr);
  } else {
    (void)cudaFree(p);
  }
}

inline void* device_malloc(size_t bytes, cudaStream_t stream) {
  void* p = nullptr;
  if (device_supports_memory_pools()) {
    EIGEN_CUDA_RUNTIME_CHECK(cudaMallocAsync(&p, bytes, stream));
  } else {
    EIGEN_CUDA_RUNTIME_CHECK(cudaMalloc(&p, bytes));
  }
  return p;
}

inline void device_free(void* p, cudaStream_t stream) noexcept {
  if (!p) return;
  if (device_supports_memory_pools()) {
    (void)cudaFreeAsync(p, stream);
  } else {
    (void)cudaFree(p);
  }
}

struct CudaFreeDeleter {
  // When `borrow == true`, the unique_ptr does not free the pointer. Used by
  // DeviceMatrix::view() to wrap a non-owning device pointer with the same
  // smart-pointer machinery as owning storage, without changing the type.
  bool borrow = false;
  void operator()(void* p) const noexcept {
    if (p && !borrow) device_free(p);
  }
};

struct CudaFreeHostDeleter {
  void operator()(void* p) const noexcept {
    if (p) (void)cudaFreeHost(p);
  }
};

// Borrow-aware deleter for cudaStream_t — the stream analog of CudaFreeDeleter.
// With `borrow == true` destruction is left to the stream's real owner.
// Ownership is recorded explicitly rather than inferred from the value because
// nullptr (CUDA's legacy default stream) is a valid stream to borrow.
struct CudaStreamDeleter {
  bool borrow = false;
  void operator()(cudaStream_t s) const noexcept {
    if (s && !borrow) (void)cudaStreamDestroy(s);
  }
};

// Owning-or-borrowing RAII handle for a CUDA stream; get() yields the raw
// cudaStream_t to pass to CUDA APIs. Declare it before members that release
// resources against the stream (device buffers, library handles) so it is
// destroyed after them.
using CudaStreamHandle = std::unique_ptr<std::remove_pointer_t<cudaStream_t>, CudaStreamDeleter>;

// Create a new stream owned (and eventually destroyed) by the handle.
inline CudaStreamHandle create_stream() {
  cudaStream_t s = nullptr;
  EIGEN_CUDA_RUNTIME_CHECK(cudaStreamCreate(&s));
  return CudaStreamHandle(s, CudaStreamDeleter{/*borrow=*/false});
}

// Wrap an existing stream without taking ownership (nullptr = the legacy
// default stream).
inline CudaStreamHandle borrow_stream(cudaStream_t s) noexcept {
  return CudaStreamHandle(s, CudaStreamDeleter{/*borrow=*/true});
}

// On devices without stream-ordered allocation, retain a small-buffer cache to
// avoid a synchronous cudaMalloc/cudaFree pair for every DeviceScalar. Each
// cached pointer carries an event recorded after its last use; reuse on another
// stream waits on that event instead of returning the pointer immediately.
template <size_t SmallBufferThreshold = 256, size_t MaxPoolSize = 64>
struct FallbackDeviceBufferPool {
  static constexpr size_t kSmallBufferThreshold = SmallBufferThreshold;
  static constexpr size_t kMaxPoolSize = MaxPoolSize;

  struct Entry {
    void* ptr;
    size_t bytes;
    cudaEvent_t ready;
  };

  enum class State : signed char { kNotConstructed = 0, kAlive = 1, kDestroyed = 2 };

  static State& threadState() {
    thread_local State state = State::kNotConstructed;
    return state;
  }

  FallbackDeviceBufferPool() {
    free_list_.reserve(kMaxPoolSize);
    threadState() = State::kAlive;
  }

  ~FallbackDeviceBufferPool() {
    for (auto& entry : free_list_) {
      (void)cudaEventSynchronize(entry.ready);
      (void)cudaEventDestroy(entry.ready);
      (void)cudaFree(entry.ptr);
    }
    threadState() = State::kDestroyed;
  }

  void* allocate(size_t bytes, cudaStream_t stream) {
    for (size_t i = 0; i < free_list_.size(); ++i) {
      if (free_list_[i].bytes >= bytes) {
        Entry entry = free_list_[i];
        free_list_[i] = free_list_.back();
        free_list_.pop_back();
        EIGEN_CUDA_RUNTIME_CHECK(cudaStreamWaitEvent(stream, entry.ready, 0));
        (void)cudaEventDestroy(entry.ready);
        return entry.ptr;
      }
    }
    void* p = nullptr;
    EIGEN_CUDA_RUNTIME_CHECK(cudaMalloc(&p, bytes));
    return p;
  }

  void deallocate(void* p, size_t bytes, cudaStream_t stream) noexcept {
    cudaEvent_t ready = nullptr;
    if (cudaEventCreateWithFlags(&ready, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventRecord(ready, stream) != cudaSuccess) {
      if (ready) (void)cudaEventDestroy(ready);
      (void)cudaFree(p);
      return;
    }
    if (free_list_.size() < kMaxPoolSize) {
      free_list_.push_back({p, bytes, ready});
    } else {
      (void)cudaEventSynchronize(ready);
      (void)cudaEventDestroy(ready);
      (void)cudaFree(p);
    }
  }

  static FallbackDeviceBufferPool& threadLocal() {
    thread_local FallbackDeviceBufferPool pool;
    return pool;
  }

 private:
  std::vector<Entry> free_list_;
};

struct DeviceBufferDeleter {
  size_t bytes = 0;
  cudaStream_t stream = nullptr;

  void operator()(void* p) const noexcept {
    if (!p) return;
    if (device_supports_memory_pools()) {
      device_free(p, stream);
    } else if (bytes > 0 && bytes <= FallbackDeviceBufferPool<>::kSmallBufferThreshold &&
               FallbackDeviceBufferPool<>::threadState() == FallbackDeviceBufferPool<>::State::kAlive) {
      FallbackDeviceBufferPool<>::threadLocal().deallocate(p, bytes, stream);
    } else {
      (void)cudaFree(p);
    }
  }
};

class DeviceBuffer {
 public:
  DeviceBuffer() = default;

  DeviceBuffer(size_t bytes, cudaStream_t stream) : bytes_(bytes) {
    if (bytes > 0) {
      void* p = nullptr;
      if (device_supports_memory_pools()) {
        p = device_malloc(bytes, stream);
      } else if (bytes <= FallbackDeviceBufferPool<>::kSmallBufferThreshold &&
                 FallbackDeviceBufferPool<>::threadState() != FallbackDeviceBufferPool<>::State::kDestroyed) {
        p = FallbackDeviceBufferPool<>::threadLocal().allocate(bytes, stream);
      } else {
        EIGEN_CUDA_RUNTIME_CHECK(cudaMalloc(&p, bytes));
      }
      ptr_ = std::unique_ptr<void, DeviceBufferDeleter>(p, DeviceBufferDeleter{bytes, stream});
    }
  }

  // Explicit moves so a moved-from buffer reports size() == 0 (callers use
  // size() for grow-only reuse decisions; a stale size on a null buffer would
  // suppress the reallocation).
  DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(std::move(o.ptr_)), bytes_(o.bytes_) { o.bytes_ = 0; }
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
    if (this != &o) {
      ptr_ = std::move(o.ptr_);
      bytes_ = o.bytes_;
      o.bytes_ = 0;
    }
    return *this;
  }

  void* get() const noexcept { return ptr_.get(); }
  void* release() noexcept {
    bytes_ = 0;
    return ptr_.release();
  }
  explicit operator bool() const noexcept { return static_cast<bool>(ptr_); }

  /** Logical allocation size in bytes, tracked for adopted pointers as well. */
  size_t size() const noexcept { return bytes_; }

  // Adopt an existing device pointer of `bytes` usable bytes. Caller
  // relinquishes ownership.
  static DeviceBuffer adopt(void* p, size_t bytes, cudaStream_t stream) noexcept {
    DeviceBuffer b;
    b.ptr_ = std::unique_ptr<void, DeviceBufferDeleter>(p, DeviceBufferDeleter{bytes, stream});
    b.bytes_ = p ? bytes : 0;
    return b;
  }

 private:
  std::unique_ptr<void, DeviceBufferDeleter> ptr_;
  size_t bytes_ = 0;
};

// cudaMemcpyAsync only overlaps with compute when the host side is pinned, so
// async D2H staging goes through this buffer.
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

// cudaDataType_t lives in library_types.h, pulled in transitively by
// cuda_runtime.h, so this trait needs no NVIDIA library header of its own.
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
}  // namespace internal
}  // namespace gpu
}  // namespace Eigen

#endif  // EIGEN_GPU_SUPPORT_H
