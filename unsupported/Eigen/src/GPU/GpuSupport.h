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
#include <vector>

#include <limits>
#include <memory>
#include <type_traits>

namespace Eigen {
namespace gpu {
// Transpose/adjoint flag for BLAS-, solver-, and sparse-style calls. Each
// library's support header maps it to its own enum (cublasOperation_t,
// cusparseOperation_t, ...) via a to_<lib>_op() helper.
enum class GpuOp { NoTrans, Trans, ConjTrans };

namespace internal {
// Aborts via eigen_assert on failure, and eigen_assert throws where it is so
// configured, so this must not be used in a destructor or any other noexcept
// function: there the throw would call std::terminate.
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
// synchronization performed by cudaMalloc / cudaFree. All module allocations
// go through device_malloc / device_free on the *legacy default stream*:
// legacy-stream ordering guarantees that work enqueued later on any blocking
// stream observes the allocation, and that a free waits for all previously
// enqueued work on blocking streams — the same lifetime guarantees callers
// got from cudaMalloc / cudaFree, minus the host stalls.
//
// Caveat: streams created with cudaStreamNonBlocking do not synchronize with
// the legacy stream. When borrowing such a stream (gpu::Context(stream)),
// define EIGEN_GPU_NO_STREAM_ORDERED_ALLOC to fall back to cudaMalloc/cudaFree.
//
// Support is detected once per process, from the device current at first use.

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

/** Device properties that decide whether host-addressable storage is usable,
 * and what it costs. Probed once per process from the device current at first
 * use, matching device_supports_memory_pools(). */
struct DeviceCapabilities {
  bool integrated = false;
  bool concurrent_managed_access = false;
  bool can_map_host_memory = false;
  bool managed_memory = false;
};

inline const DeviceCapabilities& device_capabilities() {
  static const DeviceCapabilities caps = [] {
    DeviceCapabilities c;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return c;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return c;
    c.integrated = prop.integrated != 0;
    c.concurrent_managed_access = prop.concurrentManagedAccess != 0;
    c.can_map_host_memory = prop.canMapHostMemory != 0;
    c.managed_memory = prop.managedMemory != 0;
    return c;
  }();
  return caps;
}

}  // namespace internal

/** \ingroup GPU_Module
 * A block of storage, as returned by a MemoryResource.
 *
 * \a device is what kernels and NVIDIA library calls receive; \a host is a
 * host-addressable alias of the same bytes, or null for device-only memory.
 * The two are equal for some resources and different for others -- on Tegra,
 * malloc + cudaHostRegister returns distinct addresses -- so callers must not
 * assume either. */
struct Allocation {
  void* device = nullptr;
  void* host = nullptr;

  explicit operator bool() const noexcept { return device != nullptr; }
};

/** \ingroup GPU_Module
 * \class MemoryResource
 * \brief Supplies the storage behind a DeviceMatrix.
 *
 * Resources carry no per-allocation state and must outlive every buffer that
 * names them; the built-in ones are process-wide singletons. See
 * GpuMemoryResource.h for the concrete kinds. */
class MemoryResource {
 public:
  virtual ~MemoryResource() = default;

  virtual Allocation allocate(size_t bytes, cudaStream_t stream) = 0;
  virtual void deallocate(const Allocation& a, size_t bytes, cudaStream_t stream) noexcept = 0;

  /** Whether Allocation::host is non-null: the host reads and writes this
   * memory directly, so no upload or download is needed. */
  virtual bool isHostAccessible() const noexcept = 0;

  /** Whether the host may touch the memory while device work is outstanding.
   * False for managed memory wherever concurrentManagedAccess is 0. */
  virtual bool allowsConcurrentHostAccess() const noexcept { return false; }

  virtual const char* name() const noexcept = 0;
};

namespace internal {

struct CudaFreeHostDeleter {
  void operator()(void* p) const noexcept {
    if (p) (void)cudaFreeHost(p);
  }
};

// RAII CUDA stream; the ownership flag supports borrowed, caller-owned streams.
struct CudaStreamDeleter {
  bool owns = true;
  void operator()(cudaStream_t s) const noexcept {
    if (owns && s) (void)cudaStreamDestroy(s);
  }
};
using UniqueStream = std::unique_ptr<std::remove_pointer_t<cudaStream_t>, CudaStreamDeleter>;

// Recycles allocations up to kSmallBufferThreshold bytes (e.g. DeviceScalar) to
// avoid cudaMalloc/cudaFree overhead. Larger allocations bypass the pool.
template <size_t SmallBufferThreshold = 256, size_t MaxPoolSize = 64>
struct DeviceBufferPool {
  static constexpr size_t kSmallBufferThreshold = SmallBufferThreshold;
  static constexpr size_t kMaxPoolSize = MaxPoolSize;

  struct Entry {
    void* ptr;
    size_t bytes;
  };

  // Lifetime marker for the thread-local pool. thread_local destruction runs
  // in reverse construction order, so a long-lived object holding pooled
  // buffers (e.g. the thread-local gpu::Context, or a static) can be
  // destroyed *after* the pool. The marker is trivially destructible — it
  // stays readable during TLS teardown — letting the deleter fall back to a
  // direct device_free once the pool is gone instead of touching a destroyed
  // vector.
  enum class State : signed char { kNotConstructed = 0, kAlive = 1, kDestroyed = 2 };

  static State& threadState() {
    thread_local State state = State::kNotConstructed;
    return state;
  }

  DeviceBufferPool() { threadState() = State::kAlive; }

  ~DeviceBufferPool() {
    for (auto& e : free_list_) device_free(e.ptr);
    threadState() = State::kDestroyed;
  }

  void* allocate(size_t bytes) {
    for (size_t i = 0; i < free_list_.size(); ++i) {
      if (free_list_[i].bytes >= bytes) {
        void* p = free_list_[i].ptr;
        free_list_[i] = free_list_.back();
        free_list_.pop_back();
        return p;
      }
    }
    return device_malloc(bytes);
  }

  void deallocate(void* p, size_t bytes) {
    if (free_list_.size() < kMaxPoolSize) {
      free_list_.push_back({p, bytes});
    } else {
      device_free(p);
    }
  }

  static DeviceBufferPool& threadLocal() {
    thread_local DeviceBufferPool pool;
    return pool;
  }

 private:
  std::vector<Entry> free_list_;
};

// How a block of GPU-visible storage was obtained, and therefore how it must be
// given back. Recorded per block rather than per type, so one buffer class can
// hold storage from any of them.
enum class BufferKind : signed char {
  kBorrowed,  // non-owning view; whoever owns the storage frees it
  kPooled,    // default allocator, recycled through the thread-local pool
  kDirect,    // default allocator, released with device_free
  kResource,  // MemoryResource::deallocate -- only the resource knows how
};

struct DeviceBufferDeleter {
  BufferKind kind = BufferKind::kDirect;
  size_t bytes = 0;
  // Storage obtained from a MemoryResource goes back to that resource, which
  // knows how it was made: cudaFree, cudaFreeHost and unregister-then-free are
  // not interchangeable. The host alias goes back with it, since a resource
  // that page-locks ordinary memory frees the host pointer, not the device one.
  MemoryResource* resource = nullptr;
  void* host = nullptr;

  void operator()(void* p) const noexcept {
    if (!p || kind == BufferKind::kBorrowed) return;
    if (kind == BufferKind::kResource) {
      resource->deallocate(Allocation{p, host}, bytes, /*stream=*/nullptr);
    } else if (kind == BufferKind::kPooled && DeviceBufferPool<>::threadState() == DeviceBufferPool<>::State::kAlive) {
      DeviceBufferPool<>::threadLocal().deallocate(p, bytes);
    } else {
      // kDirect, or pooled storage outliving the pool during TLS teardown: the
      // pool's blocks come from device_malloc, so this is correct either way.
      device_free(p);
    }
  }
};

/** Owning handle for a block of GPU-visible storage.
 *
 * The block comes either from the module's default device allocator or from a
 * MemoryResource, and the buffer records which so it can be released the same
 * way. A resource may also supply a host-addressable alias of the same bytes --
 * hostData() -- which is what lets the host read and write the storage with no
 * transfer at all. DeviceMatrix and DeviceScalar are typed wrappers around one
 * of these, so both get every storage kind from the same place. */
class DeviceBuffer {
 public:
  /** Whether a default (resource-less) allocation may be served from the
   * thread-local small-buffer pool.
   *
   * The pool hands a freed block straight to the next requester with no stream
   * ordering. That suits scratch created and consumed on one stream, and is
   * wrong for storage a consumer on another stream may still be reading, so
   * DeviceMatrix opts out: device_malloc / device_free are ordered on the
   * legacy default stream. */
  enum class Pooling : signed char { kAllowed, kDisallowed };

  DeviceBuffer() = default;

  explicit DeviceBuffer(size_t bytes, Pooling pooling = Pooling::kAllowed) {
    allocate(bytes, /*resource=*/nullptr, pooling);
  }

  /** Allocate through \p resource, which must outlive the buffer, or from the
   * default device allocator when it is null. */
  DeviceBuffer(size_t bytes, MemoryResource* resource, Pooling pooling = Pooling::kAllowed) {
    allocate(bytes, resource, pooling);
  }

  DeviceBuffer(DeviceBuffer&&) noexcept = default;
  DeviceBuffer& operator=(DeviceBuffer&&) noexcept = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  /** Device pointer: what kernels and NVIDIA library calls receive. */
  void* get() const noexcept { return ptr_.get(); }

  /** Host-addressable alias of the same bytes, or null for device-only storage.
   * Not necessarily equal to get(): malloc + cudaHostRegister yields distinct
   * host and device addresses. */
  void* hostData() const noexcept { return ptr_ ? ptr_.get_deleter().host : nullptr; }

  /** Whether hostData() is usable. */
  bool isHostAccessible() const noexcept { return hostData() != nullptr; }

  /** The resource backing this block, or null for the default allocator. */
  MemoryResource* memoryResource() const noexcept { return ptr_ ? ptr_.get_deleter().resource : nullptr; }

  /** Whether destruction releases the storage. False for view(). */
  bool owns() const noexcept { return ptr_ && ptr_.get_deleter().kind != BufferKind::kBorrowed; }

  /** Logical allocation size in bytes, tracked for adopted pointers as well.
   * Reported as 0 once the pointer is gone -- callers use it for grow-only
   * reuse decisions, and a stale size on a moved-from buffer would suppress
   * the reallocation. */
  size_t size() const noexcept { return ptr_ ? ptr_.get_deleter().bytes : 0; }

  explicit operator bool() const noexcept { return static_cast<bool>(ptr_); }

  /** Hand the device pointer out; the caller must free it with device_free.
   * Not noexcept: the guards below are eigen_assert, which the test harness
   * turns into a throw. */
  void* release() {
    eigen_assert((owns() || !ptr_) && "cannot release a borrowed DeviceBuffer: it does not own its memory");
    eigen_assert(memoryResource() == nullptr &&
                 "cannot release resource-backed storage: only its MemoryResource can free it");
    void* p = ptr_.release();
    ptr_.get_deleter() = DeviceBufferDeleter{};
    return p;
  }

  /** Adopt an existing device pointer of \p bytes usable bytes obtained from
   * the default allocator. Caller relinquishes ownership; adopted buffers
   * bypass the pool on destruction. */
  static DeviceBuffer adopt(void* p, size_t bytes) noexcept {
    DeviceBuffer b;
    if (p) b.reset(p, DeviceBufferDeleter{BufferKind::kDirect, bytes});
    return b;
  }

  /** Non-owning view over storage that someone else owns and outlives it. */
  static DeviceBuffer view(void* p, size_t bytes) noexcept {
    DeviceBuffer b;
    if (p) b.reset(p, DeviceBufferDeleter{BufferKind::kBorrowed, bytes});
    return b;
  }

 private:
  void allocate(size_t bytes, MemoryResource* resource, Pooling pooling) {
    if (bytes == 0) return;
    if (resource != nullptr) {
      const Allocation a = resource->allocate(bytes, /*stream=*/nullptr);
      reset(a.device, DeviceBufferDeleter{BufferKind::kResource, bytes, resource, a.host});
      return;
    }
    // Bypass the pool once its thread_local has been destroyed (allocation from
    // a static or TLS destructor); the matching deleter then also takes the
    // direct device_free path.
    const bool pooled = pooling == Pooling::kAllowed && bytes <= DeviceBufferPool<>::kSmallBufferThreshold &&
                        DeviceBufferPool<>::threadState() != DeviceBufferPool<>::State::kDestroyed;
    void* p = pooled ? DeviceBufferPool<>::threadLocal().allocate(bytes) : device_malloc(bytes);
    reset(p, DeviceBufferDeleter{pooled ? BufferKind::kPooled : BufferKind::kDirect, bytes});
  }

  void reset(void* p, const DeviceBufferDeleter& deleter) {
    ptr_ = std::unique_ptr<void, DeviceBufferDeleter>(p, deleter);
  }

  std::unique_ptr<void, DeviceBufferDeleter> ptr_;
};

/** Order a host read or write of storage the device also sees.
 *
 * Which synchronization that needs is a property of the storage, so the
 * resource decides: managed memory on a device without concurrentManagedAccess
 * forbids host access while *any* device work is outstanding and needs a
 * device-wide wait, while page-locked host memory only needs the stream that
 * produced the value. Callers establish that the storage is host-accessible at
 * all before calling; a null resource is the default device allocator, whose
 * storage never is. */
inline void sync_for_host_access(const MemoryResource* resource, cudaStream_t stream) {
  if (resource != nullptr && !resource->allowsConcurrentHostAccess()) {
    EIGEN_CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
  } else {
    EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream));
  }
}

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

// Upload a column-major host matrix whose strides are in elements. Ref<const
// PlainMatrix> can bind any outer stride in place. Use a 2D DMA for ordinary
// padded layouts; copy legal negative or overlapping Eigen strides one
// contiguous column at a time because CUDA cannot express them as a pitch.
template <typename Scalar>
void upload_host_matrix(Scalar* dst, Index dst_outer_stride, const Scalar* src, Index src_outer_stride, Index rows,
                        Index cols, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) return;
  eigen_assert(dst_outer_stride >= rows);
  const size_t column_bytes = static_cast<size_t>(rows) * sizeof(Scalar);
  if (src_outer_stride >= rows) {
    EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpy2DAsync(dst, static_cast<size_t>(dst_outer_stride) * sizeof(Scalar), src,
                                               static_cast<size_t>(src_outer_stride) * sizeof(Scalar), column_bytes,
                                               static_cast<size_t>(cols), cudaMemcpyHostToDevice, stream));
  } else {
    for (Index col = 0; col < cols; ++col) {
      EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(dst + col * dst_outer_stride, src + col * src_outer_stride, column_bytes,
                                               cudaMemcpyHostToDevice, stream));
    }
  }
}

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
