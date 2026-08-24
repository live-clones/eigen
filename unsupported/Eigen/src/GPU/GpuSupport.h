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

  /** Whether the resource can serve another allocation while an earlier one
   * remains live. Custom resources conservatively default to false. */
  virtual bool supportsMultipleAllocations() const noexcept { return false; }

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

/** Device-only storage from the module's stream-ordered allocator.
 *
 * The only correct choice on a discrete GPU, and what a DeviceMatrix uses when
 * it names nothing else. device_malloc / device_free order allocation and
 * release on the legacy default stream, so a block is never handed out again
 * while a kernel enqueued on a blocking stream may still be reading it. */
class DeviceMemoryResource final : public MemoryResource {
 public:
  Allocation allocate(size_t bytes, cudaStream_t /*stream*/) override {
    Allocation a;
    a.device = device_malloc(bytes);
    return a;
  }
  void deallocate(const Allocation& a, size_t /*bytes*/, cudaStream_t /*stream*/) noexcept override {
    device_free(a.device);
  }
  bool isHostAccessible() const noexcept override { return false; }
  bool supportsMultipleAllocations() const noexcept override { return true; }
  const char* name() const noexcept override { return "device"; }
};

/** Device-only storage with small blocks recycled through DeviceBufferPool.
 *
 * The default for DeviceScalar and for the module's internal scratch: sizes
 * that churn once per iteration and would otherwise pay cudaMallocAsync every
 * time. The pool hands a freed block straight to the next requester with no
 * stream ordering, which suits storage created and consumed on one stream and
 * is exactly why DeviceMatrix names the plain device resource instead. Blocks
 * over the pool's threshold fall through to the same allocator. */
class PooledDeviceMemoryResource final : public MemoryResource {
 public:
  Allocation allocate(size_t bytes, cudaStream_t /*stream*/) override {
    Allocation a;
    // Bypass the pool once its thread_local has been destroyed (allocation from
    // a static or TLS destructor); deallocate() then also takes the direct path.
    const bool pooled = bytes <= DeviceBufferPool<>::kSmallBufferThreshold &&
                        DeviceBufferPool<>::threadState() != DeviceBufferPool<>::State::kDestroyed;
    a.device = pooled ? DeviceBufferPool<>::threadLocal().allocate(bytes) : device_malloc(bytes);
    return a;
  }
  void deallocate(const Allocation& a, size_t bytes, cudaStream_t /*stream*/) noexcept override {
    if (!a.device) return;
    if (bytes <= DeviceBufferPool<>::kSmallBufferThreshold &&
        DeviceBufferPool<>::threadState() == DeviceBufferPool<>::State::kAlive) {
      DeviceBufferPool<>::threadLocal().deallocate(a.device, bytes);
    } else {
      // Over the threshold, or pooled storage outliving the pool during TLS
      // teardown: the pool's blocks come from device_malloc, so a direct free
      // is correct either way.
      device_free(a.device);
    }
  }
  bool isHostAccessible() const noexcept override { return false; }
  bool supportsMultipleAllocations() const noexcept override { return true; }
  const char* name() const noexcept override { return "pooled-device"; }
};

}  // namespace internal

/** \ingroup GPU_Module
 * Device-only storage through the stream-ordered memory pool. What a
 * DeviceMatrix allocates from when it names no other resource. */
inline MemoryResource& deviceMemoryResource() {
  static internal::DeviceMemoryResource r;
  return r;
}

/** \ingroup GPU_Module
 * Device-only storage with small blocks recycled per thread. What a
 * DeviceScalar -- and so every reduction result -- allocates from when it names
 * no other resource. */
inline MemoryResource& pooledDeviceMemoryResource() {
  static internal::PooledDeviceMemoryResource r;
  return r;
}

namespace internal {

struct DeviceBufferDeleter {
  // The resource that will release this block: it knows how the block was made,
  // and cudaFree, cudaFreeHost and unregister-then-free are not
  // interchangeable. Null means the block is borrowed and freed by its owner.
  MemoryResource* resource = nullptr;
  size_t bytes = 0;
  // The host alias goes back with the device pointer, since a resource that
  // page-locks ordinary memory frees the host pointer, not the device one.
  void* host = nullptr;
  // The stream this block was last used on. A resource built on cudaFreeAsync
  // recycles in stream order, so releasing on the default stream would let the
  // block be handed out again while work queued on a non-blocking stream is
  // still reading it -- the two are not ordered against each other.
  cudaStream_t stream = nullptr;

  void operator()(void* p) const noexcept {
    if (p && resource != nullptr) resource->deallocate(Allocation{p, host}, bytes, stream);
  }
};

/** Owning handle for a block of GPU-visible storage.
 *
 * Every block comes from a MemoryResource, and the buffer holds the one that
 * will release it, so "no resource" is not a state the buffer has to represent
 * -- the resource is simply defaulted. A resource may also supply a
 * host-addressable alias of the same bytes -- hostData() -- which is what lets
 * the host read and write the storage with no transfer at all. DeviceMatrix and
 * DeviceScalar are typed wrappers around one of these, so both get every
 * storage kind from the same place. */
class DeviceBuffer {
 public:
  DeviceBuffer() = default;

  /** Allocate \p bytes from \p resource, which must outlive the buffer.
   *
   * The default recycles small blocks per thread, which is what the module's
   * scratch and DeviceScalar want. DeviceMatrix names deviceMemoryResource()
   * instead: the pool does not order reuse against a consumer on another
   * stream, and a matrix's ready event exists precisely so there can be one. */
  explicit DeviceBuffer(size_t bytes, MemoryResource& resource = pooledDeviceMemoryResource(),
                        cudaStream_t stream = nullptr) {
    if (bytes == 0) return;
    const Allocation a = resource.allocate(bytes, stream);
    reset(a.device, DeviceBufferDeleter{&resource, bytes, a.host, stream});
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

  /** Release this block on \p stream rather than on the default stream.
   *
   * Set it to whichever stream last read or wrote the storage: a stream-ordered
   * resource must not recycle the block ahead of work still queued there. */
  void setReleaseStream(cudaStream_t stream) noexcept {
    if (ptr_) ptr_.get_deleter().stream = stream;
  }

  /** The resource that will release this block. Null exactly when there is
   * nothing to release: an empty buffer, or a borrowed view(). */
  MemoryResource* memoryResource() const noexcept { return ptr_ ? ptr_.get_deleter().resource : nullptr; }

  /** Whether destruction releases the storage. False for view(). */
  bool owns() const noexcept { return memoryResource() != nullptr; }

  /** Logical allocation size in bytes, tracked for adopted pointers as well.
   * Reported as 0 once the pointer is gone -- callers use it for grow-only
   * reuse decisions, and a stale size on a moved-from buffer would suppress
   * the reallocation. */
  size_t size() const noexcept { return ptr_ ? ptr_.get_deleter().bytes : 0; }

  explicit operator bool() const noexcept { return static_cast<bool>(ptr_); }

  /** Hand the device pointer out, leaving the buffer empty.
   *
   * A bare pointer records nothing about how its storage was made, and whoever
   * adopts it next will free it with device_free, so only the two device-only
   * resources qualify. Prefer moving the whole buffer where the caller can.
   *
   * Not noexcept: the guard is eigen_assert, which the test harness throws. */
  void* release() {
    eigen_assert(releasableAsBarePointer() &&
                 "only device-only storage can be released as a bare pointer: a borrowed view does not own its "
                 "memory, and host-visible storage can only be freed by its MemoryResource");
    // The stale deleter is unobservable: every accessor guards on ptr_, and
    // unique_ptr never invokes a deleter for a null pointer.
    return ptr_.release();
  }

  /** Adopt an existing device pointer of \p bytes usable bytes obtained from
   * device_malloc. Caller relinquishes ownership. */
  static DeviceBuffer adopt(void* p, size_t bytes) noexcept { return wrap(p, bytes, &deviceMemoryResource()); }

  /** Non-owning view over storage that someone else owns and outlives it. */
  static DeviceBuffer view(void* p, size_t bytes) noexcept { return wrap(p, bytes, /*resource=*/nullptr); }

 private:
  // Wrap an existing pointer; a null resource means nothing frees it.
  static DeviceBuffer wrap(void* p, size_t bytes, MemoryResource* resource) noexcept {
    DeviceBuffer b;
    if (p) b.reset(p, DeviceBufferDeleter{resource, bytes});
    return b;
  }

  // Both device-only resources hand out storage that device_free releases --
  // the pool's blocks come from device_malloc like any other. An empty buffer
  // releases a null pointer, which is harmless.
  bool releasableAsBarePointer() const noexcept {
    MemoryResource* r = memoryResource();
    return !ptr_ || r == &deviceMemoryResource() || r == &pooledDeviceMemoryResource();
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
 * produced the value. Device-only storage takes the stream wait too: the host
 * cannot address it at all, so its callers have already stopped. */
inline void sync_for_host_access(const MemoryResource& resource, cudaStream_t stream) {
  if (resource.isHostAccessible() && !resource.allowsConcurrentHostAccess()) {
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
