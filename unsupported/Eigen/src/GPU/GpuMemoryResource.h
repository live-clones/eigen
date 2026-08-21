// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// The host-visible storage kinds a DeviceMatrix or DeviceScalar can be built on.
//
// internal::DeviceBuffer used to call cudaMallocAsync directly, which fixed both
// the allocation strategy and the ownership rules in the buffer. A MemoryResource
// moves that decision to the caller: the buffer holds a resource and asks it for
// storage, so device memory, managed memory and page-locked host memory are one
// axis of variation instead of one class each. Because the buffer is where it
// lives, everything built on one -- matrices, scalars, solver scratch -- gets
// every storage kind for free.
//
// The MemoryResource interface and the two device-only implementations live in
// GpuSupport.h, where the buffer can default to one of them; this header adds
// the kinds the host can address. Allocation returns *two* pointers rather
// than one. That is not gratuitous:
// on Tegra, malloc + cudaHostRegister hands back a device address different
// from the host address --
//
//   registered : host=0xaaaae7179630  device=0x1021c4630   same=0
//   hostAlloc  : host=0x1021c4000     device=0x1021c4000   same=1
//   managed    : one pointer by construction
//
// -- so a pmr-style `void* allocate()` cannot express the registered case
// without a side table mapping one to the other.

#ifndef EIGEN_GPU_MEMORY_RESOURCE_H
#define EIGEN_GPU_MEMORY_RESOURCE_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdlib>

#include "./GpuSupport.h"

namespace Eigen {
namespace gpu {

namespace internal {

/** cudaMallocManaged: one address valid on both sides.
 *
 * Cheap to allocate and simple to use, but on a device without
 * concurrentManagedAccess the host may not touch it while the GPU is busy, so
 * every host read costs a device-wide synchronize. Prefer a page-locked host
 * resource for anything the host inspects per iteration. */
class ManagedMemoryResource final : public MemoryResource {
 public:
  Allocation allocate(size_t bytes, cudaStream_t /*stream*/) override {
    Allocation a;
    if (bytes == 0) return a;
    EIGEN_CUDA_RUNTIME_CHECK(cudaMallocManaged(&a.device, bytes, cudaMemAttachGlobal));
    a.host = a.device;
    return a;
  }
  void deallocate(const Allocation& a, size_t /*bytes*/, cudaStream_t /*stream*/) noexcept override {
    if (a.device) (void)cudaFree(a.device);
  }
  bool isHostAccessible() const noexcept override { return true; }
  bool allowsConcurrentHostAccess() const noexcept override { return device_capabilities().concurrent_managed_access; }
  bool supportsMultipleAllocations() const noexcept override { return true; }
  const char* name() const noexcept override { return "managed"; }
};

/** cudaHostAlloc(cudaHostAllocMapped): page-locked host memory mapped into the
 * device address space.
 *
 * The host and device addresses come back equal on the parts measured so far,
 * but that is not contractual and the Allocation carries both regardless. No
 * restriction on concurrent host access. */
class MappedHostMemoryResource final : public MemoryResource {
 public:
  Allocation allocate(size_t bytes, cudaStream_t /*stream*/) override {
    Allocation a;
    if (bytes == 0) return a;
    eigen_assert(device_capabilities().can_map_host_memory && "device cannot map host memory");
    cudaError_t status = cudaHostAlloc(&a.host, bytes, cudaHostAllocMapped);
    EIGEN_CUDA_RUNTIME_CHECK(status);
    if (status != cudaSuccess) return a;
    status = cudaHostGetDevicePointer(&a.device, a.host, 0);
    if (status != cudaSuccess) {
      (void)cudaFreeHost(a.host);
      a.device = nullptr;
      a.host = nullptr;
    }
    EIGEN_CUDA_RUNTIME_CHECK(status);
    return a;
  }
  void deallocate(const Allocation& a, size_t /*bytes*/, cudaStream_t /*stream*/) noexcept override {
    if (a.host) (void)cudaFreeHost(a.host);
  }
  bool isHostAccessible() const noexcept override { return true; }
  bool allowsConcurrentHostAccess() const noexcept override { return true; }
  bool supportsMultipleAllocations() const noexcept override { return true; }
  const char* name() const noexcept override { return "mapped"; }
};

/** malloc + cudaHostRegister: ordinary host memory, page-locked and mapped.
 *
 * This is the resource whose device address differs from its host address, and
 * the reason Allocation carries both. Registration walks the page table, so it
 * is the most expensive of the four to allocate and the cheapest to use from
 * the host. */
class RegisteredHostMemoryResource final : public MemoryResource {
 public:
  Allocation allocate(size_t bytes, cudaStream_t /*stream*/) override {
    Allocation a;
    if (bytes == 0) return a;
    eigen_assert(device_capabilities().can_map_host_memory && "device cannot map host memory");
    a.host = std::malloc(bytes);
    eigen_assert(a.host != nullptr && "host allocation failed");
    if (a.host == nullptr) return a;
    cudaError_t status = cudaHostRegister(a.host, bytes, cudaHostRegisterMapped);
    if (status != cudaSuccess) {
      std::free(a.host);
      a.host = nullptr;
    }
    EIGEN_CUDA_RUNTIME_CHECK(status);
    if (status != cudaSuccess) return a;
    status = cudaHostGetDevicePointer(&a.device, a.host, 0);
    if (status != cudaSuccess) {
      (void)cudaHostUnregister(a.host);
      std::free(a.host);
      a.device = nullptr;
      a.host = nullptr;
    }
    EIGEN_CUDA_RUNTIME_CHECK(status);
    return a;
  }
  void deallocate(const Allocation& a, size_t /*bytes*/, cudaStream_t /*stream*/) noexcept override {
    if (!a.host) return;
    (void)cudaHostUnregister(a.host);
    std::free(a.host);
  }
  bool isHostAccessible() const noexcept override { return true; }
  bool allowsConcurrentHostAccess() const noexcept override { return true; }
  bool supportsMultipleAllocations() const noexcept override { return true; }
  const char* name() const noexcept override { return "registered"; }
};

/** Recycles allocations from an upstream resource.
 *
 * Device memory comes from a stream-ordered pool and costs ~1.7 us to obtain.
 * The host-visible kinds do not: cudaMallocManaged, cudaHostAlloc and
 * cudaHostRegister are synchronous and walk page tables, measured at 106-906 us
 * for the same sizes on an AGX Orin -- two to three orders of magnitude more.
 * A matrix allocated per iteration would spend far longer being created than
 * being computed on.
 *
 * Matching is by exact byte count, which is what a loop reusing one shape
 * needs, and avoids handing back an allocation larger than asked for whose
 * host and device pointers were derived for a different length.
 *
 * The cache is bounded in *bytes*, not entries, because what it holds is
 * page-locked: memory the OS cannot reclaim. An entry-capped cache measured
 * here held roughly 2 GiB pinned at n=4096 and slowed unrelated device-only
 * work by 37x, which is a far worse outcome than the allocation it saved.
 * Buffers above kMaxEntryBytes are never cached. The cost being amortized is
 * roughly constant in size -- 106 us for a 16 KiB registration and 906 us for a
 * 4 MiB one -- so it dominates a small matrix and disappears against a large
 * one, while the memory pinned to hold it grows with the matrix. Caching the
 * small end captures nearly all of the benefit for a small fraction of the
 * pinned footprint.
 *
 * The resource itself has process lifetime so a buffer may cross threads. Its
 * fixed-capacity free list is thread-local, like DeviceBufferPool: memory
 * released on another thread lands in that releasing thread's list. One CUDA
 * event per thread fences releases on the legacy default stream; allocations
 * reuse cached blocks only after the latest such event has completed. */
template <typename Upstream>
class CachingMemoryResource final : public MemoryResource {
 public:
  /** Total bytes the cache may hold. */
  static constexpr size_t kMaxCachedBytes = 16u << 20;
  /** Largest single allocation worth caching. */
  static constexpr size_t kMaxEntryBytes = 4u << 20;
  static constexpr size_t kMaxEntries = 32;

  explicit CachingMemoryResource(Upstream& upstream) : upstream_(&upstream) {}

  Allocation allocate(size_t bytes, cudaStream_t stream) override {
    if (!cacheable()) return upstream_->allocate(bytes, stream);
    Cache& cache = threadCache();
    if (!cache.releasesComplete()) return upstream_->allocate(bytes, stream);
    for (size_t i = 0; i < cache.size; ++i) {
      if (cache.free_list[i].bytes == bytes) {
        const Allocation a = cache.free_list[i].allocation;
        cache.cached_bytes -= cache.free_list[i].bytes;
        cache.free_list[i] = cache.free_list[--cache.size];
        return a;
      }
    }
    return upstream_->allocate(bytes, stream);
  }

  void deallocate(const Allocation& a, size_t bytes, cudaStream_t stream) noexcept override {
    if (!a.device) return;
    if (!cacheable()) {
      upstream_->deallocate(a, bytes, stream);
      return;
    }

    Cache& cache = threadCache();
    if (!cache.recordRelease()) {
      (void)cudaDeviceSynchronize();
      upstream_->deallocate(a, bytes, stream);
      return;
    }

    if (bytes <= kMaxEntryBytes && cache.size < kMaxEntries && cache.cached_bytes + bytes <= kMaxCachedBytes) {
      cache.free_list[cache.size++] = Entry{a, bytes};
      cache.cached_bytes += bytes;
      return;
    }
    cache.waitForReleases();
    upstream_->deallocate(a, bytes, stream);
  }

  bool isHostAccessible() const noexcept override { return upstream_->isHostAccessible(); }
  bool allowsConcurrentHostAccess() const noexcept override { return upstream_->allowsConcurrentHostAccess(); }
  bool supportsMultipleAllocations() const noexcept override { return upstream_->supportsMultipleAllocations(); }
  const char* name() const noexcept override { return upstream_->name(); }

 private:
  /** Storage whose host access is restricted is never held onto.
   *
   * On a device with concurrentManagedAccess == 0, keeping a managed
   * allocation alive slows *other* storage kinds process-wide: measured on an
   * AGX Orin, a host-read loop over mapped memory costs 62 us on its own and
   * 118 us -- managed's own figure -- once any managed allocation is live in
   * the process. Caching would make that permanent, so managed memory is
   * released as soon as it is finished with and pays its allocation cost each
   * time. Where the device does allow concurrent access the restriction does
   * not apply and managed allocations are cached like the rest. */
  bool cacheable() const noexcept { return upstream_->allowsConcurrentHostAccess(); }

  struct Entry {
    Allocation allocation;
    size_t bytes;
  };

  struct Cache {
    explicit Cache(Upstream& upstream) : upstream_(&upstream) {}

    ~Cache() {
      waitForReleases();
      for (size_t i = 0; i < size; ++i) upstream_->deallocate(free_list[i].allocation, free_list[i].bytes, nullptr);
      if (ready_event != nullptr) (void)cudaEventDestroy(ready_event);
    }

    bool recordRelease() noexcept {
      if (ready_event == nullptr && cudaEventCreateWithFlags(&ready_event, cudaEventDisableTiming) != cudaSuccess)
        return false;
      if (cudaEventRecord(ready_event, /*legacy default stream=*/nullptr) == cudaSuccess) return true;
      (void)cudaEventDestroy(ready_event);
      ready_event = nullptr;
      return false;
    }

    bool releasesComplete() const noexcept {
      return ready_event == nullptr || cudaEventQuery(ready_event) == cudaSuccess;
    }

    void waitForReleases() const noexcept {
      if (ready_event != nullptr) (void)cudaEventSynchronize(ready_event);
    }

    Upstream* upstream_;
    std::array<Entry, kMaxEntries> free_list;
    cudaEvent_t ready_event = nullptr;
    size_t size = 0;
    size_t cached_bytes = 0;
  };

  Cache& threadCache() {
    thread_local Cache cache(*upstream_);
    return cache;
  }

  Upstream* upstream_;
};

}  // namespace internal

/** \ingroup GPU_Module Managed storage; see internal::ManagedMemoryResource. */
inline MemoryResource& managedMemoryResource() {
  static internal::ManagedMemoryResource upstream;
  static internal::CachingMemoryResource<internal::ManagedMemoryResource> cached(upstream);
  return cached;
}

/** \ingroup GPU_Module Page-locked mapped host storage. */
inline MemoryResource& mappedHostMemoryResource() {
  static internal::MappedHostMemoryResource upstream;
  static internal::CachingMemoryResource<internal::MappedHostMemoryResource> cached(upstream);
  return cached;
}

/** \ingroup GPU_Module Registered ordinary host storage. */
inline MemoryResource& registeredHostMemoryResource() {
  static internal::RegisteredHostMemoryResource upstream;
  static internal::CachingMemoryResource<internal::RegisteredHostMemoryResource> cached(upstream);
  return cached;
}

/** \ingroup GPU_Module
 * True when the current device shares physical memory with the host
 * (cudaDeviceProp::integrated), so host storage can be made GPU-readable
 * without a copy. Reported, not acted on. */
inline bool deviceIsIntegrated() { return internal::device_capabilities().integrated; }

/** \ingroup GPU_Module
 * True when the host may access managed memory while device work is
 * outstanding. Zero on Tegra. */
inline bool deviceSupportsConcurrentManagedAccess() {
  return internal::device_capabilities().concurrent_managed_access;
}

/** \ingroup GPU_Module
 * The resource that avoids host/device copies on this device, or the plain
 * device resource when copies are the cheaper option.
 *
 * Reported rather than applied silently: on an integrated part this is
 * mappedHostMemoryResource(), on a discrete one deviceMemoryResource(), and a
 * caller who wants a specific kind should name it. */
inline MemoryResource& preferredHostVisibleMemoryResource() {
  return internal::device_capabilities().integrated ? mappedHostMemoryResource() : deviceMemoryResource();
}

/** \ingroup GPU_Module
 * \class HostMatrixResource
 * \brief Adopts an existing Eigen matrix and serves its storage to the device.
 *
 * The matrix is moved in and held here, page-locked and mapped, so the device
 * reads the caller's own bytes with no copy and no second allocation. Deletion
 * is this object's job, which is what "move the matrix in so the allocator owns
 * it" amounts to in practice.
 *
 * \code
 * MatrixXf A = ...;
 * gpu::HostMatrixResource<float> res(std::move(A));
 * gpu::DeviceMatrix<float> d(res.rows(), res.cols(), res);
 * // d.data() is the device alias of the adopted storage; d.hostData() the host one.
 * \endcode
 *
 * \b Note. This serves exactly one allocation -- the matrix it adopted -- so it
 * satisfies the MemoryResource interface without really being an allocator. The
 * fit is imperfect on purpose: Eigen's DenseStorage has no release(), and it
 * allocates through conditional_aligned_new_auto, so the buffer cannot be handed
 * to a general allocator that would later free() it. Owning the Matrix object is
 * the only way to take responsibility for its deletion. */
template <typename Scalar_>
class HostMatrixResource final : public MemoryResource {
 public:
  using Scalar = Scalar_;
  using PlainMatrix = Eigen::Matrix<Scalar, Dynamic, Dynamic, ColMajor>;

  explicit HostMatrixResource(PlainMatrix&& matrix) : matrix_(std::move(matrix)) {
    if (matrix_.size() > 0) {
      eigen_assert(internal::device_capabilities().can_map_host_memory && "device cannot map host memory");
      bytes_ = static_cast<size_t>(matrix_.size()) * sizeof(Scalar);
      const cudaError_t register_status = cudaHostRegister(matrix_.data(), bytes_, cudaHostRegisterMapped);
      EIGEN_CUDA_RUNTIME_CHECK(register_status);
      if (register_status != cudaSuccess) return;
      registered_ = true;
      const cudaError_t pointer_status = cudaHostGetDevicePointer(&device_, matrix_.data(), 0);
      if (pointer_status != cudaSuccess) {
        (void)cudaHostUnregister(matrix_.data());
        device_ = nullptr;
        registered_ = false;
      }
      EIGEN_CUDA_RUNTIME_CHECK(pointer_status);
    }
  }

  ~HostMatrixResource() override {
    waitForRelease();
    if (registered_) (void)cudaHostUnregister(matrix_.data());
  }

  HostMatrixResource(const HostMatrixResource&) = delete;
  HostMatrixResource& operator=(const HostMatrixResource&) = delete;

  Allocation allocate(size_t bytes, cudaStream_t /*stream*/) override {
    waitForRelease();
    eigen_assert(bytes <= bytes_ && "HostMatrixResource serves only the matrix it adopted");
    eigen_assert(!served_ && "HostMatrixResource serves a single allocation");
    served_ = true;
    Allocation a;
    a.device = device_;
    a.host = matrix_.data();
    return a;
  }

  void deallocate(const Allocation& /*a*/, size_t /*bytes*/, cudaStream_t /*stream*/) noexcept override {
    // The storage belongs to the adopted matrix and is released in the
    // destructor; a DeviceMatrix going out of scope must not free it.
    cudaEvent_t event = nullptr;
    if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) == cudaSuccess &&
        cudaEventRecord(event, /*legacy default stream=*/nullptr) == cudaSuccess) {
      ready_event_ = event;
    } else {
      if (event != nullptr) (void)cudaEventDestroy(event);
      (void)cudaDeviceSynchronize();
    }
    served_ = false;
  }

  bool isHostAccessible() const noexcept override { return true; }
  bool allowsConcurrentHostAccess() const noexcept override { return true; }
  bool supportsMultipleAllocations() const noexcept override { return false; }
  const char* name() const noexcept override { return "adopted-host-matrix"; }

  Index rows() const { return matrix_.rows(); }
  Index cols() const { return matrix_.cols(); }
  /** The adopted matrix, readable in place once device work has completed. */
  const PlainMatrix& matrix() const { return matrix_; }
  PlainMatrix& matrix() { return matrix_; }

 private:
  void waitForRelease() noexcept {
    if (ready_event_ == nullptr) return;
    (void)cudaEventSynchronize(ready_event_);
    (void)cudaEventDestroy(ready_event_);
    ready_event_ = nullptr;
  }

  PlainMatrix matrix_;
  void* device_ = nullptr;
  cudaEvent_t ready_event_ = nullptr;
  size_t bytes_ = 0;
  bool registered_ = false;
  bool served_ = false;
};

}  // namespace gpu
}  // namespace Eigen

#endif  // EIGEN_GPU_MEMORY_RESOURCE_H
