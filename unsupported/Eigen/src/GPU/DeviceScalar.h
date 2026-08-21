// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// Device-resident scalar for deferred host synchronization.
//
// Reductions (dot, nrm2) write their result straight to device memory under
// CUBLAS_POINTER_MODE_DEVICE, so no host sync happens until the value is read.
// Conversion to Scalar is that read. Because the first conversion flushes the
// stream, later conversions in the same expression only download: a CG iteration
// costs one sync rather than three.
//
// The storage comes from the same internal::DeviceBuffer that backs
// DeviceMatrix, so a scalar can be placed on any gpu::MemoryResource. On a
// host-accessible one the value is written where the host can already address
// it and the download disappears, leaving only the synchronization.

#ifndef EIGEN_GPU_DEVICE_SCALAR_H
#define EIGEN_GPU_DEVICE_SCALAR_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

#include "./GpuSupport.h"
#include "./DeviceScalarOps.h"

namespace Eigen {
namespace gpu {

template <typename Scalar_>
class DeviceScalar {
 public:
  using Scalar = Scalar_;

  /** Allocate an uninitialized device scalar. Contents are undefined until
   * written, e.g. by cuBLAS dot/nrm2 under POINTER_MODE_DEVICE. */
  explicit DeviceScalar(cudaStream_t stream = nullptr) : d_val_(sizeof(Scalar)), stream_(stream) {}

  /** As above, but allocated through \p resource, which must outlive this
   * scalar. A host-accessible resource makes get() a synchronization and a
   * load instead of a synchronization and a device-to-host copy. */
  explicit DeviceScalar(MemoryResource& resource, cudaStream_t stream = nullptr)
      : d_val_(sizeof(Scalar), &resource), stream_(stream) {}

  DeviceScalar(Scalar host_val, cudaStream_t stream) : d_val_(sizeof(Scalar)), stream_(stream) { upload(host_val); }

  /** Initialized to \p host_val, in storage from \p resource. */
  DeviceScalar(Scalar host_val, MemoryResource& resource, cudaStream_t stream)
      : d_val_(sizeof(Scalar), &resource), stream_(stream) {
    upload(host_val);
  }

  /** Uninitialized storage from \p resource, or from the default device
   * allocator when it is null.
   *
   * The nullable form a constructor cannot offer -- DeviceScalar(cudaStream_t)
   * would be ambiguous with it -- so that a result can inherit the storage kind
   * of the operand it was computed from. */
  static DeviceScalar fromResource(MemoryResource* resource, cudaStream_t stream) {
    return DeviceScalar(internal::DeviceBuffer(sizeof(Scalar), resource), stream);
  }

  /** As fromResource(), initialized to \p host_val. */
  static DeviceScalar fromResource(MemoryResource* resource, Scalar host_val, cudaStream_t stream) {
    DeviceScalar s = fromResource(resource, stream);
    s.upload(host_val);
    return s;
  }

  DeviceScalar(DeviceScalar&& o) noexcept : d_val_(std::move(o.d_val_)), stream_(o.stream_) { o.stream_ = nullptr; }

  DeviceScalar& operator=(DeviceScalar&& o) noexcept {
    if (this != &o) {
      d_val_ = std::move(o.d_val_);
      stream_ = o.stream_;
      o.stream_ = nullptr;
    }
    return *this;
  }

  DeviceScalar(const DeviceScalar&) = delete;
  DeviceScalar& operator=(const DeviceScalar&) = delete;

  /** Read the value on the host, synchronizing first. */
  Scalar get() const {
    if (const Scalar* host = hostData()) {
      // The value is already in memory the host can address; ordering the read
      // against the device work that produced it is all that is left.
      internal::sync_for_host_access(memoryResource(), stream_);
      return *host;
    }
    Scalar result;
    EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(&result, d_val_.get(), sizeof(Scalar), cudaMemcpyDeviceToHost, stream_));
    EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream_));
    return result;
  }

  /** Implicit conversion, enabling `Scalar alpha = deviceScalar` and
   * `if (deviceScalar < threshold)`. Triggers a sync. */
  operator Scalar() const { return get(); }

  Scalar* devicePtr() { return static_cast<Scalar*>(d_val_.get()); }
  const Scalar* devicePtr() const { return static_cast<const Scalar*>(d_val_.get()); }

  /** Host-addressable alias of devicePtr(), or null for device-only storage.
   * Reading it requires the same synchronization get() performs. */
  Scalar* hostData() { return static_cast<Scalar*>(d_val_.hostData()); }
  const Scalar* hostData() const { return static_cast<const Scalar*>(d_val_.hostData()); }

  /** Whether hostData() is usable. */
  bool isHostAccessible() const { return d_val_.isHostAccessible(); }

  /** The resource backing this scalar, or null for the default allocator. */
  MemoryResource* memoryResource() const { return d_val_.memoryResource(); }

  cudaStream_t stream() const { return stream_; }

  // The arithmetic below keeps results on device via the NPP helpers in
  // DeviceScalarOps.h, and covers real types only; complex division falls back to
  // the implicit conversion and its host sync. Unlike DeviceMatrix, DeviceScalar
  // tracks no cross-stream readiness, so all operands must share one stream.

  // Each result is placed on the left operand's resource, so a chain of
  // device-side arithmetic stays exactly as host-readable as the reduction that
  // started it: with a device-only operand nothing changes.

  friend DeviceScalar operator/(const DeviceScalar& a, const DeviceScalar& b) {
    eigen_assert(a.stream_ == b.stream_ && "DeviceScalar operator/: operands must share the same stream");
    DeviceScalar result = fromResource(a.memoryResource(), a.stream_);
    gpu::internal::device_scalar_div(a.devicePtr(), b.devicePtr(), result.devicePtr(), a.stream_);
    return result;
  }

  friend DeviceScalar operator/(Scalar a, const DeviceScalar& b) {
    return fromResource(b.memoryResource(), a, b.stream_) / b;
  }

  friend DeviceScalar operator/(const DeviceScalar& a, Scalar b) {
    return a / fromResource(a.memoryResource(), b, a.stream_);
  }

  DeviceScalar operator-() const {
    DeviceScalar result = fromResource(memoryResource(), stream_);
    gpu::internal::device_scalar_neg(devicePtr(), result.devicePtr(), stream_);
    return result;
  }

 private:
  DeviceScalar(internal::DeviceBuffer&& storage, cudaStream_t stream) : d_val_(std::move(storage)), stream_(stream) {}

  // Always a stream-ordered copy, even into host-accessible storage: a caching
  // resource can hand back a block whose previous owner is still being written
  // on the device, and a plain host store would race with that write where the
  // copy queues behind it.
  void upload(Scalar host_val) {
    EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(d_val_.get(), &host_val, sizeof(Scalar), cudaMemcpyHostToDevice, stream_));
  }

  internal::DeviceBuffer d_val_;
  cudaStream_t stream_ = nullptr;
};

}  // namespace gpu
}  // namespace Eigen

#endif  // EIGEN_GPU_DEVICE_SCALAR_H
