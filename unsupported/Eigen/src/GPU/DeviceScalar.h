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
   * written, e.g. by cuBLAS dot/nrm2 under POINTER_MODE_DEVICE. The raw-stream
   * overloads borrow: the stream must stay valid for this scalar's lifetime.
   * The handle overloads (used by the Context-driven reductions) share stream
   * ownership, so the scalar may outlive the Context that produced it. */
  explicit DeviceScalar(cudaStream_t stream = nullptr) : DeviceScalar(internal::borrow_stream(stream)) {}

  DeviceScalar(Scalar host_val, cudaStream_t stream) : DeviceScalar(host_val, internal::borrow_stream(stream)) {}

  explicit DeviceScalar(internal::CudaStreamHandle stream) : d_val_(sizeof(Scalar), std::move(stream)) {}

  DeviceScalar(Scalar host_val, internal::CudaStreamHandle stream) : d_val_(sizeof(Scalar), std::move(stream)) {
    EIGEN_CUDA_RUNTIME_CHECK(
        cudaMemcpyAsync(d_val_.get(), &host_val, sizeof(Scalar), cudaMemcpyHostToDevice, streamHandle()));
  }

  DeviceScalar(DeviceScalar&& o) noexcept = default;
  DeviceScalar& operator=(DeviceScalar&& o) noexcept = default;

  DeviceScalar(const DeviceScalar&) = delete;
  DeviceScalar& operator=(const DeviceScalar&) = delete;

  /** Download from device, synchronizing the stream. */
  Scalar get() const {
    Scalar result;
    EIGEN_CUDA_RUNTIME_CHECK(
        cudaMemcpyAsync(&result, d_val_.get(), sizeof(Scalar), cudaMemcpyDeviceToHost, streamHandle()));
    EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(streamHandle()));
    return result;
  }

  /** Implicit conversion, enabling `Scalar alpha = deviceScalar` and
   * `if (deviceScalar < threshold)`. Triggers a sync. */
  operator Scalar() const { return get(); }

  Scalar* devicePtr() { return static_cast<Scalar*>(d_val_.get()); }
  const Scalar* devicePtr() const { return static_cast<const Scalar*>(d_val_.get()); }
  cudaStream_t stream() const { return streamHandle().get(); }

  /** The stream this scalar is bound to. Held by the allocation itself, so a
   * scalar that shares stream ownership keeps an owned stream alive. */
  const internal::CudaStreamHandle& streamHandle() const { return d_val_.streamHandle(); }

  // The arithmetic below keeps results on device via the NPP helpers in
  // DeviceScalarOps.h, and covers real types only; complex division falls back to
  // the implicit conversion and its host sync. Unlike DeviceMatrix, DeviceScalar
  // tracks no cross-stream readiness, so all operands must share one stream.

  friend DeviceScalar operator/(const DeviceScalar& a, const DeviceScalar& b) {
    eigen_assert(a.streamHandle() == b.streamHandle() && "DeviceScalar operator/: operands must share the same stream");
    DeviceScalar result(a.streamHandle());
    gpu::internal::device_scalar_div(a.devicePtr(), b.devicePtr(), result.devicePtr(), a.streamHandle());
    return result;
  }

  friend DeviceScalar operator/(Scalar a, const DeviceScalar& b) {
    DeviceScalar d_a(a, b.streamHandle());
    return d_a / b;
  }

  friend DeviceScalar operator/(const DeviceScalar& a, Scalar b) {
    DeviceScalar d_b(b, a.streamHandle());
    return a / d_b;
  }

  DeviceScalar operator-() const {
    DeviceScalar result(streamHandle());
    gpu::internal::device_scalar_neg(devicePtr(), result.devicePtr(), streamHandle());
    return result;
  }

 private:
  internal::DeviceBuffer d_val_;
};

}  // namespace gpu
}  // namespace Eigen

#endif  // EIGEN_GPU_DEVICE_SCALAR_H
