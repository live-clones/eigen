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

#include "./FwdDecl.h"
#include "./GpuSupport.h"
#include "./DeviceScalarOps.h"

namespace Eigen {
namespace gpu {

/** \brief RAII wrapper for a scalar in GPU device memory.
 *
 * Accesses are ordered across streams exactly as for DeviceMatrix, with the same
 * prepareRead/finishRead/prepareWrite/finishWrite protocol for your own kernels.
 * Arithmetic between DeviceScalars stays on device (real types) and runs on the
 * first operand's stream().
 */
template <typename Scalar_>
class DeviceScalar {
 public:
  using Scalar = Scalar_;

  /** Allocate an uninitialized device scalar on the thread-local Context.
   * Contents are undefined until written, e.g. by cuBLAS dot/nrm2 under
   * POINTER_MODE_DEVICE. */
  DeviceScalar();

  /** Allocate an uninitialized device scalar on \p ctx. */
  explicit DeviceScalar(Context& ctx);

  /** Upload \p value on \p ctx. */
  DeviceScalar(Context& ctx, Scalar value);

  DeviceScalar(DeviceScalar&& o) noexcept = default;
  DeviceScalar& operator=(DeviceScalar&& o) noexcept = default;

  /** Deep copy: a device-to-device cudaMemcpyAsync on the source's stream(), no
   * host round trip. Provided so that generic code returning a DeviceScalar by
   * value from a const reference (numext::real in Eigen's iterative solver
   * templates) compiles; explicit code should move instead. */
  DeviceScalar(const DeviceScalar& o) : DeviceScalar(o.d_val_.streamHandle()) {
    EIGEN_CUDA_RUNTIME_CHECK(
        cudaMemcpyAsync(devicePtr(), o.devicePtr(), sizeof(Scalar), cudaMemcpyDeviceToDevice, stream()));
    d_val_.finishWrite(/*record_event=*/false);
  }

  /** Copy assignment adopts the source's stream: it copies into a fresh
   * allocation and releases the previous one. */
  DeviceScalar& operator=(const DeviceScalar& o) {
    if (this != &o) *this = DeviceScalar(o);
    return *this;
  }

  /** Download from device on stream(), blocking until the value is available. */
  Scalar get() const {
    Scalar result;
    EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(&result, devicePtr(), sizeof(Scalar), cudaMemcpyDeviceToHost, stream()));
    EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream()));
    return result;
  }

  /** Implicit conversion, enabling `Scalar alpha = deviceScalar` and
   * `if (deviceScalar < threshold)`. Triggers a sync. */
  operator Scalar() const { return get(); }

  Scalar* devicePtr() { return static_cast<Scalar*>(d_val_.get()); }
  const Scalar* devicePtr() const { return static_cast<const Scalar*>(d_val_.get()); }

  /** The stream of the last write; see DeviceMatrix::stream(). */
  cudaStream_t stream() const { return d_val_.stream(); }

  /** See DeviceMatrix::prepareRead(). */
  void prepareRead(Context& ctx) const;
  /** See DeviceMatrix::finishRead(). */
  void finishRead(Context& ctx) const;
  /** See DeviceMatrix::prepareWrite(). */
  void prepareWrite(Context& ctx);
  /** See DeviceMatrix::finishWrite(). A scalar defers the event to the first
   * read from another stream, which then also waits for later work on stream(). */
  void finishWrite(Context& ctx);

  // The arithmetic below keeps results on device via the NPP helpers in
  // DeviceScalarOps.h, and covers real types only; complex division falls back
  // to the implicit conversion and its host sync.

  friend DeviceScalar operator/(const DeviceScalar& a, const DeviceScalar& b) {
    const internal::StreamHandle& s = a.d_val_.streamHandle();
    DeviceScalar result(s);
    b.d_val_.prepareRead(s);
    gpu::internal::device_scalar_div(a.devicePtr(), b.devicePtr(), result.devicePtr(), s.get());
    b.d_val_.finishRead(s);
    result.d_val_.finishWrite(/*record_event=*/false);
    return result;
  }

  friend DeviceScalar operator/(Scalar a, const DeviceScalar& b) {
    DeviceScalar d_a(b.d_val_.streamHandle(), a);
    return d_a / b;
  }

  friend DeviceScalar operator/(const DeviceScalar& a, Scalar b) {
    DeviceScalar d_b(a.d_val_.streamHandle(), b);
    return a / d_b;
  }

  DeviceScalar operator-() const {
    DeviceScalar result(d_val_.streamHandle());
    gpu::internal::device_scalar_neg(devicePtr(), result.devicePtr(), stream());
    result.d_val_.finishWrite(/*record_event=*/false);
    return result;
  }

 private:
  explicit DeviceScalar(const internal::StreamHandle& stream) : d_val_(sizeof(Scalar), stream) {}

  // The host value is pageable, so the copy has been staged when the call returns.
  DeviceScalar(const internal::StreamHandle& stream, Scalar value) : DeviceScalar(stream) {
    EIGEN_CUDA_RUNTIME_CHECK(
        cudaMemcpyAsync(devicePtr(), &value, sizeof(Scalar), cudaMemcpyHostToDevice, stream.get()));
    d_val_.finishWrite(/*record_event=*/false);
  }

  internal::DeviceBuffer d_val_;
};

}  // namespace gpu
}  // namespace Eigen

#endif  // EIGEN_GPU_DEVICE_SCALAR_H
