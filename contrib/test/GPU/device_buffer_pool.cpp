// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Eigen Authors
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// Tests for internal::DeviceBufferPool: a released block is recycled only once
// the device has retired the work enqueued on its home stream before the
// release. The pool is driven directly, since DeviceBuffer only routes through
// it on devices without memory pools.

#define EIGEN_USE_GPU
#include "main.h"
#include <contrib/Eigen/GPU>
#include <atomic>
#include <chrono>
#include <thread>

#include "./gpu_test_helpers.h"

using namespace Eigen;

namespace {

using Pool = gpu::internal::DeviceBufferPool<>;
constexpr size_t kBytes = Pool::kSmallBufferThreshold / 4;

void spin_until(const std::atomic<bool>& flag) {
  while (!flag.load(std::memory_order_acquire)) std::this_thread::yield();
}

// A host function parked on a stream keeps everything enqueued after it there,
// including a release event the pool records, pending until the gate opens.
struct StreamGate {
  std::atomic<bool> entered{false};
  std::atomic<bool> release{false};
};

void CUDART_CB wait_for_gate(void* data) {
  StreamGate* gate = static_cast<StreamGate*>(data);
  gate->entered.store(true, std::memory_order_release);
  spin_until(gate->release);
}

// An idle device recycles a released block on the next allocation that fits.
void test_idle_reuse() {
  Pool& pool = Pool::threadLocal();
  const gpu::internal::StreamHandle ctx_stream = gpu::internal::make_owned_stream();
  cudaStream_t stream = ctx_stream.get();
  void* const p = pool.allocate(kBytes, stream);
  VERIFY(p != nullptr);
  pool.deallocate(p, kBytes, stream);
  EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream));
  void* const q = pool.allocate(kBytes, stream);
  VERIFY_IS_EQUAL(q, p);
  pool.deallocate(q, kBytes, stream);
  EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream));
}

// A block released while its home stream is still busy stays out of
// circulation until that stream drains; afterwards it is recycled again.
void test_reuse_waits_for_in_flight_work() {
  Pool& pool = Pool::threadLocal();
  EIGEN_CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
  const gpu::internal::StreamHandle owned_stream = gpu::internal::make_owned_stream();
  cudaStream_t stream = owned_stream.get();

  void* const p = pool.allocate(kBytes, stream);

  StreamGate gate;
  EIGEN_CUDA_RUNTIME_CHECK(cudaLaunchHostFunc(stream, wait_for_gate, &gate));
  std::thread release_thread([&gate]() {
    spin_until(gate.entered);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    gate.release.store(true, std::memory_order_release);
  });

  // Released while `stream` is parked: the release event cannot have completed.
  pool.deallocate(p, kBytes, stream);
  void* const q = pool.allocate(kBytes, stream);
  VERIFY(q != p);

  release_thread.join();
  EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream));

  // Both blocks are free and retired; the older release is recycled first.
  pool.deallocate(q, kBytes, stream);
  void* const r = pool.allocate(kBytes, stream);
  VERIFY_IS_EQUAL(r, p);
  pool.deallocate(r, kBytes, stream);
  EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream));
}

// Release events on different streams retire out of order: a block released on
// a parked stream must not hide a later-released block whose stream is idle.
void test_reuse_skips_pending_release() {
  Pool& pool = Pool::threadLocal();
  EIGEN_CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
  const gpu::internal::StreamHandle parked_stream = gpu::internal::make_owned_stream();
  const gpu::internal::StreamHandle idle_stream = gpu::internal::make_owned_stream();
  cudaStream_t parked = parked_stream.get();
  cudaStream_t idle = idle_stream.get();

  void* const p = pool.allocate(kBytes, parked);
  void* const q = pool.allocate(kBytes, idle);

  StreamGate gate;
  EIGEN_CUDA_RUNTIME_CHECK(cudaLaunchHostFunc(parked, wait_for_gate, &gate));
  pool.deallocate(p, kBytes, parked);
  pool.deallocate(q, kBytes, idle);
  EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(idle));

  // p's release is older but still pending; q's has retired.
  void* const r = pool.allocate(kBytes, idle);
  VERIFY_IS_EQUAL(r, q);

  gate.release.store(true, std::memory_order_release);
  EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(parked));
  pool.deallocate(r, kBytes, idle);
  EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(idle));
}

}  // namespace

EIGEN_DECLARE_TEST(gpu_device_buffer_pool) {
  gpu_test::require_cuda_device();
  CALL_SUBTEST_1(test_idle_reuse());
  CALL_SUBTEST_1(test_reuse_waits_for_in_flight_work());
  CALL_SUBTEST_1(test_reuse_skips_pending_release());
}
