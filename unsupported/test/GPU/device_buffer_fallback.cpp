// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// Integration tests for the cudaMalloc/cudaFree fallback allocator, forced on
// every device by defining EIGEN_GPU_NO_STREAM_ORDERED_ALLOC before the module
// headers. Unlike the unit test that drives FallbackDeviceBufferPool directly,
// these go through DeviceBuffer / DeviceBufferDeleter / DeviceScalar, so the
// threshold routing, thread-state gating, event fencing, and the plain-cudaFree
// large-buffer branch are all exercised even on pool-capable hardware.
#define EIGEN_GPU_NO_STREAM_ORDERED_ALLOC

#define EIGEN_USE_GPU
#include "main.h"
#include <unsupported/Eigen/GPU>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

using namespace Eigen;

namespace {

struct StreamGate {
  std::atomic<bool> open{false};
};

void CUDART_CB wait_for_stream_gate(void* user_data) {
  auto* gate = static_cast<StreamGate*>(user_data);
  while (!gate->open.load(std::memory_order_acquire)) std::this_thread::yield();
}

}  // namespace

// Small buffers route through the thread-local pool: free followed by a
// same-size allocate must return the cached pointer, via DeviceBuffer itself
// (not the pool driven directly). Runs first so the pool holds no entries of
// this size yet.
void test_small_buffer_routes_through_pool() {
  gpu::internal::CudaStreamHandle stream = gpu::internal::create_stream();
  constexpr size_t bytes = gpu::internal::FallbackDeviceBufferPool<>::kSmallBufferThreshold;
  void* first = nullptr;
  {
    gpu::internal::DeviceBuffer buf(bytes, stream);
    first = buf.get();
    VERIFY(first != nullptr);
  }
  for (int i = 0; i < 4; ++i) {
    gpu::internal::DeviceBuffer again(bytes, stream);
    VERIFY_IS_EQUAL(again.get(), first);
  }
}

// Buffers above the threshold take the plain cudaMalloc/cudaFree branch; check
// a full write/read round trip.
void test_large_buffer_roundtrip() {
  gpu::internal::CudaStreamHandle stream = gpu::internal::create_stream();
  const size_t n = 1 << 16;
  Eigen::VectorXf h_in = Eigen::VectorXf::LinSpaced(n, 0.0f, 1.0f), h_out(n);
  gpu::internal::DeviceBuffer buf(n * sizeof(float), stream);
  EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(buf.get(), h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream));
  EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(h_out.data(), buf.get(), n * sizeof(float), cudaMemcpyDeviceToHost, stream));
  EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream));
  VERIFY_IS_APPROX(h_out, h_in);
}

// Cross-stream reuse through DeviceBuffer: the deleter records an event on the
// producing stream, and an allocation on another stream must wait on it before
// touching the recycled pointer. The host-function gate stalls the producer so
// any missing fence would let the consumer's write overtake the producer's
// pending read.
void test_cross_stream_reuse_through_device_buffer() {
  gpu::internal::CudaStreamHandle producer = gpu::internal::create_stream();
  gpu::internal::CudaStreamHandle consumer = gpu::internal::create_stream();

  float* observed = nullptr;
  EIGEN_CUDA_RUNTIME_CHECK(cudaMallocHost(&observed, sizeof(float)));
  *observed = 0.0f;
  float* inputs = nullptr;
  EIGEN_CUDA_RUNTIME_CHECK(cudaMallocHost(&inputs, 2 * sizeof(float)));
  inputs[0] = 1.0f;
  inputs[1] = 2.0f;

  constexpr size_t bytes = sizeof(float);
  StreamGate gate;
  void* first_ptr = nullptr;
  {
    gpu::internal::DeviceBuffer buf(bytes, producer);
    first_ptr = buf.get();
    EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(buf.get(), &inputs[0], bytes, cudaMemcpyHostToDevice, producer));
    EIGEN_CUDA_RUNTIME_CHECK(cudaLaunchHostFunc(producer, wait_for_stream_gate, &gate));
    EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(observed, buf.get(), bytes, cudaMemcpyDeviceToHost, producer));
  }  // deleter pools the pointer with an event recorded on `producer`

  gpu::internal::DeviceBuffer reused(bytes, consumer);
  EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(reused.get(), &inputs[1], bytes, cudaMemcpyHostToDevice, consumer));

  std::thread release_gate([&gate] {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    gate.open.store(true, std::memory_order_release);
  });
  const cudaError_t consumer_status = cudaStreamSynchronize(consumer);
  const cudaError_t producer_status = cudaStreamSynchronize(producer);
  release_gate.join();

  const float observed_value = *observed;
  EIGEN_CUDA_RUNTIME_CHECK(cudaFreeHost(inputs));
  EIGEN_CUDA_RUNTIME_CHECK(cudaFreeHost(observed));

  VERIFY_IS_EQUAL(reused.get(), first_ptr);
  VERIFY_IS_EQUAL(observed_value, 1.0f);
  VERIFY_IS_EQUAL(consumer_status, cudaSuccess);
  VERIFY_IS_EQUAL(producer_status, cudaSuccess);
}

// Sequential DeviceScalar churn past kMaxPoolSize: exercises pooled reuse, the
// overflow (plain cudaFree) branch, and allocation from a full free list.
void test_device_scalar_churn_overflows_pool() {
  constexpr int kCount = 100;  // > FallbackDeviceBufferPool<>::kMaxPoolSize
  gpu::Context ctx;
  Eigen::VectorXf h_x = Eigen::VectorXf::LinSpaced(32, 1.0f, 32.0f);
  const float expected = h_x.squaredNorm();
  gpu::DeviceMatrix<float> d_x = gpu::DeviceMatrix<float>::fromHost(h_x, ctx.stream());
  {
    std::vector<gpu::DeviceScalar<float>> live;
    live.reserve(kCount);
    for (int i = 0; i < kCount; ++i) live.push_back(d_x.dot(ctx, d_x));
    for (auto& r : live) VERIFY_IS_APPROX(static_cast<float>(r), expected);
  }  // kCount deallocations: fills the pool, then overflows it
  VERIFY_IS_APPROX(static_cast<float>(d_x.dot(ctx, d_x)), expected);
}

// A DeviceScalar may outlive its Context on the fallback path too: the shared
// stream handle keeps the stream valid for the deleter's event record.
void test_device_scalar_outlives_context() {
  Eigen::VectorXf h_x = Eigen::VectorXf::LinSpaced(64, 1.0f, 64.0f);
  gpu::DeviceScalar<float> result;
  {
    gpu::Context ctx;
    gpu::DeviceMatrix<float> d_x = gpu::DeviceMatrix<float>::fromHost(h_x, ctx.stream());
    result = d_x.dot(ctx, d_x);
  }
  VERIFY_IS_APPROX(static_cast<float>(result), h_x.squaredNorm());
}

// ensure_sized drains the stream before replacing a live buffer on the
// fallback path, so the plain cudaFree cannot overtake queued work.
void test_ensure_sized_growth_drains() {
  gpu::internal::CudaStreamHandle stream = gpu::internal::create_stream();
  const size_t small = 1 << 20, large = 1 << 21;
  gpu::internal::DeviceBuffer buf(small, stream);
  EIGEN_CUDA_RUNTIME_CHECK(cudaMemsetAsync(buf.get(), 1, buf.size(), stream));
  gpu::internal::ensure_sized(buf, large, stream);
  VERIFY(buf.size() >= large);
  EIGEN_CUDA_RUNTIME_CHECK(cudaMemsetAsync(buf.get(), 2, buf.size(), stream));
  EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream));
}

// End-to-end dense solve under the fallback allocator, including the scratch
// and host-workspace growth paths (factor a larger matrix second).
void test_llt_solve_fallback() {
  for (Eigen::Index n : {16, 48}) {
    Eigen::MatrixXf M = Eigen::MatrixXf::Random(n, n);
    Eigen::MatrixXf A = M.transpose() * M + Eigen::MatrixXf::Identity(n, n) * static_cast<float>(n);
    Eigen::MatrixXf B = Eigen::MatrixXf::Random(n, 2);

    gpu::LLT<float> gpu_llt(A);
    VERIFY_IS_EQUAL(gpu_llt.info(), Eigen::Success);
    Eigen::MatrixXf X_gpu = gpu_llt.solve(B);
    Eigen::MatrixXf X_cpu = Eigen::LLT<Eigen::MatrixXf>(A).solve(B);
    const float tol = 8.0f * static_cast<float>(n) * Eigen::NumTraits<float>::epsilon() * A.norm();
    VERIFY((X_gpu - X_cpu).norm() < tol);
  }
}

EIGEN_DECLARE_TEST(gpu_device_buffer_fallback) {
  CALL_SUBTEST(test_small_buffer_routes_through_pool());
  CALL_SUBTEST(test_large_buffer_roundtrip());
  CALL_SUBTEST(test_cross_stream_reuse_through_device_buffer());
  CALL_SUBTEST(test_device_scalar_churn_overflows_pool());
  CALL_SUBTEST(test_device_scalar_outlives_context());
  CALL_SUBTEST(test_ensure_sized_growth_drains());
  CALL_SUBTEST(test_llt_solve_fallback());
}
