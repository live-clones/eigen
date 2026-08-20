// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// Tests for gpu::DeviceMatrix and HostTransfer: typed RAII GPU memory wrapper.
// No cuSOLVER dependency — only CUDA runtime.

#define EIGEN_USE_GPU
#include "main.h"
#include <Eigen/Sparse>
#include <unsupported/Eigen/GPU>
#include "gpu_test_helpers.h"

#include <chrono>
#include <thread>

using namespace Eigen;
using gpu_test::StreamGate;
using gpu_test::wait_for_stream_gate;

// ---- Default construction ---------------------------------------------------

void test_default_construct() {
  gpu::DeviceMatrix<double> dm;
  VERIFY(dm.empty());
  VERIFY_IS_EQUAL(dm.rows(), 0);
  VERIFY_IS_EQUAL(dm.cols(), 0);
  VERIFY(dm.data() == nullptr);
  VERIFY_IS_EQUAL(dm.sizeInBytes(), size_t(0));
}

// ---- Allocate uninitialized -------------------------------------------------

template <typename Scalar>
void test_allocate(Index rows, Index cols) {
  gpu::DeviceMatrix<Scalar> dm(rows, cols);
  VERIFY(!dm.empty());
  VERIFY_IS_EQUAL(dm.rows(), rows);
  VERIFY_IS_EQUAL(dm.cols(), cols);
  VERIFY(dm.data() != nullptr);
  VERIFY_IS_EQUAL(dm.sizeInBytes(), size_t(rows) * size_t(cols) * sizeof(Scalar));
}

// ---- fromHost / toHost roundtrip (synchronous) ------------------------------

template <typename Scalar>
void test_roundtrip(Index rows, Index cols) {
  using MatrixType = Eigen::Matrix<Scalar, Dynamic, Dynamic>;
  MatrixType host = MatrixType::Random(rows, cols);

  auto dm = gpu::DeviceMatrix<Scalar>::fromHost(host);
  VERIFY_IS_EQUAL(dm.rows(), rows);
  VERIFY_IS_EQUAL(dm.cols(), cols);
  VERIFY(!dm.empty());

  MatrixType result = dm.toHost();
  VERIFY_IS_EQUAL(result.rows(), rows);
  VERIFY_IS_EQUAL(result.cols(), cols);
  VERIFY_IS_APPROX(result, host);
}

// ---- fromHostAsync / toHostAsync roundtrip -----------------------------------

template <typename Scalar>
void test_roundtrip_async(Index rows, Index cols) {
  using MatrixType = Eigen::Matrix<Scalar, Dynamic, Dynamic>;
  MatrixType host = MatrixType::Random(rows, cols);

  cudaStream_t stream;
  EIGEN_CUDA_RUNTIME_CHECK(cudaStreamCreate(&stream));

  // Async upload from raw pointer.
  auto dm = gpu::DeviceMatrix<Scalar>::fromHostAsync(host.data(), rows, cols, stream);
  VERIFY_IS_EQUAL(dm.rows(), rows);
  VERIFY_IS_EQUAL(dm.cols(), cols);

  // Async download via HostTransfer future.
  auto transfer = dm.toHostAsync(stream);

  // get() blocks and returns the matrix.
  MatrixType result = transfer.get();
  VERIFY_IS_APPROX(result, host);

  EIGEN_CUDA_RUNTIME_CHECK(cudaStreamDestroy(stream));

  cudaStream_t producer_stream;
  cudaStream_t consumer_stream;
  EIGEN_CUDA_RUNTIME_CHECK(cudaStreamCreate(&producer_stream));
  EIGEN_CUDA_RUNTIME_CHECK(cudaStreamCreate(&consumer_stream));

  auto cross_stream_dm = gpu::DeviceMatrix<Scalar>::fromHostAsync(host.data(), rows, cols, producer_stream);
  auto cross_stream_transfer = cross_stream_dm.toHostAsync(consumer_stream);
  MatrixType cross_stream_result = cross_stream_transfer.get();
  VERIFY_IS_APPROX(cross_stream_result, host);

  EIGEN_CUDA_RUNTIME_CHECK(cudaStreamDestroy(consumer_stream));
  EIGEN_CUDA_RUNTIME_CHECK(cudaStreamDestroy(producer_stream));
}

// ---- HostTransfer::ready() and idempotent get() -----------------------------

void test_host_transfer_ready() {
  using MatrixType = Eigen::Matrix<double, Dynamic, Dynamic>;
  MatrixType host = MatrixType::Random(100, 100);

  auto dm = gpu::DeviceMatrix<double>::fromHost(host);
  auto transfer = dm.toHostAsync();

  // After get(), ready() must return true.
  MatrixType result = transfer.get();
  VERIFY(transfer.ready());
  VERIFY_IS_APPROX(result, host);

  // get() is idempotent.
  MatrixType& result2 = transfer.get();
  VERIFY_IS_APPROX(result2, host);
}

// ---- HostTransfer move ------------------------------------------------------

void test_host_transfer_move() {
  using MatrixType = Eigen::Matrix<double, Dynamic, Dynamic>;
  MatrixType host = MatrixType::Random(50, 50);

  auto dm = gpu::DeviceMatrix<double>::fromHost(host);
  auto transfer = dm.toHostAsync();

  gpu::HostTransfer<double> moved(std::move(transfer));
  MatrixType result = moved.get();
  VERIFY_IS_APPROX(result, host);
}

// ---- clone() produces independent copy --------------------------------------

template <typename Scalar>
void test_clone(Index rows, Index cols) {
  using MatrixType = Eigen::Matrix<Scalar, Dynamic, Dynamic>;
  MatrixType host = MatrixType::Random(rows, cols);

  auto dm = gpu::DeviceMatrix<Scalar>::fromHost(host);
  auto cloned = dm.clone();

  // Overwrite original with different data.
  MatrixType other = MatrixType::Random(rows, cols);
  dm = gpu::DeviceMatrix<Scalar>::fromHost(other);

  // Clone still holds the original data.
  MatrixType clone_result = cloned.toHost();
  VERIFY_IS_APPROX(clone_result, host);

  // Original holds the new data.
  MatrixType dm_result = dm.toHost();
  VERIFY_IS_APPROX(dm_result, other);
}

// ---- Move construct ---------------------------------------------------------

template <typename Scalar>
void test_move_construct(Index rows, Index cols) {
  using MatrixType = Eigen::Matrix<Scalar, Dynamic, Dynamic>;
  MatrixType host = MatrixType::Random(rows, cols);

  auto dm = gpu::DeviceMatrix<Scalar>::fromHost(host);
  gpu::DeviceMatrix<Scalar> moved(std::move(dm));

  VERIFY(dm.empty());
  VERIFY(dm.data() == nullptr);

  VERIFY_IS_EQUAL(moved.rows(), rows);
  VERIFY_IS_EQUAL(moved.cols(), cols);
  MatrixType result = moved.toHost();
  VERIFY_IS_APPROX(result, host);
}

// ---- Move assign ------------------------------------------------------------

template <typename Scalar>
void test_move_assign(Index rows, Index cols) {
  using MatrixType = Eigen::Matrix<Scalar, Dynamic, Dynamic>;
  MatrixType host = MatrixType::Random(rows, cols);

  auto dm = gpu::DeviceMatrix<Scalar>::fromHost(host);
  gpu::DeviceMatrix<Scalar> dest;
  dest = std::move(dm);

  VERIFY(dm.empty());
  VERIFY_IS_EQUAL(dest.rows(), rows);
  MatrixType result = dest.toHost();
  VERIFY_IS_APPROX(result, host);
}

// ---- resize() ---------------------------------------------------------------

void test_resize() {
  gpu::DeviceMatrix<double> dm(10, 20);
  VERIFY_IS_EQUAL(dm.rows(), 10);
  VERIFY_IS_EQUAL(dm.cols(), 20);

  dm.resize(50, 30);
  VERIFY_IS_EQUAL(dm.rows(), 50);
  VERIFY_IS_EQUAL(dm.cols(), 30);
  VERIFY(dm.data() != nullptr);

  // Resize to same dimensions is a no-op.
  double* ptr_before = dm.data();
  dm.resize(50, 30);
  VERIFY(dm.data() == ptr_before);
}

// ---- Empty / 0x0 matrix -----------------------------------------------------

void test_empty() {
  using MatrixType = Eigen::Matrix<double, Dynamic, Dynamic>;
  MatrixType empty_mat(0, 0);

  auto dm = gpu::DeviceMatrix<double>::fromHost(empty_mat);
  VERIFY(dm.empty());
  VERIFY_IS_EQUAL(dm.rows(), 0);
  VERIFY_IS_EQUAL(dm.cols(), 0);

  MatrixType result = dm.toHost();
  VERIFY_IS_EQUAL(result.rows(), 0);
  VERIFY_IS_EQUAL(result.cols(), 0);
}

// ---- DeviceBuffer stream ordering -----------------------------------------

void test_device_buffer_uses_owner_stream() {
  if (!gpu::internal::device_supports_memory_pools()) return;

  cudaStream_t allocation_stream = nullptr;
  EIGEN_CUDA_RUNTIME_CHECK(cudaStreamCreate(&allocation_stream));

  cudaGraph_t graph = nullptr;
  EIGEN_CUDA_RUNTIME_CHECK(cudaStreamBeginCapture(allocation_stream, cudaStreamCaptureModeThreadLocal));
  {
    gpu::internal::DeviceBuffer buffer(1024, gpu::internal::borrow_stream(allocation_stream));
    EIGEN_CUDA_RUNTIME_CHECK(cudaMemsetAsync(buffer.get(), 0, buffer.size(), allocation_stream));
  }
  EIGEN_CUDA_RUNTIME_CHECK(cudaStreamEndCapture(allocation_stream, &graph));

  cudaGraphExec_t graph_exec = nullptr;
  EIGEN_CUDA_RUNTIME_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  EIGEN_CUDA_RUNTIME_CHECK(cudaGraphLaunch(graph_exec, allocation_stream));
  EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(allocation_stream));

  EIGEN_CUDA_RUNTIME_CHECK(cudaGraphExecDestroy(graph_exec));
  EIGEN_CUDA_RUNTIME_CHECK(cudaGraphDestroy(graph));
  EIGEN_CUDA_RUNTIME_CHECK(cudaStreamDestroy(allocation_stream));
}

void test_fallback_pool_cross_stream_reuse() {
  using Pool = gpu::internal::FallbackDeviceBufferPool<64, 4>;
  Pool& pool = Pool::threadLocal();

  cudaStream_t producer_stream = nullptr;
  cudaStream_t consumer_stream = nullptr;
  EIGEN_CUDA_RUNTIME_CHECK(cudaStreamCreate(&producer_stream));
  EIGEN_CUDA_RUNTIME_CHECK(cudaStreamCreate(&consumer_stream));

  float* observed = nullptr;
  EIGEN_CUDA_RUNTIME_CHECK(cudaMallocHost(&observed, sizeof(float)));
  *observed = 0.0f;

  float* inputs = nullptr;
  EIGEN_CUDA_RUNTIME_CHECK(cudaMallocHost(&inputs, 2 * sizeof(float)));
  inputs[0] = 1.0f;
  inputs[1] = 2.0f;

  constexpr size_t bytes = sizeof(float);
  void* first_ptr = pool.allocate(bytes, producer_stream);
  EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(first_ptr, &inputs[0], bytes, cudaMemcpyHostToDevice, producer_stream));

  StreamGate gate;
  EIGEN_CUDA_RUNTIME_CHECK(cudaLaunchHostFunc(producer_stream, wait_for_stream_gate, &gate));
  EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(observed, first_ptr, bytes, cudaMemcpyDeviceToHost, producer_stream));
  pool.deallocate(first_ptr, bytes, producer_stream);

  void* reused_ptr = pool.allocate(bytes, consumer_stream);
  EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(reused_ptr, &inputs[1], bytes, cudaMemcpyHostToDevice, consumer_stream));

  cudaError_t consumer_status = cudaSuccess;
  cudaError_t producer_status = cudaSuccess;
  {
    gpu_test::StreamGateOpener opener(gate, /*delay_ms=*/50);
    consumer_status = cudaStreamSynchronize(consumer_stream);
    producer_status = cudaStreamSynchronize(producer_stream);
  }

  const float observed_value = *observed;
  const cudaError_t free_status = cudaFree(reused_ptr);
  const cudaError_t inputs_free_status = cudaFreeHost(inputs);
  const cudaError_t host_free_status = cudaFreeHost(observed);
  const cudaError_t consumer_destroy_status = cudaStreamDestroy(consumer_stream);
  const cudaError_t producer_destroy_status = cudaStreamDestroy(producer_stream);

  VERIFY(first_ptr == reused_ptr);
  VERIFY_IS_EQUAL(observed_value, 1.0f);
  VERIFY_IS_EQUAL(consumer_status, cudaSuccess);
  VERIFY_IS_EQUAL(producer_status, cudaSuccess);
  VERIFY_IS_EQUAL(free_status, cudaSuccess);
  VERIFY_IS_EQUAL(inputs_free_status, cudaSuccess);
  VERIFY_IS_EQUAL(host_free_status, cudaSuccess);
  VERIFY_IS_EQUAL(consumer_destroy_status, cudaSuccess);
  VERIFY_IS_EQUAL(producer_destroy_status, cudaSuccess);
}

// ---- Cross-stream free ordering (two-stream immediate release) --------------

// Regression: the stream-aware deleter enqueues cudaFreeAsync on the
// allocation's bound stream S. A consumer on stream T that read the matrix via
// waitReady(T) must complete before that free executes, even when the matrix
// is released immediately after the read is enqueued — by destruction
// (mode 0), a reallocating resize (mode 1), or a move-overwrite (mode 2). The
// consumer stream is held closed by a host-function gate while the storage is
// released and a same-size allocation on S is clobbered; without the reverse
// S-waits-for-T dependency, the pool may recycle the block and the clobber
// lands under the still-pending read.
void test_cross_stream_free_ordering(int teardown_mode) {
  if (!gpu::internal::device_supports_memory_pools()) return;

  const Index n = 4096;
  const size_t bytes = static_cast<size_t>(n) * sizeof(float);

  Eigen::MatrixXf h_src = Eigen::MatrixXf::Random(n, 1);

  gpu::internal::CudaStreamHandle producer = gpu::internal::create_stream();
  cudaStream_t consumer = nullptr;
  EIGEN_CUDA_RUNTIME_CHECK(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking));

  StreamGate gate;
  gpu::HostTransfer<float> transfer = [&]() {
    gpu::internal::DeviceBuffer buf(bytes, producer);
    EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(buf.get(), h_src.data(), bytes, cudaMemcpyHostToDevice, producer));
    auto dm = gpu::DeviceMatrix<float>::adopt(static_cast<float*>(buf.release()), n, 1, producer);
    dm.recordReady(producer);

    // Hold the consumer stream closed, then enqueue the cross-stream read.
    EIGEN_CUDA_RUNTIME_CHECK(cudaLaunchHostFunc(consumer, wait_for_stream_gate, &gate));
    gpu::HostTransfer<float> pending = dm.toHostAsync(consumer);

    // Release the storage while the read is still gated.
    if (teardown_mode == 1) {
      dm.resize(2 * n, 2);  // exceeds capacity: frees and reallocates
    } else if (teardown_mode == 2) {
      dm = gpu::DeviceMatrix<float>(4, 4);  // move-overwrite frees the old block
    }
    return pending;  // mode 0: dm goes out of scope here (destruction)
  }();

  // Same-size allocation on S: once the free above is stream-ordered on S, the
  // pool may recycle the block, so this memset would corrupt the gated read if
  // the free were not ordered after it.
  gpu::internal::DeviceBuffer clobber(bytes, producer);
  EIGEN_CUDA_RUNTIME_CHECK(cudaMemsetAsync(clobber.get(), 0xFF, bytes, producer));

  {
    gpu_test::StreamGateOpener opener(gate, /*delay_ms=*/50);
    Eigen::MatrixXf result = transfer.get();
    VERIFY_IS_EQUAL(result, h_src);  // exact: pure copies, no arithmetic
  }

  clobber = gpu::internal::DeviceBuffer();
  EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(consumer));
  EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(producer.get()));
  EIGEN_CUDA_RUNTIME_CHECK(cudaStreamDestroy(consumer));
}

// ---- Stream lifetime -------------------------------------------------------

// A DeviceScalar produced through a Context may outlive it: the shared stream
// handle in its buffer deleter keeps the (owned) stream alive until the free
// is enqueued, and get() can still synchronize on it.
void test_device_scalar_outlives_context() {
  Eigen::VectorXf h_x = Eigen::VectorXf::LinSpaced(64, 1.0f, 64.0f);
  gpu::DeviceScalar<float> result;
  {
    gpu::Context ctx;
    gpu::DeviceMatrix<float> d_x = gpu::DeviceMatrix<float>::fromHost(h_x, ctx.stream());
    result = d_x.dot(ctx, d_x);
  }
  // Context (and its owned stream's primary reference) is gone; the read must
  // still work and the buffer free must target a live stream.
  VERIFY_IS_APPROX(static_cast<float>(result), h_x.squaredNorm());
}

// A DeviceBuffer bound to an owned stream keeps that stream alive after the
// last CudaStreamHandle owner is destroyed; the stream-ordered free then still
// has a valid target.
void test_device_buffer_outlives_stream_handle() {
  gpu::internal::DeviceBuffer buffer;
  {
    gpu::internal::CudaStreamHandle stream = gpu::internal::create_stream();
    buffer = gpu::internal::DeviceBuffer(1024, stream);
    EIGEN_CUDA_RUNTIME_CHECK(cudaMemsetAsync(buffer.get(), 0, buffer.size(), stream));
  }
  buffer = gpu::internal::DeviceBuffer();  // free enqueued on the still-live stream
  EIGEN_CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
}

// ---- Per-scalar driver ------------------------------------------------------

template <typename Scalar>
void test_scalar() {
  // Square.
  CALL_SUBTEST(test_roundtrip<Scalar>(1, 1));
  CALL_SUBTEST(test_roundtrip<Scalar>(64, 64));
  CALL_SUBTEST(test_roundtrip<Scalar>(256, 256));

  // Rectangular.
  CALL_SUBTEST(test_roundtrip<Scalar>(100, 7));
  CALL_SUBTEST(test_roundtrip<Scalar>(7, 100));

  // Async roundtrip.
  CALL_SUBTEST(test_roundtrip_async<Scalar>(64, 64));
  CALL_SUBTEST(test_roundtrip_async<Scalar>(100, 7));

  CALL_SUBTEST(test_clone<Scalar>(64, 64));
  CALL_SUBTEST(test_move_construct<Scalar>(64, 64));
  CALL_SUBTEST(test_move_assign<Scalar>(64, 64));
}

// ---- BLAS-1: dot product ----------------------------------------------------

template <typename Scalar>
void test_blas1(Index n) {
  using Vec = Matrix<Scalar, Dynamic, 1>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  // All BLAS-1 ops share one gpu::Context — same stream, zero event overhead.
  gpu::Context ctx;

  RealScalar tol = RealScalar(10) * RealScalar(n) * NumTraits<Scalar>::epsilon();

  // dot
  {
    Vec a = Vec::Random(n);
    Vec b = Vec::Random(n);
    auto d_a = gpu::DeviceMatrix<Scalar>::fromHost(a, ctx.stream());
    auto d_b = gpu::DeviceMatrix<Scalar>::fromHost(b, ctx.stream());
    Scalar gpu_dot = d_a.dot(ctx, d_b);
    Scalar cpu_dot = a.dot(b);
    VERIFY(numext::abs(gpu_dot - cpu_dot) < tol * numext::abs(cpu_dot) + tol);
  }

  // norm / squaredNorm
  {
    Vec a = Vec::Random(n);
    auto d_a = gpu::DeviceMatrix<Scalar>::fromHost(a, ctx.stream());
    RealScalar gpu_norm = d_a.norm(ctx);
    RealScalar cpu_norm = a.norm();
    VERIFY(numext::abs(gpu_norm - cpu_norm) < tol * cpu_norm + tol);
    RealScalar gpu_sqnorm = d_a.squaredNorm(ctx);
    RealScalar cpu_sqnorm = a.squaredNorm();
    VERIFY(numext::abs(gpu_sqnorm - cpu_sqnorm) < tol * cpu_sqnorm + tol);
  }

  // addScaled (axpy)
  {
    Vec x = Vec::Random(n);
    Vec y = Vec::Random(n);
    Scalar alpha(2.5);
    Vec y_ref = y + alpha * x;
    auto d_y = gpu::DeviceMatrix<Scalar>::fromHost(y, ctx.stream());
    auto d_x = gpu::DeviceMatrix<Scalar>::fromHost(x, ctx.stream());
    d_y.addScaled(ctx, alpha, d_x);
    Vec y_gpu = d_y.toHost(ctx.stream());
    VERIFY((y_gpu - y_ref).norm() < tol * y_ref.norm() + tol);
  }

  // scale (scal)
  {
    Vec x = Vec::Random(n);
    Scalar alpha(3.0);
    Vec x_ref = alpha * x;
    auto d_x = gpu::DeviceMatrix<Scalar>::fromHost(x, ctx.stream());
    d_x.scale(ctx, alpha);
    Vec x_gpu = d_x.toHost(ctx.stream());
    VERIFY((x_gpu - x_ref).norm() < tol * x_ref.norm() + tol);
  }

  // copyFrom
  {
    Vec x = Vec::Random(n);
    auto d_x = gpu::DeviceMatrix<Scalar>::fromHost(x, ctx.stream());
    gpu::DeviceMatrix<Scalar> d_y;
    d_y.copyFrom(ctx, d_x);
    Vec y = d_y.toHost(ctx.stream());
    VERIFY_IS_APPROX(y, x);
  }

  // setZero
  {
    Vec x = Vec::Random(n);
    auto d_x = gpu::DeviceMatrix<Scalar>::fromHost(x, ctx.stream());
    d_x.setZero(ctx);
    Vec result = d_x.toHost(ctx.stream());
    VERIFY_IS_EQUAL(result, Vec::Zero(n));
  }
}

// ---- BLAS-1 operator overloads (CG-style) -----------------------------------

template <typename Scalar>
void test_cg_operators(Index n) {
  using Vec = Matrix<Scalar, Dynamic, 1>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  RealScalar tol = RealScalar(10) * RealScalar(n) * NumTraits<Scalar>::epsilon();

  Vec x = Vec::Random(n);
  Vec p = Vec::Random(n);
  Vec tmp = Vec::Random(n);
  Vec z = Vec::Random(n);
  Scalar alpha(2.5);
  Scalar beta(0.7);

  // Test: x += alpha * p
  {
    Vec x_ref = x + alpha * p;
    auto d_x = gpu::DeviceMatrix<Scalar>::fromHost(x);
    auto d_p = gpu::DeviceMatrix<Scalar>::fromHost(p);
    d_x += alpha * d_p;
    Vec x_gpu = d_x.toHost();
    VERIFY((x_gpu - x_ref).norm() < tol * x_ref.norm() + tol);
  }

  // Test: r -= alpha * tmp
  {
    Vec r = Vec::Random(n);
    Vec r_ref = r - alpha * tmp;
    auto d_r = gpu::DeviceMatrix<Scalar>::fromHost(r);
    auto d_tmp = gpu::DeviceMatrix<Scalar>::fromHost(tmp);
    d_r -= alpha * d_tmp;
    Vec r_gpu = d_r.toHost();
    VERIFY((r_gpu - r_ref).norm() < tol * r_ref.norm() + tol);
  }

  // Test: p = z + beta * p  (cuBLAS geam)
  {
    Vec p_copy = p;
    Vec p_ref = z + beta * p_copy;
    auto d_p = gpu::DeviceMatrix<Scalar>::fromHost(p_copy);
    auto d_z = gpu::DeviceMatrix<Scalar>::fromHost(z);
    d_p = d_z + beta * d_p;
    Vec p_gpu = d_p.toHost();
    VERIFY((p_gpu - p_ref).norm() < tol * p_ref.norm() + tol);
  }

  // Test: operator+= and operator-= with gpu::DeviceMatrix (no scalar)
  {
    Vec a = Vec::Random(n);
    Vec b = Vec::Random(n);
    Vec a_ref = a + b;
    auto d_a = gpu::DeviceMatrix<Scalar>::fromHost(a);
    auto d_b = gpu::DeviceMatrix<Scalar>::fromHost(b);
    d_a += d_b;
    VERIFY((d_a.toHost() - a_ref).norm() < tol * a_ref.norm() + tol);
  }
}

// ---- gpu::DeviceScalar: deferred sync -------------------------------------------

template <typename Scalar>
void test_device_scalar() {
  using Vec = Matrix<Scalar, Dynamic, 1>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  const Index n = 256;
  Vec a = Vec::Random(n);
  Vec b = Vec::Random(n);

  gpu::Context ctx;
  auto d_a = gpu::DeviceMatrix<Scalar>::fromHost(a, ctx.stream());
  auto d_b = gpu::DeviceMatrix<Scalar>::fromHost(b, ctx.stream());

  // dot() returns gpu::DeviceScalar — implicit conversion to Scalar syncs.
  Scalar gpu_dot = d_a.dot(ctx, d_b);
  Scalar cpu_dot = a.dot(b);
  RealScalar tol = RealScalar(10) * RealScalar(n) * NumTraits<Scalar>::epsilon();
  VERIFY(numext::abs(gpu_dot - cpu_dot) < tol * numext::abs(cpu_dot) + tol);

  // squaredNorm() returns host RealScalar directly (syncs internally).
  RealScalar gpu_sqnorm = d_a.squaredNorm(ctx);
  RealScalar cpu_sqnorm = a.squaredNorm();
  VERIFY(numext::abs(gpu_sqnorm - cpu_sqnorm) < tol * cpu_sqnorm + tol);

  // norm() returns gpu::DeviceScalar<RealScalar> — implicit conversion syncs.
  RealScalar gpu_norm = d_a.norm(ctx);
  RealScalar cpu_norm = a.norm();
  VERIFY(numext::abs(gpu_norm - cpu_norm) < tol * cpu_norm + tol);

  // Convenience overloads (thread-local context).
  gpu::Context::setThreadLocal(&ctx);
  Scalar gpu_dot2 = d_a.dot(d_b);
  VERIFY(numext::abs(gpu_dot2 - cpu_dot) < tol * numext::abs(cpu_dot) + tol);
  gpu::Context::setThreadLocal(nullptr);

  // Empty vectors: dot and norm must return zero.
  {
    gpu::DeviceMatrix<Scalar> d_empty(0, 1);
    gpu::DeviceMatrix<Scalar> d_empty2(0, 1);
    Scalar empty_dot = d_empty.dot(ctx, d_empty2);
    VERIFY_IS_EQUAL(empty_dot, Scalar(0));
    RealScalar empty_sqnorm = d_empty.squaredNorm(ctx);
    VERIFY_IS_EQUAL(empty_sqnorm, RealScalar(0));
    RealScalar empty_norm = d_empty.norm(ctx);
    VERIFY_IS_EQUAL(empty_norm, RealScalar(0));
  }
}

// ---- cwiseProduct -----------------------------------------------------------

template <typename Scalar>
void test_cwiseProduct() {
  using Vec = Matrix<Scalar, Dynamic, 1>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  const Index n = 256;
  Vec a = Vec::Random(n);
  Vec b = Vec::Random(n);
  Vec ref = a.array() * b.array();

  gpu::Context ctx;
  auto d_a = gpu::DeviceMatrix<Scalar>::fromHost(a, ctx.stream());
  auto d_b = gpu::DeviceMatrix<Scalar>::fromHost(b, ctx.stream());
  auto d_c = d_a.cwiseProduct(ctx, d_b);
  Vec result = d_c.toHost(ctx.stream());

  RealScalar tol = RealScalar(10) * RealScalar(n) * NumTraits<Scalar>::epsilon();
  VERIFY((result - ref).norm() < tol * ref.norm() + tol);
}

EIGEN_DECLARE_TEST(gpu_device_matrix) {
  CALL_SUBTEST(test_default_construct());
  CALL_SUBTEST(test_empty());
  CALL_SUBTEST(test_device_buffer_uses_owner_stream());
  CALL_SUBTEST(test_fallback_pool_cross_stream_reuse());
  CALL_SUBTEST(test_cross_stream_free_ordering(0));
  CALL_SUBTEST(test_cross_stream_free_ordering(1));
  CALL_SUBTEST(test_cross_stream_free_ordering(2));
  CALL_SUBTEST(test_device_scalar_outlives_context());
  CALL_SUBTEST(test_device_buffer_outlives_stream_handle());
  CALL_SUBTEST(test_resize());
  CALL_SUBTEST(test_host_transfer_ready());
  CALL_SUBTEST(test_host_transfer_move());
  CALL_SUBTEST((test_allocate<float>(100, 50)));
  CALL_SUBTEST((test_allocate<double>(100, 50)));
  CALL_SUBTEST(test_scalar<float>());
  CALL_SUBTEST(test_scalar<double>());
  CALL_SUBTEST(test_scalar<std::complex<float>>());
  CALL_SUBTEST(test_scalar<std::complex<double>>());
  CALL_SUBTEST(test_blas1<float>(256));
  CALL_SUBTEST(test_blas1<double>(256));
  CALL_SUBTEST(test_blas1<std::complex<float>>(256));
  CALL_SUBTEST(test_blas1<std::complex<double>>(256));
  CALL_SUBTEST(test_cg_operators<float>(256));
  CALL_SUBTEST(test_cg_operators<double>(256));
  CALL_SUBTEST(test_cg_operators<std::complex<float>>(256));
  CALL_SUBTEST(test_cg_operators<std::complex<double>>(256));
  CALL_SUBTEST(test_device_scalar<float>());
  CALL_SUBTEST(test_device_scalar<double>());
  CALL_SUBTEST(test_device_scalar<std::complex<float>>());
  CALL_SUBTEST(test_device_scalar<std::complex<double>>());
  CALL_SUBTEST(test_cwiseProduct<float>());
  CALL_SUBTEST(test_cwiseProduct<double>());
}
