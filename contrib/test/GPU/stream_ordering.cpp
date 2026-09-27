// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// Cross-context ordering of DeviceMatrix and DeviceScalar accesses (RAW, WAR,
// WAW, free after read), and no use of the legacy default stream. An ordering
// test parks one context's stream behind a StreamGate and enqueues the access
// under test on another: an access not ordered after the parked one runs first.

#define EIGEN_USE_GPU
#include "main.h"
#include <Eigen/Sparse>
#include <contrib/Eigen/GPU>

#include "./gpu_test_helpers.h"

using namespace Eigen;
using gpu_test::LegacyStreamSentinel;
using gpu_test::StreamGate;

using Vec = VectorXd;
using DeviceVec = gpu::DeviceMatrix<double>;
constexpr Index kSize = 4096;
constexpr std::chrono::milliseconds kGateDelay(100);

void wait_for(gpu::Context& ctx) { EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(ctx.stream())); }

// Each scenario runs twice, parking the stream only on the second pass. The
// first pass creates the lazily created library handles and loads the kernels
// the scenario launches: both can synchronize the device (CUDA loads a module
// on its first launch by default since 12.2), which would wait on the gate
// forever.
class MaybeParked {
 public:
  MaybeParked(cudaStream_t stream, bool park) : gate_(park ? new StreamGate(stream) : nullptr) {}
  void openAfter(std::chrono::milliseconds delay) {
    if (gate_) gate_->openAfter(delay);
  }

 private:
  std::unique_ptr<StreamGate> gate_;
};

// A read on another context waits for the pending write.
void test_read_after_write() {
  gpu::Context writer, reader;
  const Vec a = Vec::Random(kSize);
  for (bool park : {false, true}) {
    DeviceVec d_a = DeviceVec::fromHost(writer, a);
    DeviceVec d_b;
    {
      MaybeParked gate(writer.stream(), park);
      d_a.scale(writer, 2.0);
      d_b = d_a.clone(reader);
      gate.openAfter(kGateDelay);
      wait_for(reader);
    }
    VERIFY_IS_APPROX(d_b.toHost(reader), Vec(2.0 * a));
  }
}

// A write waits for a pending read on another context.
void test_write_after_read() {
  gpu::Context writer, reader;
  const Vec a = Vec::Random(kSize);
  for (bool park : {false, true}) {
    DeviceVec d_a = DeviceVec::fromHost(writer, a);
    DeviceVec d_b;
    {
      MaybeParked gate(reader.stream(), park);
      d_b = d_a.clone(reader);
      d_a.setZero(writer);
      gate.openAfter(kGateDelay);
      wait_for(writer);
    }
    VERIFY_IS_APPROX(d_b.toHost(reader), a);
    VERIFY(d_a.toHost(writer).isZero(0));
  }
}

// A write on a new context waits for the reads that the old home stream still
// has queued: those are never recorded, since same-stream order covers them.
void test_write_after_home_stream_read() {
  gpu::Context home, other;
  const Vec a = Vec::Random(kSize);
  for (bool park : {false, true}) {
    DeviceVec d_a = DeviceVec::fromHost(home, a);
    DeviceVec d_b;
    {
      MaybeParked gate(home.stream(), park);
      d_b = d_a.clone(home);
      d_a.setZero(other);
      gate.openAfter(kGateDelay);
      wait_for(other);
    }
    VERIFY_IS_APPROX(d_b.toHost(home), a);
    VERIFY(d_a.toHost(other).isZero(0));
    VERIFY(d_a.stream() == other.stream());
  }
}

// Freeing a matrix waits for a pending read on another context. The same-size
// allocation right after the free is where the pool hands the block out again.
void test_free_after_read() {
  gpu::Context writer, reader;
  const Vec a = Vec::Random(kSize);
  for (bool park : {false, true}) {
    DeviceVec d_b;
    DeviceVec d_reuse;
    {
      DeviceVec d_a = DeviceVec::fromHost(writer, a);
      MaybeParked gate(reader.stream(), park);
      d_b = d_a.clone(reader);
      d_a = DeviceVec();
      d_reuse = DeviceVec(writer, kSize, 1);
      d_reuse.setZero(writer);
      gate.openAfter(kGateDelay);
      wait_for(writer);
    }
    VERIFY_IS_APPROX(d_b.toHost(reader), a);
  }
}

// Scalar arithmetic runs on its first operand's stream and waits for a second
// operand still being written on another context.
void test_device_scalar_across_contexts() {
  gpu::Context producer, consumer;
  const Vec a = Vec::Random(kSize);
  const DeviceVec d_a = DeviceVec::fromHost(producer, a);
  for (bool park : {false, true}) {
    gpu::DeviceScalar<double> ratio(consumer);
    {
      MaybeParked gate(producer.stream(), park);
      gpu::DeviceScalar<double> sq = d_a.squaredNorm(producer);
      ratio = gpu::DeviceScalar<double>(consumer, 1.0) / sq;
      gate.openAfter(kGateDelay);
      wait_for(consumer);
    }
    VERIFY(ratio.stream() == consumer.stream());
    VERIFY_IS_APPROX(ratio.get(), 1.0 / a.squaredNorm());
  }
}

// A solver that adopts a matrix by move waits for the matrix's pending write.
void test_solver_adopts_pending_matrix() {
  gpu::Context producer, solver_ctx;
  const Index n = 64;
  const MatrixXd M = MatrixXd::Random(n, n);
  const MatrixXd A = M * M.transpose() + MatrixXd::Identity(n, n) * double(n);
  const MatrixXd B = MatrixXd::Random(n, 3);
  gpu::LLT<double> llt(solver_ctx);
  for (bool park : {false, true}) {
    gpu::DeviceMatrix<double> d_A = gpu::DeviceMatrix<double>::fromHost(producer, A);
    MatrixXd X;
    {
      MaybeParked gate(producer.stream(), park);
      d_A.scale(producer, 2.0);
      llt.compute(std::move(d_A));
      gate.openAfter(kGateDelay);
      X = llt.solve(B);
    }
    VERIFY_IS_APPROX(MatrixXd(2.0 * A * X), B);
  }
}

// Alternating writes on one context and reads on another keep one read mark:
// the next read reuses the mark a write retired.
void test_read_marks_are_reused() {
  gpu::Context writer, reader;
  DeviceVec d_a = DeviceVec::fromHost(writer, Vec::Random(kSize));
  for (int i = 0; i < 16; ++i) {
    d_a.scale(writer, 0.5);
    DeviceVec d_b = d_a.clone(reader);
  }
  VERIFY_IS_EQUAL(gpu::internal::DeviceMatrixAccess::buffer(d_a).readMarkCount(), std::size_t(1));
}

// Threads, each on its own Context, read one matrix at the same time. A race on
// its read-side state shows up only probabilistically, so this is a stress
// test rather than a proof.
void test_concurrent_reads() {
  const Vec a = Vec::Random(kSize);
  const DeviceVec d_a = DeviceVec::fromHost(a);
  std::atomic<int> mismatches{0};
  auto read_many = [&] {
    gpu::Context ctx;
    for (int i = 0; i < 200; ++i) {
      const DeviceVec d_b = d_a.clone(ctx);
      if (i % 50 == 0 && !d_b.toHost(ctx).isApprox(a)) ++mismatches;
    }
    wait_for(ctx);
  };
  std::thread t1(read_many), t2(read_many), t3(read_many);
  t1.join();
  t2.join();
  t3.join();
  VERIFY_IS_EQUAL(mismatches.load(), 0);
}

// Destroying a view makes its stream wait for the reads made through it, so
// the owner's free there waits for them too.
void test_free_after_read_through_view() {
  gpu::Context writer, reader;
  const Vec a = Vec::Random(kSize);
  for (bool park : {false, true}) {
    DeviceVec owner = DeviceVec::fromHost(writer, a);
    DeviceVec d_b;
    DeviceVec d_reuse;
    {
      MaybeParked gate(reader.stream(), park);
      {
        const DeviceVec view = DeviceVec::view(writer, owner.data(), owner.rows(), owner.cols());
        d_b = view.clone(reader);
      }
      owner = DeviceVec();
      d_reuse = DeviceVec(writer, kSize, 1);
      d_reuse.setZero(writer);
      gate.openAfter(kGateDelay);
      wait_for(writer);
    }
    VERIFY_IS_APPROX(d_b.toHost(reader), a);
  }
}

// release() hands the pending work to a Context the caller holds, so it works
// after the Context that wrote the matrix, and with it the matrix's stream, is gone.
void test_release_after_context() {
  const Vec a = Vec::Random(kSize);
  DeviceVec matrix;
  {
    gpu::Context writer;
    matrix = DeviceVec::fromHost(writer, a);
    matrix.scale(writer, 2.0);
  }
  gpu::Context ctx;
  double* p = matrix.release(ctx);
  VERIFY(matrix.empty() && matrix.data() == nullptr);
  Vec back(kSize);
  EIGEN_CUDA_RUNTIME_CHECK(
      cudaMemcpyAsync(back.data(), p, sizeof(double) * kSize, cudaMemcpyDeviceToHost, ctx.stream()));
  wait_for(ctx);
  EIGEN_CUDA_RUNTIME_CHECK(cudaFree(p));
  VERIFY_IS_APPROX(back, Vec(2.0 * a));
}

// A read mark retires when its read completes, without a write: short-lived
// readers of a constant matrix leave a single mark behind.
void test_read_marks_retire_when_done() {
  gpu::Context home;
  const DeviceVec d_a = DeviceVec::fromHost(home, Vec::Random(kSize));
  for (int i = 0; i < 32; ++i) {
    gpu::Context reader;
    const DeviceVec d_b = d_a.clone(reader);
    wait_for(reader);
  }
  VERIFY_IS_EQUAL(gpu::internal::DeviceMatrixAccess::buffer(d_a).readMarkCount(), std::size_t(1));
}

// Matrices outlive the Context that wrote them: its stream stays alive for
// their reads and their stream-ordered frees.
void test_matrix_outlives_context() {
  const Vec a = Vec::Random(kSize);
  DeviceVec d_a;
  gpu::DeviceScalar<double> norm;
  {
    gpu::Context scoped;
    d_a = DeviceVec::fromHost(scoped, a);
    d_a.scale(scoped, 2.0);
    norm = d_a.norm(scoped);
  }
  VERIFY_IS_APPROX(d_a.toHost(), Vec(2.0 * a));
  VERIFY_IS_APPROX(norm.get(), 2.0 * a.norm());
}

// The sentinel itself: a legacy-stream operation invalidates it.
void test_sentinel_detects_legacy_stream() {
  void* p = nullptr;
  EIGEN_CUDA_RUNTIME_CHECK(cudaMalloc(&p, 16));
  {
    LegacyStreamSentinel sentinel;
    (void)cudaMemsetAsync(p, 0, 16, cudaStreamLegacy);
    VERIFY(!sentinel.end());
  }
  (void)cudaGetLastError();
  EIGEN_CUDA_RUNTIME_CHECK(cudaFree(p));
}

// Arithmetic that exists for real scalars only (NPP).
template <typename Scalar>
std::enable_if_t<!NumTraits<Scalar>::IsComplex> real_only_workload(gpu::Context& ctx,
                                                                   const gpu::DeviceMatrix<Scalar>& d_x,
                                                                   const gpu::DeviceMatrix<Scalar>& d_y) {
  gpu::DeviceScalar<Scalar> ratio = d_x.dot(ctx, d_y) / d_y.squaredNorm(ctx);
  gpu::DeviceScalar<Scalar> neg = -ratio;
  gpu::DeviceMatrix<Scalar> d_z = d_x.cwiseProduct(ctx, d_y);
  d_z /= Scalar(3);
  d_z *= neg;
  VERIFY((numext::isfinite)(Scalar(neg)));
}

template <typename Scalar>
std::enable_if_t<NumTraits<Scalar>::IsComplex> real_only_workload(gpu::Context&, const gpu::DeviceMatrix<Scalar>&,
                                                                  const gpu::DeviceMatrix<Scalar>&) {}

// Every kind of module operation, across the thread-local and two explicit
// contexts. The contexts, solvers, and FFT are members because creating them
// creates library handles, which synchronize the device and are rejected while
// the sentinel's capture is open; run() creates and destroys everything else.
template <typename Scalar>
struct ModuleWorkload {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using Complex = std::complex<RealScalar>;
  using ComplexVec = Matrix<Complex, Dynamic, 1>;

  const Index n = 48;
  const Mat M = Mat::Random(n, n);
  const Mat A = M * M.adjoint() + Mat::Identity(n, n) * RealScalar(n);
  const Mat B = Mat::Random(n, 2);
  const SparseMatrix<Scalar> S = A.sparseView();
  const ComplexVec z = ComplexVec::Random(64);

  gpu::Context ctx1, ctx2;
  gpu::LLT<Scalar> llt;
  gpu::LU<Scalar> lu{ctx1};
  gpu::QR<Scalar> qr{ctx2};
  gpu::SVD<Scalar> svd;
  gpu::SelfAdjointEigenSolver<Scalar> eig{ctx1};
  gpu::SparseContext<Scalar> sparse{ctx1};
  gpu::SparseContext<Scalar> standalone;
  gpu::FFT<RealScalar> fft{ctx2};

  void run() {
    gpu::DeviceMatrix<Scalar> d_A = gpu::DeviceMatrix<Scalar>::fromHost(ctx1, A);
    gpu::DeviceMatrix<Scalar> d_B = gpu::DeviceMatrix<Scalar>::fromHostAsync(ctx2, B.data(), n, 2);
    gpu::DeviceMatrix<Scalar> d_M = gpu::DeviceMatrix<Scalar>::fromHost(M);

    // Expressions on all three contexts, with operands from the others.
    gpu::DeviceMatrix<Scalar> d_C = d_A * d_B;
    d_C.device(ctx2) += d_M * d_B;
    gpu::DeviceMatrix<Scalar> d_D;
    d_D.device(ctx1) = d_A + Scalar(2) * d_M;
    d_D.device(ctx2) = d_A.adjoint() * d_M;
    gpu::DeviceMatrix<Scalar> d_X = d_A.llt().solve(d_B);
    d_X.device(ctx2) = d_A.lu().solve(d_C);
    d_X.device(ctx1) = d_A.template triangularView<Lower>().solve(d_B);
    d_X.device(ctx2) = d_A.template selfadjointView<Lower>() * d_B;
    gpu::DeviceMatrix<Scalar> d_S;
    d_S.template selfadjointView<Lower>().rankUpdate(d_B);
    d_S.resize(ctx2, 2 * n, 2 * n);
    d_S.setZero(ctx2);

    // BLAS-1 and device scalars.
    gpu::DeviceMatrix<Scalar> d_x = d_B.clone(ctx1), d_y = d_C.clone(ctx2);
    d_x.resize(ctx1, 2 * n, 1);
    d_y.resize(ctx2, 2 * n, 1);
    d_x.addScaled(ctx2, Scalar(2), d_y);
    d_x += Scalar(3) * d_y;
    d_x -= d_y;
    d_x.scale(ctx1, Scalar(0.5));
    gpu::DeviceScalar<Scalar> dot = d_x.dot(ctx2, d_y);
    gpu::DeviceScalar<RealScalar> nrm = d_x.norm(ctx1);
    d_y.copyFrom(ctx1, d_x);
    real_only_workload(ctx2, d_x, d_y);
    VERIFY((numext::isfinite)(numext::abs(Scalar(dot))) && (numext::isfinite)(RealScalar(nrm)));

    // Transfers.
    auto transfer = d_C.toHostAsync(ctx2);
    VERIFY_IS_APPROX(Mat(d_C.toHost(ctx1)), transfer.get());

    // Dense solvers: private and bound contexts, host and device inputs, moves.
    llt.compute(A);
    lu.compute(d_A);
    qr.compute(A);
    svd.compute(d_A);
    eig.compute(A);
    VERIFY(llt.info() == Success && lu.info() == Success && qr.info() == Success);
    VERIFY_IS_APPROX(Mat(A * llt.solve(B)), B);
    VERIFY_IS_APPROX(Mat(A * lu.solve(d_B).toHost()), B);
    VERIFY_IS_APPROX(Mat(A * qr.solve(d_B).toHost(ctx2)), B);
    VERIFY_IS_APPROX(Mat(A * svd.solve(B)), B);
    VERIFY(svd.singularValues().size() == n && svd.matrixU().rows() == n);
    VERIFY(eig.eigenvalues().size() == n && eig.eigenvectors().rows() == n);
    {
      gpu::DeviceMatrix<Scalar> d_U = svd.d_matrixU();
      gpu::DeviceMatrix<Scalar> d_UB;
      d_UB.device(ctx2) = d_U.adjoint() * d_B;
    }
    llt.compute(gpu::DeviceMatrix<Scalar>::fromHost(ctx2, A));
    gpu::DeviceMatrix<Scalar> d_Y = llt.solve(d_B.clone(ctx2));
    VERIFY_IS_APPROX(Mat(A * d_Y.toHost()), B);

    // Sparse products.
    VERIFY_IS_APPROX(Mat(standalone.multiply(S, B.col(0))), Mat(A * B.col(0)));
    auto view = sparse.deviceView(S);
    gpu::DeviceMatrix<Scalar> d_b = d_B.clone(ctx2);
    d_b.resize(ctx2, n, 1);
    d_b.setZero(ctx2);
    gpu::DeviceMatrix<Scalar> d_Sx = view * d_b;
    VERIFY(d_Sx.toHost(ctx2).isZero(0));

    // FFT, host and device.
    VERIFY_IS_APPROX(fft.inv(fft.fwd(z)), z);
    gpu::DeviceMatrix<Complex> d_z = gpu::DeviceMatrix<Complex>::fromHost(ctx1, z), d_Z, d_zz;
    fft.fwd(d_z, d_Z);
    fft.inv(d_Z, d_zz);
    VERIFY_IS_APPROX(d_zz.toHost(ctx1), z);
  }
};

// Runs ModuleWorkload once to create every lazily created handle, plan, and
// buffer, then again under a sentinel, from a thread of its own so that its
// thread-local Context is fresh as well.
template <typename Scalar>
void test_no_legacy_stream() {
  bool clean = false;
  std::thread worker([&clean] {
    gpu::Context::threadLocal();
    ModuleWorkload<Scalar> workload;
    workload.run();
    LegacyStreamSentinel sentinel;
    workload.run();
    clean = sentinel.end();
  });
  worker.join();
  VERIFY(clean);
}

EIGEN_DECLARE_TEST(gpu_stream_ordering) {
  gpu_test::require_cuda_device();
  CALL_SUBTEST_1(test_read_after_write());
  CALL_SUBTEST_1(test_write_after_read());
  CALL_SUBTEST_1(test_write_after_home_stream_read());
  CALL_SUBTEST_1(test_free_after_read());
  CALL_SUBTEST_1(test_device_scalar_across_contexts());
  CALL_SUBTEST_1(test_solver_adopts_pending_matrix());
  CALL_SUBTEST_1(test_matrix_outlives_context());
  CALL_SUBTEST_1(test_read_marks_are_reused());
  CALL_SUBTEST_1(test_concurrent_reads());
  CALL_SUBTEST_2(test_sentinel_detects_legacy_stream());
  CALL_SUBTEST_2(test_no_legacy_stream<double>());
  CALL_SUBTEST_2(test_no_legacy_stream<std::complex<float>>());
  CALL_SUBTEST_3(test_free_after_read_through_view());
  CALL_SUBTEST_3(test_release_after_context());
  CALL_SUBTEST_3(test_read_marks_retire_when_done());
}
