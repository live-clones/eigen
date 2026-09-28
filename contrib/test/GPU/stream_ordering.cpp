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
// WAW, free after read), by the module and by direct accesses bracketed with
// prepare*/finish*, and no use of the legacy default stream. An ordering
// test parks one context's stream behind a StreamGate and enqueues the access
// under test on another: an access not ordered after the parked one runs first.

#define EIGEN_USE_GPU
#include "main.h"
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
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

// Direct accesses to device memory, outside the module, on ctx's stream.
void copy_on(gpu::Context& ctx, void* dst, const void* src, std::size_t bytes) {
  EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, ctx.stream()));
}
void zero_on(gpu::Context& ctx, void* dst, std::size_t bytes) {
  EIGEN_CUDA_RUNTIME_CHECK(cudaMemsetAsync(dst, 0, bytes, ctx.stream()));
}
constexpr std::size_t kBytes = sizeof(double) * kSize;

// Each scenario runs twice, parking the stream only on the second pass. The
// first pass creates the lazily created library handles and loads the kernels
// the scenario launches: both can synchronize the device (CUDA loads a module
// on its first launch by default since 12.2), which would wait on the gate
// forever. Without memory pools nothing is parked: cudaMalloc and cudaFree
// synchronize the device, so the scenarios run but are no longer deterministic.
class MaybeParked {
 public:
  MaybeParked(cudaStream_t stream, bool park)
      : gate_(park && gpu::internal::device_supports_memory_pools() ? new StreamGate(stream) : nullptr) {}
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
  DeviceVec d_a = DeviceVec::fromHost(producer, a);
  double scale = 1;
  for (bool park : {false, true}) {
    gpu::DeviceScalar<double> ratio(consumer);
    {
      MaybeParked gate(producer.stream(), park);
      // A new value each pass, so that reading the previous pass's result,
      // whose block the allocator may hand out again, fails the check.
      d_a.scale(producer, 2.0);
      gpu::DeviceScalar<double> sq = d_a.squaredNorm(producer);
      ratio = gpu::DeviceScalar<double>(consumer, 1.0) / sq;
      gate.openAfter(kGateDelay);
      wait_for(consumer);
    }
    scale *= 2;
    VERIFY(ratio.stream() == consumer.stream());
    VERIFY_IS_APPROX(ratio.get(), 1.0 / (scale * scale * a.squaredNorm()));
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

// Alternating writes on one context and reads on another keep one read mark.
void test_read_marks_are_reused() {
  gpu::Context writer, reader;
  DeviceVec d_a = DeviceVec::fromHost(writer, Vec::Random(kSize));
  for (int i = 0; i < 16; ++i) {
    d_a.scale(writer, 0.5);
    DeviceVec d_b = d_a.clone(reader);
  }
  VERIFY_IS_EQUAL(gpu::internal::DeviceMatrixAccess::buffer(d_a).readMarkCount(), std::size_t(1));
}

// A write retires the read marks it waited for: a read from another stream
// reuses the retired mark while the read it recorded is still pending.
void test_write_retires_read_marks() {
  gpu::Context writer, reader1, reader2;
  for (bool park : {false, true}) {
    DeviceVec d_a = DeviceVec::fromHost(writer, Vec::Random(kSize));
    std::size_t marks = 0;
    {
      MaybeParked gate(reader1.stream(), park);
      DeviceVec d_b = d_a.clone(reader1);
      d_a.scale(writer, 0.5);
      DeviceVec d_c = d_a.clone(reader2);
      marks = gpu::internal::DeviceMatrixAccess::buffer(d_a).readMarkCount();
      gate.openAfter(kGateDelay);
      wait_for(reader2);
    }
    VERIFY_IS_EQUAL(marks, std::size_t(1));
  }
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

// release() orders the matrix's pending write before the work the caller then
// enqueues on the Context it passes.
void test_release_orders_pending_write() {
  gpu::Context writer, ctx;
  const Vec a = Vec::Random(kSize);
  for (bool park : {false, true}) {
    DeviceVec matrix = DeviceVec::fromHost(writer, a);
    DeviceVec out(ctx, kSize, 1);
    double* p = nullptr;
    {
      MaybeParked gate(writer.stream(), park);
      matrix.scale(writer, 2.0);
      p = matrix.release(ctx);
      copy_on(ctx, out.data(), p, kBytes);
      gate.openAfter(kGateDelay);
      wait_for(ctx);
    }
    EIGEN_CUDA_RUNTIME_CHECK(cudaFree(p));
    VERIFY_IS_APPROX(out.toHost(ctx), Vec(2.0 * a));
  }
}

// A view written on another stream hands that write back when destroyed: the
// owner's later work on the view's Context follows it.
void test_write_through_view() {
  gpu::Context home, other;
  const Vec a = Vec::Random(kSize), c = Vec::Random(kSize);
  const DeviceVec d_c = DeviceVec::fromHost(home, c);
  for (bool park : {false, true}) {
    DeviceVec owner = DeviceVec::fromHost(home, a);
    {
      MaybeParked gate(other.stream(), park);
      {
        DeviceVec view = DeviceVec::view(home, owner.data(), owner.rows(), owner.cols());
        view.scale(other, 2.0);
      }
      owner.copyFrom(home, d_c);
      gate.openAfter(kGateDelay);
      wait_for(home);
    }
    VERIFY_IS_APPROX(owner.toHost(home), c);
  }
}

// A one-shot solve into its own matrix operand that grows the matrix frees the
// operand's storage; the free waits for the pending copy of the operand. A
// second thread reuses the freed block meanwhile, as another allocation would.
void test_oneshot_solve_into_operand() {
  gpu::Context home, solver;
  const Index n = 64;
  const MatrixXd M = MatrixXd::Random(n, n);
  const MatrixXd A = M * M.transpose() + MatrixXd::Identity(n, n) * double(n);
  const MatrixXd B = MatrixXd::Random(n, n + 8);
  const gpu::DeviceMatrix<double> d_B = gpu::DeviceMatrix<double>::fromHost(home, B);
  for (bool park : {false, true}) {
    gpu::DeviceMatrix<double> d_A = gpu::DeviceMatrix<double>::fromHost(home, A);
    {
      MaybeParked gate(solver.stream(), park);
      // Opened from the start: in debug builds the solve itself waits for the solver stream.
      gate.openAfter(kGateDelay);
      std::thread reuse([&] {
        std::this_thread::sleep_for(kGateDelay / 4);
        gpu::DeviceMatrix<double> junk(home, n, n);
        junk.setZero(home);
        wait_for(home);
      });
      d_A.device(solver) = d_A.llt().solve(d_B);
      reuse.join();
    }
    VERIFY_IS_APPROX(MatrixXd(A * d_A.toHost(solver)), B);
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

// The tests below bracket direct accesses with the public prepare*/finish*
// calls, as user code does; each check fails with its bracket removed.

// prepareRead: a direct read on another context waits for the module's pending write.
void test_direct_read_after_write() {
  gpu::Context writer, reader;
  const Vec a = Vec::Random(kSize);
  for (bool park : {false, true}) {
    DeviceVec d_a = DeviceVec::fromHost(writer, a);
    DeviceVec d_b(reader, kSize, 1);
    {
      MaybeParked gate(writer.stream(), park);
      d_a.scale(writer, 2.0);
      d_a.prepareRead(reader);
      d_b.prepareWrite(reader);
      copy_on(reader, d_b.data(), d_a.data(), kBytes);
      d_a.finishRead(reader);
      d_b.finishWrite(reader);
      gate.openAfter(kGateDelay);
      wait_for(reader);
    }
    VERIFY_IS_APPROX(d_b.toHost(reader), Vec(2.0 * a));
  }
}

// finishRead: the module's write waits for a pending direct read on another context.
void test_write_after_direct_read() {
  gpu::Context writer, reader;
  const Vec a = Vec::Random(kSize);
  for (bool park : {false, true}) {
    DeviceVec d_a = DeviceVec::fromHost(writer, a);
    DeviceVec d_b(reader, kSize, 1);
    {
      MaybeParked gate(reader.stream(), park);
      d_a.prepareRead(reader);
      d_b.prepareWrite(reader);
      copy_on(reader, d_b.data(), d_a.data(), kBytes);
      d_a.finishRead(reader);
      d_b.finishWrite(reader);
      d_a.setZero(writer);
      gate.openAfter(kGateDelay);
      wait_for(writer);
    }
    VERIFY_IS_APPROX(d_b.toHost(reader), a);
    VERIFY(d_a.toHost(writer).isZero(0));
  }
}

// prepareWrite on stream(): a direct write waits for the module's pending read
// on another context.
void test_direct_write_after_read() {
  gpu::Context writer, reader;
  const Vec a = Vec::Random(kSize);
  for (bool park : {false, true}) {
    DeviceVec d_a = DeviceVec::fromHost(writer, a);
    DeviceVec d_b;
    {
      MaybeParked gate(reader.stream(), park);
      d_b = d_a.clone(reader);
      d_a.prepareWrite(writer);
      zero_on(writer, d_a.data(), kBytes);
      d_a.finishWrite(writer);
      gate.openAfter(kGateDelay);
      wait_for(writer);
    }
    VERIFY_IS_APPROX(d_b.toHost(reader), a);
    VERIFY(d_a.toHost(writer).isZero(0));
  }
}

// prepareWrite on another stream: a direct write there waits for the work
// queued on stream(), reads included, and its stream becomes stream().
void test_direct_write_moves_stream() {
  gpu::Context home, other;
  const Vec a = Vec::Random(kSize);
  for (bool park : {false, true}) {
    DeviceVec d_a = DeviceVec::fromHost(home, a);
    DeviceVec d_b;
    {
      MaybeParked gate(home.stream(), park);
      d_b = d_a.clone(home);
      d_a.prepareWrite(other);
      zero_on(other, d_a.data(), kBytes);
      d_a.finishWrite(other);
      gate.openAfter(kGateDelay);
      wait_for(other);
    }
    VERIFY_IS_APPROX(d_b.toHost(home), a);
    VERIFY(d_a.toHost(other).isZero(0));
    VERIFY(d_a.stream() == other.stream());
  }
}

// finishWrite: the module's read on another context waits for a direct write.
// The matrix's previous write event has completed, so without finishWrite the
// read would wait on that one.
void test_read_after_direct_write() {
  gpu::Context writer, reader;
  const Vec a = Vec::Random(kSize);
  for (bool park : {false, true}) {
    DeviceVec d_a = DeviceVec::fromHost(writer, a);
    d_a.scale(writer, 2.0);
    wait_for(writer);
    DeviceVec d_b;
    {
      MaybeParked gate(writer.stream(), park);
      d_a.prepareWrite(writer);
      zero_on(writer, d_a.data(), kBytes);
      d_a.finishWrite(writer);
      d_b = d_a.clone(reader);
      gate.openAfter(kGateDelay);
      wait_for(reader);
    }
    VERIFY(d_b.toHost(reader).isZero(0));
  }
}

// DeviceScalar::finishWrite records no event but drops the previous one, which
// an earlier read on another context recorded: the next such read records a
// fresh event, behind the direct write.
void test_device_scalar_direct_write() {
  gpu::Context producer, consumer;
  for (bool park : {false, true}) {
    gpu::DeviceScalar<double> s(producer, 1.0), four(producer, 4.0);
    const gpu::DeviceScalar<double> one(consumer, 1.0);
    VERIFY_IS_APPROX((one / s).get(), 1.0);
    gpu::DeviceScalar<double> ratio(consumer);
    {
      MaybeParked gate(producer.stream(), park);
      four.prepareRead(producer);
      s.prepareWrite(producer);
      copy_on(producer, s.devicePtr(), four.devicePtr(), sizeof(double));
      four.finishRead(producer);
      s.finishWrite(producer);
      ratio = one / s;
      gate.openAfter(kGateDelay);
      wait_for(consumer);
    }
    VERIFY_IS_APPROX(ratio.get(), 0.25);
  }
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

// Arithmetic that exists for real scalars only (NPP), and Eigen's
// ConjugateGradient class, whose device form is real-only.
template <typename Scalar>
std::enable_if_t<!NumTraits<Scalar>::IsComplex> real_only_workload(gpu::Context& ctx,
                                                                   const gpu::DeviceMatrix<Scalar>& d_x,
                                                                   const gpu::DeviceMatrix<Scalar>& d_y,
                                                                   const gpu::DeviceSparseView<Scalar>& view,
                                                                   const gpu::DeviceMatrix<Scalar>& d_b) {
  gpu::DeviceScalar<Scalar> ratio = d_x.dot(ctx, d_y) / d_y.squaredNorm(ctx);
  gpu::DeviceScalar<Scalar> neg = -ratio;
  gpu::DeviceMatrix<Scalar> d_z = d_x.cwiseProduct(ctx, d_y);
  d_z /= Scalar(3);
  d_z *= neg;
  d_z += ratio * d_y;
  d_z -= ratio * d_x;
  VERIFY((numext::isfinite)(Scalar(neg)));

  ConjugateGradient<gpu::DeviceSparseView<Scalar>, Lower | Upper, IdentityPreconditioner> cg;
  cg.setMaxIterations(3);
  cg.compute(view);
  gpu::DeviceMatrix<Scalar> d_sol(ctx, d_b.rows(), 1);
  d_sol.setZero(ctx);
  cg.solveWithGuessInPlace(d_b, d_sol);
  VERIFY(cg.iterations() > 0);
}

template <typename Scalar>
std::enable_if_t<NumTraits<Scalar>::IsComplex> real_only_workload(gpu::Context&, const gpu::DeviceMatrix<Scalar>&,
                                                                  const gpu::DeviceMatrix<Scalar>&,
                                                                  const gpu::DeviceSparseView<Scalar>&,
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
  using ComplexMat = Matrix<Complex, Dynamic, Dynamic>;
  using RealVec = Matrix<RealScalar, Dynamic, 1>;
  using Vec = Matrix<Scalar, Dynamic, 1>;
  using Bsm = BlockSparseMatrix<Scalar, RowMajor, 2, 2, int>;

  const Index n = 48;
  const Mat M = Mat::Random(n, n);
  const Mat A = M * M.adjoint() + Mat::Identity(n, n) * RealScalar(n);
  const Mat B = Mat::Random(n, 2);
  const Vec b = Vec::Random(n);
  const SparseMatrix<Scalar> S = A.sparseView();
  const ComplexVec z = ComplexVec::Random(64);
  const RealVec r = RealVec::Random(64);
  const ComplexMat Z = ComplexMat::Random(8, 8);
  const Bsm block_identity = make_block_identity();

  gpu::Context ctx1, ctx2;
  gpu::LLT<Scalar> llt;
  gpu::LU<Scalar> lu{ctx1};
  gpu::QR<Scalar> qr{ctx2};
  gpu::SVD<Scalar> svd;
  gpu::SelfAdjointEigenSolver<Scalar> eig{ctx1};
  gpu::SparseContext<Scalar> sparse{ctx1};
  gpu::SparseContext<Scalar> standalone;
  gpu::FFT<RealScalar> fft{ctx2};

  Bsm make_block_identity() const {
    std::vector<typename Bsm::TripletType> triplets;
    for (int k = 0; k < int(n / 2); ++k) triplets.emplace_back(k, k, Bsm::BlockType::Identity());
    Bsm identity(n / 2, n / 2);
    identity.setFromTriplets(triplets.begin(), triplets.end());
    return identity;
  }

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
    VERIFY((numext::isfinite)(numext::abs(Scalar(dot))) && (numext::isfinite)(RealScalar(nrm)));

    // Copies, release, and adoption.
    gpu::DeviceMatrix<Scalar> d_copy = d_x;
    d_copy = d_y;
    Scalar* raw = d_copy.release(ctx2);
    gpu::DeviceMatrix<Scalar> d_adopted = gpu::DeviceMatrix<Scalar>::adopt(ctx2, raw, 2 * n, 1);
    d_adopted.scale(ctx1, Scalar(2));

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
    VERIFY_IS_APPROX(Mat(A.transpose() * lu.solve(d_B, gpu::GpuOp::Trans).toHost()), B);
    VERIFY(svd.solve(d_B, Index(n / 2)).rows() == n && svd.solve(d_B, RealScalar(0.1)).rows() == n);

    // Sparse products: CSC and BSR, SpMV, SpMM, and the affine form.
    VERIFY_IS_APPROX(Mat(standalone.multiply(S, B.col(0))), Mat(A * B.col(0)));
    auto view = sparse.deviceView(S);
    gpu::DeviceMatrix<Scalar> d_b = gpu::DeviceMatrix<Scalar>::fromHost(ctx2, b);
    gpu::DeviceMatrix<Scalar> d_Sx = view * d_b;
    gpu::DeviceMatrix<Scalar> d_SX = view * d_B;
    gpu::DeviceMatrix<Scalar> d_r = d_b - view * d_Sx;
    VERIFY_IS_APPROX(Vec(d_r.toHost(ctx2)), Vec(b - A * A * b));
    VERIFY_IS_APPROX(Mat(d_SX.toHost(ctx1)), Mat(A * B));
    auto block_view = standalone.deviceView(block_identity);
    gpu::DeviceMatrix<Scalar> d_Ib = block_view * d_b;
    VERIFY_IS_APPROX(Vec(d_Ib.toHost(ctx2)), b);
    real_only_workload(ctx2, d_x, d_y, view, d_b);

    // FFT, host and device: C2C, R2C / C2R, and 2D.
    VERIFY_IS_APPROX(fft.inv(fft.fwd(z)), z);
    gpu::DeviceMatrix<Complex> d_z = gpu::DeviceMatrix<Complex>::fromHost(ctx1, z), d_Z, d_zz;
    fft.fwd(d_z, d_Z);
    fft.inv(d_Z, d_zz);
    VERIFY_IS_APPROX(d_zz.toHost(ctx1), z);
    gpu::DeviceMatrix<RealScalar> d_r64 = gpu::DeviceMatrix<RealScalar>::fromHost(ctx1, r), d_rr;
    gpu::DeviceMatrix<Complex> d_R;
    fft.fwd(d_r64, d_R);
    fft.invReal(d_R, d_rr, r.size());
    VERIFY_IS_APPROX(d_rr.toHost(ctx2), r);
    gpu::DeviceMatrix<Complex> d_Z2 = gpu::DeviceMatrix<Complex>::fromHost(ctx1, Z), d_F2, d_ZZ2;
    fft.fwd2(d_Z2, d_F2);
    fft.inv2(d_F2, d_ZZ2);
    VERIFY_IS_APPROX(d_ZZ2.toHost(ctx1), Z);
  }
};

// Runs ModuleWorkload once to create every lazily created handle, plan, and
// buffer, then again under a sentinel, from a thread of its own so that its
// thread-local Context is fresh as well.
template <typename Scalar>
void test_no_legacy_stream() {
  // The cudaMalloc/cudaFree fallback synchronizes the device, which the capture rejects.
  if (!gpu::internal::device_supports_memory_pools()) return;
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
  CALL_SUBTEST(test_read_after_write());
  CALL_SUBTEST(test_write_after_read());
  CALL_SUBTEST(test_write_after_home_stream_read());
  CALL_SUBTEST(test_free_after_read());
  CALL_SUBTEST(test_device_scalar_across_contexts());
  CALL_SUBTEST(test_solver_adopts_pending_matrix());
  CALL_SUBTEST(test_matrix_outlives_context());
  CALL_SUBTEST(test_read_marks_are_reused());
  CALL_SUBTEST(test_write_retires_read_marks());
  CALL_SUBTEST(test_concurrent_reads());
  CALL_SUBTEST(test_sentinel_detects_legacy_stream());
  CALL_SUBTEST(test_no_legacy_stream<double>());
  CALL_SUBTEST(test_no_legacy_stream<std::complex<float>>());
  CALL_SUBTEST(test_free_after_read_through_view());
  CALL_SUBTEST(test_release_after_context());
  CALL_SUBTEST(test_release_orders_pending_write());
  CALL_SUBTEST(test_write_through_view());
  CALL_SUBTEST(test_oneshot_solve_into_operand());
  CALL_SUBTEST(test_read_marks_retire_when_done());
  CALL_SUBTEST(test_direct_read_after_write());
  CALL_SUBTEST(test_write_after_direct_read());
  CALL_SUBTEST(test_direct_write_after_read());
  CALL_SUBTEST(test_direct_write_moves_stream());
  CALL_SUBTEST(test_read_after_direct_write());
  CALL_SUBTEST(test_device_scalar_direct_write());
}
