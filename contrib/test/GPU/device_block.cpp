// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// gpu::DeviceBlock: column views of a DeviceMatrix that share its memory and
// its cross-context access ordering. The ordering tests park one context's
// stream behind a StreamGate, as in stream_ordering.cpp: an access through a
// block that is not ordered after the parked one runs first.

#define EIGEN_USE_GPU
#include "main.h"
#include <contrib/Eigen/GPU>

#include "./gpu_test_helpers.h"

using namespace Eigen;
using gpu_test::StreamGate;

namespace {

constexpr std::chrono::milliseconds kGateDelay(100);

void wait_for(gpu::Context& ctx) { EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(ctx.stream())); }

// Each ordering scenario runs twice, parking the stream only on the second
// pass, so the first creates the library handles and loads the kernels: both
// can synchronize the device, which would wait on the gate forever.
class MaybeParked {
 public:
  MaybeParked(cudaStream_t stream, bool park) : gate_(park ? new StreamGate(stream) : nullptr) {}
  void openAfter(std::chrono::milliseconds delay) {
    if (gate_) gate_->openAfter(delay);
  }

 private:
  std::unique_ptr<StreamGate> gate_;
};

}  // namespace

template <typename Scalar>
void test_block_values(Index rows, Index cols) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using DeviceMat = gpu::DeviceMatrix<Scalar>;
  gpu::Context ctx;
  const Mat A = Mat::Random(rows, cols);
  DeviceMat d_A = DeviceMat::fromHost(ctx, A);
  const Index j = cols / 2, start = cols / 3, n = cols - start - 1;

  VERIFY_IS_EQUAL(d_A.col(j).rows(), rows);
  VERIFY_IS_EQUAL(d_A.col(j).cols(), Index(1));
  VERIFY(d_A.col(j).data() == d_A.data() + j * rows);
  VERIFY_IS_EQUAL(d_A.col(j).toHost(ctx), Mat(A.col(j)));
  VERIFY_IS_EQUAL(d_A.leftCols(2).toHost(ctx), Mat(A.leftCols(2)));
  VERIFY_IS_EQUAL(d_A.middleCols(start, n).toHost(ctx), Mat(A.middleCols(start, n)));
  VERIFY_IS_EQUAL(d_A.rightCols(3).toHost(ctx), Mat(A.rightCols(3)));
  VERIFY(d_A.middleCols(start, 0).empty());
  // A block of a block is a block of the parent.
  VERIFY_IS_EQUAL(d_A.middleCols(start, n).col(1).toHost(ctx), Mat(A.col(start + 1)));

  const DeviceMat& c_A = d_A;
  VERIFY_IS_EQUAL(c_A.col(j).toHost(ctx), Mat(A.col(j)));

  // Copying a block into a DeviceMatrix makes an owning copy; copying a block
  // makes another view.
  DeviceMat copy = d_A.col(j);
  VERIFY(copy.data() != d_A.col(j).data());
  DeviceMat assigned;
  assigned = d_A.col(j);
  VERIFY(assigned.data() != d_A.col(j).data());
  auto view = d_A.col(j);
  auto view_copy = view;
  VERIFY(view_copy.data() == view.data());
  d_A.setZero(ctx);
  VERIFY_IS_EQUAL(copy.toHost(ctx), Mat(A.col(j)));
  VERIFY_IS_EQUAL(assigned.toHost(ctx), Mat(A.col(j)));
  VERIFY(view_copy.toHost(ctx).isZero(0));
}

template <typename Scalar>
void test_block_writes(Index rows, Index cols) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using Vec = Matrix<Scalar, Dynamic, 1>;
  using DeviceMat = gpu::DeviceMatrix<Scalar>;
  eigen_assert(cols >= 9);
  gpu::Context ctx;
  const Mat A = Mat::Random(rows, cols);
  const Vec x = Vec::Random(rows);
  const Mat B = Mat::Random(rows, 5), D = Mat::Random(5, 2);
  DeviceMat d_A = DeviceMat::fromHost(ctx, A);
  const DeviceMat d_x = DeviceMat::fromHost(ctx, x);
  const DeviceMat d_B = DeviceMat::fromHost(ctx, B), d_D = DeviceMat::fromHost(ctx, D);
  Mat expected = A;

  d_A.col(1).setZero(ctx);
  expected.col(1).setZero();
  d_A.col(2).scale(ctx, Scalar(2));
  expected.col(2) *= Scalar(2);
  d_A.col(3).addScaled(ctx, Scalar(3), d_x);
  expected.col(3) += Scalar(3) * x;
  // Assigning to a block copies into the parent, also from a temporary, which
  // a block cannot adopt, and from another block.
  d_A.col(4) = d_x;
  expected.col(4) = x;
  d_A.col(5) = d_x.clone(ctx);
  expected.col(5) = x;
  d_A.col(6) = d_A.col(0);
  expected.col(6) = expected.col(0);
  d_A.middleCols(7, 2).device(ctx) = d_B * d_D;
  expected.middleCols(7, 2) = B * D;

  VERIFY_IS_APPROX(d_A.toHost(ctx), expected);
}

// The classical Gram-Schmidt step of a Lanczos reorthogonalization on column
// blocks: h = V(:, 0:j)^H v, v -= V(:, 0:j) h.
template <typename Scalar>
void test_block_operands(Index n, Index k) {
  using RealScalar = typename NumTraits<Scalar>::Real;
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using Vec = Matrix<Scalar, Dynamic, 1>;
  using DeviceMat = gpu::DeviceMatrix<Scalar>;
  gpu::Context ctx;
  const Mat V = Mat::Random(n, k);
  Vec v = Vec::Random(n);
  const DeviceMat d_V = DeviceMat::fromHost(ctx, V);
  DeviceMat d_v = DeviceMat::fromHost(ctx, v);
  const Index j = k - 1;

  DeviceMat d_h;
  d_h.device(ctx) = d_V.leftCols(j).adjoint() * d_v;
  d_v.device(ctx) -= d_V.leftCols(j) * d_h;
  const Vec h = V.leftCols(j).adjoint() * v;
  v -= V.leftCols(j) * h;
  VERIFY_IS_APPROX(Vec(d_h.toHost(ctx)), h);
  VERIFY_IS_APPROX(Vec(d_v.toHost(ctx)), v);

  const Scalar dot = d_V.col(1).dot(ctx, d_V.col(2));
  VERIFY_IS_APPROX(dot, V.col(1).dot(V.col(2)));
  const RealScalar norm = d_V.col(3).norm(ctx);
  VERIFY_IS_APPROX(norm, V.col(3).norm());
}

// A read of the parent on another context waits for a write through a block.
void test_parent_read_after_block_write() {
  using DeviceMat = gpu::DeviceMatrix<double>;
  gpu::Context writer, reader;
  const MatrixXd A = MatrixXd::Random(4096, 4);
  for (bool park : {false, true}) {
    DeviceMat d_A = DeviceMat::fromHost(writer, A);
    DeviceMat d_B;
    {
      MaybeParked gate(writer.stream(), park);
      d_A.col(2).scale(writer, 2.0);
      d_B = d_A.clone(reader);
      gate.openAfter(kGateDelay);
      wait_for(reader);
    }
    MatrixXd expected = A;
    expected.col(2) *= 2.0;
    VERIFY_IS_APPROX(d_B.toHost(reader), expected);
  }
}

// A write through one block waits for a pending read through another block on
// another context: blocks share the parent's access ordering.
void test_block_write_after_block_read() {
  using DeviceMat = gpu::DeviceMatrix<double>;
  gpu::Context writer, reader;
  const MatrixXd A = MatrixXd::Random(4096, 4);
  for (bool park : {false, true}) {
    DeviceMat d_A = DeviceMat::fromHost(writer, A);
    DeviceMat d_c;
    {
      MaybeParked gate(reader.stream(), park);
      d_c = d_A.col(0).clone(reader);
      d_A.leftCols(2).setZero(writer);
      gate.openAfter(kGateDelay);
      wait_for(writer);
    }
    VERIFY_IS_APPROX(d_c.toHost(reader), MatrixXd(A.col(0)));
    VERIFY(d_A.leftCols(2).toHost(writer).isZero(0));
  }
}

void test_block_cannot_own() {
  gpu::Context ctx;
  gpu::DeviceMatrix<double> d_A = gpu::DeviceMatrix<double>::fromHost(ctx, MatrixXd::Random(8, 4));
  auto block = d_A.col(1);
  // Safe here where the module otherwise avoids VERIFY_RAISES_ASSERT: both
  // assertions fire before any RAII object is constructed.
  VERIFY_RAISES_ASSERT(block.resize(ctx, 9, 1));
  VERIFY_RAISES_ASSERT(gpu::internal::DeviceMatrixAccess::take(block));
  VERIFY_IS_EQUAL(block.rows(), Index(8));
  VERIFY(block.data() == d_A.data() + 8);
}

template <typename Scalar>
void test_scalar() {
  CALL_SUBTEST(test_block_values<Scalar>(64, 12));
  CALL_SUBTEST(test_block_writes<Scalar>(64, 12));
  CALL_SUBTEST(test_block_operands<Scalar>(500, 20));
}

EIGEN_DECLARE_TEST(gpu_device_block) {
  gpu_test::require_cuda_device();
  CALL_SUBTEST_1(test_scalar<float>());
  CALL_SUBTEST_1(test_scalar<double>());
  CALL_SUBTEST_2(test_scalar<std::complex<float>>());
  CALL_SUBTEST_2(test_scalar<std::complex<double>>());
  CALL_SUBTEST_3(test_parent_read_after_block_write());
  CALL_SUBTEST_3(test_block_write_after_block_read());
  CALL_SUBTEST_3(test_block_cannot_own());
}
