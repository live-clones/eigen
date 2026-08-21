// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// Tests for gpu::MemoryResource and the types that allocate through one.
//
// The property under test is that where the memory comes from is orthogonal to
// everything else: a DeviceMatrix built on managed, mapped or registered
// storage is the same type, works with the same expressions and solvers, and
// differs only in whether hostData() is non-null and what syncHost() has to do.
// Because the resource lives in the buffer both types share, the same holds for
// a DeviceScalar and for the reductions that produce one.

#define EIGEN_USE_GPU
#include "main.h"
#include <unsupported/Eigen/GPU>

#include "./gpu_test_helpers.h"

using namespace Eigen;

namespace {

// Every host-accessible resource. The default device resource is covered
// separately since it has no host pointer.
std::vector<gpu::MemoryResource*> hostAccessibleResources() {
  return {&gpu::managedMemoryResource(), &gpu::mappedHostMemoryResource(), &gpu::registeredHostMemoryResource()};
}

}  // namespace

// ---- The resources describe themselves consistently ------------------------

void test_resource_properties() {
  VERIFY(!gpu::deviceMemoryResource().isHostAccessible());
  VERIFY(!gpu::pooledDeviceMemoryResource().isHostAccessible());
  for (gpu::MemoryResource* r : hostAccessibleResources()) {
    VERIFY(r->isHostAccessible());
    VERIFY(r->name() != nullptr);
  }
  // Managed memory is the one kind whose host access is restricted, and only on
  // a device that says so.
  VERIFY_IS_EQUAL(gpu::managedMemoryResource().allowsConcurrentHostAccess(),
                  gpu::deviceSupportsConcurrentManagedAccess());
  VERIFY(gpu::mappedHostMemoryResource().allowsConcurrentHostAccess());
  VERIFY(gpu::registeredHostMemoryResource().allowsConcurrentHostAccess());
}

// ---- Allocation round trip through each resource ---------------------------

template <typename Scalar>
void test_allocate_and_free(Index n) {
  for (gpu::MemoryResource* r : hostAccessibleResources()) {
    gpu::DeviceMatrix<Scalar> m(n, n, *r);
    VERIFY_IS_EQUAL(m.rows(), n);
    VERIFY(m.data() != nullptr);
    VERIFY(m.isHostAccessible());
    VERIFY(m.hostData() != nullptr);
    VERIFY(m.memoryResource() == r);
    // Writable from the host without any transfer.
    Eigen::Map<Eigen::Matrix<Scalar, Dynamic, Dynamic>>(m.hostData(), n, n).setConstant(Scalar(1));
  }
  // Naming no resource is naming the default one, not an absence.
  gpu::DeviceMatrix<Scalar> d(n, n);
  VERIFY(!d.isHostAccessible());
  VERIFY(d.hostData() == nullptr);
  VERIFY(d.memoryResource() == &gpu::deviceMemoryResource());
}

// ---- The storage kind is orthogonal to the expression layer -----------------

template <typename Scalar>
void test_gemm_on_each_resource(Index n) {
  using MatrixType = Eigen::Matrix<Scalar, Dynamic, Dynamic>;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using HostMap = Eigen::Map<MatrixType>;

  const MatrixType hA = MatrixType::Random(n, n);
  const MatrixType hB = MatrixType::Random(n, n);
  const MatrixType expected = hA * hB;
  const RealScalar tol = RealScalar(20) * RealScalar(n) * NumTraits<Scalar>::epsilon();

  for (gpu::MemoryResource* r : hostAccessibleResources()) {
    gpu::Context ctx;
    gpu::DeviceMatrix<Scalar> A(n, n, *r), B(n, n, *r), C(n, n, *r);
    HostMap(A.hostData(), n, n) = hA;  // filled in place: no upload
    HostMap(B.hostData(), n, n) = hB;

    C.device(ctx) = A * B;
    C.syncHost(ctx);

    VERIFY((HostMap(C.hostData(), n, n) - expected).norm() / (hA.norm() * hB.norm()) < tol);
  }
}

// A solver must work on resource-backed storage too, not just BLAS-3.
template <typename Scalar>
void test_llt_on_resource(Index n) {
  using MatrixType = Eigen::Matrix<Scalar, Dynamic, Dynamic>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  const MatrixType M = MatrixType::Random(n, n);
  const MatrixType spd = M.adjoint() * M + MatrixType::Identity(n, n) * static_cast<Scalar>(n);

  gpu::DeviceMatrix<Scalar> A(n, n, gpu::mappedHostMemoryResource());
  Eigen::Map<MatrixType>(A.hostData(), n, n) = spd;

  gpu::LLT<Scalar> llt;
  llt.compute(A);
  VERIFY_IS_EQUAL(llt.info(), Success);

  const MatrixType B = MatrixType::Random(n, 3);
  const MatrixType X = llt.solve(B);
  const RealScalar tol = RealScalar(20) * RealScalar(n) * NumTraits<Scalar>::epsilon();
  VERIFY((spd * X - B).norm() / B.norm() < tol);
}

// A solver can take ownership of resource-backed storage. releaseBuffer() hands
// the resource across with the block, so factoring a matrix in place no longer
// requires it to be device-only, and an in-place solve gives the caller back a
// result on the resource it supplied.
template <typename Scalar>
void test_solver_adopts_resource_backed(Index n) {
  using MatrixType = Eigen::Matrix<Scalar, Dynamic, Dynamic>;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using HostMap = Eigen::Map<MatrixType>;

  const MatrixType M = MatrixType::Random(n, n);
  const MatrixType spd = M.adjoint() * M + MatrixType::Identity(n, n) * static_cast<Scalar>(n);
  const MatrixType hb = MatrixType::Random(n, 1);
  const RealScalar tol = RealScalar(20) * RealScalar(n) * NumTraits<Scalar>::epsilon();

  gpu::Context ctx;
  gpu::MemoryResource& r = gpu::mappedHostMemoryResource();

  gpu::DeviceMatrix<Scalar> A(n, n, r);
  HostMap(A.hostData(), n, n) = spd;

  gpu::LLT<Scalar> llt;
  llt.compute(std::move(A));  // factored in place in A's page-locked storage
  VERIFY_IS_EQUAL(llt.info(), Success);
  VERIFY(A.data() == nullptr);

  gpu::DeviceMatrix<Scalar> d_b(n, 1, r);
  HostMap(d_b.hostData(), n, 1) = hb;

  gpu::DeviceMatrix<Scalar> d_x = llt.solve(std::move(d_b));
  VERIFY(d_x.isHostAccessible());
  VERIFY(d_x.memoryResource() == &r);

  d_x.syncHost(ctx);
  VERIFY((spd * HostMap(d_x.hostData(), n, 1) - hb).norm() / hb.norm() < tol);
}

// ---- Adopting a matrix the caller already has -------------------------------

template <typename Scalar>
void test_adopted_host_matrix(Index n) {
  using MatrixType = Eigen::Matrix<Scalar, Dynamic, Dynamic>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  MatrixType hA = MatrixType::Random(n, n);
  const MatrixType hB = MatrixType::Random(n, n);
  const MatrixType expected = hA * hB;
  const Scalar* original_storage = hA.data();

  gpu::HostMatrixResource<Scalar> adopted(std::move(hA));
  // Moved in, not copied: the resource serves the caller's original storage.
  VERIFY(adopted.matrix().data() == original_storage);
  VERIFY_IS_EQUAL(adopted.rows(), n);

  gpu::Context ctx;
  gpu::DeviceMatrix<Scalar> A(n, n, adopted);
  VERIFY(A.hostData() == original_storage);

  gpu::DeviceMatrix<Scalar> B(n, n, gpu::mappedHostMemoryResource());
  Eigen::Map<MatrixType>(B.hostData(), n, n) = hB;

  gpu::DeviceMatrix<Scalar> C(n, n, gpu::mappedHostMemoryResource());
  C.device(ctx) = A * B;
  C.syncHost(ctx);

  const RealScalar tol = RealScalar(20) * RealScalar(n) * NumTraits<Scalar>::epsilon();
  VERIFY((Eigen::Map<MatrixType>(C.hostData(), n, n) - expected).norm() / expected.norm() < tol);
}

// ---- The same storage kinds, one type down ---------------------------------

// A DeviceScalar allocates through the buffer DeviceMatrix uses, so it reaches
// every resource too. On a host-accessible one the value is written where the
// host can already address it and get() drops its device-to-host copy.
template <typename Scalar>
void test_device_scalar_on_each_resource() {
  gpu::Context ctx;
  const Scalar value = Scalar(3);

  for (gpu::MemoryResource* r : hostAccessibleResources()) {
    gpu::DeviceScalar<Scalar> s(value, *r, ctx.stream());
    VERIFY(s.isHostAccessible());
    VERIFY(s.hostData() != nullptr);
    VERIFY(&s.memoryResource() == r);
    VERIFY_IS_EQUAL(s.get(), value);
    // Same bytes both ways: the device pointer is an alias, not a copy.
    VERIFY_IS_EQUAL(*s.hostData(), value);
  }

  // Naming no resource gives the module default: pooled device storage.
  gpu::DeviceScalar<Scalar> d(value, ctx.stream());
  VERIFY(!d.isHostAccessible());
  VERIFY(d.hostData() == nullptr);
  VERIFY(&d.memoryResource() == &gpu::pooledDeviceMemoryResource());
  VERIFY_IS_EQUAL(d.get(), value);
}

// A reduction asked for a host-accessible result gives one, and it is right.
template <typename Scalar>
void test_reduction_result_on_resource(Index n) {
  using MatrixType = Eigen::Matrix<Scalar, Dynamic, Dynamic>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  const MatrixType hx = MatrixType::Random(n, 1);
  const MatrixType hy = MatrixType::Random(n, 1);
  const RealScalar tol = RealScalar(20) * RealScalar(n) * NumTraits<Scalar>::epsilon();

  gpu::Context ctx;
  gpu::DeviceMatrix<Scalar> x = gpu::DeviceMatrix<Scalar>::fromHost(hx, ctx.stream());
  gpu::DeviceMatrix<Scalar> y = gpu::DeviceMatrix<Scalar>::fromHost(hy, ctx.stream());

  for (gpu::MemoryResource* r : hostAccessibleResources()) {
    gpu::DeviceScalar<Scalar> d_dot = x.dot(ctx, y, *r);
    VERIFY(d_dot.isHostAccessible());
    VERIFY(&d_dot.memoryResource() == r);
    VERIFY_IS_APPROX(Scalar(d_dot), Scalar(hx.col(0).dot(hy.col(0))));

    gpu::DeviceScalar<RealScalar> d_norm = x.norm(ctx, *r);
    VERIFY(d_norm.isHostAccessible());
    VERIFY(numext::abs(RealScalar(d_norm) - hx.norm()) < tol * hx.norm());

    gpu::DeviceScalar<RealScalar> d_sq = x.squaredNorm(ctx, *r);
    VERIFY(d_sq.isHostAccessible());
    VERIFY(numext::abs(RealScalar(d_sq) - hx.squaredNorm()) < tol * hx.squaredNorm());
  }

  // Naming the resource is optional, and leaving it out is the old behaviour:
  // a device-only result from the module's pooled storage. Both the explicit
  // Context form and the thread-local one default the same way.
  gpu::DeviceScalar<Scalar> plain = x.dot(ctx, y);
  VERIFY(!plain.isHostAccessible());
  VERIFY(&plain.memoryResource() == &gpu::pooledDeviceMemoryResource());
  VERIFY_IS_APPROX(Scalar(plain), Scalar(hx.col(0).dot(hy.col(0))));
  VERIFY(!x.norm().isHostAccessible());
  VERIFY(!x.squaredNorm(ctx).isHostAccessible());
}

// Device-side scalar arithmetic keeps the result where its left operand lives,
// so a whole chain stays readable without a copy. Real types only: the NPP
// helpers behind operator/ and unary minus do not cover complex.
template <typename Scalar>
void test_device_scalar_arithmetic_keeps_resource() {
  gpu::Context ctx;
  gpu::MemoryResource& r = gpu::mappedHostMemoryResource();

  gpu::DeviceScalar<Scalar> a(Scalar(6), r, ctx.stream());
  gpu::DeviceScalar<Scalar> b(Scalar(4), r, ctx.stream());

  gpu::DeviceScalar<Scalar> q = a / b;
  VERIFY(q.isHostAccessible());
  VERIFY(&q.memoryResource() == &r);
  VERIFY_IS_APPROX(Scalar(q), Scalar(1.5));

  gpu::DeviceScalar<Scalar> neg = -a;
  VERIFY(neg.isHostAccessible());
  VERIFY_IS_APPROX(Scalar(neg), Scalar(-6));

  // A device-only operand still yields device-only storage.
  gpu::DeviceScalar<Scalar> c(Scalar(6), ctx.stream());
  VERIFY(!(-c).isHostAccessible());
}

// ---- Ownership guards -------------------------------------------------------

// Resource-backed storage must go back to its resource. release() hands out a
// bare pointer that the adopting overloads free with device_free, which is
// right only for the default allocator.
void test_release_rejects_resource_backed() {
  gpu::DeviceMatrix<double> m(8, 8, gpu::mappedHostMemoryResource());
  VERIFY_RAISES_ASSERT(m.release());
}

template <typename Scalar>
void test_scalar() {
  CALL_SUBTEST(test_allocate_and_free<Scalar>(64));
  CALL_SUBTEST(test_gemm_on_each_resource<Scalar>(64));
  CALL_SUBTEST(test_gemm_on_each_resource<Scalar>(128));
  CALL_SUBTEST(test_llt_on_resource<Scalar>(64));
  CALL_SUBTEST(test_solver_adopts_resource_backed<Scalar>(64));
  CALL_SUBTEST(test_adopted_host_matrix<Scalar>(64));
  CALL_SUBTEST(test_device_scalar_on_each_resource<Scalar>());
  CALL_SUBTEST(test_reduction_result_on_resource<Scalar>(97));
}

EIGEN_DECLARE_TEST(gpu_memory_resource) {
  gpu_test::require_cuda_device();
  CALL_SUBTEST_1(test_resource_properties());
  CALL_SUBTEST_1(test_release_rejects_resource_backed());
  CALL_SUBTEST_1(test_device_scalar_arithmetic_keeps_resource<float>());
  CALL_SUBTEST_2(test_device_scalar_arithmetic_keeps_resource<double>());
  CALL_SUBTEST_1(test_scalar<float>());
  CALL_SUBTEST_2(test_scalar<double>());
  CALL_SUBTEST_3(test_scalar<std::complex<float>>());
  CALL_SUBTEST_4(test_scalar<std::complex<double>>());
}
