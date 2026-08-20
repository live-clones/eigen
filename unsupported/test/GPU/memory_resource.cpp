// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// Tests for gpu::MemoryResource and the DeviceMatrix overload that allocates
// through one.
//
// The property under test is that where the memory comes from is orthogonal to
// everything else: a DeviceMatrix built on managed, mapped or registered
// storage is the same type, works with the same expressions and solvers, and
// differs only in whether hostData() is non-null and what syncHost() has to do.

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
  // The default allocator is device-only and unchanged.
  gpu::DeviceMatrix<Scalar> d(n, n);
  VERIFY(!d.isHostAccessible());
  VERIFY(d.hostData() == nullptr);
  VERIFY(d.memoryResource() == nullptr);
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
  CALL_SUBTEST(test_adopted_host_matrix<Scalar>(64));
}

EIGEN_DECLARE_TEST(gpu_memory_resource) {
  gpu_test::require_cuda_device();
  CALL_SUBTEST_1(test_resource_properties());
  CALL_SUBTEST_1(test_release_rejects_resource_backed());
  CALL_SUBTEST_1(test_scalar<float>());
  CALL_SUBTEST_2(test_scalar<double>());
  CALL_SUBTEST_3(test_scalar<std::complex<float>>());
  CALL_SUBTEST_4(test_scalar<std::complex<double>>());
}
