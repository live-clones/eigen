// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// cuSOLVER-specific support types. Generic CUDA runtime utilities (DeviceBuffer,
// EIGEN_CUDA_RUNTIME_CHECK) live in GpuSupport.h.

#ifndef EIGEN_GPU_CUSOLVER_SUPPORT_H
#define EIGEN_GPU_CUSOLVER_SUPPORT_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

#include "./GpuSupport.h"
#include <cusolverDn.h>
#include <cstdio>

namespace Eigen {
namespace gpu {
namespace internal {

// cuSOLVER's public API has no cusolverGetErrorString(), so failed asserts would
// otherwise carry a bare numeric code.
inline const char* cusolver_status_name(cusolverStatus_t s) {
  switch (s) {
    case CUSOLVER_STATUS_SUCCESS:
      return "CUSOLVER_STATUS_SUCCESS";
    case CUSOLVER_STATUS_NOT_INITIALIZED:
      return "CUSOLVER_STATUS_NOT_INITIALIZED";
    case CUSOLVER_STATUS_ALLOC_FAILED:
      return "CUSOLVER_STATUS_ALLOC_FAILED";
    case CUSOLVER_STATUS_INVALID_VALUE:
      return "CUSOLVER_STATUS_INVALID_VALUE";
    case CUSOLVER_STATUS_ARCH_MISMATCH:
      return "CUSOLVER_STATUS_ARCH_MISMATCH";
    case CUSOLVER_STATUS_EXECUTION_FAILED:
      return "CUSOLVER_STATUS_EXECUTION_FAILED";
    case CUSOLVER_STATUS_INTERNAL_ERROR:
      return "CUSOLVER_STATUS_INTERNAL_ERROR";
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    case CUSOLVER_STATUS_NOT_SUPPORTED:
      return "CUSOLVER_STATUS_NOT_SUPPORTED";
    default:
      return "CUSOLVER_STATUS_UNKNOWN";
  }
}

inline bool report_cusolver_failure(cusolverStatus_t s, const char* expr, const char* file, int line) {
  std::fprintf(stderr,
               "cuSOLVER call failed\n"
               "  expr:   %s\n"
               "  status: %s (%d)\n"
               "  at:     %s:%d\n",
               expr, cusolver_status_name(s), static_cast<int>(s), file, line);
  return false;
}

#define EIGEN_CUSOLVER_CHECK(expr)                                                                \
  do {                                                                                            \
    cusolverStatus_t _s = (expr);                                                                 \
    eigen_assert(_s == CUSOLVER_STATUS_SUCCESS ||                                                 \
                 ::Eigen::gpu::internal::report_cusolver_failure(_s, #expr, __FILE__, __LINE__)); \
  } while (0)

struct CusolverParams {
  cusolverDnParams_t p = nullptr;

  CusolverParams() { EIGEN_CUSOLVER_CHECK(cusolverDnCreateParams(&p)); }

  ~CusolverParams() {
    if (p) (void)cusolverDnDestroyParams(p);  // noexcept context: cannot propagate
  }

  CusolverParams(CusolverParams&& o) noexcept : p(o.p) { o.p = nullptr; }
  CusolverParams& operator=(CusolverParams&& o) noexcept {
    if (this != &o) {
      if (p) (void)cusolverDnDestroyParams(p);  // noexcept context: cannot propagate
      p = o.p;
      o.p = nullptr;
    }
    return *this;
  }

  CusolverParams(const CusolverParams&) = delete;
  CusolverParams& operator=(const CusolverParams&) = delete;
};

// RAII cuSOLVER dense handle; the ownership flag supports handles borrowed from a gpu::Context.
struct CusolverHandleDeleter {
  bool owns = true;
  void operator()(cusolverDnHandle_t h) const noexcept {
    if (owns && h) (void)cusolverDnDestroy(h);
  }
};
using UniqueCusolverHandle = std::unique_ptr<std::remove_pointer_t<cusolverDnHandle_t>, CusolverHandleDeleter>;

// Alias kept for compatibility; cuda_data_type<> in GpuSupport.h is canonical.
template <typename Scalar>
using cusolver_data_type = cuda_data_type<Scalar>;

// cuSOLVER always interprets the matrix as column-major, so callers must pass the
// triangle that holds the data in column-major layout.
template <int UpLo>
struct cusolver_fill_mode;

template <>
struct cusolver_fill_mode<Lower> {
  static constexpr cublasFillMode_t value = CUBLAS_FILL_MODE_LOWER;
};
template <>
struct cusolver_fill_mode<Upper> {
  static constexpr cublasFillMode_t value = CUBLAS_FILL_MODE_UPPER;
};

// The Bunch-Kaufman routines factor real symmetric or complex-symmetric (A = A^T)
// matrices; cuSOLVER has no Hermitian hetrf. A complex Scalar therefore has to
// name that choice with Eigen's Symmetric mode bit (Lower | Symmetric), so a
// Hermitian matrix cannot be handed over by mistake.
template <typename Scalar, int UpLo>
struct ldlt_fill_mode {
  static constexpr int triangle = UpLo & (Lower | Upper);
  static_assert(triangle == Lower || triangle == Upper,
                "gpu::LDLT: the mode must contain exactly one of Lower or Upper");
  static_assert((UpLo & ~(Lower | Upper | Symmetric)) == 0,
                "gpu::LDLT: only the Lower, Upper and Symmetric mode bits apply");
  static_assert(!NumTraits<Scalar>::IsComplex || (UpLo & Symmetric) != 0,
                "gpu::LDLT factors complex-symmetric (A = A^T) matrices, not Hermitian ones: opt in with "
                "Lower | Symmetric or Upper | Symmetric, or use gpu::LU for a Hermitian indefinite matrix");
  static constexpr cublasFillMode_t value = cusolver_fill_mode<triangle>::value;
};

// cuSOLVER ships no generic X variant for ormqr/unmqr, so these overloads supply
// one: real → ormqr (orthogonal Q), complex → unmqr (unitary Q).
inline cusolverStatus_t cusolverDnXormqr(cusolverDnHandle_t h, cublasSideMode_t side, cublasOperation_t trans, int m,
                                         int n, int k, const float* A, int lda, const float* tau, float* C, int ldc,
                                         float* work, int lwork, int* info) {
  return cusolverDnSormqr(h, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info);
}
inline cusolverStatus_t cusolverDnXormqr(cusolverDnHandle_t h, cublasSideMode_t side, cublasOperation_t trans, int m,
                                         int n, int k, const double* A, int lda, const double* tau, double* C, int ldc,
                                         double* work, int lwork, int* info) {
  return cusolverDnDormqr(h, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info);
}
inline cusolverStatus_t cusolverDnXormqr(cusolverDnHandle_t h, cublasSideMode_t side, cublasOperation_t trans, int m,
                                         int n, int k, const std::complex<float>* A, int lda,
                                         const std::complex<float>* tau, std::complex<float>* C, int ldc,
                                         std::complex<float>* work, int lwork, int* info) {
  return cusolverDnCunmqr(h, side, trans, m, n, k, reinterpret_cast<const cuComplex*>(A), lda,
                          reinterpret_cast<const cuComplex*>(tau), reinterpret_cast<cuComplex*>(C), ldc,
                          reinterpret_cast<cuComplex*>(work), lwork, info);
}
inline cusolverStatus_t cusolverDnXormqr(cusolverDnHandle_t h, cublasSideMode_t side, cublasOperation_t trans, int m,
                                         int n, int k, const std::complex<double>* A, int lda,
                                         const std::complex<double>* tau, std::complex<double>* C, int ldc,
                                         std::complex<double>* work, int lwork, int* info) {
  return cusolverDnZunmqr(h, side, trans, m, n, k, reinterpret_cast<const cuDoubleComplex*>(A), lda,
                          reinterpret_cast<const cuDoubleComplex*>(tau), reinterpret_cast<cuDoubleComplex*>(C), ldc,
                          reinterpret_cast<cuDoubleComplex*>(work), lwork, info);
}

inline cusolverStatus_t cusolverDnXormqr_bufferSize(cusolverDnHandle_t h, cublasSideMode_t side,
                                                    cublasOperation_t trans, int m, int n, int k, const float* A,
                                                    int lda, const float* tau, const float* C, int ldc, int* lwork) {
  return cusolverDnSormqr_bufferSize(h, side, trans, m, n, k, A, lda, tau, C, ldc, lwork);
}
inline cusolverStatus_t cusolverDnXormqr_bufferSize(cusolverDnHandle_t h, cublasSideMode_t side,
                                                    cublasOperation_t trans, int m, int n, int k, const double* A,
                                                    int lda, const double* tau, const double* C, int ldc, int* lwork) {
  return cusolverDnDormqr_bufferSize(h, side, trans, m, n, k, A, lda, tau, C, ldc, lwork);
}
inline cusolverStatus_t cusolverDnXormqr_bufferSize(cusolverDnHandle_t h, cublasSideMode_t side,
                                                    cublasOperation_t trans, int m, int n, int k,
                                                    const std::complex<float>* A, int lda,
                                                    const std::complex<float>* tau, const std::complex<float>* C,
                                                    int ldc, int* lwork) {
  return cusolverDnCunmqr_bufferSize(h, side, trans, m, n, k, reinterpret_cast<const cuComplex*>(A), lda,
                                     reinterpret_cast<const cuComplex*>(tau), reinterpret_cast<const cuComplex*>(C),
                                     ldc, lwork);
}
inline cusolverStatus_t cusolverDnXormqr_bufferSize(cusolverDnHandle_t h, cublasSideMode_t side,
                                                    cublasOperation_t trans, int m, int n, int k,
                                                    const std::complex<double>* A, int lda,
                                                    const std::complex<double>* tau, const std::complex<double>* C,
                                                    int ldc, int* lwork) {
  return cusolverDnZunmqr_bufferSize(h, side, trans, m, n, k, reinterpret_cast<const cuDoubleComplex*>(A), lda,
                                     reinterpret_cast<const cuDoubleComplex*>(tau),
                                     reinterpret_cast<const cuDoubleComplex*>(C), ldc, lwork);
}

// The Bunch-Kaufman routines exist only in cuSOLVER's legacy 32-bit API, so these
// overloads supply generic entry points the same way. The complex variants factor
// complex-symmetric matrices (A = A^T), like LAPACK's csytrf/zsytrf; cuSOLVER has
// no Hermitian hetrf.
inline cusolverStatus_t cusolverDnXsytrf_bufferSize(cusolverDnHandle_t h, int n, float* A, int lda, int* lwork) {
  return cusolverDnSsytrf_bufferSize(h, n, A, lda, lwork);
}
inline cusolverStatus_t cusolverDnXsytrf_bufferSize(cusolverDnHandle_t h, int n, double* A, int lda, int* lwork) {
  return cusolverDnDsytrf_bufferSize(h, n, A, lda, lwork);
}
inline cusolverStatus_t cusolverDnXsytrf_bufferSize(cusolverDnHandle_t h, int n, std::complex<float>* A, int lda,
                                                    int* lwork) {
  return cusolverDnCsytrf_bufferSize(h, n, reinterpret_cast<cuComplex*>(A), lda, lwork);
}
inline cusolverStatus_t cusolverDnXsytrf_bufferSize(cusolverDnHandle_t h, int n, std::complex<double>* A, int lda,
                                                    int* lwork) {
  return cusolverDnZsytrf_bufferSize(h, n, reinterpret_cast<cuDoubleComplex*>(A), lda, lwork);
}

inline cusolverStatus_t cusolverDnXsytrf(cusolverDnHandle_t h, cublasFillMode_t uplo, int n, float* A, int lda,
                                         int* ipiv, float* work, int lwork, int* info) {
  return cusolverDnSsytrf(h, uplo, n, A, lda, ipiv, work, lwork, info);
}
inline cusolverStatus_t cusolverDnXsytrf(cusolverDnHandle_t h, cublasFillMode_t uplo, int n, double* A, int lda,
                                         int* ipiv, double* work, int lwork, int* info) {
  return cusolverDnDsytrf(h, uplo, n, A, lda, ipiv, work, lwork, info);
}
inline cusolverStatus_t cusolverDnXsytrf(cusolverDnHandle_t h, cublasFillMode_t uplo, int n, std::complex<float>* A,
                                         int lda, int* ipiv, std::complex<float>* work, int lwork, int* info) {
  return cusolverDnCsytrf(h, uplo, n, reinterpret_cast<cuComplex*>(A), lda, ipiv, reinterpret_cast<cuComplex*>(work),
                          lwork, info);
}
inline cusolverStatus_t cusolverDnXsytrf(cusolverDnHandle_t h, cublasFillMode_t uplo, int n, std::complex<double>* A,
                                         int lda, int* ipiv, std::complex<double>* work, int lwork, int* info) {
  return cusolverDnZsytrf(h, uplo, n, reinterpret_cast<cuDoubleComplex*>(A), lda, ipiv,
                          reinterpret_cast<cuDoubleComplex*>(work), lwork, info);
}

inline cusolverStatus_t cusolverDnXsytri_bufferSize(cusolverDnHandle_t h, cublasFillMode_t uplo, int n, float* A,
                                                    int lda, const int* ipiv, int* lwork) {
  return cusolverDnSsytri_bufferSize(h, uplo, n, A, lda, ipiv, lwork);
}
inline cusolverStatus_t cusolverDnXsytri_bufferSize(cusolverDnHandle_t h, cublasFillMode_t uplo, int n, double* A,
                                                    int lda, const int* ipiv, int* lwork) {
  return cusolverDnDsytri_bufferSize(h, uplo, n, A, lda, ipiv, lwork);
}
inline cusolverStatus_t cusolverDnXsytri_bufferSize(cusolverDnHandle_t h, cublasFillMode_t uplo, int n,
                                                    std::complex<float>* A, int lda, const int* ipiv, int* lwork) {
  return cusolverDnCsytri_bufferSize(h, uplo, n, reinterpret_cast<cuComplex*>(A), lda, ipiv, lwork);
}
inline cusolverStatus_t cusolverDnXsytri_bufferSize(cusolverDnHandle_t h, cublasFillMode_t uplo, int n,
                                                    std::complex<double>* A, int lda, const int* ipiv, int* lwork) {
  return cusolverDnZsytri_bufferSize(h, uplo, n, reinterpret_cast<cuDoubleComplex*>(A), lda, ipiv, lwork);
}

inline cusolverStatus_t cusolverDnXsytri(cusolverDnHandle_t h, cublasFillMode_t uplo, int n, float* A, int lda,
                                         const int* ipiv, float* work, int lwork, int* info) {
  return cusolverDnSsytri(h, uplo, n, A, lda, ipiv, work, lwork, info);
}
inline cusolverStatus_t cusolverDnXsytri(cusolverDnHandle_t h, cublasFillMode_t uplo, int n, double* A, int lda,
                                         const int* ipiv, double* work, int lwork, int* info) {
  return cusolverDnDsytri(h, uplo, n, A, lda, ipiv, work, lwork, info);
}
inline cusolverStatus_t cusolverDnXsytri(cusolverDnHandle_t h, cublasFillMode_t uplo, int n, std::complex<float>* A,
                                         int lda, const int* ipiv, std::complex<float>* work, int lwork, int* info) {
  return cusolverDnCsytri(h, uplo, n, reinterpret_cast<cuComplex*>(A), lda, ipiv, reinterpret_cast<cuComplex*>(work),
                          lwork, info);
}
inline cusolverStatus_t cusolverDnXsytri(cusolverDnHandle_t h, cublasFillMode_t uplo, int n, std::complex<double>* A,
                                         int lda, const int* ipiv, std::complex<double>* work, int lwork, int* info) {
  return cusolverDnZsytri(h, uplo, n, reinterpret_cast<cuDoubleComplex*>(A), lda, ipiv,
                          reinterpret_cast<cuDoubleComplex*>(work), lwork, info);
}

}  // namespace internal
}  // namespace gpu
}  // namespace Eigen

#endif  // EIGEN_GPU_CUSOLVER_SUPPORT_H
