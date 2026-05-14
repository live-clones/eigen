// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// cuBLAS-specific support types:
//   - Error-checking macro
//   - Operation enum and mapping to cublasOperation_t
//
// Generic CUDA runtime utilities (DeviceBuffer, cuda_data_type) are in GpuSupport.h.

#ifndef EIGEN_GPU_CUBLAS_SUPPORT_H
#define EIGEN_GPU_CUBLAS_SUPPORT_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

#include "./GpuSupport.h"
#include <cublas_v2.h>
#include <cublasLt.h>
#include <climits>
#include <cstring>

namespace Eigen {
namespace gpu {
namespace internal {

// ---- Error-checking macro ---------------------------------------------------

#define EIGEN_CUBLAS_CHECK(expr)                                       \
  do {                                                                 \
    cublasStatus_t _s = (expr);                                        \
    eigen_assert(_s == CUBLAS_STATUS_SUCCESS && "cuBLAS call failed"); \
  } while (0)

constexpr cublasOperation_t to_cublas_op(GpuOp op) {
  switch (op) {
    case GpuOp::Trans:
      return CUBLAS_OP_T;
    case GpuOp::ConjTrans:
      return CUBLAS_OP_C;
    default:
      return CUBLAS_OP_N;
  }
}

// ---- Scalar → cublasComputeType_t -------------------------------------------
// cublasLtMatmul requires a compute type (separate from the data type).
//
// Precision policy:
//   - Default: tensor core algorithms enabled via cublasLtMatmul heuristics.
//     For double, cuBLAS may use Ozaki emulation on sm_80+ tensor cores.
//   - EIGEN_CUDA_TF32: opt-in to TF32 for float (~2x faster, 10-bit mantissa).
//   - EIGEN_NO_CUDA_TENSOR_OPS: disables all tensor core usage. Uses pedantic
//     compute types. For bit-exact reproducibility.

// Single-precision (real or complex) and double-precision (real or complex) each
// pick their compute type from one set of preprocessor switches; specializations
// just dispatch to the right precision tag.
namespace cuda_compute_type_detail {
#if defined(EIGEN_NO_CUDA_TENSOR_OPS)
constexpr cublasComputeType_t kFloat = CUBLAS_COMPUTE_32F_PEDANTIC;
constexpr cublasComputeType_t kDouble = CUBLAS_COMPUTE_64F_PEDANTIC;
#elif defined(EIGEN_CUDA_TF32)
constexpr cublasComputeType_t kFloat = CUBLAS_COMPUTE_32F_FAST_TF32;
constexpr cublasComputeType_t kDouble = CUBLAS_COMPUTE_64F;
#else
constexpr cublasComputeType_t kFloat = CUBLAS_COMPUTE_32F;
constexpr cublasComputeType_t kDouble = CUBLAS_COMPUTE_64F;
#endif
}  // namespace cuda_compute_type_detail

template <typename Scalar>
struct cuda_compute_type;

template <>
struct cuda_compute_type<float> {
  static constexpr cublasComputeType_t value = cuda_compute_type_detail::kFloat;
};
template <>
struct cuda_compute_type<double> {
  static constexpr cublasComputeType_t value = cuda_compute_type_detail::kDouble;
};
template <>
struct cuda_compute_type<std::complex<float>> {
  static constexpr cublasComputeType_t value = cuda_compute_type_detail::kFloat;
};
template <>
struct cuda_compute_type<std::complex<double>> {
  static constexpr cublasComputeType_t value = cuda_compute_type_detail::kDouble;
};

// ---- cublasLt GEMM dispatch -------------------------------------------------
// Wraps cublasLtMatmul with descriptor setup, heuristic algorithm selection,
// and a small per-context plan cache. Supports 64-bit dimensions natively.
//
// The workspace buffer (DeviceBuffer*) is grown lazily to match the selected
// algorithm's actual requirement. Growth is monotonic.
//
// EIGEN_NO_CUDA_TENSOR_OPS: pedantic compute types prevent cublasLt from
// selecting tensor core algorithms, matching the previous cublasGemmEx behavior.
//
// Thread safety: the workspace buffer and plan cache are not thread-safe. All
// GEMM calls sharing them must be on the same CUDA stream (guaranteed by
// gpu::Context's single-stream design and by each solver owning its own stream).

#define EIGEN_CUBLASLT_CHECK(expr)                                       \
  do {                                                                   \
    cublasStatus_t _s = (expr);                                          \
    eigen_assert(_s == CUBLAS_STATUS_SUCCESS && "cuBLASLt call failed"); \
  } while (0)

// Maximum workspace the heuristic is allowed to consider. This is a preference
// ceiling, not an allocation — actual allocation matches the selected algorithm.
static constexpr size_t kCublasLtMaxWorkspaceBytes = 32 * 1024 * 1024;  // 32 MB

// cublasGemmEx fallback algorithm hint (used when cublasLt heuristic returns no results).
constexpr cublasGemmAlgo_t cuda_gemm_algo() {
#ifdef EIGEN_NO_CUDA_TENSOR_OPS
  return CUBLAS_GEMM_DEFAULT;
#else
  return CUBLAS_GEMM_DEFAULT_TENSOR_OP;
#endif
}

// ---- cublasLt plan cache ----------------------------------------------------
// Caches matmul descriptors, matrix layouts, and the selected algorithm for
// repeated GEMM calls with the same (m, n, k, dtype, transA, transB) shape.
// Eliminates per-call descriptor creation and heuristic lookup overhead, which
// can be 5-35% of total time for small/medium matrices.
//
// The cache uses a small fixed-size array with linear scan and an LRU
// eviction policy backed by a per-entry tick stamp. GEMM shapes in typical
// workloads (CG iteration, chained solves) have very low cardinality (usually
// 1-3 distinct shapes), so a simple array beats a hash map.

struct CublasLtPlanEntry {
  int64_t m, n, k;
  int64_t lda, ldb, ldc;
  cudaDataType_t dtype;
  cublasOperation_t transA, transB;

  cublasLtMatmulDesc_t matmul_desc;
  cublasLtMatrixLayout_t layout_A, layout_B, layout_C;
  cublasLtMatmulAlgo_t algo;
  size_t workspace_size;
  bool use_cublaslt;  // false → cublasGemmEx fallback

  // Per-entry tick stamp used for LRU eviction. Updated on hit (find) and
  // on creation (insert). Zero means the slot is empty.
  uint64_t last_used;
};

class CublasLtPlanCache {
 public:
  static constexpr int kMaxEntries = 8;

  CublasLtPlanCache() = default;
  ~CublasLtPlanCache() { clear(); }

  // Non-copyable; movable so containing types (e.g. GpuSolverContext) can be moved.
  CublasLtPlanCache(const CublasLtPlanCache&) = delete;
  CublasLtPlanCache& operator=(const CublasLtPlanCache&) = delete;

  CublasLtPlanCache(CublasLtPlanCache&& o) noexcept : size_(o.size_), tick_(o.tick_) {
    for (int i = 0; i < o.size_; ++i) {
      entries_[i] = o.entries_[i];
      o.entries_[i].matmul_desc = nullptr;
      o.entries_[i].layout_A = o.entries_[i].layout_B = o.entries_[i].layout_C = nullptr;
    }
    o.size_ = 0;
    o.tick_ = 0;
  }

  CublasLtPlanCache& operator=(CublasLtPlanCache&& o) noexcept {
    if (this != &o) {
      clear();
      size_ = o.size_;
      tick_ = o.tick_;
      for (int i = 0; i < o.size_; ++i) {
        entries_[i] = o.entries_[i];
        o.entries_[i].matmul_desc = nullptr;
        o.entries_[i].layout_A = o.entries_[i].layout_B = o.entries_[i].layout_C = nullptr;
      }
      o.size_ = 0;
      o.tick_ = 0;
    }
    return *this;
  }

  /** Look up a cached plan. Returns nullptr on miss. Bumps the entry's LRU
   * tick on hit; the linear scan is unchanged, so this stays O(n). */
  const CublasLtPlanEntry* find(int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc,
                                cudaDataType_t dtype, cublasOperation_t transA, cublasOperation_t transB) {
    for (int i = 0; i < size_; ++i) {
      auto& e = entries_[i];
      if (e.m == m && e.n == n && e.k == k && e.lda == lda && e.ldb == ldb && e.ldc == ldc && e.dtype == dtype &&
          e.transA == transA && e.transB == transB) {
        e.last_used = ++tick_;
        return &e;
      }
    }
    return nullptr;
  }

  /** Insert a new entry, evicting the least-recently-used if full. */
  CublasLtPlanEntry* insert(cublasLtHandle_t lt_handle, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb,
                            int64_t ldc, cudaDataType_t dtype, cublasComputeType_t compute, cudaDataType_t alpha_type,
                            cublasOperation_t transA, cublasOperation_t transB) {
    int slot;
    if (size_ < kMaxEntries) {
      slot = size_++;
    } else {
      // Single linear scan for the entry with the smallest tick. No array
      // shift — the evicted slot is reused in place.
      slot = 0;
      for (int i = 1; i < size_; ++i) {
        if (entries_[i].last_used < entries_[slot].last_used) slot = i;
      }
      destroy_entry(entries_[slot]);
    }
    auto& e = entries_[slot];
    e.m = m;
    e.n = n;
    e.k = k;
    e.lda = lda;
    e.ldb = ldb;
    e.ldc = ldc;
    e.dtype = dtype;
    e.transA = transA;
    e.transB = transB;
    e.use_cublaslt = false;

    EIGEN_CUBLASLT_CHECK(cublasLtMatmulDescCreate(&e.matmul_desc, compute, alpha_type));
    EIGEN_CUBLASLT_CHECK(
        cublasLtMatmulDescSetAttribute(e.matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    EIGEN_CUBLASLT_CHECK(
        cublasLtMatmulDescSetAttribute(e.matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));

    // Layout dimensions are the physical (rows, cols) of the column-major operand;
    // the leading dimension is the actual stride between columns (lda/ldb/ldc),
    // which may exceed the active row count (e.g., a thin view of a wider buffer).
    const int64_t a_rows = (transA == CUBLAS_OP_N) ? m : k;
    const int64_t a_cols = (transA == CUBLAS_OP_N) ? k : m;
    const int64_t b_rows = (transB == CUBLAS_OP_N) ? k : n;
    const int64_t b_cols = (transB == CUBLAS_OP_N) ? n : k;
    EIGEN_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&e.layout_A, dtype, a_rows, a_cols, lda));
    EIGEN_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&e.layout_B, dtype, b_rows, b_cols, ldb));
    EIGEN_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&e.layout_C, dtype, m, n, ldc));

    cublasLtMatmulPreference_t preference = nullptr;
    EIGEN_CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    size_t max_ws = kCublasLtMaxWorkspaceBytes;
    EIGEN_CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                              &max_ws, sizeof(max_ws)));

    cublasLtMatmulHeuristicResult_t result;
    int returned_results = 0;
    cublasStatus_t heuristic_status =
        cublasLtMatmulAlgoGetHeuristic(lt_handle, e.matmul_desc, e.layout_A, e.layout_B, e.layout_C, e.layout_C,
                                       preference, 1, &result, &returned_results);

    EIGEN_CUBLASLT_CHECK(cublasLtMatmulPreferenceDestroy(preference));

    if (heuristic_status == CUBLAS_STATUS_SUCCESS && returned_results > 0) {
      e.algo = result.algo;
      e.workspace_size = result.workspaceSize;
      e.use_cublaslt = true;
    } else {
      e.workspace_size = 0;
      e.use_cublaslt = false;
    }

    e.last_used = ++tick_;
    return &e;
  }

  void clear() {
    for (int i = 0; i < size_; ++i) destroy_entry(entries_[i]);
    size_ = 0;
  }

 private:
  // Value-initialize so destroy_entry() sees null handles in any slot that an
  // earlier insert() abandoned mid-construction (e.g., if cublasLt*Create
  // fired an EIGEN_CUBLASLT_CHECK assertion that was compiled away in
  // EIGEN_NO_DEBUG builds).
  CublasLtPlanEntry entries_[kMaxEntries] = {};
  int size_ = 0;
  uint64_t tick_ = 0;

  static void destroy_entry(CublasLtPlanEntry& e) {
    if (e.layout_C) cublasLtMatrixLayoutDestroy(e.layout_C);
    if (e.layout_B) cublasLtMatrixLayoutDestroy(e.layout_B);
    if (e.layout_A) cublasLtMatrixLayoutDestroy(e.layout_A);
    if (e.matmul_desc) cublasLtMatmulDescDestroy(e.matmul_desc);
    e.matmul_desc = nullptr;
    e.layout_A = e.layout_B = e.layout_C = nullptr;
  }
};

// cublasLtMatmul GEMM with shape-keyed plan cache and lazy workspace.
//
// Falls back to cublasGemmEx for shapes/types where the cublasLt heuristic
// returns no results.
template <typename Scalar>
void cublaslt_gemm(cublasLtHandle_t lt_handle, cublasHandle_t cublas_handle, cublasOperation_t transA,
                   cublasOperation_t transB, int64_t m, int64_t n, int64_t k, const Scalar* alpha, const Scalar* A,
                   int64_t lda, const Scalar* B, int64_t ldb, const Scalar* beta, Scalar* C, int64_t ldc,
                   DeviceBuffer* workspace, CublasLtPlanCache* plan_cache, cudaStream_t stream) {
  constexpr cudaDataType_t dtype = cuda_data_type<Scalar>::value;
  constexpr cublasComputeType_t compute = cuda_compute_type<Scalar>::value;
  constexpr cudaDataType_t alpha_type = cuda_data_type<Scalar>::value;

  // Look up or create a cached plan for this shape (key includes leading dims so
  // strided views — e.g. SVD's thin VT/U slices — get distinct cache entries).
  const CublasLtPlanEntry* entry = plan_cache->find(m, n, k, lda, ldb, ldc, dtype, transA, transB);
  if (!entry) {
    entry = plan_cache->insert(lt_handle, m, n, k, lda, ldb, ldc, dtype, compute, alpha_type, transA, transB);
  }

  if (entry->use_cublaslt) {
    const size_t needed = entry->workspace_size;
    if (needed > workspace->size()) {
      // Sync only when freeing an existing buffer that may be in use.
      if (workspace->get()) EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream));
      *workspace = DeviceBuffer(needed);
    }

    EIGEN_CUBLASLT_CHECK(cublasLtMatmul(lt_handle, entry->matmul_desc, alpha, A, entry->layout_A, B, entry->layout_B,
                                        beta, C, entry->layout_C, C, entry->layout_C, &entry->algo, workspace->get(),
                                        needed, stream));
  } else {
    // Fallback: cublasGemmEx for shapes/types that cublasLt cannot handle.
    // cublasGemmEx takes int dimensions; cublaslt_gemm itself supports int64_t.
    eigen_assert(m <= INT_MAX && n <= INT_MAX && k <= INT_MAX && lda <= INT_MAX && ldb <= INT_MAX && ldc <= INT_MAX &&
                 "cublasGemmEx fallback: dimensions exceed int range");
    EIGEN_CUBLAS_CHECK(cublasGemmEx(cublas_handle, transA, transB, static_cast<int>(m), static_cast<int>(n),
                                    static_cast<int>(k), alpha, A, dtype, static_cast<int>(lda), B, dtype,
                                    static_cast<int>(ldb), beta, C, dtype, static_cast<int>(ldc), compute,
                                    cuda_gemm_algo()));
  }
}

// ---- Type-specific cuBLAS wrappers ------------------------------------------
// cuBLAS uses separate functions per type (Sgemm, Dgemm, etc.).
// These overloaded wrappers allow calling cublasXgemm/cublasXtrsm/etc.
// with any supported scalar type.

// GEMM wrappers
inline cublasStatus_t cublasXgemm(cublasHandle_t h, cublasOperation_t transA, cublasOperation_t transB, int m, int n,
                                  int k, const float* alpha, const float* A, int lda, const float* B, int ldb,
                                  const float* beta, float* C, int ldc) {
  return cublasSgemm(h, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
inline cublasStatus_t cublasXgemm(cublasHandle_t h, cublasOperation_t transA, cublasOperation_t transB, int m, int n,
                                  int k, const double* alpha, const double* A, int lda, const double* B, int ldb,
                                  const double* beta, double* C, int ldc) {
  return cublasDgemm(h, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
static_assert(sizeof(cuComplex) == sizeof(std::complex<float>), "cuComplex and std::complex<float> layout mismatch");
static_assert(sizeof(cuDoubleComplex) == sizeof(std::complex<double>),
              "cuDoubleComplex and std::complex<double> layout mismatch");

// Complex scalar args (alpha, beta) are type-punned from std::complex<T>*
// to cuComplex*/cuDoubleComplex*.  A reinterpret_cast violates strict
// aliasing: when inlined, clang/MSVC can elide the caller's store (the
// compiler no longer sees a read through the original type), causing
// segfaults.  We use memcpy — the standard-blessed type-pun — for scalars.
// Device array pointers (A, B, C) are opaque to the host compiler, so
// reinterpret_cast is safe there.
inline cublasStatus_t cublasXgemm(cublasHandle_t h, cublasOperation_t transA, cublasOperation_t transB, int m, int n,
                                  int k, const std::complex<float>* alpha, const std::complex<float>* A, int lda,
                                  const std::complex<float>* B, int ldb, const std::complex<float>* beta,
                                  std::complex<float>* C, int ldc) {
  cuComplex a, b;
  std::memcpy(&a, alpha, sizeof(a));
  std::memcpy(&b, beta, sizeof(b));
  return cublasCgemm(h, transA, transB, m, n, k, &a, reinterpret_cast<const cuComplex*>(A), lda,
                     reinterpret_cast<const cuComplex*>(B), ldb, &b, reinterpret_cast<cuComplex*>(C), ldc);
}
inline cublasStatus_t cublasXgemm(cublasHandle_t h, cublasOperation_t transA, cublasOperation_t transB, int m, int n,
                                  int k, const std::complex<double>* alpha, const std::complex<double>* A, int lda,
                                  const std::complex<double>* B, int ldb, const std::complex<double>* beta,
                                  std::complex<double>* C, int ldc) {
  cuDoubleComplex a, b;
  std::memcpy(&a, alpha, sizeof(a));
  std::memcpy(&b, beta, sizeof(b));
  return cublasZgemm(h, transA, transB, m, n, k, &a, reinterpret_cast<const cuDoubleComplex*>(A), lda,
                     reinterpret_cast<const cuDoubleComplex*>(B), ldb, &b, reinterpret_cast<cuDoubleComplex*>(C), ldc);
}

// TRSM wrappers
inline cublasStatus_t cublasXtrsm(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo,
                                  cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float* alpha,
                                  const float* A, int lda, float* B, int ldb) {
  return cublasStrsm(h, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}
inline cublasStatus_t cublasXtrsm(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo,
                                  cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double* alpha,
                                  const double* A, int lda, double* B, int ldb) {
  return cublasDtrsm(h, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}
inline cublasStatus_t cublasXtrsm(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo,
                                  cublasOperation_t trans, cublasDiagType_t diag, int m, int n,
                                  const std::complex<float>* alpha, const std::complex<float>* A, int lda,
                                  std::complex<float>* B, int ldb) {
  cuComplex a;
  std::memcpy(&a, alpha, sizeof(a));
  return cublasCtrsm(h, side, uplo, trans, diag, m, n, &a, reinterpret_cast<const cuComplex*>(A), lda,
                     reinterpret_cast<cuComplex*>(B), ldb);
}
inline cublasStatus_t cublasXtrsm(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo,
                                  cublasOperation_t trans, cublasDiagType_t diag, int m, int n,
                                  const std::complex<double>* alpha, const std::complex<double>* A, int lda,
                                  std::complex<double>* B, int ldb) {
  cuDoubleComplex a;
  std::memcpy(&a, alpha, sizeof(a));
  return cublasZtrsm(h, side, uplo, trans, diag, m, n, &a, reinterpret_cast<const cuDoubleComplex*>(A), lda,
                     reinterpret_cast<cuDoubleComplex*>(B), ldb);
}

// SYMM wrappers (real → symm, complex → hemm)
inline cublasStatus_t cublasXsymm(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n,
                                  const float* alpha, const float* A, int lda, const float* B, int ldb,
                                  const float* beta, float* C, int ldc) {
  return cublasSsymm(h, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}
inline cublasStatus_t cublasXsymm(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n,
                                  const double* alpha, const double* A, int lda, const double* B, int ldb,
                                  const double* beta, double* C, int ldc) {
  return cublasDsymm(h, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}
inline cublasStatus_t cublasXsymm(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n,
                                  const std::complex<float>* alpha, const std::complex<float>* A, int lda,
                                  const std::complex<float>* B, int ldb, const std::complex<float>* beta,
                                  std::complex<float>* C, int ldc) {
  cuComplex a, b;
  std::memcpy(&a, alpha, sizeof(a));
  std::memcpy(&b, beta, sizeof(b));
  return cublasChemm(h, side, uplo, m, n, &a, reinterpret_cast<const cuComplex*>(A), lda,
                     reinterpret_cast<const cuComplex*>(B), ldb, &b, reinterpret_cast<cuComplex*>(C), ldc);
}
inline cublasStatus_t cublasXsymm(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n,
                                  const std::complex<double>* alpha, const std::complex<double>* A, int lda,
                                  const std::complex<double>* B, int ldb, const std::complex<double>* beta,
                                  std::complex<double>* C, int ldc) {
  cuDoubleComplex a, b;
  std::memcpy(&a, alpha, sizeof(a));
  std::memcpy(&b, beta, sizeof(b));
  return cublasZhemm(h, side, uplo, m, n, &a, reinterpret_cast<const cuDoubleComplex*>(A), lda,
                     reinterpret_cast<const cuDoubleComplex*>(B), ldb, &b, reinterpret_cast<cuDoubleComplex*>(C), ldc);
}

// GEAM wrappers: C = alpha * op(A) + beta * op(B)
inline cublasStatus_t cublasXgeam(cublasHandle_t h, cublasOperation_t transA, cublasOperation_t transB, int m, int n,
                                  const float* alpha, const float* A, int lda, const float* beta, const float* B,
                                  int ldb, float* C, int ldc) {
  return cublasSgeam(h, transA, transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}
inline cublasStatus_t cublasXgeam(cublasHandle_t h, cublasOperation_t transA, cublasOperation_t transB, int m, int n,
                                  const double* alpha, const double* A, int lda, const double* beta, const double* B,
                                  int ldb, double* C, int ldc) {
  return cublasDgeam(h, transA, transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}
inline cublasStatus_t cublasXgeam(cublasHandle_t h, cublasOperation_t transA, cublasOperation_t transB, int m, int n,
                                  const std::complex<float>* alpha, const std::complex<float>* A, int lda,
                                  const std::complex<float>* beta, const std::complex<float>* B, int ldb,
                                  std::complex<float>* C, int ldc) {
  cuComplex a, b;
  std::memcpy(&a, alpha, sizeof(a));
  std::memcpy(&b, beta, sizeof(b));
  return cublasCgeam(h, transA, transB, m, n, &a, reinterpret_cast<const cuComplex*>(A), lda, &b,
                     reinterpret_cast<const cuComplex*>(B), ldb, reinterpret_cast<cuComplex*>(C), ldc);
}
inline cublasStatus_t cublasXgeam(cublasHandle_t h, cublasOperation_t transA, cublasOperation_t transB, int m, int n,
                                  const std::complex<double>* alpha, const std::complex<double>* A, int lda,
                                  const std::complex<double>* beta, const std::complex<double>* B, int ldb,
                                  std::complex<double>* C, int ldc) {
  cuDoubleComplex a, b;
  std::memcpy(&a, alpha, sizeof(a));
  std::memcpy(&b, beta, sizeof(b));
  return cublasZgeam(h, transA, transB, m, n, &a, reinterpret_cast<const cuDoubleComplex*>(A), lda, &b,
                     reinterpret_cast<const cuDoubleComplex*>(B), ldb, reinterpret_cast<cuDoubleComplex*>(C), ldc);
}

// SYRK wrappers (real → syrk, complex → herk)
inline cublasStatus_t cublasXsyrk(cublasHandle_t h, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,
                                  const float* alpha, const float* A, int lda, const float* beta, float* C, int ldc) {
  return cublasSsyrk(h, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}
inline cublasStatus_t cublasXsyrk(cublasHandle_t h, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,
                                  const double* alpha, const double* A, int lda, const double* beta, double* C,
                                  int ldc) {
  return cublasDsyrk(h, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}
inline cublasStatus_t cublasXsyrk(cublasHandle_t h, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,
                                  const float* alpha, const std::complex<float>* A, int lda, const float* beta,
                                  std::complex<float>* C, int ldc) {
  return cublasCherk(h, uplo, trans, n, k, alpha, reinterpret_cast<const cuComplex*>(A), lda, beta,
                     reinterpret_cast<cuComplex*>(C), ldc);
}
inline cublasStatus_t cublasXsyrk(cublasHandle_t h, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,
                                  const double* alpha, const std::complex<double>* A, int lda, const double* beta,
                                  std::complex<double>* C, int ldc) {
  return cublasZherk(h, uplo, trans, n, k, alpha, reinterpret_cast<const cuDoubleComplex*>(A), lda, beta,
                     reinterpret_cast<cuDoubleComplex*>(C), ldc);
}

// SCAL wrappers: x = alpha * x.
// For complex x, alpha is real-valued (Csscal/Zdscal) — this matches the
// 1/n inverse-FFT scaling pattern, where the scale is intrinsically real.
inline cublasStatus_t cublasXscal(cublasHandle_t h, int n, const float* alpha, float* x, int incx) {
  return cublasSscal(h, n, alpha, x, incx);
}
inline cublasStatus_t cublasXscal(cublasHandle_t h, int n, const double* alpha, double* x, int incx) {
  return cublasDscal(h, n, alpha, x, incx);
}
inline cublasStatus_t cublasXscal(cublasHandle_t h, int n, const float* alpha, std::complex<float>* x, int incx) {
  return cublasCsscal(h, n, alpha, reinterpret_cast<cuComplex*>(x), incx);
}
inline cublasStatus_t cublasXscal(cublasHandle_t h, int n, const double* alpha, std::complex<double>* x, int incx) {
  return cublasZdscal(h, n, alpha, reinterpret_cast<cuDoubleComplex*>(x), incx);
}

// By-value alpha overloads: convenience for callers that hold the scale as a
// scalar rather than a host pointer (e.g. inverse-FFT 1/n normalization).
inline cublasStatus_t cublasXscal(cublasHandle_t h, int n, float alpha, float* x, int incx) {
  return cublasSscal(h, n, &alpha, x, incx);
}
inline cublasStatus_t cublasXscal(cublasHandle_t h, int n, double alpha, double* x, int incx) {
  return cublasDscal(h, n, &alpha, x, incx);
}
inline cublasStatus_t cublasXscal(cublasHandle_t h, int n, float alpha, std::complex<float>* x, int incx) {
  return cublasCsscal(h, n, &alpha, reinterpret_cast<cuComplex*>(x), incx);
}
inline cublasStatus_t cublasXscal(cublasHandle_t h, int n, double alpha, std::complex<double>* x, int incx) {
  return cublasZdscal(h, n, &alpha, reinterpret_cast<cuDoubleComplex*>(x), incx);
}

// DGMM wrappers: C = A * diag(x)  (side=RIGHT) or C = diag(x) * A  (side=LEFT).
// Useful for applying a diagonal scaling without materialising diag(x) as a
// dense matrix. cuBLAS docs guarantee in-place is safe when C == A.
inline cublasStatus_t cublasXdgmm(cublasHandle_t h, cublasSideMode_t side, int m, int n, const float* A, int lda,
                                  const float* x, int incx, float* C, int ldc) {
  return cublasSdgmm(h, side, m, n, A, lda, x, incx, C, ldc);
}
inline cublasStatus_t cublasXdgmm(cublasHandle_t h, cublasSideMode_t side, int m, int n, const double* A, int lda,
                                  const double* x, int incx, double* C, int ldc) {
  return cublasDdgmm(h, side, m, n, A, lda, x, incx, C, ldc);
}
inline cublasStatus_t cublasXdgmm(cublasHandle_t h, cublasSideMode_t side, int m, int n, const std::complex<float>* A,
                                  int lda, const std::complex<float>* x, int incx, std::complex<float>* C, int ldc) {
  return cublasCdgmm(h, side, m, n, reinterpret_cast<const cuComplex*>(A), lda, reinterpret_cast<const cuComplex*>(x),
                     incx, reinterpret_cast<cuComplex*>(C), ldc);
}
inline cublasStatus_t cublasXdgmm(cublasHandle_t h, cublasSideMode_t side, int m, int n, const std::complex<double>* A,
                                  int lda, const std::complex<double>* x, int incx, std::complex<double>* C, int ldc) {
  return cublasZdgmm(h, side, m, n, reinterpret_cast<const cuDoubleComplex*>(A), lda,
                     reinterpret_cast<const cuDoubleComplex*>(x), incx, reinterpret_cast<cuDoubleComplex*>(C), ldc);
}

// ---- cuBLAS Level-1 wrappers ------------------------------------------------
// Type-dispatched wrappers for BLAS-1 vector operations: dot, axpy, nrm2, scal, copy.
// These work with CUBLAS_POINTER_MODE_HOST or CUBLAS_POINTER_MODE_DEVICE depending
// on the caller's configuration. For device pointer mode, scalar result pointers
// (dot, nrm2) must point to device memory.

// dot: result = x^T * y (real) or x^H * y (complex conjugate dot)
inline cublasStatus_t cublasXdot(cublasHandle_t h, int n, const float* x, int incx, const float* y, int incy,
                                 float* result) {
  return cublasSdot(h, n, x, incx, y, incy, result);
}
inline cublasStatus_t cublasXdot(cublasHandle_t h, int n, const double* x, int incx, const double* y, int incy,
                                 double* result) {
  return cublasDdot(h, n, x, incx, y, incy, result);
}
inline cublasStatus_t cublasXdot(cublasHandle_t h, int n, const std::complex<float>* x, int incx,
                                 const std::complex<float>* y, int incy, std::complex<float>* result) {
  return cublasCdotc(h, n, reinterpret_cast<const cuComplex*>(x), incx, reinterpret_cast<const cuComplex*>(y), incy,
                     reinterpret_cast<cuComplex*>(result));
}
inline cublasStatus_t cublasXdot(cublasHandle_t h, int n, const std::complex<double>* x, int incx,
                                 const std::complex<double>* y, int incy, std::complex<double>* result) {
  return cublasZdotc(h, n, reinterpret_cast<const cuDoubleComplex*>(x), incx,
                     reinterpret_cast<const cuDoubleComplex*>(y), incy, reinterpret_cast<cuDoubleComplex*>(result));
}

// nrm2: result = ||x||_2 (always returns real)
inline cublasStatus_t cublasXnrm2(cublasHandle_t h, int n, const float* x, int incx, float* result) {
  return cublasSnrm2(h, n, x, incx, result);
}
inline cublasStatus_t cublasXnrm2(cublasHandle_t h, int n, const double* x, int incx, double* result) {
  return cublasDnrm2(h, n, x, incx, result);
}
inline cublasStatus_t cublasXnrm2(cublasHandle_t h, int n, const std::complex<float>* x, int incx, float* result) {
  return cublasScnrm2(h, n, reinterpret_cast<const cuComplex*>(x), incx, result);
}
inline cublasStatus_t cublasXnrm2(cublasHandle_t h, int n, const std::complex<double>* x, int incx, double* result) {
  return cublasDznrm2(h, n, reinterpret_cast<const cuDoubleComplex*>(x), incx, result);
}

// axpy: y += alpha * x
inline cublasStatus_t cublasXaxpy(cublasHandle_t h, int n, const float* alpha, const float* x, int incx, float* y,
                                  int incy) {
  return cublasSaxpy(h, n, alpha, x, incx, y, incy);
}
inline cublasStatus_t cublasXaxpy(cublasHandle_t h, int n, const double* alpha, const double* x, int incx, double* y,
                                  int incy) {
  return cublasDaxpy(h, n, alpha, x, incx, y, incy);
}
inline cublasStatus_t cublasXaxpy(cublasHandle_t h, int n, const std::complex<float>* alpha,
                                  const std::complex<float>* x, int incx, std::complex<float>* y, int incy) {
  cuComplex a;
  std::memcpy(&a, alpha, sizeof(a));
  return cublasCaxpy(h, n, &a, reinterpret_cast<const cuComplex*>(x), incx, reinterpret_cast<cuComplex*>(y), incy);
}
inline cublasStatus_t cublasXaxpy(cublasHandle_t h, int n, const std::complex<double>* alpha,
                                  const std::complex<double>* x, int incx, std::complex<double>* y, int incy) {
  cuDoubleComplex a;
  std::memcpy(&a, alpha, sizeof(a));
  return cublasZaxpy(h, n, &a, reinterpret_cast<const cuDoubleComplex*>(x), incx, reinterpret_cast<cuDoubleComplex*>(y),
                     incy);
}

// SCAL with complex alpha on complex vectors (Cscal/Zscal). The real-alpha
// overloads (Sscal/Dscal/Csscal/Zdscal) live above with the FFT-scaling forms.
inline cublasStatus_t cublasXscal(cublasHandle_t h, int n, const std::complex<float>* alpha, std::complex<float>* x,
                                  int incx) {
  cuComplex a;
  std::memcpy(&a, alpha, sizeof(a));
  return cublasCscal(h, n, &a, reinterpret_cast<cuComplex*>(x), incx);
}
inline cublasStatus_t cublasXscal(cublasHandle_t h, int n, const std::complex<double>* alpha, std::complex<double>* x,
                                  int incx) {
  cuDoubleComplex a;
  std::memcpy(&a, alpha, sizeof(a));
  return cublasZscal(h, n, &a, reinterpret_cast<cuDoubleComplex*>(x), incx);
}

// copy: y = x
inline cublasStatus_t cublasXcopy(cublasHandle_t h, int n, const float* x, int incx, float* y, int incy) {
  return cublasScopy(h, n, x, incx, y, incy);
}
inline cublasStatus_t cublasXcopy(cublasHandle_t h, int n, const double* x, int incx, double* y, int incy) {
  return cublasDcopy(h, n, x, incx, y, incy);
}
inline cublasStatus_t cublasXcopy(cublasHandle_t h, int n, const std::complex<float>* x, int incx,
                                  std::complex<float>* y, int incy) {
  return cublasCcopy(h, n, reinterpret_cast<const cuComplex*>(x), incx, reinterpret_cast<cuComplex*>(y), incy);
}
inline cublasStatus_t cublasXcopy(cublasHandle_t h, int n, const std::complex<double>* x, int incx,
                                  std::complex<double>* y, int incy) {
  return cublasZcopy(h, n, reinterpret_cast<const cuDoubleComplex*>(x), incx, reinterpret_cast<cuDoubleComplex*>(y),
                     incy);
}

}  // namespace internal
}  // namespace gpu
}  // namespace Eigen

#endif  // EIGEN_GPU_CUBLAS_SUPPORT_H
