// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// Dispatch functions mapping DeviceMatrix expressions to NVIDIA library calls,
// plus the DeviceMatrix members that need a complete gpu::Context. The
// expression argument selects the dispatch() overload.

#ifndef EIGEN_GPU_DEVICE_DISPATCH_H
#define EIGEN_GPU_DEVICE_DISPATCH_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

#include <cstdint>

#include "./DeviceMatrix.h"
#include "./DeviceScalar.h"
#include "./DeviceExpr.h"
#include "./DeviceBlasExpr.h"
#include "./DeviceSolverExpr.h"
#include "./GpuContext.h"
#include "./CuSolverSupport.h"

namespace Eigen {
namespace gpu {
namespace internal {
template <typename Scalar>
bool aliases_device_memory(const DeviceMatrix<Scalar>& a, const DeviceMatrix<Scalar>& b) {
  return a.data() != nullptr && a.data() == b.data();
}

template <typename Lhs, typename Rhs>
void dispatch(Context& ctx, DeviceMatrix<scalar_type_t<Lhs>>& dst, const GemmExpr<Lhs, Rhs>& expr,
              scalar_type_t<Lhs> beta_val, scalar_type_t<Lhs> alpha_scale = scalar_type_t<Lhs>(1)) {
  using Scalar = scalar_type_t<Lhs>;
  using traits_lhs = device_expr_traits<Lhs>;
  using traits_rhs = device_expr_traits<Rhs>;

  const DeviceMatrix<Scalar>& A = traits_lhs::matrix(expr.lhs());
  const DeviceMatrix<Scalar>& B = traits_rhs::matrix(expr.rhs());

  // cuBLAS leaves C aliasing A or B undefined.
  eigen_assert(!aliases_device_memory(dst, A) && "GEMM: output aliases left operand (use a temporary)");
  eigen_assert(!aliases_device_memory(dst, B) && "GEMM: output aliases right operand (use a temporary)");

  constexpr cublasOperation_t transA = to_cublas_op(traits_lhs::op);
  constexpr cublasOperation_t transB = to_cublas_op(traits_rhs::op);

  const int64_t m = (traits_lhs::op == GpuOp::NoTrans) ? A.rows() : A.cols();
  const int64_t k = (traits_lhs::op == GpuOp::NoTrans) ? A.cols() : A.rows();
  const int64_t n = (traits_rhs::op == GpuOp::NoTrans) ? B.cols() : B.rows();
  const int64_t rhs_k = (traits_rhs::op == GpuOp::NoTrans) ? B.rows() : B.cols();

  eigen_assert(k == rhs_k && "DeviceMatrix GEMM dimension mismatch");

  const int64_t lda = A.rows();
  const int64_t ldb = B.rows();

  const bool resized = dst.empty() || dst.rows() != m || dst.cols() != n;
  dst.resize(ctx, m, n);
  const int64_t ldc = dst.rows();

  Scalar alpha_local = alpha_scale * traits_lhs::alpha(expr.lhs()) * traits_rhs::alpha(expr.rhs());

  A.prepareRead(ctx);
  B.prepareRead(ctx);
  dst.prepareWrite(ctx);

  if (resized && beta_val != Scalar(0) && dst.sizeInBytes() > 0) {
    EIGEN_CUDA_RUNTIME_CHECK(cudaMemsetAsync(dst.data(), 0, dst.sizeInBytes(), ctx.stream()));
  }

  // cuBLAS reads alpha and beta through host pointers. Holding them in an array
  // keeps the compiler from eliding their stack slots — at -O1+ clang and MSVC
  // otherwise drop the stores for complex types, leaving cuBLAS with a dangling
  // pointer.
  Scalar scalars[2] = {alpha_local, beta_val};
  cublaslt_gemm(ctx.cublasLtHandle(), ctx.cublasHandle(), transA, transB, m, n, k, &scalars[0], A.data(), lda, B.data(),
                ldb, &scalars[1], dst.data(), ldc, ctx.gemmWorkspace(), ctx.gemmPlanCache(),
                ctx.cublasLtMaxWorkspaceBytes(), ctx.streamHandle());

  A.finishRead(ctx);
  B.finishRead(ctx);
  dst.finishWrite(ctx);
}

// Debug-build status check shared by the one-shot solver dispatches: syncs
// the stream and asserts on the two info words (factorize, solve). Release
// builds skip both the check and the sync — one-shot expressions are then
// fully async with no failure detection; use gpu::LLT / gpu::LU + info()
// when failures must be detected.
inline void oneshot_check_info(Context& ctx, OneShotSolverScratch& scratch, const char* what) {
#ifdef EIGEN_NO_DEBUG
  EIGEN_UNUSED_VARIABLE(ctx);
  EIGEN_UNUSED_VARIABLE(scratch);
  EIGEN_UNUSED_VARIABLE(what);
#else
  int* info_words = static_cast<int*>(scratch.h_info.get());
  EIGEN_CUDA_RUNTIME_CHECK(
      cudaMemcpyAsync(info_words, scratch.d_info.get(), kOneShotInfoBytes, cudaMemcpyDeviceToHost, ctx.stream()));
  EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(ctx.stream()));
  eigen_assert(info_words[0] == 0 && "cuSOLVER one-shot factorization failed" && what);
  eigen_assert(info_words[1] == 0 && "cuSOLVER one-shot solve failed" && what);
  EIGEN_UNUSED_VARIABLE(what);
#endif
}

template <typename Scalar, int UpLo>
void dispatch(Context& ctx, DeviceMatrix<Scalar>& dst, const LltSolveExpr<Scalar, UpLo>& expr) {
  const DeviceMatrix<Scalar>& A = expr.matrix();
  const DeviceMatrix<Scalar>& B = expr.rhs();

  eigen_assert(A.rows() == A.cols() && "LLT requires a square matrix");
  eigen_assert(B.rows() == A.rows() && "LLT solve: RHS rows must match matrix size");

  if (A.rows() == 0 || B.cols() == 0) {
    dst.resize(ctx, A.rows(), B.cols());
    return;
  }

  A.prepareRead(ctx);
  B.prepareRead(ctx);

  // thread_local: must outlive the async kernels (no end-of-call sync), and
  // only TUs that instantiate the one-shot path pull in cuSOLVER symbols.
  static thread_local CusolverParams params;
  constexpr cublasFillMode_t uplo = cusolver_fill_mode<UpLo>::value;
  const int64_t n = static_cast<int64_t>(A.rows());
  constexpr cudaDataType_t dtype = cuda_data_type<Scalar>::value;
  OneShotSolverScratch& scratch = ctx.oneshotSolverScratch();
  {
    const size_t mat_bytes = A.sizeInBytes();
    // Context-owned grow-only scratch: no per-call allocation, no end-of-call sync.
    ensure_sized(scratch.d_factor, mat_bytes, ctx.streamHandle());
    EIGEN_CUDA_RUNTIME_CHECK(
        cudaMemcpyAsync(scratch.d_factor.get(), A.data(), mat_bytes, cudaMemcpyDeviceToDevice, ctx.stream()));
  }
  const int64_t lda = static_cast<int64_t>(A.rows());
  size_t dev_ws = 0;
  size_t host_ws = 0;
  EIGEN_CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(ctx.cusolverHandle(), params.p, uplo, n, dtype,
                                                   scratch.d_factor.get(), lda, dtype, &dev_ws, &host_ws));
  ensure_sized(scratch.d_workspace, dev_ws, ctx.streamHandle());
  if (scratch.h_workspace.size() < host_ws) scratch.h_workspace.resize(host_ws);
  // Two info slots (potrf, potrs) so both kernels queue back-to-back. If potrf
  // fails, potrs runs on garbage but the debug check catches both at once.
  int* d_info_potrf = static_cast<int*>(scratch.d_info.get());
  int* d_info_potrs = d_info_potrf + 1;
  EIGEN_CUSOLVER_CHECK(cusolverDnXpotrf(ctx.cusolverHandle(), params.p, uplo, n, dtype, scratch.d_factor.get(), lda,
                                        dtype, scratch.d_workspace.get(), dev_ws,
                                        host_ws > 0 ? scratch.h_workspace.data() : nullptr, host_ws, d_info_potrf));

  dst.resize(ctx, n, B.cols());
  dst.prepareWrite(ctx);
  const size_t rhs_bytes = B.sizeInBytes();
  EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(dst.data(), B.data(), rhs_bytes, cudaMemcpyDeviceToDevice, ctx.stream()));

  const int64_t nrhs = static_cast<int64_t>(B.cols());
  EIGEN_CUSOLVER_CHECK(cusolverDnXpotrs(ctx.cusolverHandle(), params.p, uplo, n, nrhs, dtype, scratch.d_factor.get(),
                                        lda, dtype, dst.data(), static_cast<int64_t>(dst.rows()), d_info_potrs));
  A.finishRead(ctx);
  B.finishRead(ctx);
  dst.finishWrite(ctx);
  oneshot_check_info(ctx, scratch, "llt");
}

template <typename Scalar>
void dispatch(Context& ctx, DeviceMatrix<Scalar>& dst, const LuSolveExpr<Scalar>& expr) {
  const DeviceMatrix<Scalar>& A = expr.matrix();
  const DeviceMatrix<Scalar>& B = expr.rhs();

  eigen_assert(A.rows() == A.cols() && "LU requires a square matrix");
  eigen_assert(B.rows() == A.rows() && "LU solve: RHS rows must match matrix size");

  if (A.rows() == 0 || B.cols() == 0) {
    dst.resize(ctx, A.rows(), B.cols());
    return;
  }

  A.prepareRead(ctx);
  B.prepareRead(ctx);

  // thread_local: must outlive the async kernels (no end-of-call sync), and
  // only TUs that instantiate the one-shot path pull in cuSOLVER symbols.
  static thread_local CusolverParams params;
  const int64_t n = static_cast<int64_t>(A.rows());
  constexpr cudaDataType_t dtype = cuda_data_type<Scalar>::value;
  OneShotSolverScratch& scratch = ctx.oneshotSolverScratch();
  {
    const size_t mat_bytes = A.sizeInBytes();
    // Context-owned grow-only scratch: no per-call allocation, no end-of-call sync.
    ensure_sized(scratch.d_factor, mat_bytes, ctx.streamHandle());
    EIGEN_CUDA_RUNTIME_CHECK(
        cudaMemcpyAsync(scratch.d_factor.get(), A.data(), mat_bytes, cudaMemcpyDeviceToDevice, ctx.stream()));
  }
  ensure_sized(scratch.d_ipiv, static_cast<size_t>(n) * sizeof(int64_t), ctx.streamHandle());
  const int64_t lda = static_cast<int64_t>(A.rows());
  size_t dev_ws = 0;
  size_t host_ws = 0;
  EIGEN_CUSOLVER_CHECK(cusolverDnXgetrf_bufferSize(ctx.cusolverHandle(), params.p, n, n, dtype, scratch.d_factor.get(),
                                                   lda, dtype, &dev_ws, &host_ws));
  ensure_sized(scratch.d_workspace, dev_ws, ctx.streamHandle());
  if (scratch.h_workspace.size() < host_ws) scratch.h_workspace.resize(host_ws);
  int* d_info_getrf = static_cast<int*>(scratch.d_info.get());
  int* d_info_getrs = d_info_getrf + 1;
  EIGEN_CUSOLVER_CHECK(cusolverDnXgetrf(ctx.cusolverHandle(), params.p, n, n, dtype, scratch.d_factor.get(), lda,
                                        static_cast<int64_t*>(scratch.d_ipiv.get()), dtype, scratch.d_workspace.get(),
                                        dev_ws, host_ws > 0 ? scratch.h_workspace.data() : nullptr, host_ws,
                                        d_info_getrf));

  dst.resize(ctx, n, B.cols());
  dst.prepareWrite(ctx);
  const size_t rhs_bytes = B.sizeInBytes();
  EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(dst.data(), B.data(), rhs_bytes, cudaMemcpyDeviceToDevice, ctx.stream()));

  const int64_t nrhs = static_cast<int64_t>(B.cols());
  EIGEN_CUSOLVER_CHECK(cusolverDnXgetrs(ctx.cusolverHandle(), params.p, CUBLAS_OP_N, n, nrhs, dtype,
                                        scratch.d_factor.get(), lda, static_cast<const int64_t*>(scratch.d_ipiv.get()),
                                        dtype, dst.data(), static_cast<int64_t>(dst.rows()), d_info_getrs));
  A.finishRead(ctx);
  B.finishRead(ctx);
  dst.finishWrite(ctx);
  oneshot_check_info(ctx, scratch, "lu");
}

template <typename Scalar, int UpLo>
void dispatch(Context& ctx, DeviceMatrix<Scalar>& dst, const TrsmExpr<Scalar, UpLo>& expr) {
  const DeviceMatrix<Scalar>& A = expr.matrix();
  const DeviceMatrix<Scalar>& B = expr.rhs();

  eigen_assert(A.rows() == A.cols() && "TRSM requires a square triangular matrix");
  eigen_assert(B.rows() == A.rows() && "TRSM: RHS rows must match matrix size");

  const int64_t n = A.rows();
  const int64_t nrhs = B.cols();

  if (n == 0 || nrhs == 0) {
    dst.resize(ctx, n, B.cols());
    return;
  }

  eigen_assert(!aliases_device_memory(dst, A) && "DeviceMatrix TRSM destination aliases triangular operand");
  eigen_assert(!aliases_device_memory(dst, B) && "DeviceMatrix TRSM destination aliases RHS operand");
  dst.resize(ctx, n, B.cols());
  A.prepareRead(ctx);
  B.prepareRead(ctx);
  dst.prepareWrite(ctx);
  const size_t rhs_bytes = static_cast<size_t>(dst.rows()) * static_cast<size_t>(nrhs) * sizeof(Scalar);
  EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(dst.data(), B.data(), rhs_bytes, cudaMemcpyDeviceToDevice, ctx.stream()));

  constexpr cublasFillMode_t uplo = (UpLo == Lower) ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
  Scalar alpha(1);

  EIGEN_CUBLAS_CHECK(cublasXtrsm(ctx.cublasHandle(), CUBLAS_SIDE_LEFT, uplo, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, nrhs,
                                 &alpha, A.data(), A.rows(), dst.data(), dst.rows()));

  A.finishRead(ctx);
  B.finishRead(ctx);
  dst.finishWrite(ctx);
}

template <typename Scalar, int UpLo>
void dispatch(Context& ctx, DeviceMatrix<Scalar>& dst, const SymmExpr<Scalar, UpLo>& expr) {
  const DeviceMatrix<Scalar>& A = expr.matrix();
  const DeviceMatrix<Scalar>& B = expr.rhs();

  eigen_assert(A.rows() == A.cols() && "SYMM requires a square matrix");
  eigen_assert(B.rows() == A.rows() && "SYMM: RHS rows must match matrix size");

  const int64_t m = A.rows();
  const int64_t n = B.cols();

  if (m == 0 || n == 0) {
    dst.resize(ctx, m, B.cols());
    return;
  }

  eigen_assert(!aliases_device_memory(dst, A) && "DeviceMatrix SYMM destination aliases self-adjoint operand");
  eigen_assert(!aliases_device_memory(dst, B) && "DeviceMatrix SYMM destination aliases RHS operand");
  dst.resize(ctx, m, n);
  A.prepareRead(ctx);
  B.prepareRead(ctx);
  dst.prepareWrite(ctx);

  constexpr cublasFillMode_t uplo = (UpLo == Lower) ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
  // The array keeps the host-pointer stack slots alive; see the GEMM dispatch.
  Scalar scalars[2] = {Scalar(1), Scalar(0)};

  EIGEN_CUBLAS_CHECK(cublasXsymm(ctx.cublasHandle(), CUBLAS_SIDE_LEFT, uplo, m, n, &scalars[0], A.data(), A.rows(),
                                 B.data(), B.rows(), &scalars[1], dst.data(), dst.rows()));

  A.finishRead(ctx);
  B.finishRead(ctx);
  dst.finishWrite(ctx);
}

template <typename Scalar, int UpLo>
void dispatch(Context& ctx, DeviceMatrix<Scalar>& dst, const SyrkExpr<Scalar, UpLo>& expr,
              typename NumTraits<Scalar>::Real alpha_val, typename NumTraits<Scalar>::Real beta_val) {
  using RealScalar = typename NumTraits<Scalar>::Real;
  const DeviceMatrix<Scalar>& A = expr.matrix();

  const int64_t n = A.rows();
  const int64_t k = A.cols();

  if (n == 0) {
    dst.resize(ctx, 0, 0);
    return;
  }

  eigen_assert(!aliases_device_memory(dst, A) && "DeviceMatrix SYRK destination aliases input operand");
  const bool resized = dst.empty() || dst.rows() != n || dst.cols() != n;
  dst.resize(ctx, n, n);
  A.prepareRead(ctx);
  dst.prepareWrite(ctx);
  if (resized && beta_val != RealScalar(0)) {
    EIGEN_CUDA_RUNTIME_CHECK(cudaMemsetAsync(dst.data(), 0, dst.sizeInBytes(), ctx.stream()));
  }

  constexpr cublasFillMode_t uplo = (UpLo == Lower) ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;

  EIGEN_CUBLAS_CHECK(cublasXsyrk(ctx.cublasHandle(), uplo, CUBLAS_OP_N, n, k, &alpha_val, A.data(), A.rows(), &beta_val,
                                 dst.data(), dst.rows()));

  A.finishRead(ctx);
  dst.finishWrite(ctx);
}

// DeviceAddExpr → cublasXgeam: dst = alpha * A + beta * B. Safe when dst
// aliases A and/or B (geam supports in-place operation with equal leading
// dimensions, which always holds here since DeviceMatrix is fully dense).

template <typename Scalar>
void dispatch(Context& ctx, DeviceMatrix<Scalar>& dst, const DeviceAddExpr<Scalar>& expr) {
  const DeviceMatrix<Scalar>& A = expr.A();
  const DeviceMatrix<Scalar>& B = expr.B();
  eigen_assert(A.rows() == B.rows() && A.cols() == B.cols());
  const int64_t m = A.rows();
  const int64_t n = A.cols();
  // dst may be A or B (same shape), in which case resize keeps the allocation.
  dst.resize(ctx, A.rows(), A.cols());
  if (m > 0 && n > 0) {
    A.prepareRead(ctx);
    B.prepareRead(ctx);
    dst.prepareWrite(ctx);
    // See the GEMM dispatch: array prevents compiler from eliding host-pointer stack slots.
    Scalar scalars[2] = {expr.alpha(), expr.beta()};
    EIGEN_CUBLAS_CHECK(cublasXgeam(ctx.cublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, m, n, &scalars[0], A.data(), m,
                                   &scalars[1], B.data(), m, dst.data(), m));
    A.finishRead(ctx);
    B.finishRead(ctx);
    dst.finishWrite(ctx);
  }
}
}  // namespace internal

template <typename Scalar_>
class Assignment {
 public:
  using Scalar = Scalar_;

  Assignment(DeviceMatrix<Scalar>& dst, Context& ctx) : dst_(dst), ctx_(ctx) {}

  template <typename Lhs, typename Rhs>
  DeviceMatrix<Scalar>& operator=(const GemmExpr<Lhs, Rhs>& expr) {
    internal::dispatch(ctx_, dst_, expr, Scalar(0));
    return dst_;
  }

  template <typename Lhs, typename Rhs>
  DeviceMatrix<Scalar>& operator+=(const GemmExpr<Lhs, Rhs>& expr) {
    internal::dispatch(ctx_, dst_, expr, Scalar(1));
    return dst_;
  }

  template <typename Lhs, typename Rhs>
  DeviceMatrix<Scalar>& operator-=(const GemmExpr<Lhs, Rhs>& expr) {
    internal::dispatch(ctx_, dst_, expr, Scalar(1), Scalar(-1));
    return dst_;
  }

  template <int UpLo>
  DeviceMatrix<Scalar>& operator=(const LltSolveExpr<Scalar, UpLo>& expr) {
    internal::dispatch(ctx_, dst_, expr);
    return dst_;
  }

  DeviceMatrix<Scalar>& operator=(const LuSolveExpr<Scalar>& expr) {
    internal::dispatch(ctx_, dst_, expr);
    return dst_;
  }

  template <int UpLo>
  DeviceMatrix<Scalar>& operator=(const TrsmExpr<Scalar, UpLo>& expr) {
    internal::dispatch(ctx_, dst_, expr);
    return dst_;
  }

  template <int UpLo>
  DeviceMatrix<Scalar>& operator=(const SymmExpr<Scalar, UpLo>& expr) {
    internal::dispatch(ctx_, dst_, expr);
    return dst_;
  }

  DeviceMatrix<Scalar>& operator=(const DeviceAddExpr<Scalar>& expr) {
    internal::dispatch(ctx_, dst_, expr);
    return dst_;
  }

  DeviceMatrix<Scalar>& operator=(const Scaled<DeviceMatrix<Scalar>>& expr) {
    // geam with beta == 0: cuBLAS documents B as unread, so pass A twice.
    internal::dispatch(ctx_, dst_, DeviceAddExpr<Scalar>(expr.scalar(), expr.inner(), Scalar(0), expr.inner()));
    return dst_;
  }

  template <typename Expr>
  DeviceMatrix<Scalar>& operator=(const Expr&) {
    static_assert(sizeof(Expr) == 0,
                  "DeviceMatrix expression not supported: no cuBLAS/cuSOLVER mapping. "
                  "Supported: GEMM (A*B), geam (A + alpha*B, alpha*A), "
                  "TRSM (.triangularView().solve()), SYMM (.selfadjointView()*B), "
                  "LLT (.llt().solve()), LU (.lu().solve()).");
    return dst_;
  }

 private:
  DeviceMatrix<Scalar>& dst_;
  Context& ctx_;
};

// The definitions below call Context::threadLocal(), so they cannot live in
// DeviceMatrix.h, where Context is still incomplete.

template <typename Scalar_>
template <typename Lhs, typename Rhs>
DeviceMatrix<Scalar_>& DeviceMatrix<Scalar_>::operator=(const GemmExpr<Lhs, Rhs>& expr) {
  device(Context::threadLocal()) = expr;
  return *this;
}

template <typename Scalar_>
template <typename Lhs, typename Rhs>
DeviceMatrix<Scalar_>& DeviceMatrix<Scalar_>::operator+=(const GemmExpr<Lhs, Rhs>& expr) {
  device(Context::threadLocal()) += expr;
  return *this;
}

template <typename Scalar_>
template <typename Lhs, typename Rhs>
DeviceMatrix<Scalar_>& DeviceMatrix<Scalar_>::operator-=(const GemmExpr<Lhs, Rhs>& expr) {
  device(Context::threadLocal()) -= expr;
  return *this;
}

template <typename Scalar_>
template <int UpLo>
DeviceMatrix<Scalar_>& DeviceMatrix<Scalar_>::operator=(const LltSolveExpr<Scalar_, UpLo>& expr) {
  device(Context::threadLocal()) = expr;
  return *this;
}

template <typename Scalar_>
DeviceMatrix<Scalar_>& DeviceMatrix<Scalar_>::operator=(const LuSolveExpr<Scalar_>& expr) {
  device(Context::threadLocal()) = expr;
  return *this;
}

template <typename Scalar_>
template <int UpLo>
DeviceMatrix<Scalar_>& DeviceMatrix<Scalar_>::operator=(const TrsmExpr<Scalar_, UpLo>& expr) {
  device(Context::threadLocal()) = expr;
  return *this;
}

template <typename Scalar_>
template <int UpLo>
DeviceMatrix<Scalar_>& DeviceMatrix<Scalar_>::operator=(const SymmExpr<Scalar_, UpLo>& expr) {
  device(Context::threadLocal()) = expr;
  return *this;
}

template <typename Scalar_>
DeviceMatrix<Scalar_>& DeviceMatrix<Scalar_>::operator=(const Scaled<DeviceMatrix>& expr) {
  device(Context::threadLocal()) = expr;
  return *this;
}

// Enable copy-initialization straight from an expression, e.g.
//   DeviceMatrix<double> d_C = d_A * d_B;
// Each default-constructs and delegates to the matching operator=.

template <typename Scalar_>
template <typename Lhs, typename Rhs>
DeviceMatrix<Scalar_>::DeviceMatrix(const GemmExpr<Lhs, Rhs>& expr) : DeviceMatrix() {
  *this = expr;
}

template <typename Scalar_>
DeviceMatrix<Scalar_>::DeviceMatrix(const Scaled<DeviceMatrix>& expr) : DeviceMatrix() {
  *this = expr;
}

template <typename Scalar_>
DeviceMatrix<Scalar_>::DeviceMatrix(const DeviceAddExpr<Scalar_>& expr) : DeviceMatrix() {
  *this = expr;
}

template <typename Scalar_>
template <int UpLo>
DeviceMatrix<Scalar_>::DeviceMatrix(const LltSolveExpr<Scalar_, UpLo>& expr) : DeviceMatrix() {
  *this = expr;
}

template <typename Scalar_>
DeviceMatrix<Scalar_>::DeviceMatrix(const LuSolveExpr<Scalar_>& expr) : DeviceMatrix() {
  *this = expr;
}

template <typename Scalar_>
template <int UpLo>
DeviceMatrix<Scalar_>::DeviceMatrix(const TrsmExpr<Scalar_, UpLo>& expr) : DeviceMatrix() {
  *this = expr;
}

template <typename Scalar_>
template <int UpLo>
DeviceMatrix<Scalar_>::DeviceMatrix(const SymmExpr<Scalar_, UpLo>& expr) : DeviceMatrix() {
  *this = expr;
}

template <typename Scalar_, int UpLo_>
void SelfAdjointView<Scalar_, UpLo_>::rankUpdate(const DeviceMatrix<Scalar_>& A, RealScalar alpha) {
  SyrkExpr<Scalar_, UpLo_> expr(A);
  RealScalar beta = matrix().empty() ? RealScalar(0) : RealScalar(1);
  internal::dispatch(Context::threadLocal(), matrix(), expr, alpha, beta);
}

namespace internal {
// Runs `f` with the handle temporarily in CUBLAS_POINTER_MODE_DEVICE, restoring
// the caller's mode afterwards.
template <typename F>
void with_device_pointer_mode(cublasHandle_t h, F&& f) {
  cublasPointerMode_t prev;
  EIGEN_CUBLAS_CHECK(cublasGetPointerMode(h, &prev));
  EIGEN_CUBLAS_CHECK(cublasSetPointerMode(h, CUBLAS_POINTER_MODE_DEVICE));
  f();
  EIGEN_CUBLAS_CHECK(cublasSetPointerMode(h, prev));
}
}  // namespace internal

// Allocation, transfers, and the access protocol. Every method that enqueues
// work orders it with prepareRead/prepareWrite before and finishRead/finishWrite
// after, which is all it takes to be safe across contexts.

template <typename Scalar_>
DeviceMatrix<Scalar_>::DeviceMatrix(Context& ctx, Index rows, Index cols) {
  internal::DeviceMatrixAccess::resize(*this, ctx.streamHandle(), rows, cols);
}

template <typename Scalar_>
DeviceMatrix<Scalar_>::DeviceMatrix(Index rows, Index cols) : DeviceMatrix(Context::threadLocal(), rows, cols) {}

template <typename Scalar_>
DeviceMatrix<Scalar_>::DeviceMatrix(Index n) : DeviceMatrix(Context::threadLocal(), n, 1) {}

template <typename Scalar_>
void DeviceMatrix<Scalar_>::resize(Context& ctx, Index rows, Index cols) {
  internal::DeviceMatrixAccess::resize(*this, ctx.streamHandle(), rows, cols);
}

template <typename Scalar_>
void DeviceMatrix<Scalar_>::resize(Index rows, Index cols) {
  resize(Context::threadLocal(), rows, cols);
}

template <typename Scalar_>
void DeviceMatrix<Scalar_>::prepareRead(Context& ctx) const {
  buf_.prepareRead(ctx.streamHandle());
}

template <typename Scalar_>
void DeviceMatrix<Scalar_>::finishRead(Context& ctx) const {
  buf_.finishRead(ctx.streamHandle());
}

template <typename Scalar_>
void DeviceMatrix<Scalar_>::prepareWrite(Context& ctx) {
  buf_.prepareWrite(ctx.streamHandle());
}

template <typename Scalar_>
void DeviceMatrix<Scalar_>::finishWrite(Context& ctx) {
  eigen_assert((!buf_ || ctx.stream() == buf_.stream()) && "DeviceMatrix::finishWrite: call prepareWrite(ctx) first");
  EIGEN_UNUSED_VARIABLE(ctx);
  buf_.finishWrite();
}

template <typename Scalar_>
template <typename Derived>
DeviceMatrix<Scalar_> DeviceMatrix<Scalar_>::fromHost(Context& ctx, const DenseBase<Derived>& host) {
  // Ref binds any column-major direct-access input in place (no host copy);
  // row-major layouts and expressions evaluate into its temporary. A bound
  // block keeps its parent's outer stride, so the upload must honour
  // outerStride() rather than assume rows() -- see upload_host_matrix.
  const Ref<const PlainMatrix> mat(host.derived());
  DeviceMatrix dm(ctx, mat.rows(), mat.cols());
  if (dm.sizeInBytes() > 0) {
    internal::upload_host_matrix(dm.data(), mat.rows(), mat.data(), mat.outerStride(), mat.rows(), mat.cols(),
                                 ctx.stream());
    dm.finishWrite(ctx);
    EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(ctx.stream()));
  }
  return dm;
}

template <typename Scalar_>
template <typename Derived>
DeviceMatrix<Scalar_> DeviceMatrix<Scalar_>::fromHost(const DenseBase<Derived>& host) {
  return fromHost(Context::threadLocal(), host);
}

template <typename Scalar_>
DeviceMatrix<Scalar_> DeviceMatrix<Scalar_>::fromHostAsync(Context& ctx, const Scalar* host_data, Index rows,
                                                           Index cols) {
  eigen_assert(rows >= 0 && cols >= 0);
  eigen_assert(host_data != nullptr || (rows == 0 || cols == 0));
  DeviceMatrix dm(ctx, rows, cols);
  if (dm.sizeInBytes() > 0) {
    EIGEN_CUDA_RUNTIME_CHECK(
        cudaMemcpyAsync(dm.data(), host_data, dm.sizeInBytes(), cudaMemcpyHostToDevice, ctx.stream()));
    dm.finishWrite(ctx);
  }
  return dm;
}

template <typename Scalar_>
DeviceMatrix<Scalar_> DeviceMatrix<Scalar_>::fromHostAsync(const Scalar* host_data, Index rows, Index cols) {
  return fromHostAsync(Context::threadLocal(), host_data, rows, cols);
}

template <typename Scalar_>
typename DeviceMatrix<Scalar_>::PlainMatrix DeviceMatrix<Scalar_>::toHost(Context& ctx) const {
  PlainMatrix host_buf(rows_, cols_);
  if (sizeInBytes() > 0) {
    prepareRead(ctx);
    EIGEN_CUDA_RUNTIME_CHECK(
        cudaMemcpyAsync(host_buf.data(), data(), sizeInBytes(), cudaMemcpyDeviceToHost, ctx.stream()));
    finishRead(ctx);
    EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(ctx.stream()));
  }
  return host_buf;
}

template <typename Scalar_>
typename DeviceMatrix<Scalar_>::PlainMatrix DeviceMatrix<Scalar_>::toHost() const {
  return toHost(Context::threadLocal());
}

template <typename Scalar_>
HostTransfer<Scalar_> DeviceMatrix<Scalar_>::toHostAsync(Context& ctx) const {
  PlainMatrix host_buf(rows_, cols_);
  internal::PinnedHostBuffer pinned_buf(sizeInBytes());
  if (sizeInBytes() > 0) {
    prepareRead(ctx);
    EIGEN_CUDA_RUNTIME_CHECK(
        cudaMemcpyAsync(pinned_buf.get(), data(), sizeInBytes(), cudaMemcpyDeviceToHost, ctx.stream()));
    finishRead(ctx);
  }
  cudaEvent_t transfer_event;
  EIGEN_CUDA_RUNTIME_CHECK(cudaEventCreateWithFlags(&transfer_event, cudaEventDisableTiming));
  EIGEN_CUDA_RUNTIME_CHECK(cudaEventRecord(transfer_event, ctx.stream()));
  return HostTransfer<Scalar>(std::move(host_buf), std::move(pinned_buf), transfer_event);
}

template <typename Scalar_>
HostTransfer<Scalar_> DeviceMatrix<Scalar_>::toHostAsync() const {
  return toHostAsync(Context::threadLocal());
}

template <typename Scalar_>
DeviceMatrix<Scalar_> DeviceMatrix<Scalar_>::clone(Context& ctx) const {
  DeviceMatrix result(ctx, rows_, cols_);
  if (sizeInBytes() > 0) {
    prepareRead(ctx);
    EIGEN_CUDA_RUNTIME_CHECK(
        cudaMemcpyAsync(result.data(), data(), sizeInBytes(), cudaMemcpyDeviceToDevice, ctx.stream()));
    finishRead(ctx);
    result.finishWrite(ctx);
  }
  return result;
}

template <typename Scalar_>
DeviceMatrix<Scalar_> DeviceMatrix<Scalar_>::clone() const {
  return clone(Context::threadLocal());
}

template <typename Scalar_>
DeviceMatrix<Scalar_> DeviceMatrix<Scalar_>::adopt(Context& ctx, Scalar* device_ptr, Index rows, Index cols) {
  eigen_assert(rows >= 0 && cols >= 0);
  const size_t bytes = static_cast<size_t>(rows) * static_cast<size_t>(cols) * sizeof(Scalar);
  return internal::DeviceMatrixAccess::wrap<Scalar>(
      internal::DeviceBuffer::adopt(device_ptr, bytes, ctx.streamHandle()), rows, cols);
}

template <typename Scalar_>
DeviceMatrix<Scalar_> DeviceMatrix<Scalar_>::adopt(Scalar* device_ptr, Index rows, Index cols) {
  return adopt(Context::threadLocal(), device_ptr, rows, cols);
}

template <typename Scalar_>
DeviceMatrix<Scalar_> DeviceMatrix<Scalar_>::view(Context& ctx, Scalar* device_ptr, Index rows, Index cols) {
  eigen_assert(rows >= 0 && cols >= 0);
  const size_t bytes = static_cast<size_t>(rows) * static_cast<size_t>(cols) * sizeof(Scalar);
  return internal::DeviceMatrixAccess::wrap<Scalar>(
      internal::DeviceBuffer::borrow(device_ptr, bytes, ctx.streamHandle()), rows, cols);
}

template <typename Scalar_>
DeviceMatrix<Scalar_> DeviceMatrix<Scalar_>::view(Scalar* device_ptr, Index rows, Index cols) {
  return view(Context::threadLocal(), device_ptr, rows, cols);
}

template <typename Scalar_>
Scalar_* DeviceMatrix<Scalar_>::release(Context& ctx) {
  Scalar* p = static_cast<Scalar*>(buf_.release(ctx.streamHandle()));
  rows_ = 0;
  cols_ = 0;
  return p;
}

template <typename Scalar_>
Scalar_* DeviceMatrix<Scalar_>::release() {
  return release(Context::threadLocal());
}

template <typename Scalar_>
DeviceScalar<Scalar_>::DeviceScalar() : DeviceScalar(Context::threadLocal()) {}

template <typename Scalar_>
DeviceScalar<Scalar_>::DeviceScalar(Context& ctx) : DeviceScalar(ctx.streamHandle()) {}

template <typename Scalar_>
DeviceScalar<Scalar_>::DeviceScalar(Context& ctx, Scalar value) : DeviceScalar(ctx.streamHandle(), value) {}

template <typename Scalar_>
void DeviceScalar<Scalar_>::prepareRead(Context& ctx) const {
  d_val_.prepareRead(ctx.streamHandle());
}

template <typename Scalar_>
void DeviceScalar<Scalar_>::finishRead(Context& ctx) const {
  d_val_.finishRead(ctx.streamHandle());
}

template <typename Scalar_>
void DeviceScalar<Scalar_>::prepareWrite(Context& ctx) {
  d_val_.prepareWrite(ctx.streamHandle());
}

template <typename Scalar_>
void DeviceScalar<Scalar_>::finishWrite(Context& ctx) {
  eigen_assert((!d_val_ || ctx.stream() == d_val_.stream()) &&
               "DeviceScalar::finishWrite: call prepareWrite(ctx) first");
  EIGEN_UNUSED_VARIABLE(ctx);
  d_val_.finishWrite(/*record_event=*/false);
}

// The reductions below (dot, norm, squaredNorm) run under
// CUBLAS_POINTER_MODE_DEVICE: the scalar result is written to device memory and
// stays there until DeviceScalar's conversion to Scalar syncs and reads it.

namespace internal {
inline int64_t blas1_size(Index rows, Index cols) { return static_cast<int64_t>(rows) * static_cast<int64_t>(cols); }
}  // namespace internal

template <typename Scalar_>
DeviceScalar<typename DeviceMatrix<Scalar_>::Scalar> DeviceMatrix<Scalar_>::dot(Context& ctx,
                                                                                const DeviceMatrix& other) const {
  const int64_t n = internal::blas1_size(rows_, cols_);
  eigen_assert(n == internal::blas1_size(other.rows_, other.cols_));
  if (n > 0) {
    // Allocated uninitialized: cublasXdot overwrites the slot, so uploading a
    // zero first would be a wasted H2D transfer per reduction.
    DeviceScalar<Scalar> result(ctx);
    prepareRead(ctx);
    other.prepareRead(ctx);
    internal::with_device_pointer_mode(ctx.cublasHandle(), [&] {
      EIGEN_CUBLAS_CHECK(internal::cublasXdot(ctx.cublasHandle(), n, data(), 1, other.data(), 1, result.devicePtr()));
    });
    finishRead(ctx);
    other.finishRead(ctx);
    result.finishWrite(ctx);
    return result;
  }
  return DeviceScalar<Scalar>(ctx, Scalar(0));
}

namespace internal {
// For real Scalar, dot(x,x) already has type DeviceScalar<RealScalar>, so a move
// suffices and nothing syncs.
template <typename Scalar, typename RealScalar>
std::enable_if_t<std::is_same<Scalar, RealScalar>::value, DeviceScalar<RealScalar>> squaredNorm_from_dot(
    DeviceScalar<Scalar>&& d, Context&) {
  return std::move(d);
}
// Complex must sync to extract the real part: DeviceScalar arithmetic is real-only.
template <typename Scalar, typename RealScalar>
std::enable_if_t<!std::is_same<Scalar, RealScalar>::value, DeviceScalar<RealScalar>> squaredNorm_from_dot(
    DeviceScalar<Scalar>&& d, Context& ctx) {
  return DeviceScalar<RealScalar>(ctx, numext::real(Scalar(d)));
}
}  // namespace internal

template <typename Scalar_>
DeviceScalar<typename NumTraits<Scalar_>::Real> DeviceMatrix<Scalar_>::squaredNorm(Context& ctx) const {
  // dot(x,x) rather than nrm2()^2: the dot kernel is ~4.5x faster. It has no
  // overflow protection, so callers guard the scale of x themselves; Eigen's
  // iterative solver templates call stableNorm() instead.
  return internal::squaredNorm_from_dot<Scalar_, RealScalar>(dot(ctx, *this), ctx);
}

template <typename Scalar_>
DeviceScalar<typename NumTraits<Scalar_>::Real> DeviceMatrix<Scalar_>::norm(Context& ctx) const {
  const int64_t n = internal::blas1_size(rows_, cols_);
  if (n > 0) {
    // See dot(): uninitialized on purpose, cublasXnrm2 overwrites the slot.
    DeviceScalar<RealScalar> result(ctx);
    prepareRead(ctx);
    internal::with_device_pointer_mode(ctx.cublasHandle(), [&] {
      EIGEN_CUBLAS_CHECK(internal::cublasXnrm2(ctx.cublasHandle(), n, data(), 1, result.devicePtr()));
    });
    finishRead(ctx);
    result.finishWrite(ctx);
    return result;
  }
  return DeviceScalar<RealScalar>(ctx, RealScalar(0));
}

template <typename Scalar_>
void DeviceMatrix<Scalar_>::setZero(Context& ctx) {
  if (sizeInBytes() > 0) {
    prepareWrite(ctx);
    EIGEN_CUDA_RUNTIME_CHECK(cudaMemsetAsync(data(), 0, sizeInBytes(), ctx.stream()));
    finishWrite(ctx);
  }
}

template <typename Scalar_>
void DeviceMatrix<Scalar_>::addScaled(Context& ctx, Scalar alpha, const DeviceMatrix& x) {
  const int64_t n = internal::blas1_size(rows_, cols_);
  eigen_assert(n == internal::blas1_size(x.rows_, x.cols_));
  if (n > 0) {
    x.prepareRead(ctx);
    prepareWrite(ctx);
    EIGEN_CUBLAS_CHECK(internal::cublasXaxpy(ctx.cublasHandle(), n, &alpha, x.data(), 1, data(), 1));
    x.finishRead(ctx);
    finishWrite(ctx);
  }
}

template <typename Scalar_>
void DeviceMatrix<Scalar_>::scale(Context& ctx, Scalar alpha) {
  const int64_t n = internal::blas1_size(rows_, cols_);
  if (n > 0) {
    prepareWrite(ctx);
    EIGEN_CUBLAS_CHECK(internal::cublasXscal(ctx.cublasHandle(), n, &alpha, data(), 1));
    finishWrite(ctx);
  }
}

template <typename Scalar_>
void DeviceMatrix<Scalar_>::copyFrom(Context& ctx, const DeviceMatrix& other) {
  resize(ctx, other.rows_, other.cols_);
  const int64_t n = internal::blas1_size(rows_, cols_);
  if (n > 0) {
    other.prepareRead(ctx);
    prepareWrite(ctx);
    EIGEN_CUBLAS_CHECK(internal::cublasXcopy(ctx.cublasHandle(), n, other.data(), 1, data(), 1));
    other.finishRead(ctx);
    finishWrite(ctx);
  }
}

// this += alpha * x  (axpy)
template <typename Scalar_>
DeviceMatrix<Scalar_>& DeviceMatrix<Scalar_>::operator+=(const Scaled<DeviceMatrix>& expr) {
  addScaled(Context::threadLocal(), expr.scalar(), internal::device_expr_traits<DeviceMatrix>::matrix(expr.inner()));
  return *this;
}

// this -= alpha * x  (axpy with negated alpha)
template <typename Scalar_>
DeviceMatrix<Scalar_>& DeviceMatrix<Scalar_>::operator-=(const Scaled<DeviceMatrix>& expr) {
  addScaled(Context::threadLocal(), -expr.scalar(), internal::device_expr_traits<DeviceMatrix>::matrix(expr.inner()));
  return *this;
}

// this += x  (axpy with alpha=1)
template <typename Scalar_>
DeviceMatrix<Scalar_>& DeviceMatrix<Scalar_>::operator+=(const DeviceMatrix& other) {
  Scalar one(1);
  addScaled(Context::threadLocal(), one, other);
  return *this;
}

// this -= x  (axpy with alpha=-1)
template <typename Scalar_>
DeviceMatrix<Scalar_>& DeviceMatrix<Scalar_>::operator-=(const DeviceMatrix& other) {
  Scalar neg_one(-1);
  addScaled(Context::threadLocal(), neg_one, other);
  return *this;
}

// this *= alpha  (scal, host pointer)
template <typename Scalar_>
DeviceMatrix<Scalar_>& DeviceMatrix<Scalar_>::operator*=(Scalar alpha) {
  scale(Context::threadLocal(), alpha);
  return *this;
}

namespace internal {
// x[i] /= alpha. Real Scalar: NPP divides in place. Complex Scalar: cuBLAS scal by the
// host reciprocal, which std::complex computes with scaling, so |alpha| > sqrt(max)
// does not overflow; one extra rounding per element.
inline void divide_in_place(Context& ctx, float* x, int64_t n, float alpha) {
  device_divC(alpha, x, Eigen::internal::convert_index<int>(n), ctx.stream());
}
inline void divide_in_place(Context& ctx, double* x, int64_t n, double alpha) {
  device_divC(alpha, x, Eigen::internal::convert_index<int>(n), ctx.stream());
}
template <typename Real>
void divide_in_place(Context& ctx, std::complex<Real>* x, int64_t n, std::complex<Real> alpha) {
  const std::complex<Real> inv = std::complex<Real>(1) / alpha;
  EIGEN_CUBLAS_CHECK(cublasXscal(ctx.cublasHandle(), n, &inv, x, 1));
}
}  // namespace internal

template <typename Scalar_>
void DeviceMatrix<Scalar_>::divide(Context& ctx, Scalar alpha) {
  const int64_t n = internal::blas1_size(rows_, cols_);
  if (n > 0) {
    prepareWrite(ctx);
    internal::divide_in_place(ctx, data(), n, alpha);
    finishWrite(ctx);
  }
}

// this /= alpha
template <typename Scalar_>
DeviceMatrix<Scalar_>& DeviceMatrix<Scalar_>::operator/=(Scalar alpha) {
  divide(Context::threadLocal(), alpha);
  return *this;
}

// Deep copies: device-to-device cuBLAS copy on the thread-local Context.
template <typename Scalar_>
DeviceMatrix<Scalar_>::DeviceMatrix(const DeviceMatrix& other) : DeviceMatrix() {
  copyFrom(Context::threadLocal(), other);
}

template <typename Scalar_>
DeviceMatrix<Scalar_>& DeviceMatrix<Scalar_>::operator=(const DeviceMatrix& other) {
  if (this != &other) copyFrom(Context::threadLocal(), other);
  return *this;
}

template <typename Scalar_>
DeviceScalar<typename NumTraits<Scalar_>::Real> DeviceMatrix<Scalar_>::stableNorm(Context& ctx) const {
  return norm(ctx);
}

template <typename Scalar_>
DeviceScalar<typename NumTraits<Scalar_>::Real> DeviceMatrix<Scalar_>::stableNorm() const {
  return norm(Context::threadLocal());
}

// this *= alpha  (scal, device pointer — avoids host sync)
template <typename Scalar_>
DeviceMatrix<Scalar_>& DeviceMatrix<Scalar_>::operator*=(const DeviceScalar<Scalar>& alpha) {
  const int64_t n = internal::blas1_size(rows_, cols_);
  if (n > 0) {
    Context& ctx = Context::threadLocal();
    alpha.prepareRead(ctx);
    prepareWrite(ctx);
    internal::with_device_pointer_mode(ctx.cublasHandle(), [&] {
      EIGEN_CUBLAS_CHECK(internal::cublasXscal(ctx.cublasHandle(), n, alpha.devicePtr(), data(), 1));
    });
    alpha.finishRead(ctx);
    finishWrite(ctx);
  }
  return *this;
}

// this += DeviceScalar * x  (axpy with CUBLAS_POINTER_MODE_DEVICE)
template <typename Scalar_>
DeviceMatrix<Scalar_>& DeviceMatrix<Scalar_>::operator+=(const DeviceScaledDevice<Scalar_>& expr) {
  const int64_t n = internal::blas1_size(rows_, cols_);
  const auto& x = expr.matrix();
  eigen_assert(n == internal::blas1_size(x.rows_, x.cols_));
  if (n > 0) {
    Context& ctx = Context::threadLocal();
    expr.alpha().prepareRead(ctx);
    x.prepareRead(ctx);
    prepareWrite(ctx);
    internal::with_device_pointer_mode(ctx.cublasHandle(), [&] {
      EIGEN_CUBLAS_CHECK(
          internal::cublasXaxpy(ctx.cublasHandle(), n, expr.alpha().devicePtr(), x.data(), 1, data(), 1));
    });
    expr.alpha().finishRead(ctx);
    x.finishRead(ctx);
    finishWrite(ctx);
  }
  return *this;
}

// this -= DeviceScalar * x  (axpy with negated device scalar)
template <typename Scalar_>
DeviceMatrix<Scalar_>& DeviceMatrix<Scalar_>::operator-=(const DeviceScaledDevice<Scalar_>& expr) {
  auto neg_alpha = -expr.alpha();
  DeviceScaledDevice<Scalar_> neg_expr(neg_alpha, expr.matrix());
  return operator+=(neg_expr);
}

// this = alpha * A + beta * B  (cuBLAS geam)
template <typename Scalar_>
DeviceMatrix<Scalar_>& DeviceMatrix<Scalar_>::operator=(const DeviceAddExpr<Scalar_>& expr) {
  internal::dispatch(Context::threadLocal(), *this, expr);
  return *this;
}

// cwiseProduct (allocating).
template <typename Scalar_>
DeviceMatrix<Scalar_> DeviceMatrix<Scalar_>::cwiseProduct(Context& ctx, const DeviceMatrix& other) const {
  const int64_t n = internal::blas1_size(rows_, cols_);
  eigen_assert(n == internal::blas1_size(other.rows_, other.cols_));
  DeviceMatrix result(ctx, rows_, cols_);
  if (n > 0) {
    prepareRead(ctx);
    other.prepareRead(ctx);
    internal::device_cwiseProduct(data(), other.data(), result.data(), Eigen::internal::convert_index<int>(n),
                                  ctx.stream());
    finishRead(ctx);
    other.finishRead(ctx);
    result.finishWrite(ctx);
  }
  return result;
}

// In-place cwiseProduct: this = a .* b (reuses this buffer, no allocation).
template <typename Scalar_>
void DeviceMatrix<Scalar_>::cwiseProduct(Context& ctx, const DeviceMatrix& a, const DeviceMatrix& b) {
  const int64_t n = internal::blas1_size(a.rows_, a.cols_);
  eigen_assert(n == internal::blas1_size(b.rows_, b.cols_));
  resize(ctx, a.rows_, a.cols_);
  if (n > 0) {
    a.prepareRead(ctx);
    b.prepareRead(ctx);
    prepareWrite(ctx);
    internal::device_cwiseProduct(a.data(), b.data(), data(), Eigen::internal::convert_index<int>(n), ctx.stream());
    a.finishRead(ctx);
    b.finishRead(ctx);
    finishWrite(ctx);
  }
}

// Convenience overloads using thread-local default Context.
template <typename Scalar_>
DeviceScalar<typename DeviceMatrix<Scalar_>::Scalar> DeviceMatrix<Scalar_>::dot(const DeviceMatrix& other) const {
  return dot(Context::threadLocal(), other);
}

template <typename Scalar_>
DeviceScalar<typename NumTraits<Scalar_>::Real> DeviceMatrix<Scalar_>::squaredNorm() const {
  return squaredNorm(Context::threadLocal());
}

template <typename Scalar_>
DeviceScalar<typename NumTraits<Scalar_>::Real> DeviceMatrix<Scalar_>::norm() const {
  return norm(Context::threadLocal());
}

template <typename Scalar_>
void DeviceMatrix<Scalar_>::setZero() {
  setZero(Context::threadLocal());
}
}  // namespace gpu
}  // namespace Eigen

#endif  // EIGEN_GPU_DEVICE_DISPATCH_H
