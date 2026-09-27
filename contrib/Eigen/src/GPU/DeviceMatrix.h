// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// Typed RAII wrapper for a dense column-major matrix in GPU device memory.
//
// Cross-stream safety is automatic: the storage is an internal::DeviceBuffer,
// which orders reads after the last write, writes after earlier reads and
// writes, and the free after every access, on whichever streams they ran.

#ifndef EIGEN_GPU_DEVICE_MATRIX_H
#define EIGEN_GPU_DEVICE_MATRIX_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

#include <cstring>

#include "./FwdDecl.h"
#include "./GpuSupport.h"

namespace Eigen {
namespace gpu {

/** \ingroup GPU_Module
 * \class HostTransfer
 * \brief Future for an asynchronous device-to-host matrix transfer.
 *
 * Returned by gpu::DeviceMatrix::toHostAsync(). The transfer runs asynchronously
 * on the given CUDA stream. Call get() to block until complete and retrieve
 * the host matrix, or ready() to poll without blocking.
 */
template <typename Scalar_>
class HostTransfer {
 public:
  using Scalar = Scalar_;
  using PlainMatrix = Eigen::Matrix<Scalar, Dynamic, Dynamic, ColMajor>;

  /** Block until the transfer completes and return the host matrix.
   * Idempotent: subsequent calls return the same matrix without re-syncing. */
  PlainMatrix& get() {
    if (!synced_) {
      EIGEN_CUDA_RUNTIME_CHECK(cudaEventSynchronize(event_));
      if (pinned_buf_ && host_buf_.size() > 0) {
        std::memcpy(host_buf_.data(), pinned_buf_.get(), static_cast<size_t>(host_buf_.size()) * sizeof(Scalar));
      }
      pinned_buf_ = internal::PinnedHostBuffer();  // free pinned memory early
      synced_ = true;
    }
    return host_buf_;
  }

  /** Non-blocking check: has the transfer completed? */
  bool ready() const {
    if (synced_) return true;
    const cudaError_t err = cudaEventQuery(event_);
    if (err == cudaSuccess) return true;
    if (err != cudaErrorNotReady)
      EIGEN_GPU_CHECK_FAILED(cudaGetErrorName(err), "cudaEventQuery(event_)", __FILE__, __LINE__);
    return false;
  }

  // The pinned staging buffer's cudaFreeHost is not stream-ordered, so a
  // pending transfer is waited for before the buffer goes.
  ~HostTransfer() { discard(); }

  HostTransfer(HostTransfer&& o) noexcept
      : host_buf_(std::move(o.host_buf_)), pinned_buf_(std::move(o.pinned_buf_)), event_(o.event_), synced_(o.synced_) {
    o.event_ = nullptr;
    o.synced_ = true;
  }

  HostTransfer& operator=(HostTransfer&& o) noexcept {
    if (this != &o) {
      discard();
      host_buf_ = std::move(o.host_buf_);
      pinned_buf_ = std::move(o.pinned_buf_);
      event_ = o.event_;
      synced_ = o.synced_;
      o.event_ = nullptr;
      o.synced_ = true;
    }
    return *this;
  }

  HostTransfer(const HostTransfer&) = delete;
  HostTransfer& operator=(const HostTransfer&) = delete;

 private:
  template <typename>
  friend class DeviceMatrix;

  HostTransfer(PlainMatrix&& buf, internal::PinnedHostBuffer&& pinned, cudaEvent_t event)
      : host_buf_(std::move(buf)), pinned_buf_(std::move(pinned)), event_(event), synced_(false) {}

  // Unchecked: runs from the destructor and a noexcept operator, where eigen_assert may not throw.
  void discard() noexcept {
    if (!event_) return;
    if (!synced_) (void)cudaEventSynchronize(event_);
    (void)cudaEventDestroy(event_);
    event_ = nullptr;
  }

  PlainMatrix host_buf_;                   // final destination (pageable)
  internal::PinnedHostBuffer pinned_buf_;  // staging buffer for async DMA
  cudaEvent_t event_ = nullptr;
  bool synced_ = false;
};

/** \ingroup GPU_Module
 * \class DeviceMatrix
 * \brief RAII wrapper for a dense column-major matrix in GPU device memory.
 *
 * \tparam Scalar_  Element type: float, double, complex<float>, complex<double>
 *
 * Owns a stream-ordered device allocation with tracked dimensions (the leading
 * dimension is always rows()). Every operation that enqueues work runs on a
 * gpu::Context: the one passed as the first argument, or Context::threadLocal()
 * for the overloads and operators that take none.
 *
 * Accesses are ordered across contexts automatically. The matrix remembers the
 * stream of its last write (stream()) and the streams that have read it since:
 * a read on another stream waits for that write, a write waits for every
 * earlier access, and the memory is freed on stream() after all of them —
 * without blocking the host. Several threads, each on its own Context, may
 * read one matrix concurrently; a write needs exclusive access.
 *
 * To access data() from your own kernels, bracket the launch on ctx.stream()
 * the same way the library does:
 * \code
 * d_A.prepareRead(ctx);  d_C.prepareWrite(ctx);
 * my_kernel<<<grid, block, 0, ctx.stream()>>>(d_A.data(), d_C.data());
 * d_A.finishRead(ctx);   d_C.finishWrite(ctx);
 * \endcode
 *
 * Transfers come in synchronous and asynchronous variants: fromHost() /
 * fromHostAsync() and toHost() / toHostAsync().
 */
template <typename Scalar_>
class DeviceMatrix {
 public:
  using Scalar = Scalar_;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using PlainObject = DeviceMatrix;  // owning type, as generic solver code expects
  using PlainMatrix = Eigen::Matrix<Scalar, Dynamic, Dynamic, ColMajor>;

  /** Default: empty (0x0, no allocation). */
  DeviceMatrix() = default;

  /** Allocate an uninitialized column vector on the thread-local Context,
   * mirroring Matrix<Scalar,Dynamic,1>(n) so generic solver code compiles unchanged. */
  explicit DeviceMatrix(Index n);

  /** Allocate uninitialized device memory for a rows x cols matrix on the thread-local Context. */
  DeviceMatrix(Index rows, Index cols);

  /** Allocate uninitialized device memory for a rows x cols matrix on \p ctx. */
  DeviceMatrix(Context& ctx, Index rows, Index cols);

  // Copy-initialization from a device expression, mirroring the Eigen CPU idiom
  // `DeviceMatrix<double> d_C = d_A * d_B;`. Each delegates to the corresponding
  // operator= on the thread-local Context, and is defined out-of-line in
  // DeviceDispatch.h — GpuSparseContext.h for SpMV — where Context is complete.

  template <typename Lhs, typename Rhs>
  DeviceMatrix(const GemmExpr<Lhs, Rhs>& expr);
  DeviceMatrix(const Scaled<DeviceMatrix>& expr);
  DeviceMatrix(const DeviceAddExpr<Scalar>& expr);
  template <int UpLo>
  DeviceMatrix(const LltSolveExpr<Scalar, UpLo>& expr);
  DeviceMatrix(const LuSolveExpr<Scalar>& expr);
  template <int UpLo>
  DeviceMatrix(const TrsmExpr<Scalar, UpLo>& expr);
  template <int UpLo>
  DeviceMatrix(const SymmExpr<Scalar, UpLo>& expr);
  DeviceMatrix(const SpMVExpr<Scalar>& expr);
  DeviceMatrix(const SpMVAffineExpr<Scalar>& expr);

  DeviceMatrix(DeviceMatrix&& o) noexcept : buf_(std::move(o.buf_)), rows_(o.rows_), cols_(o.cols_) {
    o.rows_ = 0;
    o.cols_ = 0;
  }

  DeviceMatrix& operator=(DeviceMatrix&& o) noexcept {
    if (this != &o) {
      buf_ = std::move(o.buf_);
      rows_ = o.rows_;
      cols_ = o.cols_;
      o.rows_ = 0;
      o.cols_ = 0;
    }
    return *this;
  }

  /** Deep copy: a device-to-device cuBLAS copy on the thread-local Context,
   * asynchronous and without a host transfer. Copies exist so that generic Eigen
   * algorithm code with value semantics (`p = precond.solve(residual)` in
   * internal::conjugate_gradient) compiles against DeviceMatrix; code that
   * manages contexts explicitly should prefer copyFrom(ctx, other). Defined
   * out-of-line in DeviceDispatch.h, where Context is complete. */
  DeviceMatrix(const DeviceMatrix& other);
  DeviceMatrix& operator=(const DeviceMatrix& other);

  /** Upload a host Eigen matrix to device memory on \p ctx, synchronously: the
   * copy has completed on return, so \p host may then be modified. Plain
   * column-major input (any outer stride) is transferred directly; other
   * expressions are first evaluated into a contiguous temporary. */
  template <typename Derived>
  static DeviceMatrix fromHost(Context& ctx, const DenseBase<Derived>& host);

  /** fromHost() on the thread-local Context. */
  template <typename Derived>
  static DeviceMatrix fromHost(const DenseBase<Derived>& host);

  /** Upload contiguous column-major host data asynchronously on \p ctx. The
   * caller must keep \p host_data alive and unmodified until the copy has
   * executed, e.g. until a later synchronizing call on \p ctx. */
  static DeviceMatrix fromHostAsync(Context& ctx, const Scalar* host_data, Index rows, Index cols);

  /** fromHostAsync() on the thread-local Context. */
  static DeviceMatrix fromHostAsync(const Scalar* host_data, Index rows, Index cols);

  /** Download to host memory on \p ctx; blocks until the copy has completed. */
  PlainMatrix toHost(Context& ctx) const;

  /** toHost() on the thread-local Context. */
  PlainMatrix toHost() const;

  /** Enqueue a device-to-host transfer on \p ctx and return a future;
   * HostTransfer::get() blocks and returns the host matrix. */
  HostTransfer<Scalar> toHostAsync(Context& ctx) const;

  /** toHostAsync() on the thread-local Context. */
  HostTransfer<Scalar> toHostAsync() const;

  /** Deep copy on device, enqueued on \p ctx without a host sync. */
  DeviceMatrix clone(Context& ctx) const;

  /** clone() on the thread-local Context. */
  DeviceMatrix clone() const;

  /** Discard contents and resize to (rows x cols); contents are undefined
   * afterwards. Keeps the existing allocation when it is large enough, so
   * cycling through same-or-smaller shapes does not allocate; otherwise the old
   * allocation is freed (stream-ordered) and a new one is made on \p ctx. */
  void resize(Context& ctx, Index rows, Index cols);

  /** resize() allocating on the thread-local Context. */
  void resize(Index rows, Index cols);

  Scalar* data() { return static_cast<Scalar*>(buf_.get()); }
  const Scalar* data() const { return static_cast<const Scalar*>(buf_.get()); }
  Index rows() const { return rows_; }
  Index cols() const { return cols_; }
  bool empty() const { return rows_ == 0 || cols_ == 0; }

  /** Size of the matrix data in bytes. */
  size_t sizeInBytes() const { return static_cast<size_t>(rows_) * static_cast<size_t>(cols_) * sizeof(Scalar); }

  /** The stream of the last write: the matrix's pending work is ordered on it,
   * and its memory is freed there. */
  cudaStream_t stream() const { return buf_.stream(); }

  /** Before reading data() on \p ctx's stream: wait for the last write. */
  void prepareRead(Context& ctx) const;

  /** After enqueuing a read on \p ctx's stream: later writes and the free wait for it. */
  void finishRead(Context& ctx) const;

  /** Before writing data() on \p ctx's stream (including a read-modify-write):
   * wait for every earlier access. \p ctx's stream becomes stream(). */
  void prepareWrite(Context& ctx);

  /** After enqueuing a write on \p ctx's stream: later reads on other streams wait for it. */
  void finishWrite(Context& ctx);

  /** Adjoint view: maps to a GEMM operand with ConjTrans. */
  AdjointView<Scalar> adjoint() const { return AdjointView<Scalar>(*this); }

  /** Transpose view: maps to a GEMM operand with Trans. */
  TransposeView<Scalar> transpose() const { return TransposeView<Scalar>(*this); }

  /** Bind this matrix to a Context for expression assignment:
   * `d_C.device(ctx) = d_A * d_B;` */
  Assignment<Scalar> device(Context& ctx) { return Assignment<Scalar>(*this, ctx); }

  template <typename Lhs, typename Rhs>
  DeviceMatrix& operator=(const GemmExpr<Lhs, Rhs>& expr);

  template <typename Lhs, typename Rhs>
  DeviceMatrix& operator+=(const GemmExpr<Lhs, Rhs>& expr);

  /** Subtract a GEMM expression using the thread-local default Context. */
  template <typename Lhs, typename Rhs>
  DeviceMatrix& operator-=(const GemmExpr<Lhs, Rhs>& expr);

  /** Cholesky view: d_A.llt().solve(d_B) → LltSolveExpr. */
  LLTView<Scalar, Lower> llt() const { return LLTView<Scalar, Lower>(*this); }

  /** Cholesky view with explicit triangle: d_A.llt<Upper>().solve(d_B). */
  template <int UpLo>
  LLTView<Scalar, UpLo> llt() const {
    return LLTView<Scalar, UpLo>(*this);
  }

  /** LU view: d_A.lu().solve(d_B) → LuSolveExpr. */
  LUView<Scalar> lu() const { return LUView<Scalar>(*this); }

  template <int UpLo>
  DeviceMatrix& operator=(const LltSolveExpr<Scalar, UpLo>& expr);

  DeviceMatrix& operator=(const LuSolveExpr<Scalar>& expr);

  /** Triangular view: d_A.triangularView<Lower>().solve(d_B) → TrsmExpr. */
  template <int UpLo>
  TriangularView<Scalar, UpLo> triangularView() const {
    return TriangularView<Scalar, UpLo>(*this);
  }

  /** Self-adjoint view (mutable): d_C.selfadjointView<Lower>().rankUpdate(d_A). */
  template <int UpLo>
  SelfAdjointView<Scalar, UpLo> selfadjointView() {
    return SelfAdjointView<Scalar, UpLo>(*this);
  }

  /** Self-adjoint view (const): d_A.selfadjointView<Lower>() * d_B → SymmExpr. */
  template <int UpLo>
  ConstSelfAdjointView<Scalar, UpLo> selfadjointView() const {
    return ConstSelfAdjointView<Scalar, UpLo>(*this);
  }

  template <int UpLo>
  DeviceMatrix& operator=(const TrsmExpr<Scalar, UpLo>& expr);

  template <int UpLo>
  DeviceMatrix& operator=(const SymmExpr<Scalar, UpLo>& expr);

  // A DeviceMatrix is always dense (lda == rows) and a vector is one with
  // cols == 1, so the BLAS-1 methods below simply run over the flat rows*cols
  // array and serve both. Passing an explicit Context& lets callers keep every
  // operation on one stream, which elides the cross-stream event waits.

  /** Dot product: this^H * other. The result stays on device until read through
   * DeviceScalar's conversion to Scalar, which syncs. */
  DeviceScalar<Scalar> dot(Context& ctx, const DeviceMatrix& other) const;

  /** Squared L2 norm via dot(x, x). For real types the result stays on device;
   * for complex it syncs, since DeviceScalar arithmetic is real-only. */
  DeviceScalar<typename NumTraits<Scalar>::Real> squaredNorm(Context& ctx) const;

  /** L2 norm, without a host sync. */
  DeviceScalar<typename NumTraits<Scalar>::Real> norm(Context& ctx) const;

  /** Overflow-safe L2 norm, the same as norm(): cuBLAS nrm2 already runs a
   * scaled sum of squares. Provided so that Eigen's iterative solver templates,
   * which call stableNorm(), compile against DeviceMatrix. */
  DeviceScalar<typename NumTraits<Scalar>::Real> stableNorm(Context& ctx) const;

  /** Set all elements to zero. */
  void setZero(Context& ctx);

  /** this += alpha * x (cuBLAS axpy). Requires same total size. */
  void addScaled(Context& ctx, Scalar alpha, const DeviceMatrix& x);

  /** this *= alpha (cuBLAS scal). */
  void scale(Context& ctx, Scalar alpha);

  /** this /= alpha. A true division for real Scalar (NPP divide-by-constant: one
   * rounding per element, and no overflow of 1/alpha for subnormal alpha); complex
   * Scalar scales by the host reciprocal, one extra rounding. */
  void divide(Context& ctx, Scalar alpha);

  /** Deep copy: this = other (cuBLAS copy). Resizes if needed. */
  void copyFrom(Context& ctx, const DeviceMatrix& other);

  DeviceScalar<Scalar> dot(const DeviceMatrix& other) const;
  DeviceScalar<typename NumTraits<Scalar>::Real> squaredNorm() const;
  DeviceScalar<typename NumTraits<Scalar>::Real> norm() const;
  DeviceScalar<typename NumTraits<Scalar>::Real> stableNorm() const;
  void setZero();

  // The operators below let iterative-solver code written against Matrix — say
  // `x += alpha * p` — compile unchanged against DeviceMatrix, dispatching to
  // cuBLAS axpy/scal. `alpha * DeviceMatrix` yields Scaled<DeviceMatrix<Scalar>>
  // from DeviceExpr.h.

  /** this += alpha * x (cuBLAS axpy). */
  DeviceMatrix& operator+=(const Scaled<DeviceMatrix>& expr);

  /** this -= alpha * x (cuBLAS axpy with negated alpha). */
  DeviceMatrix& operator-=(const Scaled<DeviceMatrix>& expr);

  /** this += x (cuBLAS axpy with alpha=1). */
  DeviceMatrix& operator+=(const DeviceMatrix& other);

  /** this -= x (cuBLAS axpy with alpha=-1). */
  DeviceMatrix& operator-=(const DeviceMatrix& other);

  /** this *= alpha (cuBLAS scal, host pointer mode). */
  DeviceMatrix& operator*=(Scalar alpha);

  /** this /= alpha, see divide(). */
  DeviceMatrix& operator/=(Scalar alpha);

  /** this *= alpha (cuBLAS scal, device pointer mode). Avoids a host sync. */
  DeviceMatrix& operator*=(const DeviceScalar<Scalar>& alpha);

  /** Element-wise product: result[i] = this[i] * other[i]. */
  DeviceMatrix cwiseProduct(Context& ctx, const DeviceMatrix& other) const;

  /** In-place element-wise product: this[i] = a[i] * b[i].
   * Reuses this matrix's buffer when sizes match, avoiding cudaMalloc. */
  void cwiseProduct(Context& ctx, const DeviceMatrix& a, const DeviceMatrix& b);

  /** this += DeviceScalar * x (cuBLAS axpy with POINTER_MODE_DEVICE). */
  DeviceMatrix& operator+=(const DeviceScaledDevice<Scalar>& expr);

  /** this -= DeviceScalar * x (cuBLAS axpy with negated device scalar). */
  DeviceMatrix& operator-=(const DeviceScaledDevice<Scalar>& expr);

  /** Assign from an SpMV expression: d_y = d_A * d_x (one cuSPARSE call). */
  DeviceMatrix& operator=(const SpMVExpr<Scalar>& expr);

  /** Assign from an SpMV expression with a dense addend: d_y = d_b - d_A * d_x and
   * the other sign combinations. Copies the addend into d_y (skipped when it is
   * d_y), then one cuSPARSE call with beta = ±1. The addend must have the shape
   * of the product, and d_x must not be d_y. */
  DeviceMatrix& operator=(const SpMVAffineExpr<Scalar>& expr);

  /** Assign from an add expression: d_C = alpha * d_A + beta * d_B (cuBLAS geam). */
  DeviceMatrix& operator=(const DeviceAddExpr<Scalar>& expr);

  /** Assign from a scaled matrix: d_C = alpha * d_A (cuBLAS geam with beta=0).
   * Also covers unary minus: d_C = -d_A. Safe when d_C aliases d_A. */
  DeviceMatrix& operator=(const Scaled<DeviceMatrix>& expr);

  /** No-op — every DeviceMatrix assignment is already implicitly noalias.
   *
   * Eigen's Matrix falls back to a temporary when .noalias() is omitted, but
   * DeviceMatrix dispatches straight to NVIDIA library calls, which offer no
   * aliasing protection. For GEMM and SpMV the caller must therefore keep the
   * operands clear of the destination; geam (`d_C = d_A + alpha * d_B`) is safe
   * under aliasing. Debug asserts catch violations.
   *
   * The method exists so `tmp.noalias() = mat * p` compiles for both types. */
  DeviceMatrix& noalias() { return *this; }

  /** Take ownership of \p device_ptr, which must come from cudaMalloc or
   * cudaMallocAsync and have its pending work complete or enqueued on \p ctx's
   * stream. It is freed there, stream-ordered, when the matrix is destroyed. */
  static DeviceMatrix adopt(Context& ctx, Scalar* device_ptr, Index rows, Index cols);

  /** adopt() on the thread-local Context. */
  static DeviceMatrix adopt(Scalar* device_ptr, Index rows, Index cols);

  /** Construct a non-owning view over an existing device pointer, whose pending
   * writes must be complete or enqueued on \p ctx's stream.
   *
   * The pointer is *borrowed*: destruction does not free, and the underlying
   * storage must outlive this view. This chains decomposition outputs (e.g.
   * `svd.d_matrixU()`) into downstream cuBLAS expressions without an intervening
   * D2D copy, and supports the full read interface. The view orders its own
   * accesses and, when destroyed, makes \p ctx's stream wait for the reads made
   * through it: the owner's writes and free are ordered after those reads only
   * if they run on \p ctx's stream after the view is gone. Do not assign through
   * a view: the borrowed pointer would be silently replaced, leaving the owner
   * intact. */
  static DeviceMatrix view(Context& ctx, Scalar* device_ptr, Index rows, Index cols);

  /** view() on the thread-local Context. */
  static DeviceMatrix view(Scalar* device_ptr, Index rows, Index cols);

  /** Transfer ownership of the device pointer out and leave the matrix empty.
   * Every pending access is first ordered before later work on \p ctx's
   * stream: use the pointer, and free it (cudaFreeAsync on ctx.stream(), or
   * cudaFree), only after that. */
  Scalar* release(Context& ctx);

  /** release() to the thread-local Context. */
  Scalar* release();

 private:
  friend struct internal::DeviceMatrixAccess;

  DeviceMatrix(internal::DeviceBuffer&& buf, Index rows, Index cols) : buf_(std::move(buf)), rows_(rows), cols_(cols) {}

  internal::DeviceBuffer buf_;
  Index rows_ = 0;
  Index cols_ = 0;
};

namespace internal {
// Storage access for the solvers and dispatchers that hand a DeviceMatrix's
// allocation to an internal owner or wrap an internal buffer as a result,
// keeping the buffer's stream ordering intact across the handoff.
struct DeviceMatrixAccess {
  template <typename Scalar>
  static const DeviceBuffer& buffer(const DeviceMatrix<Scalar>& m) {
    return m.buf_;
  }

  /** Moves the allocation out, leaving \p m empty. */
  template <typename Scalar>
  static DeviceBuffer take(DeviceMatrix<Scalar>& m) {
    m.rows_ = 0;
    m.cols_ = 0;
    return std::move(m.buf_);
  }

  template <typename Scalar>
  static DeviceMatrix<Scalar> wrap(DeviceBuffer&& buf, Index rows, Index cols) {
    return DeviceMatrix<Scalar>(std::move(buf), rows, cols);
  }

  /** Reallocates \p m on \p stream unless its owned allocation already fits rows x cols. */
  template <typename Scalar>
  static void resize(DeviceMatrix<Scalar>& m, const StreamHandle& stream, Index rows, Index cols) {
    eigen_assert(rows >= 0 && cols >= 0);
    if (rows == m.rows_ && cols == m.cols_) return;
    const size_t bytes = static_cast<size_t>(rows) * static_cast<size_t>(cols) * sizeof(Scalar);
    if (bytes > 0 && bytes <= m.buf_.size() && m.buf_.owns()) {
      // The next access's prepareWrite orders the reuse after pending ones.
      m.rows_ = rows;
      m.cols_ = cols;
      return;
    }
    // Free first so a large reallocation can reuse the memory.
    m.buf_.reset();
    m.buf_ = DeviceBuffer(bytes, stream);
    m.rows_ = rows;
    m.cols_ = cols;
  }
};
}  // namespace internal
}  // namespace gpu
}  // namespace Eigen

#endif  // EIGEN_GPU_DEVICE_MATRIX_H
