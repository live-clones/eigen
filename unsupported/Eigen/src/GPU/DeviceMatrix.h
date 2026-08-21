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
// Cross-stream safety is automatic: an internal CUDA event records when the last
// write completed, and consumers on a different stream wait on it before reading.

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
    cudaError_t err = cudaEventQuery(event_);
    if (err == cudaSuccess) return true;
    eigen_assert(err == cudaErrorNotReady && "cudaEventQuery failed");
    return false;
  }

  ~HostTransfer() {
    if (event_) (void)cudaEventDestroy(event_);
  }

  HostTransfer(HostTransfer&& o) noexcept
      : host_buf_(std::move(o.host_buf_)), pinned_buf_(std::move(o.pinned_buf_)), event_(o.event_), synced_(o.synced_) {
    o.event_ = nullptr;
    o.synced_ = true;
  }

  HostTransfer& operator=(HostTransfer&& o) noexcept {
    if (this != &o) {
      // Unchecked like the destructor: eigen_assert may throw, and this operator is noexcept.
      if (event_) (void)cudaEventDestroy(event_);
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
 * Owns a device allocation with tracked dimensions and leading dimension.
 * An internal CUDA event records when the data was last written, enabling
 * safe cross-stream consumption without user-visible synchronization.
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

  /** Allocate an uninitialized column vector, mirroring
   * Matrix<Scalar,Dynamic,1>(n) so generic solver code compiles unchanged. */
  explicit DeviceMatrix(Index n) : rows_(n), cols_(1) {
    eigen_assert(n >= 0);
    allocate(sizeInBytes());
  }

  /** Allocate uninitialized memory for a rows x cols matrix.
   *
   * With a host-accessible \p resource -- managed, mapped or registered
   * storage -- hostData() is non-null and the host reads and writes the same
   * bytes the device does, so fromHost() / toHost() are not needed at all.
   * \p resource must outlive this matrix. */
  DeviceMatrix(Index rows, Index cols, MemoryResource& resource = deviceMemoryResource())
      : rows_(rows), cols_(cols), resource_(&resource) {
    eigen_assert(rows >= 0 && cols >= 0);
    allocate(sizeInBytes());
  }

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

  ~DeviceMatrix() {
    // cudaEventDestroy on a pending event is non-blocking: the runtime defers
    // teardown until the event completes. data_ then goes back to its resource;
    // for the device-only kinds that release is stream-ordered on the legacy
    // default stream, so the buffer outlives any in-flight kernel still
    // touching it.
    if (ready_event_) (void)cudaEventDestroy(ready_event_);
  }

  DeviceMatrix(DeviceMatrix&& o) noexcept
      : data_(std::move(o.data_)),
        rows_(o.rows_),
        cols_(o.cols_),
        ready_event_(o.ready_event_),
        ready_stream_(o.ready_stream_),
        retained_buffer_(std::move(o.retained_buffer_)),
        resource_(o.resource_) {
    o.rows_ = 0;
    o.cols_ = 0;
    o.ready_event_ = nullptr;
    o.ready_stream_ = nullptr;
    o.resource_ = &deviceMemoryResource();
  }

  DeviceMatrix& operator=(DeviceMatrix&& o) noexcept {
    if (this != &o) {
      // Unchecked like the destructor: eigen_assert may throw, and this operator is noexcept.
      if (ready_event_) (void)cudaEventDestroy(ready_event_);
      data_ = std::move(o.data_);
      rows_ = o.rows_;
      cols_ = o.cols_;
      ready_event_ = o.ready_event_;
      ready_stream_ = o.ready_stream_;
      retained_buffer_ = std::move(o.retained_buffer_);
      resource_ = o.resource_;
      o.rows_ = 0;
      o.cols_ = 0;
      o.ready_event_ = nullptr;
      o.ready_stream_ = nullptr;
      o.resource_ = &deviceMemoryResource();
    }
    return *this;
  }

  DeviceMatrix(const DeviceMatrix&) = delete;
  DeviceMatrix& operator=(const DeviceMatrix&) = delete;

  /** Upload a host Eigen matrix to device memory (synchronous).
   *
   * Copies to device via cudaMemcpyAsync on \p stream and synchronizes before
   * returning. Plain contiguous column-major input is transferred directly;
   * other expressions are first evaluated into a contiguous temporary.
   *
   * \param host   Any Eigen dense expression.
   * \param stream CUDA stream for the transfer (default: stream 0).
   */
  template <typename Derived>
  static DeviceMatrix fromHost(const DenseBase<Derived>& host, cudaStream_t stream = nullptr) {
    // Ref binds any column-major direct-access input in place (no host copy);
    // row-major layouts and expressions evaluate into its temporary. A bound
    // block keeps its parent's outer stride, so the upload must honour
    // outerStride() rather than assume rows() -- see upload_host_matrix.
    const Ref<const PlainMatrix> mat(host.derived());
    DeviceMatrix dm(mat.rows(), mat.cols());
    if (dm.sizeInBytes() > 0) {
      internal::upload_host_matrix(dm.data(), mat.rows(), mat.data(), mat.outerStride(), mat.rows(), mat.cols(),
                                   stream);
      EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream));
    }
    return dm;
  }

  /** Upload from a raw host pointer to device memory (asynchronous).
   *
   * Enqueues an async H2D copy on \p stream and records an internal event.
   * The caller must keep \p host_data alive until the transfer completes.
   *
   * \param host_data  Pointer to contiguous column-major host data.
   * \param rows       Number of rows.
   * \param cols       Number of columns.
   * \param stream     CUDA stream for the transfer.
   */
  static DeviceMatrix fromHostAsync(const Scalar* host_data, Index rows, Index cols, cudaStream_t stream) {
    eigen_assert(rows >= 0 && cols >= 0);
    eigen_assert(host_data != nullptr || (rows == 0 || cols == 0));
    DeviceMatrix dm(rows, cols);
    if (dm.sizeInBytes() > 0) {
      EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(dm.data(), host_data, dm.sizeInBytes(), cudaMemcpyHostToDevice, stream));
      dm.recordReady(stream);
    }
    return dm;
  }

  /** Download device matrix to host memory (synchronous).
   *
   * Waits on the internal ready event, enqueues a D2H copy on \p stream,
   * synchronizes, and returns the host matrix directly.
   *
   * \param stream CUDA stream for the transfer (default: stream 0).
   */
  PlainMatrix toHost(cudaStream_t stream = nullptr) const {
    PlainMatrix host_buf(rows_, cols_);
    if (sizeInBytes() > 0) {
      waitReady(stream);
      EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(host_buf.data(), data(), sizeInBytes(), cudaMemcpyDeviceToHost, stream));
      EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream));
    }
    return host_buf;
  }

  /** Enqueue an async device-to-host transfer and return a future.
   *
   * Waits on the internal ready event (if any) to ensure the device data is
   * valid, then enqueues the D2H copy on \p stream. Call HostTransfer::get() to
   * block and retrieve the host matrix.
   *
   * \param stream CUDA stream for the transfer (default: stream 0).
   */
  HostTransfer<Scalar> toHostAsync(cudaStream_t stream = nullptr) const {
    PlainMatrix host_buf(rows_, cols_);
    internal::PinnedHostBuffer pinned_buf(sizeInBytes());
    if (sizeInBytes() > 0) {
      waitReady(stream);
      EIGEN_CUDA_RUNTIME_CHECK(
          cudaMemcpyAsync(pinned_buf.get(), data(), sizeInBytes(), cudaMemcpyDeviceToHost, stream));
    }
    cudaEvent_t transfer_event;
    EIGEN_CUDA_RUNTIME_CHECK(cudaEventCreateWithFlags(&transfer_event, cudaEventDisableTiming));
    EIGEN_CUDA_RUNTIME_CHECK(cudaEventRecord(transfer_event, stream));
    return HostTransfer<Scalar>(std::move(host_buf), std::move(pinned_buf), transfer_event);
  }

  /** Deep copy on device. Fully async — records an event on the result, no sync.
   *
   * \param stream CUDA stream for the D2D copy (default: stream 0).
   */
  DeviceMatrix clone(cudaStream_t stream = nullptr) const {
    DeviceMatrix result(rows_, cols_);
    if (sizeInBytes() > 0) {
      waitReady(stream);
      EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(result.data(), data(), sizeInBytes(), cudaMemcpyDeviceToDevice, stream));
      result.recordReady(stream);
    }
    return result;
  }

  /** Discard contents and resize to (rows x cols). Contents are undefined
   * afterwards. Keeps the existing allocation when it is large enough
   * (capacity-aware: no cudaMalloc/cudaFree churn when cycling through
   * same-or-smaller shapes); otherwise reallocates and clears the ready
   * event. */
  void resize(Index rows, Index cols) {
    eigen_assert(rows >= 0 && cols >= 0);
    if (rows == rows_ && cols == cols_) return;
    const size_t bytes = static_cast<size_t>(rows) * static_cast<size_t>(cols) * sizeof(Scalar);
    if (bytes > 0 && bytes <= data_.size() && data_.owns()) {
      // Reuse the allocation; the ready event still orders any in-flight
      // writes to this buffer ahead of its next producer. A borrowed view owns
      // nothing to reuse, so it reallocates.
      rows_ = rows;
      cols_ = cols;
      return;
    }
    data_ = internal::DeviceBuffer();
    clearReady();
    retained_buffer_ = internal::DeviceBuffer();
    rows_ = rows;
    cols_ = cols;
    allocate(bytes);
  }

  Scalar* data() { return static_cast<Scalar*>(data_.get()); }
  const Scalar* data() const { return static_cast<const Scalar*>(data_.get()); }

  /** Host-addressable alias of data(), or null when the storage is device-only.
   *
   * Non-null exactly when the matrix was built on a host-accessible
   * MemoryResource. Not necessarily equal to data(): malloc + cudaHostRegister
   * yields distinct host and device addresses. */
  Scalar* hostData() { return static_cast<Scalar*>(data_.hostData()); }
  const Scalar* hostData() const { return static_cast<const Scalar*>(data_.hostData()); }

  /** Whether hostData() is usable. */
  bool isHostAccessible() const { return data_.isHostAccessible(); }

  /** The resource this matrix allocates from; never null, since naming none
   * means deviceMemoryResource().
   *
   * This is "where the next block comes from", which is not the same question
   * as "who frees the current one": a view() borrows storage allocated
   * elsewhere, and an empty matrix has no block at all. */
  MemoryResource& memoryResource() const { return *resource_; }

  /** Make device writes visible through hostData().
   *
   * Which synchronization that needs is a property of the storage, so the
   * resource decides -- see internal::sync_for_host_access(). Callers do not
   * have to know which kind they have. */
  void syncHost(Context& ctx);
  Index rows() const { return rows_; }
  Index cols() const { return cols_; }
  bool empty() const { return rows_ == 0 || cols_ == 0; }

  /** Size of the device allocation in bytes. */
  size_t sizeInBytes() const { return static_cast<size_t>(rows_) * static_cast<size_t>(cols_) * sizeof(Scalar); }

  /** Record that device data is ready after work on \p stream. */
  void recordReady(cudaStream_t stream) {
    ensureEvent();
    EIGEN_CUDA_RUNTIME_CHECK(cudaEventRecord(ready_event_, stream));
    ready_stream_ = stream;
  }

  /** Make \p stream wait until the device data is ready.
   * No-op if no event recorded, or if the consumer stream is the same as the
   * producer stream (CUDA guarantees in-order execution within a stream). */
  void waitReady(cudaStream_t stream) const {
    if (ready_event_ && stream != ready_stream_) {
      EIGEN_CUDA_RUNTIME_CHECK(cudaStreamWaitEvent(stream, ready_event_, 0));
    }
  }

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

  // Every reduction takes the resource its result lives in, defaulted so that
  // naming one is optional. With a host-accessible resource cuBLAS writes the
  // value where the host can already address it, and reading it back costs a
  // synchronization and a load rather than a synchronization and a
  // device-to-host copy. The storage is named rather than inherited from *this:
  // a matrix on an adopted host buffer has nowhere to put a second allocation,
  // and a reduction that silently changed where its result lived would be
  // harder to reason about than one that was asked to.

  /** Dot product: this^H * other. The result stays on device until read through
   * DeviceScalar's conversion to Scalar, which syncs. */
  DeviceScalar<Scalar> dot(Context& ctx, const DeviceMatrix& other,
                           MemoryResource& resource = pooledDeviceMemoryResource()) const;

  /** Squared L2 norm via dot(x, x). For real types the result stays on device;
   * for complex it syncs, since DeviceScalar arithmetic is real-only. */
  DeviceScalar<typename NumTraits<Scalar>::Real> squaredNorm(
      Context& ctx, MemoryResource& resource = pooledDeviceMemoryResource()) const;

  /** L2 norm, without a host sync. */
  DeviceScalar<typename NumTraits<Scalar>::Real> norm(Context& ctx,
                                                      MemoryResource& resource = pooledDeviceMemoryResource()) const;

  /** Set all elements to zero. */
  void setZero(Context& ctx);
  void setZero(cudaStream_t stream);

  /** this += alpha * x (cuBLAS axpy). Requires same total size. */
  void addScaled(Context& ctx, Scalar alpha, const DeviceMatrix& x);

  /** this *= alpha (cuBLAS scal). */
  void scale(Context& ctx, Scalar alpha);

  /** Deep copy: this = other (cuBLAS copy). Resizes if needed. */
  void copyFrom(Context& ctx, const DeviceMatrix& other);

  DeviceScalar<Scalar> dot(const DeviceMatrix& other, MemoryResource& resource = pooledDeviceMemoryResource()) const;
  DeviceScalar<typename NumTraits<Scalar>::Real> squaredNorm(
      MemoryResource& resource = pooledDeviceMemoryResource()) const;
  DeviceScalar<typename NumTraits<Scalar>::Real> norm(MemoryResource& resource = pooledDeviceMemoryResource()) const;
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

  /** Assign from an SpMV expression: d_y = d_A * d_x. */
  DeviceMatrix& operator=(const SpMVExpr<Scalar>& expr);

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

  /** Adopt an existing device pointer. Caller relinquishes ownership. */
  static DeviceMatrix adopt(Scalar* device_ptr, Index rows, Index cols) {
    DeviceMatrix dm;
    dm.rows_ = rows;
    dm.cols_ = cols;
    dm.data_ = internal::DeviceBuffer::adopt(device_ptr, dm.sizeInBytes());
    return dm;
  }

  /** Adopt an existing buffer as a rows x cols matrix.
   *
   * Resource-backed storage goes back to its resource. Pooled device scratch
   * is promoted to stream-ordered device ownership because a public matrix may
   * be destroyed after use on another stream. \p buffer must hold at least
   * rows * cols * sizeof(Scalar) bytes. */
  static DeviceMatrix adopt(internal::DeviceBuffer&& buffer, Index rows, Index cols) {
    DeviceMatrix dm;
    dm.rows_ = rows;
    dm.cols_ = cols;
    eigen_assert(buffer.size() >= dm.sizeInBytes() && "adopted buffer is too small for the requested shape");
    eigen_assert((buffer.owns() || !buffer) && "cannot adopt a borrowed buffer; use view()");
    MemoryResource* resource = buffer.memoryResource();
    if (resource == &pooledDeviceMemoryResource()) {
      const size_t bytes = buffer.size();
      void* pointer = buffer.release();
      buffer = internal::DeviceBuffer::adopt(pointer, bytes);
      resource = &deviceMemoryResource();
    }
    if (resource != nullptr) dm.resource_ = resource;
    dm.data_ = std::move(buffer);
    return dm;
  }

  /** Construct a non-owning view over an existing device pointer.
   *
   * The pointer is *borrowed*: destruction does not free, and the underlying
   * storage must outlive this view. This chains decomposition outputs (e.g.
   * `svd.d_matrixU()`) into downstream cuBLAS expressions without an intervening
   * D2D copy, and supports the full read interface. Do not assign through a view:
   * the borrowed pointer would be silently replaced, leaving the owner intact. */
  static DeviceMatrix view(Scalar* device_ptr, Index rows, Index cols) {
    DeviceMatrix dm;
    dm.rows_ = rows;
    dm.cols_ = cols;
    dm.data_ = internal::DeviceBuffer::view(device_ptr, dm.sizeInBytes());
    return dm;
  }

  /** Hand the storage out as a buffer, leaving this matrix empty.
   *
   * The buffer keeps the resource that made it, so page-locked and pooled
   * storage both survive the handoff. This is how the solvers take ownership of
   * a matrix moved into compute(): pairs with adopt(DeviceBuffer&&, ...). */
  internal::DeviceBuffer releaseBuffer() {
    internal::DeviceBuffer buffer = std::move(data_);
    rows_ = 0;
    cols_ = 0;
    resource_ = &deviceMemoryResource();
    clearReady();
    return buffer;
  }

  /** Transfer ownership of the device pointer out. Zeros internal state.
   *
   * A bare pointer loses how the storage was made, so DeviceBuffer::release()
   * rejects both a borrowed view() and host-visible storage: whoever adopts the
   * pointer next frees it with device_free. Prefer releaseBuffer(), which has
   * neither restriction. */
  Scalar* release() { return static_cast<Scalar*>(releaseBuffer().release()); }

 private:
  // Fresh owning allocation of `bytes` (no-op for empty), replacing whatever
  // data_ held -- including a borrowed view, whose deleter must not follow the
  // new owned pointer.
  void allocate(size_t bytes) { data_ = internal::DeviceBuffer(bytes, *resource_); }

  void ensureEvent() {
    if (!ready_event_) {
      EIGEN_CUDA_RUNTIME_CHECK(cudaEventCreateWithFlags(&ready_event_, cudaEventDisableTiming));
    }
  }

  void clearReady() {
    if (ready_event_) {
      EIGEN_CUDA_RUNTIME_CHECK(cudaEventDestroy(ready_event_));
      ready_event_ = nullptr;
    }
    ready_stream_ = nullptr;
  }

  void retainBuffer(internal::DeviceBuffer&& buffer) { retained_buffer_ = std::move(buffer); }

  internal::DeviceBuffer data_;  // device storage, its host alias and its allocator
  Index rows_ = 0;
  Index cols_ = 0;
  cudaEvent_t ready_event_ = nullptr;       // internal: tracks last write completion
  cudaStream_t ready_stream_ = nullptr;     // stream that recorded ready_event_ (for same-stream skip)
  internal::DeviceBuffer retained_buffer_;  // internal: keeps async aux buffers alive
  // What this matrix allocates from. Never null; deviceMemoryResource() rather
  // than the pooled one because a matrix's ready event lets another stream read
  // its data, and the pool does not order reuse against that.
  MemoryResource* resource_ = &deviceMemoryResource();
};
}  // namespace gpu
}  // namespace Eigen

#endif  // EIGEN_GPU_DEVICE_MATRIX_H
