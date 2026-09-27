// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// Generic CUDA runtime support shared by all GPU library integrations.
// Depends only on <cuda_runtime.h>; no NVIDIA library headers.

#ifndef EIGEN_GPU_SUPPORT_H
#define EIGEN_GPU_SUPPORT_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <limits>
#include <memory>
#include <mutex>
#include <type_traits>

namespace Eigen {
namespace gpu {
// Transpose/adjoint flag for BLAS-, solver-, and sparse-style calls. Each
// library's support header maps it to its own enum (cublasOperation_t,
// cusparseOperation_t, ...) via a to_<lib>_op() helper.
enum class GpuOp { NoTrans, Trans, ConjTrans };

namespace internal {
// Prints `file:line: call: error` to stderr and stops: std::abort() where
// assertions are compiled out, a failed eigen_assert otherwise. No build goes
// on past a failed call: its work was not done, and a sticky error leaves the
// context unusable for every later call.
inline void gpu_check_failed(const char* error, const char* expression, const char* file, int line) {
  std::fprintf(stderr, "%s:%d: %s: %s\n", file, line, expression, error);
#if defined(EIGEN_NO_DEBUG)
  std::abort();
#else
  eigen_assert(false && "GPU runtime or library call failed");
#endif
}

// Every check macro of the module reports failures here, in every build. It may
// be defined to throw, so no destructor or noexcept function uses the checks.
#ifndef EIGEN_GPU_CHECK_FAILED
#define EIGEN_GPU_CHECK_FAILED(error, expression, file, line) \
  ::Eigen::gpu::internal::gpu_check_failed(error, expression, file, line)
#endif

#define EIGEN_CUDA_RUNTIME_CHECK(expr)                                                              \
  do {                                                                                              \
    const cudaError_t _e = (expr);                                                                  \
    if (_e != cudaSuccess) EIGEN_GPU_CHECK_FAILED(cudaGetErrorName(_e), #expr, __FILE__, __LINE__); \
  } while (0)

// For the libraries without a status-to-string function: "<library> status <code>".
inline void gpu_check_failed_code(const char* library, int status, const char* expression, const char* file, int line) {
  char error[64];
  std::snprintf(error, sizeof(error), "%s status %d", library, status);
  // A user-defined EIGEN_GPU_CHECK_FAILED need not use every argument.
  EIGEN_UNUSED_VARIABLE(expression);
  EIGEN_UNUSED_VARIABLE(file);
  EIGEN_UNUSED_VARIABLE(line);
  EIGEN_GPU_CHECK_FAILED(error, expression, file, line);
}

// cuBLAS and the legacy cuSOLVER APIs take dimensions and leading dimensions as
// 32-bit `int`, while Eigen's Index is 64-bit by default and GPU allocations can
// exceed INT_MAX in one dimension. Narrow through this helper at every such call
// site so an out-of-range value asserts instead of silently overflowing.
inline int to_blas_int(int64_t v) {
  eigen_assert(v >= 0 && v <= static_cast<int64_t>((std::numeric_limits<int>::max)()) &&
               "dimension exceeds the int range supported by cuBLAS / cuSOLVER");
  return static_cast<int>(v);
}

// Shared ownership of a CUDA stream. Every device buffer holds its home stream
// (see DeviceBuffer), so the stream-ordered free always has a live stream to
// run on, even after the Context that created the stream is gone. A borrowed
// stream (a caller's cudaStream_t) is held without ownership and must outlive
// every buffer whose home it becomes.
using StreamHandle = std::shared_ptr<std::remove_pointer_t<cudaStream_t>>;

struct StreamDestroyer {
  void operator()(cudaStream_t s) const noexcept {
    if (s) (void)cudaStreamDestroy(s);
  }
};

// A new stream owned by the returned handle. Non-blocking, so that it never
// synchronizes implicitly with the legacy default stream: work other code puts
// on the legacy stream cannot serialize against the module's streams, and
// capturing one of them into a CUDA graph is not invalidated by it.
inline StreamHandle make_owned_stream() {
  cudaStream_t s = nullptr;
  EIGEN_CUDA_RUNTIME_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  return StreamHandle(s, StreamDestroyer());
}

// A non-owning handle (the aliasing constructor with an empty owner allocates
// no control block).
inline StreamHandle borrow_stream(cudaStream_t s) { return StreamHandle(StreamHandle(), s); }

// Allocation and free are enqueued on an explicit stream (cudaMallocAsync /
// cudaFreeAsync, CUDA 11.2+), never on the legacy default stream. Without memory
// pools (detected once per process), or with EIGEN_GPU_NO_STREAM_ORDERED_ALLOC,
// they fall back to cudaMalloc / cudaFree, which synchronize the device.
inline bool device_supports_memory_pools() {
#ifdef EIGEN_GPU_NO_STREAM_ORDERED_ALLOC
  return false;
#else
  static const bool supported = [] {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return false;
    int v = 0;
    if (cudaDeviceGetAttribute(&v, cudaDevAttrMemoryPoolsSupported, device) != cudaSuccess) return false;
    if (v == 0) return false;
    // Keep freed memory in the pool instead of trimming at every stream
    // synchronize — repeated alloc/free cycles (temporaries in loops) then
    // recycle at user-space speed.
    cudaMemPool_t pool = nullptr;
    if (cudaDeviceGetDefaultMemPool(&pool, device) == cudaSuccess) {
      // The attribute value type is cuuint64_t; use a same-size stand-in to
      // avoid requiring the driver-API header.
      unsigned long long threshold = ~0ULL;
      (void)cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
    }
    return true;
  }();
  return supported;
#endif
}

inline void* device_malloc(size_t bytes, cudaStream_t stream) {
  void* p = nullptr;
  if (device_supports_memory_pools()) {
    EIGEN_CUDA_RUNTIME_CHECK(cudaMallocAsync(&p, bytes, stream));
  } else {
    EIGEN_CUDA_RUNTIME_CHECK(cudaMalloc(&p, bytes));
  }
  return p;
}

inline void device_free(void* p, cudaStream_t stream) noexcept {
  if (!p) return;
  if (device_supports_memory_pools()) {
    (void)cudaFreeAsync(p, stream);
  } else {
    (void)cudaFree(p);
  }
}

struct CudaFreeHostDeleter {
  void operator()(void* p) const noexcept {
    if (p) (void)cudaFreeHost(p);
  }
};

// Recycles allocations up to kSmallBufferThreshold bytes (e.g. DeviceScalar) to
// avoid cudaMalloc/cudaFree overhead on devices without memory pools; where
// cudaMallocAsync exists it is both stream-ordered and cheaper than this pool's
// release event, so DeviceBuffer bypasses the pool there (bench_overhead:
// CudaMallocAsyncFree vs PoolAllocFree). Larger allocations always bypass it.
// Invariant: a block is recycled only after its release event has completed.
// deallocate() records that event on the block's home stream, which DeviceBuffer
// has already ordered behind every access to the block. Events on different
// streams retire out of order, so allocate() checks every entry, oldest first.
template <size_t SmallBufferThreshold = 256, size_t MaxPoolSize = 64>
struct DeviceBufferPool {
  static constexpr size_t kSmallBufferThreshold = SmallBufferThreshold;
  static constexpr size_t kMaxPoolSize = MaxPoolSize;

  struct Entry {
    void* ptr;
    size_t bytes;
    cudaEvent_t release_event;
  };

  // Lifetime marker for the thread-local pool. thread_local destruction runs
  // in reverse construction order, so a long-lived object holding pooled
  // buffers (e.g. the thread-local gpu::Context, or a static) can be
  // destroyed *after* the pool. The marker is trivially destructible — it
  // stays readable during TLS teardown — letting the deleter fall back to a
  // direct device_free once the pool is gone instead of touching a destroyed
  // vector.
  enum class State : signed char { kNotConstructed = 0, kAlive = 1, kDestroyed = 2 };

  static State& threadState() {
    thread_local State state = State::kNotConstructed;
    return state;
  }

  DeviceBufferPool() { threadState() = State::kAlive; }

  ~DeviceBufferPool() {
    for (const Entry& entry : free_list_) freeBlock(entry.ptr, entry.release_event);
    for (cudaEvent_t event : spare_events_) (void)cudaEventDestroy(event);
    threadState() = State::kDestroyed;
  }

  // First fit among the retired blocks, oldest release first. The blocks are
  // plain cudaMalloc memory, so a retired one is usable on any stream.
  void* allocate(size_t bytes, cudaStream_t stream) {
    for (auto it = free_list_.begin(); it != free_list_.end(); ++it) {
      if (it->bytes >= bytes && cudaEventQuery(it->release_event) == cudaSuccess) {
        void* p = it->ptr;
        spare_events_.push_back(it->release_event);
        free_list_.erase(it);
        return p;
      }
    }
    return device_malloc(bytes, stream);
  }

  // Called from a noexcept path: every failure falls back to device_free.
  void deallocate(void* p, size_t bytes, cudaStream_t stream) noexcept {
    if (free_list_.size() >= kMaxPoolSize) {
      device_free(p, stream);
      return;
    }
    cudaEvent_t release_event = acquireEvent();
    if (release_event == nullptr) {
      device_free(p, stream);
      return;
    }
    if (cudaEventRecord(release_event, stream) != cudaSuccess) {
      freeBlock(p, release_event);
      return;
    }
    free_list_.push_back({p, bytes, release_event});
  }

  static DeviceBufferPool& threadLocal() {
    thread_local DeviceBufferPool pool;
    return pool;
  }

 private:
  // Returns a spare event, or a newly created one; nullptr if creation fails.
  cudaEvent_t acquireEvent() noexcept {
    if (!spare_events_.empty()) {
      cudaEvent_t event = spare_events_.back();
      spare_events_.pop_back();
      return event;
    }
    cudaEvent_t event = nullptr;
    if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess) return nullptr;
    return event;
  }

  // Gives up a block and the event tracking its release. Destroying a pending
  // event is non-blocking; the runtime defers it until the event completes.
  // The block is cudaMalloc memory (the pool is only used without memory
  // pools), so the free is the synchronous cudaFree and needs no stream.
  static void freeBlock(void* p, cudaEvent_t release_event) noexcept {
    (void)cudaEventDestroy(release_event);
    (void)cudaFree(p);
  }

  std::vector<Entry> free_list_;
  // Events of recycled entries; free_list_.size() + spare_events_.size() <= kMaxPoolSize.
  std::vector<cudaEvent_t> spare_events_;
};

/** \brief Internal RAII owner of an untyped device allocation, ordered on its home stream.
 *
 * The buffer is allocated on, and freed on, its home stream. Accesses from any
 * stream s are ordered by bracketing each of them:
 *
 *     prepareRead(s)  ... read ...  finishRead(s)
 *     prepareWrite(s) ... write ... finishWrite()   (any access that writes)
 *
 * which enforces
 *   - RAW: a read on s != home waits for the last write;
 *   - WAR, WAW: a write on s waits for all earlier work on home and for every
 *     read recorded on another stream, and s becomes the home stream;
 *   - free: home waits for the recorded reads, then cudaFreeAsync on home. A
 *     borrowed buffer's home waits for them too, so that the owner's later
 *     writes and free there follow reads made through the borrowed pointer.
 * Accesses on the home stream itself need no events. Internal scratch that is
 * only ever touched on its home stream may skip the protocol entirely. Reads
 * may run concurrently from several threads; a write needs exclusive access.
 */
class DeviceBuffer {
 public:
  DeviceBuffer() = default;

  DeviceBuffer(size_t bytes, StreamHandle stream) : bytes_(bytes), owns_(true), stream_(std::move(stream)) {
    if (bytes > 0) {
      // The pool serves small blocks only on the cudaMalloc fallback path, and
      // not once its thread_local has been destroyed (allocation from a
      // static/TLS destructor).
      pooled_ = bytes <= DeviceBufferPool<>::kSmallBufferThreshold && !device_supports_memory_pools() &&
                DeviceBufferPool<>::threadState() != DeviceBufferPool<>::State::kDestroyed;
      ptr_ = pooled_ ? DeviceBufferPool<>::threadLocal().allocate(bytes, stream_.get())
                     : device_malloc(bytes, stream_.get());
    }
  }

  ~DeviceBuffer() { reset(); }

  // Explicit moves so a moved-from buffer reports size() == 0 (callers use
  // size() for grow-only reuse decisions; a stale size on a null buffer would
  // suppress the reallocation).
  DeviceBuffer(DeviceBuffer&& o) noexcept { moveFrom(o); }
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
    if (this != &o) {
      reset();
      moveFrom(o);
    }
    return *this;
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  /** Takes ownership of \p p (cudaMalloc or cudaMallocAsync memory whose
   * pending work is complete or enqueued on \p stream); it is freed on \p stream. */
  static DeviceBuffer adopt(void* p, size_t bytes, StreamHandle stream) {
    DeviceBuffer b;
    b.ptr_ = p;
    b.bytes_ = p ? bytes : 0;
    b.owns_ = true;
    b.stream_ = std::move(stream);
    return b;
  }

  /** Wraps \p p without taking ownership: the buffer never frees it. */
  static DeviceBuffer borrow(void* p, size_t bytes, StreamHandle stream) {
    DeviceBuffer b = adopt(p, bytes, std::move(stream));
    b.owns_ = false;
    return b;
  }

  void* get() const noexcept { return ptr_; }
  explicit operator bool() const noexcept { return ptr_ != nullptr; }

  /** Logical allocation size in bytes, tracked for adopted pointers as well. */
  size_t size() const noexcept { return bytes_; }

  /** False for a borrowed pointer, which is never freed or reallocated. */
  bool owns() const noexcept { return owns_; }

  cudaStream_t stream() const noexcept { return stream_.get(); }
  const StreamHandle& streamHandle() const noexcept { return stream_; }

  void prepareRead(const StreamHandle& s) const {
    if (!ptr_ || s.get() == stream_.get()) return;
    std::lock_guard<std::mutex> lock(read_mutex_);
    // A buffer never written through the protocol still has its allocation
    // (or an adopted producer's work) pending on the home stream.
    if (!write_recorded_) recordWriteEvent();
    EIGEN_CUDA_RUNTIME_CHECK(cudaStreamWaitEvent(s.get(), write_event_, 0));
  }

  void finishRead(const StreamHandle& s) const {
    if (!ptr_ || s.get() == stream_.get()) return;
    std::lock_guard<std::mutex> lock(read_mutex_);
    // The stream's pending mark, else a retired one. A mark retires when a
    // write has waited for it or its read has completed, so the marks number
    // at most the reader streams that ever had reads in flight at once.
    ReadMark* mark = nullptr;
    for (ReadMark& r : reads_) {
      if (r.pending && r.stream.get() == s.get()) mark = &r;
    }
    for (ReadMark& r : reads_) {
      if (mark) break;
      if (r.pending && cudaEventQuery(r.event) == cudaSuccess) {
        r.pending = false;
        r.stream.reset();
      }
      if (!r.pending) mark = &r;
    }
    if (!mark) {
      reads_.push_back(ReadMark{nullptr, nullptr, false});
      mark = &reads_.back();
      EIGEN_CUDA_RUNTIME_CHECK(cudaEventCreateWithFlags(&mark->event, cudaEventDisableTiming));
    }
    mark->stream = s;
    EIGEN_CUDA_RUNTIME_CHECK(cudaEventRecord(mark->event, s.get()));
    mark->pending = true;
  }

  /** Read marks held, pending or retired for reuse. */
  std::size_t readMarkCount() const {
    std::lock_guard<std::mutex> lock(read_mutex_);
    return reads_.size();
  }

  void prepareWrite(const StreamHandle& s) {
    if (!ptr_) {
      stream_ = s;
      return;
    }
    if (s.get() != stream_.get()) {
      // Everything enqueued on the old home so far, including reads there that
      // were never recorded: the handoff subsumes the last write.
      recordWriteEvent();
      EIGEN_CUDA_RUNTIME_CHECK(cudaStreamWaitEvent(s.get(), write_event_, 0));
      stream_ = s;
      write_recorded_ = false;
    }
    for (ReadMark& r : reads_) {
      if (r.pending && r.stream.get() != s.get()) EIGEN_CUDA_RUNTIME_CHECK(cudaStreamWaitEvent(s.get(), r.event, 0));
      r.pending = false;
      r.stream.reset();
    }
  }

  // record_event = false defers the event to the first read from another
  // stream, which then also waits for later work on home: cheaper when such
  // reads are rare, as for DeviceScalar.
  void finishWrite(bool record_event = true) {
    if (!ptr_) return;
    if (record_event) {
      recordWriteEvent();
    } else {
      write_recorded_ = false;
    }
  }

  /** Gives up ownership without freeing. \p target is first ordered after every
   * pending access, as for a write, so the caller only has to order its own use
   * and the free after \p target's work; the buffer's own stream, which may die
   * with the buffer, is not needed. */
  void* release(const StreamHandle& target) {
    prepareWrite(target);
    void* p = ptr_;
    ptr_ = nullptr;
    reset();
    return p;
  }

  /** Frees the allocation (if owned) on the home stream, after the recorded
   * reads; stream-ordered, so nothing blocks the host. */
  void reset() noexcept {
    if (ptr_) {
      for (const ReadMark& r : reads_) {
        if (r.pending && r.stream.get() != stream_.get()) (void)cudaStreamWaitEvent(stream_.get(), r.event, 0);
      }
    }
    if (ptr_ && owns_) {
      // Pooled blocks go back to the pool only while it is alive; a
      // pool-allocated block is cudaMalloc memory, so device_free handles it
      // afterwards too.
      if (pooled_ && DeviceBufferPool<>::threadState() == DeviceBufferPool<>::State::kAlive) {
        DeviceBufferPool<>::threadLocal().deallocate(ptr_, bytes_, stream_.get());
      } else {
        device_free(ptr_, stream_.get());
      }
    }
    // Destroying a pending event is non-blocking; the runtime defers it.
    for (const ReadMark& r : reads_) (void)cudaEventDestroy(r.event);
    reads_.clear();
    if (write_event_) (void)cudaEventDestroy(write_event_);
    write_event_ = nullptr;
    write_recorded_ = false;
    ptr_ = nullptr;
    bytes_ = 0;
    owns_ = false;
    pooled_ = false;
    stream_.reset();
  }

 private:
  struct ReadMark {
    StreamHandle stream;  // keeps the reader's stream alive while the mark is pending
    cudaEvent_t event;
    bool pending;
  };

  void recordWriteEvent() const {
    if (!write_event_) EIGEN_CUDA_RUNTIME_CHECK(cudaEventCreateWithFlags(&write_event_, cudaEventDisableTiming));
    EIGEN_CUDA_RUNTIME_CHECK(cudaEventRecord(write_event_, stream_.get()));
    write_recorded_ = true;
  }

  void moveFrom(DeviceBuffer& o) noexcept {
    ptr_ = o.ptr_;
    bytes_ = o.bytes_;
    owns_ = o.owns_;
    pooled_ = o.pooled_;
    stream_ = std::move(o.stream_);
    write_event_ = o.write_event_;
    write_recorded_ = o.write_recorded_;
    reads_ = std::move(o.reads_);
    o.ptr_ = nullptr;
    o.bytes_ = 0;
    o.owns_ = false;
    o.pooled_ = false;
    o.write_event_ = nullptr;
    o.write_recorded_ = false;
    o.reads_.clear();
  }

  void* ptr_ = nullptr;
  size_t bytes_ = 0;
  bool owns_ = false;
  bool pooled_ = false;
  StreamHandle stream_;
  // Mutable: prepareRead/finishRead are logically const, like the reads they
  // bracket, and read_mutex_ lets them run from several threads at once.
  mutable std::mutex read_mutex_;
  mutable cudaEvent_t write_event_ = nullptr;
  mutable bool write_recorded_ = false;
  mutable std::vector<ReadMark> reads_;
};

// cudaMemcpyAsync only overlaps with compute when the host side is pinned, so
// async D2H staging goes through this buffer.
class PinnedHostBuffer {
 public:
  PinnedHostBuffer() = default;

  explicit PinnedHostBuffer(size_t bytes) {
    if (bytes > 0) {
      void* p = nullptr;
      EIGEN_CUDA_RUNTIME_CHECK(cudaMallocHost(&p, bytes));
      ptr_.reset(p);
    }
  }

  void* get() const noexcept { return ptr_.get(); }
  explicit operator bool() const noexcept { return static_cast<bool>(ptr_); }

 private:
  std::unique_ptr<void, CudaFreeHostDeleter> ptr_;
};

// Upload a column-major host matrix whose strides are in elements. Ref<const
// PlainMatrix> can bind any outer stride in place. Use a 2D DMA for ordinary
// padded layouts; copy legal negative or overlapping Eigen strides one
// contiguous column at a time because CUDA cannot express them as a pitch.
template <typename Scalar>
void upload_host_matrix(Scalar* dst, Index dst_outer_stride, const Scalar* src, Index src_outer_stride, Index rows,
                        Index cols, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) return;
  eigen_assert(dst_outer_stride >= rows);
  const size_t column_bytes = static_cast<size_t>(rows) * sizeof(Scalar);
  if (src_outer_stride >= rows) {
    EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpy2DAsync(dst, static_cast<size_t>(dst_outer_stride) * sizeof(Scalar), src,
                                               static_cast<size_t>(src_outer_stride) * sizeof(Scalar), column_bytes,
                                               static_cast<size_t>(cols), cudaMemcpyHostToDevice, stream));
  } else {
    for (Index col = 0; col < cols; ++col) {
      EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(dst + col * dst_outer_stride, src + col * src_outer_stride, column_bytes,
                                               cudaMemcpyHostToDevice, stream));
    }
  }
}

// cudaDataType_t lives in library_types.h, pulled in transitively by
// cuda_runtime.h, so this trait needs no NVIDIA library header of its own.
template <typename Scalar>
struct cuda_data_type;

template <>
struct cuda_data_type<float> {
  static constexpr cudaDataType_t value = CUDA_R_32F;
};
template <>
struct cuda_data_type<double> {
  static constexpr cudaDataType_t value = CUDA_R_64F;
};
template <>
struct cuda_data_type<std::complex<float>> {
  static constexpr cudaDataType_t value = CUDA_C_32F;
};
template <>
struct cuda_data_type<std::complex<double>> {
  static constexpr cudaDataType_t value = CUDA_C_64F;
};
}  // namespace internal
}  // namespace gpu
}  // namespace Eigen

#endif  // EIGEN_GPU_SUPPORT_H
