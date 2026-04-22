// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_PARALLELIZER_H
#define EIGEN_PARALLELIZER_H

// IWYU pragma: private
#include "../InternalHeaderCheck.h"

// Note that in the following, there are 3 different uses of the concept
// "number of threads":
//  1. Max number of threads used by OpenMP or ThreadPool.
//     * For OpenMP this is typically the value set by the OMP_NUM_THREADS
//       environment variable, or by a call to omp_set_num_threads() prior to
//       calling Eigen.
//     * For ThreadPool, this is the number of threads in the ThreadPool.
//  2. Max number of threads currently allowed to be used by parallel Eigen
//     operations. This is set by setNbThreads(), and cannot exceed the value
//     in 1.
//  3. The actual number of threads used for a given parallel Eigen operation.
//     This is typically computed on the fly using a cost model and cannot exceed
//     the value in 2.
//     * For OpenMP, this is typically the number of threads specified in individual
//       "omp parallel" pragmas associated with an Eigen operation.
//     * For ThreadPool, it is the number of concurrent tasks scheduled in the
//       threadpool for a given Eigen operation. Notice that since the threadpool
//       uses task stealing, there is no way to limit the number of concurrently
//       executing tasks to below the number in 1. except by limiting the total
//       number of tasks in flight.

#if defined(EIGEN_HAS_OPENMP) && defined(EIGEN_GEMM_THREADPOOL)
#error "EIGEN_HAS_OPENMP and EIGEN_GEMM_THREADPOOL may not both be defined."
#endif

namespace Eigen {

namespace internal {
inline void manage_multi_threading(Action action, int* v);
}

// Public APIs.

/** Must be call first when calling Eigen from multiple threads */
EIGEN_DEPRECATED_WITH_REASON("Initialization is no longer needed.") inline void initParallel() {}

/** \returns the max number of threads reserved for Eigen
 * \sa setNbThreads */
inline int nbThreads() {
  int ret;
  internal::manage_multi_threading(GetAction, &ret);
  return ret;
}

/** Sets the max number of threads reserved for Eigen
 * \sa nbThreads */
inline void setNbThreads(int v) { internal::manage_multi_threading(SetAction, &v); }

#ifdef EIGEN_GEMM_THREADPOOL
namespace internal {
inline std::atomic<ThreadPool*>& gemm_thread_pool_storage() {
  static std::atomic<ThreadPool*> pool{nullptr};
  return pool;
}
}  // namespace internal

// Sets the ThreadPool used by Eigen parallel Gemm.
//
// The caller is responsible for ensuring no Eigen parallel Gemm is in flight
// on the previous pool when swapping pools; the atomic here only guarantees
// that concurrent reads and writes of the pointer itself are well-defined.
inline ThreadPool* setGemmThreadPool(ThreadPool* new_pool) {
  auto& slot = internal::gemm_thread_pool_storage();
  if (new_pool != nullptr) {
    slot.store(new_pool, std::memory_order_release);
    // Reset the number of threads to the number of threads on the new pool.
    setNbThreads(new_pool->NumThreads());
  }
  return slot.load(std::memory_order_acquire);
}

// Gets the ThreadPool used by Eigen parallel Gemm.
inline ThreadPool* getGemmThreadPool() { return internal::gemm_thread_pool_storage().load(std::memory_order_acquire); }
#endif

namespace internal {

// Implementation.

#if defined(EIGEN_USE_BLAS) || (!defined(EIGEN_HAS_OPENMP) && !defined(EIGEN_GEMM_THREADPOOL))

inline void manage_multi_threading(Action action, int* v) {
  if (action == SetAction) {
    eigen_internal_assert(v != nullptr);
  } else if (action == GetAction) {
    eigen_internal_assert(v != nullptr);
    *v = 1;
  } else {
    eigen_internal_assert(false);
  }
}
template <typename Index>
struct GemmParallelInfo {};
template <bool Condition, typename Functor, typename Index>
EIGEN_STRONG_INLINE void parallelize_gemm(const Functor& func, Index rows, Index cols, Index /*unused*/,
                                          bool /*unused*/) {
  func(0, rows, 0, cols);
}

#else

template <typename Index>
struct GemmParallelTaskInfo {
  GemmParallelTaskInfo() : sync(-1), users(0), lhs_start(0), lhs_length(0) {}
  std::atomic<Index> sync;
  std::atomic<int> users;
  Index lhs_start;
  Index lhs_length;
};

template <typename Index>
struct GemmParallelInfo {
  const int logical_thread_id;
  const int num_threads;
  // 2-D partition of the output: the num_threads logical threads are arranged
  // in a (t_m x t_n) grid. For each thread, tid_m = logical_thread_id / t_n
  // indexes the row group and tid_n = logical_thread_id % t_n indexes the
  // column group. The 1-D case (historical behavior) corresponds to t_m == 1.
  const int t_m;
  const int t_n;
  GemmParallelTaskInfo<Index>* task_info;

  GemmParallelInfo(int logical_thread_id_, int num_threads_, int t_m_, int t_n_,
                   GemmParallelTaskInfo<Index>* task_info_)
      : logical_thread_id(logical_thread_id_), num_threads(num_threads_), t_m(t_m_), t_n(t_n_), task_info(task_info_) {}

  int tid_m() const { return logical_thread_id / t_n; }
  int tid_n() const { return logical_thread_id % t_n; }
};

inline void manage_multi_threading(Action action, int* v) {
  static int m_maxThreads = -1;
  if (action == SetAction) {
    eigen_internal_assert(v != nullptr);
#if defined(EIGEN_HAS_OPENMP)
    // Calling action == SetAction and *v = 0 means
    // restoring m_maxThreads to the maximum number of threads specified
    // for OpenMP.
    eigen_internal_assert(*v >= 0);
    int omp_threads = omp_get_max_threads();
    m_maxThreads = (*v == 0 ? omp_threads : numext::mini(*v, omp_threads));
#elif defined(EIGEN_GEMM_THREADPOOL)
    // Calling action == SetAction and *v = 0 means
    // restoring m_maxThreads to the number of threads in the ThreadPool,
    // which defaults to 1 if no pool was provided.
    eigen_internal_assert(*v >= 0);
    ThreadPool* pool = getGemmThreadPool();
    int pool_threads = pool != nullptr ? pool->NumThreads() : 1;
    m_maxThreads = (*v == 0 ? pool_threads : numext::mini(pool_threads, *v));
#endif
  } else if (action == GetAction) {
    eigen_internal_assert(v != nullptr);
#if defined(EIGEN_HAS_OPENMP)
    if (m_maxThreads > 0)
      *v = m_maxThreads;
    else
      *v = omp_get_max_threads();
#else
    *v = m_maxThreads;
#endif
  } else {
    eigen_internal_assert(false);
  }
}

// Computes per-thread row/column ranges for a 2-D partition of an M x N grid
// across actual_threads logical threads arranged as (t_m x t_n). The row range
// belongs to the thread's row group (shared with other threads of the same
// tid_m for cooperative LHS packing). The column range is the thread's own
// panel of the output.
template <typename Index>
struct gemm_thread_tile {
  Index lhs_start;
  Index lhs_length;
  Index c0;
  Index actual_block_cols;
};

template <typename Index, int Mr>
inline gemm_thread_tile<Index> compute_gemm_thread_tile(int tid_m, int tid_n, int t_m, int t_n, Index rows,
                                                        Index cols) {
  gemm_thread_tile<Index> tile;
  // Column panel for this thread.
  Index blockCols = (cols / t_n) & ~Index(0x3);
  tile.c0 = tid_n * blockCols;
  tile.actual_block_cols = (tid_n + 1 == t_n) ? cols - tile.c0 : blockCols;
  // Row group owned by this thread's row group (tid_m), then split among the
  // t_n threads of the group so each of them packs a slice cooperatively.
  Index groupRows = rows / t_m;
  groupRows = (groupRows / Mr) * Mr;
  Index groupStart = tid_m * groupRows;
  Index groupEnd = (tid_m + 1 == t_m) ? rows : groupStart + groupRows;
  Index groupLen = groupEnd - groupStart;
  Index sliceRows = groupLen / t_n;
  sliceRows = (sliceRows / Mr) * Mr;
  tile.lhs_start = groupStart + tid_n * sliceRows;
  tile.lhs_length = (tid_n + 1 == t_n) ? groupEnd - tile.lhs_start : sliceRows;
  return tile;
}

// Computes a (t_m, t_n) partition of `threads` threads such that:
//   * t_n is as large as possible up to the column-capacity bound,
//   * t_m absorbs the remaining thread budget up to the row-capacity bound,
//   * t_m * t_n <= threads.
// When the primary dimension can absorb all threads (t_n == threads), t_m is 1
// and the partition is bit-identical to the previous 1-D column split.
template <typename Index>
inline void compute_2d_partition(int threads, Index rows, Index cols, Index nr, Index mr, int& t_m, int& t_n,
                                 int& actual_threads) {
  Index n_cap = std::max<Index>(1, cols / nr);
  Index m_cap = std::max<Index>(1, rows / mr);
  t_n = std::min<int>(threads, static_cast<int>(n_cap));
  int m_budget = std::max<int>(1, threads / t_n);
  t_m = std::min<int>(m_budget, static_cast<int>(m_cap));
  actual_threads = t_m * t_n;
}

template <bool Condition, typename Functor, typename Index>
EIGEN_STRONG_INLINE void parallelize_gemm(const Functor& func, Index rows, Index cols, Index depth, bool transpose) {
  // Dynamically check whether we should even try to execute in parallel.
  // The conditions are:
  // - the max number of threads we can create is greater than 1
  // - we are not already in a parallel code
  // - the sizes are large enough

  // compute the maximal number of threads from the size of the product:
  // this heuristic takes into account that the product kernel is fully
  // optimized when working with nr columns at once.
  Index primary = transpose ? rows : cols;
  Index secondary = transpose ? cols : rows;
  Index n_cap = std::max<Index>(1, primary / Functor::Traits::nr);
  Index m_cap = std::max<Index>(1, secondary / Functor::Traits::mr);

  // compute the maximal number of threads from the total amount of work:
  double work = static_cast<double>(rows) * static_cast<double>(cols) * static_cast<double>(depth);
  double kMinTaskSize = 50000;  // FIXME: tune this minimum task-size heuristic based on architecture and scalar type.
  Index work_threads = std::max<Index>(1, static_cast<Index>(work / kMinTaskSize));

  // 1-D cap along the primary dim: the historical thread budget.
  int threads = static_cast<int>(std::min<Index>({Index(nbThreads()), n_cap, work_threads}));

  // If the 1-D budget is limited by the primary dim (n_cap), and both the
  // thread pool and the work budget can absorb more threads, extend the
  // partition along the secondary dim (2-D). 2-D requires a larger per-thread
  // work floor than 1-D because the second dimension of threads brings extra
  // synchronization (cooperative packing, atomic user counts, etc.) whose
  // overhead must be amortized by the added compute. When n_cap >= nbThreads()
  // or work_threads_2d <= n_cap, no extension is possible and the partition
  // remains 1-D and bit-identical to the pre-2-D behavior.
  double k2dMinTaskSize = 5.0 * kMinTaskSize;  // FIXME: tune per architecture.
  Index work_threads_2d = std::max<Index>(1, static_cast<Index>(work / k2dMinTaskSize));
  if (threads == n_cap && n_cap < nbThreads() && n_cap < work_threads_2d) {
    Index threads_2d = std::min<Index>({Index(nbThreads()), n_cap * m_cap, work_threads_2d});
    threads = static_cast<int>(threads_2d);
  }

  // if multi-threading is explicitly disabled, not useful, or if we already are
  // inside a parallel session, then abort multi-threading
  bool dont_parallelize = (!Condition) || (threads <= 1);
#if defined(EIGEN_HAS_OPENMP)
  // don't parallelize if we are executing in a parallel context already.
  dont_parallelize |= omp_get_num_threads() > 1;
#elif defined(EIGEN_GEMM_THREADPOOL)
  // don't parallelize if we have a trivial threadpool or the current thread id
  // is != -1, indicating that we are already executing on a thread inside the pool.
  // In other words, we do not allow nested parallelism, since this would lead to
  // deadlocks due to the workstealing nature of the threadpool.
  ThreadPool* pool = getGemmThreadPool();
  dont_parallelize |= (pool == nullptr || pool->CurrentThreadId() != -1);
#endif
  if (dont_parallelize) return func(0, rows, 0, cols);

  if (transpose) std::swap(rows, cols);

  // Compute 2-D (t_m x t_n) partition of the output. When cols can absorb all
  // threads along the primary split, t_m is 1 and we keep the historical 1-D
  // column partitioning bit-for-bit.
  int t_m = 1, t_n = 1, actual_threads = 1;
  compute_2d_partition<Index>(threads, rows, cols, Functor::Traits::nr, Functor::Traits::mr, t_m, t_n, actual_threads);

  // Tell the blocking machinery the number of threads that will actually share
  // the packed A buffer so it can size mc accordingly.
  func.initParallelSession(actual_threads);

  ei_declare_aligned_stack_constructed_variable(GemmParallelTaskInfo<Index>, task_info, actual_threads, 0);

  // Populate task_info (lhs_start / lhs_length for every thread) BEFORE launching
  // the parallel region. The kernel reads other threads' entries to infer its
  // row group's blockA bounds, so all entries must be visible without relying
  // on a later barrier.
  ei_declare_aligned_stack_constructed_variable(gemm_thread_tile<Index>, tiles, actual_threads, 0);
  for (int i = 0; i < actual_threads; ++i) {
    int tid_m = i / t_n;
    int tid_n = i % t_n;
    tiles[i] = compute_gemm_thread_tile<Index, Functor::Traits::mr>(tid_m, tid_n, t_m, t_n, rows, cols);
    task_info[i].lhs_start = tiles[i].lhs_start;
    task_info[i].lhs_length = tiles[i].lhs_length;
  }

#if defined(EIGEN_HAS_OPENMP)
#pragma omp parallel num_threads(actual_threads)
  {
    int i = omp_get_thread_num();
    // Note that the actual number of threads might be lower than the number of
    // requested ones
    int omp_threads = omp_get_num_threads();
    // If OpenMP gave us fewer threads than requested, the partition we precomputed
    // no longer matches and correctness would break. Bail out to serial in that
    // rare case.
    if (omp_threads != actual_threads) {
#pragma omp single
      func(0, rows, 0, cols);
    } else {
      GemmParallelInfo<Index> info(i, omp_threads, t_m, t_n, task_info);

      if (transpose)
        func(tiles[i].c0, tiles[i].actual_block_cols, 0, rows, &info);
      else
        func(0, rows, tiles[i].c0, tiles[i].actual_block_cols, &info);
    }
  }

#elif defined(EIGEN_GEMM_THREADPOOL)
  Barrier barrier(actual_threads);
  auto task = [=, &func, &barrier, &task_info, &tiles](int i) {
    GemmParallelInfo<Index> info(i, actual_threads, t_m, t_n, task_info);

    if (transpose)
      func(tiles[i].c0, tiles[i].actual_block_cols, 0, rows, &info);
    else
      func(0, rows, tiles[i].c0, tiles[i].actual_block_cols, &info);

    barrier.Notify();
  };
  // Notice that we do not schedule more than "actual_threads" tasks, which allows us to
  // limit number of running threads, even if the threadpool itself was constructed
  // with a larger number of threads.
  for (int i = 0; i < actual_threads - 1; ++i) {
    pool->Schedule([=, task = std::move(task)] { task(i); });
  }
  task(actual_threads - 1);
  barrier.Wait();
#endif
}

#endif

}  // end namespace internal
}  // end namespace Eigen

#endif  // EIGEN_PARALLELIZER_H
