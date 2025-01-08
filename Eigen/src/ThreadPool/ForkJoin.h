// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2025 Weiwei Kong <weiweikong@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_THREADPOOL_FORKJOIN_H
#define EIGEN_THREADPOOL_FORKJOIN_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

// ForkJoinScheduler provides implementations of various non-blocking
// ParallelFor algorithms for unary and binary parallel tasks. More specfically,
// the implementations follow the binary tree-based algorithm from the following
// paper:
//
//   Lea, D. (2000, June). A java fork/join framework. *In Proceedings of the
//   ACM 2000 conference on Java Grande* (pp. 36-43).
//
// Example usage #1 (synchronous):
// ```
// ThreadPool thread_pool(num_threads);
// ForkJoinScheduler::ParallelFor(0, num_tasks, granularity, std::move(parallel_task), &thread_pool);
// ```
//
// Example usage #2 (asynchronous):
// ```
// ThreadPool thread_pool(num_threads);
// Barrier barrier(num_completions);
// auto done = [&](){barrier.Notify();};
// for (int k=0; k<num_async_calls; ++k) {
//   thread_pool.Schedule([&](){
//     ForkJoinScheduler::ParallelFor(0, num_tasks, granularity, parallel_task, done, &thread_pool);
//   });
// }
// barrier.Wait();
// ```
class ForkJoinScheduler {
 public:
  // Runs `do_func` in parallel for the range [start, end) with a specified granularity. `do_func` should 
  // either be of type `std::function<void(int)>` or `std::function<void(int, int)`.
  template <typename DoFnType>
  static void ParallelForAsync(int start, int end, int granularity, DoFnType do_func,
      std::function<void()> done, Eigen::ThreadPool* thread_pool) {
    if (start >= end) return;
    ForkJoinScheduler::RunParallelForAsync(start, end, granularity, do_func, done, thread_pool);
  }

  // Synchronous variant of Async::ParallelFor.
  template <typename DoFnType>
  static void ParallelFor(int start, int end, int granularity, DoFnType do_func,
      Eigen::ThreadPool* thread_pool) {
    if (start >= end) return;
    auto dummy_done = [](){};
    Barrier barrier(1);
    thread_pool->Schedule([start, end, granularity, thread_pool, &do_func, &dummy_done, &barrier](){
        ForkJoinScheduler::ParallelForAsync(start, end, granularity, do_func, dummy_done, thread_pool);
        barrier.Notify();
    });
    barrier.Wait();
  }

 private:
  // Schedules `right_thunk`, runs `left_thunk` and runs other tasks until
  // `right_thunk` has finished.
  template <typename LeftType, typename RightType>
  static void ForkJoin(LeftType&& left_thunk, RightType&& right_thunk, Eigen::ThreadPool* thread_pool) {
    std::atomic<bool> right_done(false);
    auto execute_right = [&right_thunk, &right_done]() {
      std::forward<RightType>(right_thunk)();
      right_done.store(true, std::memory_order_release);
    };
    thread_pool->Schedule(execute_right);
    std::forward<LeftType>(left_thunk)();
    Eigen::ThreadPool::Task task;
    while (!right_done.load(std::memory_order_acquire)) {
      thread_pool->MaybeGetTask(&task);
      if (task.f) task.f();
    }
  }

  // Runs `do_func` in parallel for the range [start, end). The main recursive asynchronous runner that 
  // calls `ForkJoin`.
  static void RunParallelForAsync(int start, int end, int granularity, std::function<void(int)>& do_func,
      std::function<void()> &done, Eigen::ThreadPool* thread_pool) {
    std::function<void(int, int)> wrapped_do_func = [&do_func](int start, int end) {
      for (int i = start; i < end; ++i) do_func(i);
    };
    ForkJoinScheduler::RunParallelForAsync(start, end, granularity, wrapped_do_func, done, thread_pool);
  }

  // Variant of `RunAsyncParallelFor` that uses a do function that operates on an index range. 
  // Specifically, `do_func` takes two arguments: the start and end of the range.
  static void RunParallelForAsync(int start, int end, int granularity, std::function<void(int, int)>& do_func,
      std::function<void()> &done, Eigen::ThreadPool* thread_pool) {
    if ((end - start) <= granularity) {
      do_func(start, end);
      done();
    } else {
      const int size = end - start;
      const int mid = start + Eigen::numext::div_ceil(9 * (size + 1) / 16, granularity) * granularity;
      ForkJoinScheduler::ForkJoin([start, mid, granularity, &do_func, &done, thread_pool]() { 
                                    RunParallelForAsync(start, mid, granularity, do_func, done, thread_pool); 
                                  },
                                  [mid, end, granularity, &do_func, &done, thread_pool]() {
                                    RunParallelForAsync(mid, end, granularity, do_func, done, thread_pool);
                                  },
                                  thread_pool);
    }
  }
};

}  // namespace Eigen

#endif  // EIGEN_THREADPOOL_FORKJOIN_H
