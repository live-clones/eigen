// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// cxx11_tensor_thread_pool split: async elementwise, chip, volume_patch, reductions.

#include "cxx11_tensor_thread_pool_helpers.h"

void test_async_multithread_elementwise() {
  Tensor<float, 3> in1(200, 30, 70);
  Tensor<float, 3> in2(200, 30, 70);
  Tensor<double, 3> out(200, 30, 70);

  in1.setRandom();
  in2.setRandom();

  Eigen::ThreadPool tp(internal::random<int>(3, 11));
  Eigen::ThreadPoolDevice thread_pool_device(&tp, internal::random<int>(3, 11));

  Eigen::Barrier b(1);
  out.device(thread_pool_device, [&b]() { b.Notify(); }) = (in1 + in2 * 3.14f).cast<double>();
  b.Wait();

  for (int i = 0; i < 200; ++i) {
    for (int j = 0; j < 30; ++j) {
      for (int k = 0; k < 70; ++k) {
        VERIFY_IS_APPROX(out(i, j, k), static_cast<double>(in1(i, j, k) + in2(i, j, k) * 3.14f));
      }
    }
  }
}

void test_async_multithread_chip() {
  Tensor<float, 5> in(2, 3, 5, 7, 11);
  Tensor<float, 4> out(3, 5, 7, 11);

  in.setRandom();

  Eigen::ThreadPool tp(internal::random<int>(3, 11));
  Eigen::ThreadPoolDevice thread_pool_device(&tp, internal::random<int>(3, 11));

  Eigen::Barrier b(1);
  out.device(thread_pool_device, [&b]() { b.Notify(); }) = in.chip(1, 0);
  b.Wait();

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 5; ++j) {
      for (int k = 0; k < 7; ++k) {
        for (int l = 0; l < 11; ++l) {
          VERIFY_IS_EQUAL(out(i, j, k, l), in(1, i, j, k, l));
        }
      }
    }
  }
}

void test_async_multithread_volume_patch() {
  Tensor<float, 5> in(4, 2, 3, 5, 7);
  Tensor<float, 6> out(4, 1, 1, 1, 2 * 3 * 5, 7);

  in.setRandom();

  Eigen::ThreadPool tp(internal::random<int>(3, 11));
  Eigen::ThreadPoolDevice thread_pool_device(&tp, internal::random<int>(3, 11));

  Eigen::Barrier b(1);
  out.device(thread_pool_device, [&b]() { b.Notify(); }) = in.extract_volume_patches(1, 1, 1);
  b.Wait();

  for (int i = 0; i < in.size(); ++i) {
    VERIFY_IS_EQUAL(in.data()[i], out.data()[i]);
  }
}

template <int DataLayout>
void test_multithreaded_reductions() {
  const int num_threads = internal::random<int>(3, 11);
  ThreadPool thread_pool(num_threads);
  Eigen::ThreadPoolDevice thread_pool_device(&thread_pool, num_threads);

  const int num_rows = internal::random<int>(13, 732);
  const int num_cols = internal::random<int>(13, 732);
  Tensor<float, 2, DataLayout> t1(num_rows, num_cols);
  t1.setRandom();

  Tensor<float, 0, DataLayout> full_redux;
  full_redux = t1.sum();

  Tensor<float, 0, DataLayout> full_redux_tp;
  full_redux_tp.device(thread_pool_device) = t1.sum();

  // Check that the single threaded and the multi threaded reductions return
  // the same result.
  VERIFY_IS_APPROX(full_redux(), full_redux_tp());
}

TEST(TensorThreadPoolAsyncTest, Basic) {
  test_async_multithread_elementwise();

  test_async_multithread_chip();

  test_async_multithread_volume_patch();

  test_multithreaded_reductions<ColMajor>();
  test_multithreaded_reductions<RowMajor>();
}
