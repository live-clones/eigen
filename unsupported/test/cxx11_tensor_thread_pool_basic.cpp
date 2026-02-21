// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// cxx11_tensor_thread_pool split: elementwise, chip, volume_patch,
// compound_assignment, memcpy, random, shuffle, allocate tests.

#include "cxx11_tensor_thread_pool_helpers.h"

void test_multithread_elementwise() {
  Tensor<float, 3> in1(200, 30, 70);
  Tensor<float, 3> in2(200, 30, 70);
  Tensor<double, 3> out(200, 30, 70);

  in1.setRandom();
  in2.setRandom();

  Eigen::ThreadPool tp(internal::random<int>(3, 11));
  Eigen::ThreadPoolDevice thread_pool_device(&tp, internal::random<int>(3, 11));
  out.device(thread_pool_device) = (in1 + in2 * 3.14f).cast<double>();

  for (int i = 0; i < 200; ++i) {
    for (int j = 0; j < 30; ++j) {
      for (int k = 0; k < 70; ++k) {
        VERIFY_IS_APPROX(out(i, j, k), static_cast<double>(in1(i, j, k) + in2(i, j, k) * 3.14f));
      }
    }
  }
}

void test_multithread_chip() {
  Tensor<float, 5> in(2, 3, 5, 7, 11);
  Tensor<float, 4> out(3, 5, 7, 11);

  in.setRandom();

  Eigen::ThreadPool tp(internal::random<int>(3, 11));
  Eigen::ThreadPoolDevice thread_pool_device(&tp, internal::random<int>(3, 11));

  out.device(thread_pool_device) = in.chip(1, 0);

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

void test_multithread_volume_patch() {
  Tensor<float, 5> in(4, 2, 3, 5, 7);
  Tensor<float, 6> out(4, 1, 1, 1, 2 * 3 * 5, 7);

  in.setRandom();

  Eigen::ThreadPool tp(internal::random<int>(3, 11));
  Eigen::ThreadPoolDevice thread_pool_device(&tp, internal::random<int>(3, 11));

  out.device(thread_pool_device) = in.extract_volume_patches(1, 1, 1);

  for (int i = 0; i < in.size(); ++i) {
    VERIFY_IS_EQUAL(in.data()[i], out.data()[i]);
  }
}

void test_multithread_compound_assignment() {
  Tensor<float, 3> in1(2, 3, 7);
  Tensor<float, 3> in2(2, 3, 7);
  Tensor<float, 3> out(2, 3, 7);

  in1.setRandom();
  in2.setRandom();

  Eigen::ThreadPool tp(internal::random<int>(3, 11));
  Eigen::ThreadPoolDevice thread_pool_device(&tp, internal::random<int>(3, 11));
  out.device(thread_pool_device) = in1;
  out.device(thread_pool_device) += in2 * 3.14f;

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 7; ++k) {
        VERIFY_IS_APPROX(out(i, j, k), in1(i, j, k) + in2(i, j, k) * 3.14f);
      }
    }
  }
}

void test_memcpy() {
  for (int i = 0; i < 5; ++i) {
    const int num_threads = internal::random<int>(3, 11);
    Eigen::ThreadPool tp(num_threads);
    Eigen::ThreadPoolDevice thread_pool_device(&tp, num_threads);

    const int size = internal::random<int>(13, 7632);
    Tensor<float, 1> t1(size);
    t1.setRandom();
    std::vector<float> result(size);
    thread_pool_device.memcpy(&result[0], t1.data(), size * sizeof(float));
    for (int j = 0; j < size; j++) {
      VERIFY_IS_EQUAL(t1(j), result[j]);
    }
  }
}

void test_multithread_random() {
  Eigen::ThreadPool tp(2);
  Eigen::ThreadPoolDevice device(&tp, 2);
  Tensor<float, 1> t(1 << 20);
  t.device(device) = t.random<Eigen::internal::NormalRandomGenerator<float>>();
}

template <int DataLayout>
void test_multithread_shuffle(Allocator* allocator) {
  Tensor<float, 4, DataLayout> tensor(17, 5, 7, 11);
  tensor.setRandom();

  const int num_threads = internal::random<int>(2, 11);
  ThreadPool threads(num_threads);
  Eigen::ThreadPoolDevice device(&threads, num_threads, allocator);

  Tensor<float, 4, DataLayout> shuffle(7, 5, 11, 17);
  array<ptrdiff_t, 4> shuffles = {{2, 1, 3, 0}};
  shuffle.device(device) = tensor.shuffle(shuffles);

  for (int i = 0; i < 17; ++i) {
    for (int j = 0; j < 5; ++j) {
      for (int k = 0; k < 7; ++k) {
        for (int l = 0; l < 11; ++l) {
          VERIFY_IS_EQUAL(tensor(i, j, k, l), shuffle(k, j, l, i));
        }
      }
    }
  }
}

void test_threadpool_allocate(TestAllocator* allocator) {
  const int num_threads = internal::random<int>(2, 11);
  const int num_allocs = internal::random<int>(2, 11);
  ThreadPool threads(num_threads);
  Eigen::ThreadPoolDevice device(&threads, num_threads, allocator);

  for (int a = 0; a < num_allocs; ++a) {
    void* ptr = device.allocate(512);
    device.deallocate(ptr);
  }
  VERIFY(allocator != NULL);
  VERIFY_IS_EQUAL(allocator->alloc_count(), num_allocs);
  VERIFY_IS_EQUAL(allocator->dealloc_count(), num_allocs);
}

EIGEN_DECLARE_TEST(cxx11_tensor_thread_pool_basic) {
  test_multithread_elementwise();
  test_multithread_compound_assignment();

  test_multithread_chip();

  test_multithread_volume_patch();

  test_memcpy();
  test_multithread_random();

  TestAllocator test_allocator;
  test_multithread_shuffle<ColMajor>(NULL);
  test_multithread_shuffle<RowMajor>(&test_allocator);
  test_threadpool_allocate(&test_allocator);
}
