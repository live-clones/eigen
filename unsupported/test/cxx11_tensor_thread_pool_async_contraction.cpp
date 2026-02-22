// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// cxx11_tensor_thread_pool split: async contraction tests.

#include "cxx11_tensor_thread_pool_helpers.h"

template <int DataLayout>
void test_async_multithread_contraction_agrees_with_singlethread() {
  int contract_size = internal::random<int>(100, 500);

  Tensor<float, 3, DataLayout> left(internal::random<int>(10, 40), contract_size, internal::random<int>(10, 40));

  Tensor<float, 4, DataLayout> right(internal::random<int>(1, 20), internal::random<int>(1, 20), contract_size,
                                     internal::random<int>(1, 20));

  left.setRandom();
  right.setRandom();

  // add constants to shift values away from 0 for more precision
  left += left.constant(1.5f);
  right += right.constant(1.5f);

  typedef Tensor<float, 1>::DimensionPair DimPair;
  Eigen::array<DimPair, 1> dims({{DimPair(1, 2)}});

  Eigen::ThreadPool tp(internal::random<int>(2, 11));
  Eigen::ThreadPoolDevice thread_pool_device(&tp, internal::random<int>(8, 32));

  Tensor<float, 5, DataLayout> st_result;
  st_result = left.contract(right, dims);

  Tensor<float, 5, DataLayout> tp_result(st_result.dimensions());

  Eigen::Barrier barrier(1);
  tp_result.device(thread_pool_device, [&barrier]() { barrier.Notify(); }) = left.contract(right, dims);
  barrier.Wait();

  VERIFY(dimensions_match(st_result.dimensions(), tp_result.dimensions()));
  for (ptrdiff_t i = 0; i < st_result.size(); i++) {
    // if both of the values are very small, then do nothing (because the test
    // will fail due to numerical precision issues when values are small)
    if (numext::abs(st_result.data()[i] - tp_result.data()[i]) >= 1e-4f) {
      VERIFY_IS_APPROX(st_result.data()[i], tp_result.data()[i]);
    }
  }
}

// We are triggering 'evalShardedByInnerDim' optimization.
template <int DataLayout>
static void test_async_sharded_by_inner_dim_contraction() {
  typedef Tensor<float, 1>::DimensionPair DimPair;

  const int num_threads = internal::random<int>(4, 16);
  ThreadPool threads(num_threads);
  Eigen::ThreadPoolDevice device(&threads, num_threads);

  Tensor<float, 2, DataLayout> t_left(2, 10000);
  Tensor<float, 2, DataLayout> t_right(10000, 10);
  Tensor<float, 2, DataLayout> t_result(2, 10);

  t_left.setRandom();
  t_right.setRandom();
  // Put trash in t_result to verify contraction clears output memory.
  t_result.setRandom();

  // Add a little offset so that the results won't be close to zero.
  t_left += t_left.constant(1.0f);
  t_right += t_right.constant(1.0f);

  typedef Map<Eigen::Matrix<float, Dynamic, Dynamic, DataLayout>> MapXf;
  MapXf m_left(t_left.data(), 2, 10000);
  MapXf m_right(t_right.data(), 10000, 10);
  Eigen::Matrix<float, Dynamic, Dynamic, DataLayout> m_result(2, 10);

  // this contraction should be equivalent to a single matrix multiplication
  Eigen::array<DimPair, 1> dims({{DimPair(1, 0)}});

  // compute results by separate methods
  Eigen::Barrier barrier(1);
  t_result.device(device, [&barrier]() { barrier.Notify(); }) = t_left.contract(t_right, dims);
  barrier.Wait();

  m_result = m_left * m_right;

  for (Index i = 0; i < t_result.dimensions().TotalSize(); i++) {
    VERIFY_IS_APPROX(t_result.data()[i], m_result.data()[i]);
  }
}

// We are triggering 'evalShardedByInnerDim' optimization with output kernel.
template <int DataLayout>
static void test_async_sharded_by_inner_dim_contraction_with_output_kernel() {
  typedef Tensor<float, 1>::DimensionPair DimPair;

  const int num_threads = internal::random<int>(4, 16);
  ThreadPool threads(num_threads);
  Eigen::ThreadPoolDevice device(&threads, num_threads);

  Tensor<float, 2, DataLayout> t_left(2, 10000);
  Tensor<float, 2, DataLayout> t_right(10000, 10);
  Tensor<float, 2, DataLayout> t_result(2, 10);

  t_left.setRandom();
  t_right.setRandom();
  // Put trash in t_result to verify contraction clears output memory.
  t_result.setRandom();

  // Add a little offset so that the results won't be close to zero.
  t_left += t_left.constant(1.0f);
  t_right += t_right.constant(1.0f);

  typedef Map<Eigen::Matrix<float, Dynamic, Dynamic, DataLayout>> MapXf;
  MapXf m_left(t_left.data(), 2, 10000);
  MapXf m_right(t_right.data(), 10000, 10);
  Eigen::Matrix<float, Dynamic, Dynamic, DataLayout> m_result(2, 10);

  // this contraction should be equivalent to a single matrix multiplication
  Eigen::array<DimPair, 1> dims({{DimPair(1, 0)}});

  // compute results by separate methods
  Eigen::Barrier barrier(1);
  t_result.device(device, [&barrier]() { barrier.Notify(); }) = t_left.contract(t_right, dims, SqrtOutputKernel());
  barrier.Wait();
  m_result = m_left * m_right;

  for (Index i = 0; i < t_result.dimensions().TotalSize(); i++) {
    VERIFY_IS_APPROX(t_result.data()[i], std::sqrt(m_result.data()[i]));
  }
}

TEST(TensorThreadPoolAsyncContractionTest, Basic) {
  test_async_multithread_contraction_agrees_with_singlethread<ColMajor>();
  test_async_multithread_contraction_agrees_with_singlethread<RowMajor>();

  test_async_sharded_by_inner_dim_contraction<ColMajor>();
  test_async_sharded_by_inner_dim_contraction<RowMajor>();
  test_async_sharded_by_inner_dim_contraction_with_output_kernel<ColMajor>();
  test_async_sharded_by_inner_dim_contraction_with_output_kernel<RowMajor>();
}
