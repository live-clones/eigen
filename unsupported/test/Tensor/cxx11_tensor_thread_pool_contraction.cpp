// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// cxx11_tensor_thread_pool split: all synchronous contraction tests.

#include "cxx11_tensor_thread_pool_helpers.h"

template <int DataLayout>
void test_multithread_contraction() {
  Tensor<float, 4, DataLayout> t_left(30, 50, 37, 31);
  Tensor<float, 5, DataLayout> t_right(37, 31, 70, 2, 10);
  Tensor<float, 5, DataLayout> t_result(30, 50, 70, 2, 10);

  t_left.setRandom();
  t_right.setRandom();

  // this contraction should be equivalent to a single matrix multiplication
  typedef Tensor<float, 1>::DimensionPair DimPair;
  Eigen::array<DimPair, 2> dims({{DimPair(2, 0), DimPair(3, 1)}});

  typedef Map<Matrix<float, Dynamic, Dynamic, DataLayout>> MapXf;
  MapXf m_left(t_left.data(), 1500, 1147);
  MapXf m_right(t_right.data(), 1147, 1400);
  Matrix<float, Dynamic, Dynamic, DataLayout> m_result(1500, 1400);

  Eigen::ThreadPool tp(4);
  Eigen::ThreadPoolDevice thread_pool_device(&tp, 4);

  // compute results by separate methods
  t_result.device(thread_pool_device) = t_left.contract(t_right, dims);
  m_result = m_left * m_right;

  for (ptrdiff_t i = 0; i < t_result.size(); i++) {
    VERIFY(&t_result.data()[i] != &m_result.data()[i]);
    if (fabsf(t_result(i) - m_result(i)) < 1e-4f) {
      continue;
    }
    if (Eigen::internal::isApprox(t_result(i), m_result(i), 1e-4f)) {
      continue;
    }
    std::cout << "mismatch detected at index " << i << ": " << t_result(i) << " vs " << m_result(i) << std::endl;
    assert(false);
  }
}

template <int DataLayout>
void test_contraction_corner_cases() {
  Tensor<float, 2, DataLayout> t_left(32, 500);
  Tensor<float, 2, DataLayout> t_right(32, 28 * 28);
  Tensor<float, 2, DataLayout> t_result(500, 28 * 28);

  t_left = (t_left.constant(-0.5f) + t_left.random()) * 2.0f;
  t_right = (t_right.constant(-0.6f) + t_right.random()) * 2.0f;
  t_result = t_result.constant(NAN);

  // this contraction should be equivalent to a single matrix multiplication
  typedef Tensor<float, 1>::DimensionPair DimPair;
  Eigen::array<DimPair, 1> dims{{DimPair(0, 0)}};

  typedef Map<Matrix<float, Dynamic, Dynamic, DataLayout>> MapXf;
  MapXf m_left(t_left.data(), 32, 500);
  MapXf m_right(t_right.data(), 32, 28 * 28);
  Matrix<float, Dynamic, Dynamic, DataLayout> m_result(500, 28 * 28);

  Eigen::ThreadPool tp(12);
  Eigen::ThreadPoolDevice thread_pool_device(&tp, 12);

  // compute results by separate methods
  t_result.device(thread_pool_device) = t_left.contract(t_right, dims);
  m_result = m_left.transpose() * m_right;

  for (ptrdiff_t i = 0; i < t_result.size(); i++) {
    assert(!(numext::isnan)(t_result.data()[i]));
    if (fabsf(t_result.data()[i] - m_result.data()[i]) >= 1e-4f) {
      std::cout << "mismatch detected at index " << i << " : " << t_result.data()[i] << " vs " << m_result.data()[i]
                << std::endl;
      assert(false);
    }
  }

  t_left.resize(32, 1);
  t_left = (t_left.constant(-0.5f) + t_left.random()) * 2.0f;
  t_result.resize(1, 28 * 28);
  t_result = t_result.constant(NAN);
  t_result.device(thread_pool_device) = t_left.contract(t_right, dims);
  new (&m_left) MapXf(t_left.data(), 32, 1);
  m_result = m_left.transpose() * m_right;
  for (ptrdiff_t i = 0; i < t_result.size(); i++) {
    assert(!(numext::isnan)(t_result.data()[i]));
    if (fabsf(t_result.data()[i] - m_result.data()[i]) >= 1e-4f) {
      std::cout << "mismatch detected: " << t_result.data()[i] << " vs " << m_result.data()[i] << std::endl;
      assert(false);
    }
  }

  t_left.resize(32, 500);
  t_right.resize(32, 4);
  t_left = (t_left.constant(-0.5f) + t_left.random()) * 2.0f;
  t_right = (t_right.constant(-0.6f) + t_right.random()) * 2.0f;
  t_result.resize(500, 4);
  t_result = t_result.constant(NAN);
  t_result.device(thread_pool_device) = t_left.contract(t_right, dims);
  new (&m_left) MapXf(t_left.data(), 32, 500);
  new (&m_right) MapXf(t_right.data(), 32, 4);
  m_result = m_left.transpose() * m_right;
  for (ptrdiff_t i = 0; i < t_result.size(); i++) {
    assert(!(numext::isnan)(t_result.data()[i]));
    if (fabsf(t_result.data()[i] - m_result.data()[i]) >= 1e-4f) {
      std::cout << "mismatch detected: " << t_result.data()[i] << " vs " << m_result.data()[i] << std::endl;
      assert(false);
    }
  }

  t_left.resize(32, 1);
  t_right.resize(32, 4);
  t_left = (t_left.constant(-0.5f) + t_left.random()) * 2.0f;
  t_right = (t_right.constant(-0.6f) + t_right.random()) * 2.0f;
  t_result.resize(1, 4);
  t_result = t_result.constant(NAN);
  t_result.device(thread_pool_device) = t_left.contract(t_right, dims);
  new (&m_left) MapXf(t_left.data(), 32, 1);
  new (&m_right) MapXf(t_right.data(), 32, 4);
  m_result = m_left.transpose() * m_right;
  for (ptrdiff_t i = 0; i < t_result.size(); i++) {
    assert(!(numext::isnan)(t_result.data()[i]));
    if (fabsf(t_result.data()[i] - m_result.data()[i]) >= 1e-4f) {
      std::cout << "mismatch detected: " << t_result.data()[i] << " vs " << m_result.data()[i] << std::endl;
      assert(false);
    }
  }
}

template <int DataLayout>
void test_multithread_contraction_agrees_with_singlethread() {
  int contract_size = internal::random<int>(1, 5000);

  Tensor<float, 3, DataLayout> left(internal::random<int>(1, 80), contract_size, internal::random<int>(1, 100));

  Tensor<float, 4, DataLayout> right(internal::random<int>(1, 25), internal::random<int>(1, 37), contract_size,
                                     internal::random<int>(1, 51));

  left.setRandom();
  right.setRandom();

  // add constants to shift values away from 0 for more precision
  left += left.constant(1.5f);
  right += right.constant(1.5f);

  typedef Tensor<float, 1>::DimensionPair DimPair;
  Eigen::array<DimPair, 1> dims({{DimPair(1, 2)}});

  Eigen::ThreadPool tp(internal::random<int>(2, 11));
  Eigen::ThreadPoolDevice thread_pool_device(&tp, internal::random<int>(2, 11));

  Tensor<float, 5, DataLayout> st_result;
  st_result = left.contract(right, dims);

  Tensor<float, 5, DataLayout> tp_result(st_result.dimensions());
  tp_result.device(thread_pool_device) = left.contract(right, dims);

  VERIFY(dimensions_match(st_result.dimensions(), tp_result.dimensions()));
  for (ptrdiff_t i = 0; i < st_result.size(); i++) {
    // if both of the values are very small, then do nothing (because the test will fail
    // due to numerical precision issues when values are small)
    if (numext::abs(st_result.data()[i] - tp_result.data()[i]) >= 1e-4f) {
      VERIFY_IS_APPROX(st_result.data()[i], tp_result.data()[i]);
    }
  }
}

template <int DataLayout>
static void test_multithread_contraction_with_output_kernel() {
  typedef Tensor<float, 1>::DimensionPair DimPair;

  const int num_threads = internal::random<int>(2, 11);
  ThreadPool threads(num_threads);
  Eigen::ThreadPoolDevice device(&threads, num_threads);

  Tensor<float, 4, DataLayout> t_left(30, 50, 8, 31);
  Tensor<float, 5, DataLayout> t_right(8, 31, 7, 20, 10);
  Tensor<float, 5, DataLayout> t_result(30, 50, 7, 20, 10);

  t_left.setRandom();
  t_right.setRandom();
  // Put trash in mat4 to verify contraction clears output memory.
  t_result.setRandom();

  // Add a little offset so that the results won't be close to zero.
  t_left += t_left.constant(1.0f);
  t_right += t_right.constant(1.0f);

  typedef Map<Eigen::Matrix<float, Dynamic, Dynamic, DataLayout>> MapXf;
  MapXf m_left(t_left.data(), 1500, 248);
  MapXf m_right(t_right.data(), 248, 1400);
  Eigen::Matrix<float, Dynamic, Dynamic, DataLayout> m_result(1500, 1400);

  // this contraction should be equivalent to a single matrix multiplication
  Eigen::array<DimPair, 2> dims({{DimPair(2, 0), DimPair(3, 1)}});

  // compute results by separate methods
  t_result.device(device) = t_left.contract(t_right, dims, SqrtOutputKernel());

  m_result = m_left * m_right;

  for (Index i = 0; i < t_result.dimensions().TotalSize(); i++) {
    VERIFY(&t_result.data()[i] != &m_result.data()[i]);
    VERIFY_IS_APPROX(t_result.data()[i], std::sqrt(m_result.data()[i]));
  }
}

// We are triggering 'evalShardedByInnerDim' optimization.
template <int DataLayout>
static void test_sharded_by_inner_dim_contraction() {
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
  t_result.device(device) = t_left.contract(t_right, dims);
  m_result = m_left * m_right;

  for (Index i = 0; i < t_result.dimensions().TotalSize(); i++) {
    VERIFY_IS_APPROX(t_result.data()[i], m_result.data()[i]);
  }
}

// We are triggering 'evalShardedByInnerDim' optimization with output kernel.
template <int DataLayout>
static void test_sharded_by_inner_dim_contraction_with_output_kernel() {
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
  t_result.device(device) = t_left.contract(t_right, dims, SqrtOutputKernel());
  m_result = m_left * m_right;

  for (Index i = 0; i < t_result.dimensions().TotalSize(); i++) {
    VERIFY_IS_APPROX(t_result.data()[i], std::sqrt(m_result.data()[i]));
  }
}

template <int DataLayout>
void test_full_contraction() {
  int contract_size1 = internal::random<int>(1, 500);
  int contract_size2 = internal::random<int>(1, 500);

  Tensor<float, 2, DataLayout> left(contract_size1, contract_size2);
  Tensor<float, 2, DataLayout> right(contract_size1, contract_size2);
  left.setRandom();
  right.setRandom();

  // add constants to shift values away from 0 for more precision
  left += left.constant(1.5f);
  right += right.constant(1.5f);

  typedef Tensor<float, 2>::DimensionPair DimPair;
  Eigen::array<DimPair, 2> dims({{DimPair(0, 0), DimPair(1, 1)}});

  Eigen::ThreadPool tp(internal::random<int>(2, 11));
  Eigen::ThreadPoolDevice thread_pool_device(&tp, internal::random<int>(2, 11));

  Tensor<float, 0, DataLayout> st_result;
  st_result = left.contract(right, dims);

  Tensor<float, 0, DataLayout> tp_result;
  tp_result.device(thread_pool_device) = left.contract(right, dims);

  VERIFY(dimensions_match(st_result.dimensions(), tp_result.dimensions()));
  // if both of the values are very small, then do nothing (because the test will fail
  // due to numerical precision issues when values are small)
  if (numext::abs(st_result() - tp_result()) >= 1e-4f) {
    VERIFY_IS_APPROX(st_result(), tp_result());
  }
}

TEST(TensorThreadPoolContractionTest, Basic) {
  test_multithread_contraction<ColMajor>();
  test_multithread_contraction<RowMajor>();

  test_multithread_contraction_agrees_with_singlethread<ColMajor>();
  test_multithread_contraction_agrees_with_singlethread<RowMajor>();
  test_multithread_contraction_with_output_kernel<ColMajor>();
  test_multithread_contraction_with_output_kernel<RowMajor>();

  // Test EvalShardedByInnerDimContext parallelization strategy.
  test_sharded_by_inner_dim_contraction<ColMajor>();
  test_sharded_by_inner_dim_contraction<RowMajor>();
  test_sharded_by_inner_dim_contraction_with_output_kernel<ColMajor>();
  test_sharded_by_inner_dim_contraction_with_output_kernel<RowMajor>();

  // Exercise various cases that have been problematic in the past.
  test_contraction_corner_cases<ColMajor>();
  test_contraction_corner_cases<RowMajor>();

  test_full_contraction<ColMajor>();
  test_full_contraction<RowMajor>();
}
