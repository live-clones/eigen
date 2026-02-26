// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Shared header for split cxx11_tensor_thread_pool tests.

#ifndef EIGEN_TEST_CXX11_TENSOR_THREAD_POOL_HELPERS_H
#define EIGEN_TEST_CXX11_TENSOR_THREAD_POOL_HELPERS_H

#define EIGEN_USE_THREADS

#include "main.h"
#include <iostream>
#include <Eigen/CXX11/Tensor>

using Eigen::Tensor;

class TestAllocator : public Allocator {
 public:
  ~TestAllocator() EIGEN_OVERRIDE {}
  EIGEN_DEVICE_FUNC void* allocate(size_t num_bytes) const EIGEN_OVERRIDE {
    const_cast<TestAllocator*>(this)->alloc_count_++;
    return internal::aligned_malloc(num_bytes);
  }
  EIGEN_DEVICE_FUNC void deallocate(void* buffer) const EIGEN_OVERRIDE {
    const_cast<TestAllocator*>(this)->dealloc_count_++;
    internal::aligned_free(buffer);
  }

  int alloc_count() const { return alloc_count_; }
  int dealloc_count() const { return dealloc_count_; }

 private:
  int alloc_count_ = 0;
  int dealloc_count_ = 0;
};

// Apply Sqrt to all output elements.
struct SqrtOutputKernel {
  template <typename Index, typename Scalar>
  EIGEN_ALWAYS_INLINE void operator()(const internal::blas_data_mapper<Scalar, Index, ColMajor>& output_mapper,
                                      const TensorContractionParams&, Index, Index, Index num_rows,
                                      Index num_cols) const {
    for (int i = 0; i < num_rows; ++i) {
      for (int j = 0; j < num_cols; ++j) {
        output_mapper(i, j) = std::sqrt(output_mapper(i, j));
      }
    }
  }
};

#endif  // EIGEN_TEST_CXX11_TENSOR_THREAD_POOL_HELPERS_H
