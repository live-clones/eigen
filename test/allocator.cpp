// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <vector>
#include "main.h"

template <typename T, std::size_t N>
void test_simple_mem_resource(T&& alloc, const std::array<std::size_t, N>& sizes) {
  //  auto bytes_count_begin = alloc.bytes_allocated();
  for (int i = 0; i < 10; ++i) {
    auto* buf = alloc.allocate(100 * sizeof(int), sizes[i % N]);
    for (int j = 0; j < 100; ++j) {
      static_cast<int*>(buf)[j] = j;
      VERIFY_IS_EQUAL(static_cast<int*>(buf)[j], j);
    }
    alloc.deallocate(buf, 100 * sizeof(int), sizes[i % N]);
  }
  auto* buf = alloc.allocate(100 * sizeof(double));
  for (int j = 0; j < 100; ++j) {
    static_cast<double*>(buf)[j] = j;
    VERIFY_IS_EQUAL(static_cast<double*>(buf)[j], j);
  }
  alloc.deallocate(buf, 100 * sizeof(int));
}

template <typename T>
void test_simple_mono_buffer(T&& mono) {
  static constexpr std::array<std::size_t, 5> sizes{8, 64, 128, 32, 16};
  while (mono.blocks().size() < 5) {
    test_simple_mem_resource(mono, sizes);
  }
  mono.release();
  test_simple_mem_resource(mono, sizes);
}

EIGEN_DECLARE_TEST(default_memory_resource) {
  Eigen::memory_resource* default_resource = Eigen::get_default_resource();
  static constexpr std::array<std::size_t, 1> sizes{8};
  test_simple_mem_resource(*default_resource, sizes);

  Eigen::memory_resource* default_resource2 = Eigen::get_default_resource();
  test_simple_mem_resource(*default_resource2, sizes);
}

EIGEN_DECLARE_TEST(monotonic_buffer_resource_test) {
  Eigen::monotonic_buffer_resource mono_default;
  test_simple_mono_buffer(mono_default);

  auto* default_resource = Eigen::get_default_resource();

  Eigen::monotonic_buffer_resource mono_upstream(default_resource);
  test_simple_mono_buffer(mono_upstream);

  Eigen::monotonic_buffer_resource mono_size(2 << 8);
  test_simple_mono_buffer(mono_size);

  Eigen::monotonic_buffer_resource mono_size_upstream(1 << 8, &mono_size);
  test_simple_mono_buffer(mono_size_upstream);

  static constexpr std::size_t buffer_size = 2048;
  void* buffer[buffer_size];
  Eigen::monotonic_buffer_resource mono_buffer(buffer, buffer_size / 2);
  test_simple_mono_buffer(mono_buffer);

  Eigen::monotonic_buffer_resource mono_buffer_upstream(buffer, buffer_size, &mono_size_upstream);
  test_simple_mono_buffer(mono_buffer_upstream);
}

struct biggie {
  double r[10];
};

namespace Eigen {
namespace test {
template <typename T>
bool is_aligned(T* ptr, unsigned int bytes_aligned) {
  return (reinterpret_cast<uintptr_t>(ptr) % bytes_aligned) == 0U;
}
template <typename T>
void test_simple_alloc(T&& alloc) {
  float* float_ptr = alloc.template allocate<float>(100, 16);
  VERIFY(is_aligned(float_ptr, 16));
  for (int i = 0; i < 100; ++i) {
    float_ptr[i] = i;
    VERIFY_IS_EQUAL(float_ptr[i], static_cast<float>(i));
  }
  biggie* biggie_ptr = alloc.template allocate<biggie>(100, 8);
  VERIFY(is_aligned(biggie_ptr, 8));
  for (int i = 0; i < 100; ++i) {
    for (int j = 0; j < 10; ++j) {
      biggie_ptr[i].r[j] = j;
      VERIFY_IS_EQUAL(biggie_ptr[i].r[j], static_cast<double>(j));
    }
  }
  double* dbl_ptr = alloc.template allocate<double>(100, 32);
  VERIFY(is_aligned(dbl_ptr, 32));
  for (int i = 0; i < 100; ++i) {
    dbl_ptr[i] = i;
    VERIFY_IS_EQUAL(dbl_ptr[i], static_cast<double>(i));
  }
  int* int_ptr = alloc.template allocate<int>(100, 8);
  for (int i = 0; i < 100; ++i) {
    int_ptr[i] = i;
    VERIFY_IS_EQUAL(int_ptr[i], i);
  }
  long double* long_dbl_ptr = alloc.template allocate<long double>(100, 16);
  VERIFY(is_aligned(long_dbl_ptr, 16));
  for (int i = 0; i < 100; ++i) {
    long_dbl_ptr[i] = i;
    VERIFY_IS_EQUAL(long_dbl_ptr[i], static_cast<long double>(i));
  }
  alloc.template deallocate<int>(int_ptr, 100, 8);
  alloc.template deallocate<float>(float_ptr, 100, 16);
  alloc.template deallocate<long double>(long_dbl_ptr, 100, 16);
  alloc.template deallocate<double>(dbl_ptr, 100, 32);
  alloc.template deallocate<biggie>(biggie_ptr, 100, 8);
}

}  // namespace test
}  // namespace Eigen

EIGEN_DECLARE_TEST(polymorphic_allocator_test) {
  // Constructor tests
  using Eigen::test::test_simple_alloc;
  Eigen::polymorphic_allocator alloc_default;
  test_simple_alloc(alloc_default);
  Eigen::polymorphic_allocator alloc_copy(alloc_default);
  test_simple_alloc(alloc_copy);
  Eigen::monotonic_buffer_resource mono_default;
  Eigen::polymorphic_allocator alloc_memory_resource(&mono_default);
  test_simple_alloc(alloc_memory_resource);
}
