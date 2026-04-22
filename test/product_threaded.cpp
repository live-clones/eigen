// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2023 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_GEMM_THREADPOOL
#include "main.h"

void test_parallelize_gemm() {
  constexpr int n = 1024;
  constexpr int num_threads = 4;
  MatrixXf a = MatrixXf::Random(n, n);
  MatrixXf b = MatrixXf::Random(n, n);
  MatrixXf c = MatrixXf::Random(n, n);
  c.noalias() = a * b;

  ThreadPool pool(num_threads);
  Eigen::setGemmThreadPool(&pool);
  MatrixXf c_threaded(n, n);
  c_threaded.noalias() = a * b;

  VERIFY_IS_APPROX(c, c_threaded);
  Eigen::setGemmThreadPool(nullptr);
}

void test_parallelize_gemm_varied() {
  constexpr int num_threads = 4;
  ThreadPool pool(num_threads);

  // Non-square float
  {
    MatrixXf a = MatrixXf::Random(512, 2048);
    MatrixXf b = MatrixXf::Random(2048, 256);
    MatrixXf c_serial(512, 256);
    c_serial.noalias() = a * b;
    Eigen::setGemmThreadPool(&pool);
    MatrixXf c_threaded(512, 256);
    c_threaded.noalias() = a * b;
    Eigen::setGemmThreadPool(nullptr);
    VERIFY_IS_APPROX(c_serial, c_threaded);
  }

  // Double
  {
    MatrixXd a = MatrixXd::Random(512, 512);
    MatrixXd b = MatrixXd::Random(512, 512);
    MatrixXd c_serial(512, 512);
    c_serial.noalias() = a * b;
    Eigen::setGemmThreadPool(&pool);
    MatrixXd c_threaded(512, 512);
    c_threaded.noalias() = a * b;
    Eigen::setGemmThreadPool(nullptr);
    VERIFY_IS_APPROX(c_serial, c_threaded);
  }

  // Complex double
  {
    MatrixXcd a = MatrixXcd::Random(256, 256);
    MatrixXcd b = MatrixXcd::Random(256, 256);
    MatrixXcd c_serial(256, 256);
    c_serial.noalias() = a * b;
    Eigen::setGemmThreadPool(&pool);
    MatrixXcd c_threaded(256, 256);
    c_threaded.noalias() = a * b;
    Eigen::setGemmThreadPool(nullptr);
    VERIFY_IS_APPROX(c_serial, c_threaded);
  }
}

// Exercises the 2-D partition (t_m > 1). Tall-skinny / short-fat / batched-small
// shapes previously had cols < nr * threads (or rows < mr * threads in the
// transposed-primary-dim case) and fell back to single-threaded execution.
template <typename Mat>
void verify_threaded_matches_serial(const Mat& a, const Mat& b, Eigen::ThreadPool& pool) {
  Mat c_serial(a.rows(), b.cols());
  c_serial.noalias() = a * b;
  Eigen::setGemmThreadPool(&pool);
  Mat c_threaded(a.rows(), b.cols());
  c_threaded.noalias() = a * b;
  Eigen::setGemmThreadPool(nullptr);
  VERIFY_IS_APPROX(c_serial, c_threaded);
}

void test_parallelize_gemm_tall_skinny() {
  ThreadPool pool(4);

  // Tall-skinny: M >> N, N * depth small enough that 1-D-on-N would saturate
  // at very few threads. These are the shapes that most benefit from 2-D.
  verify_threaded_matches_serial<MatrixXf>(MatrixXf::Random(8192, 8), MatrixXf::Random(8, 8), pool);
  verify_threaded_matches_serial<MatrixXf>(MatrixXf::Random(8192, 16), MatrixXf::Random(16, 16), pool);
  verify_threaded_matches_serial<MatrixXf>(MatrixXf::Random(4096, 4), MatrixXf::Random(4, 4), pool);
  verify_threaded_matches_serial<MatrixXd>(MatrixXd::Random(4096, 8), MatrixXd::Random(8, 8), pool);

  // Short-fat: transposed primary dim. Triggers the same 2-D logic after the
  // internal row/col swap.
  verify_threaded_matches_serial<MatrixXf>(MatrixXf::Random(8, 8192), MatrixXf::Random(8192, 8192), pool);
  verify_threaded_matches_serial<MatrixXd>(MatrixXd::Random(8, 4096), MatrixXd::Random(4096, 4096), pool);

  // Medium shapes that stay 1-D (regression guard).
  verify_threaded_matches_serial<MatrixXf>(MatrixXf::Random(1024, 1024), MatrixXf::Random(1024, 1024), pool);
  verify_threaded_matches_serial<MatrixXd>(MatrixXd::Random(512, 512), MatrixXd::Random(512, 512), pool);
}

// Sanity-check that varying the pool size (including sizes that cause the
// cost model to drop below actual_threads) all produce the same result as
// serial execution.
void test_parallelize_gemm_thread_counts() {
  MatrixXf a = MatrixXf::Random(2048, 64);
  MatrixXf b = MatrixXf::Random(64, 64);
  MatrixXf c_serial(2048, 64);
  c_serial.noalias() = a * b;

  for (int nt : {1, 2, 3, 4, 8}) {
    ThreadPool pool(nt);
    Eigen::setGemmThreadPool(&pool);
    MatrixXf c_threaded(2048, 64);
    c_threaded.noalias() = a * b;
    Eigen::setGemmThreadPool(nullptr);
    VERIFY_IS_APPROX(c_serial, c_threaded);
  }
}

EIGEN_DECLARE_TEST(product_threaded) {
  CALL_SUBTEST_1(test_parallelize_gemm());
  CALL_SUBTEST_2(test_parallelize_gemm_varied());
  CALL_SUBTEST_3(test_parallelize_gemm_tall_skinny());
  CALL_SUBTEST_4(test_parallelize_gemm_thread_counts());
}
