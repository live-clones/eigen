// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Apple AMX GEMM tests. On Apple arm64 NEON builds the opt-in below turns the AMX path on; it runs where the chip is
// a known M2 or later and the products fall back to NEON elsewhere, where these tests check that path instead.
#ifndef EIGEN_ARM64_USE_APPLE_AMX
#define EIGEN_ARM64_USE_APPLE_AMX
#endif
#define EIGEN_GEMM_THREADPOOL
#include "main.h"

#if defined(__APPLE__) && defined(__aarch64__) && defined(EIGEN_VECTORIZE_NEON) && !defined(EIGEN_USE_BLAS) && \
    !defined(EIGEN_GEMM_APPLE_AMX)
#error "EIGEN_ARM64_USE_APPLE_AMX did not enable the Apple AMX GEMM in an Apple arm64 NEON build."
#endif

template <typename Scalar, int LhsOrder, int RhsOrder, int ResOrder>
void check_product(Index rows, Index cols, Index depth, Scalar alpha) {
  using Lhs = Matrix<Scalar, Dynamic, Dynamic, LhsOrder>;
  using Rhs = Matrix<Scalar, Dynamic, Dynamic, RhsOrder>;
  using Res = Matrix<Scalar, Dynamic, Dynamic, ResOrder>;
  const Lhs a = Lhs::Random(rows, depth);
  const Rhs b = Rhs::Random(depth, cols);
  Res c = Res::Random(rows, cols);
  Res ref = c;
  ref.noalias() += alpha * a.lazyProduct(b);
  c.noalias() += alpha * a * b;
  VERIFY_IS_APPROX(c, ref);
  c.noalias() = a * b;
  ref = a.lazyProduct(b);
  VERIFY_IS_APPROX(c, ref);
}

template <typename Scalar>
void check_orders(Index rows, Index cols, Index depth, Scalar alpha) {
  check_product<Scalar, ColMajor, ColMajor, ColMajor>(rows, cols, depth, alpha);
  check_product<Scalar, RowMajor, ColMajor, ColMajor>(rows, cols, depth, alpha);
  check_product<Scalar, ColMajor, RowMajor, ColMajor>(rows, cols, depth, alpha);
  check_product<Scalar, RowMajor, RowMajor, ColMajor>(rows, cols, depth, alpha);
  check_product<Scalar, ColMajor, ColMajor, RowMajor>(rows, cols, depth, alpha);
  check_product<Scalar, RowMajor, RowMajor, RowMajor>(rows, cols, depth, alpha);
}

// Full blocks, partial blocks on each side, depths that are not a multiple of 4, and results just past the crossover.
template <typename Scalar>
void test_shapes() {
  const Index shapes[][3] = {{32, 32, 32},  {64, 96, 128}, {31, 33, 17}, {33, 31, 101}, {16, 48, 7},
                             {100, 70, 3}, {257, 129, 64}, {16, 16, 128}, {40, 300, 9}, {300, 40, 9}};
  for (const auto& s : shapes) {
    check_orders<Scalar>(s[0], s[1], s[2], Scalar(1));
    check_orders<Scalar>(s[0], s[1], s[2], Scalar(-0.5));
  }
  // A long depth: several depth blocks under the default blocking.
  check_orders<Scalar>(40, 70, Index(9000 / sizeof(Scalar)), Scalar(2));
}

// Operands and results inside larger matrices: strides that are not multiples of a block, unaligned starts.
template <typename Scalar>
void test_strided() {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  const Mat big_a = Mat::Random(203, 180), big_b = Mat::Random(170, 150);
  Mat big_c = Mat::Random(210, 160);
  Mat ref = big_c;
  const auto a = big_a.block(1, 3, 97, 131);
  const auto b = big_b.block(5, 1, 131, 67);
  ref.block(3, 2, 97, 67).noalias() -= a.lazyProduct(b);
  big_c.block(3, 2, 97, 67).noalias() -= a * b;
  VERIFY_IS_APPROX(big_c, ref);
  ref.block(1, 1, 67, 97).noalias() += Scalar(3) * b.transpose().lazyProduct(a.transpose());
  big_c.block(1, 1, 67, 97).noalias() += Scalar(3) * b.transpose() * a.transpose();
  VERIFY_IS_APPROX(big_c, ref);
  // A result with an inner stride keeps the generic kernel.
  Mat storage = Mat::Random(2 * 97, 67);
  Map<Mat, 0, Stride<Dynamic, 2>> c(storage.data(), 97, 67, Stride<Dynamic, 2>(2 * 97, 2));
  Mat c_ref = c;
  c_ref.noalias() += a.lazyProduct(b);
  c.noalias() += a * b;
  VERIFY_IS_APPROX(Mat(c), c_ref);
}

// The driver under small blocks, so that every loop level of the blocking takes several iterations.
template <typename Scalar>
void test_blocking() {
  if (!internal::apple_amx::usable()) return;
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using RowMat = Matrix<Scalar, Dynamic, Dynamic, RowMajor>;
  constexpr Index kMr = internal::apple_amx::block<Scalar>::mr, kNr = internal::apple_amx::block<Scalar>::nr;
  const Index M = 3 * kMr + 5, N = 4 * kNr + 3, K = 77;
  for (int orders = 0; orders < 4; ++orders) {
    const bool a_rows = orders & 1, b_rows = orders & 2;
    const RowMat a_row = RowMat::Random(M, K), b_row = RowMat::Random(K, N);
    const Mat a_col = a_row, b_col = b_row;
    RowMat c = RowMat::Random(M, N);
    RowMat ref = c;
    ref.noalias() += Scalar(-2) * a_row.lazyProduct(b_row);
    internal::apple_amx::problem<Scalar> p;
    p.M = M;
    p.N = N;
    p.K = K;
    p.alpha = Scalar(-2);
    p.A = a_rows ? a_row.data() : a_col.data();
    p.lda = a_rows ? K : M;
    p.a_row_major = a_rows;
    p.B = b_rows ? b_row.data() : b_col.data();
    p.ldb = b_rows ? N : K;
    p.b_row_major = b_rows;
    p.C = c.data();
    p.ldc = N;
    internal::apple_amx::run(p, 2 * kMr, 2 * kNr, Index(20));
    VERIFY_IS_APPROX(c, ref);
  }
}

// The AMX path runs wherever the chip supports it, and only there.
template <typename Scalar>
void test_dispatch() {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  const Mat a = Mat::Random(64, 64), b = Mat::Random(64, 64);
  Mat c = Mat::Zero(64, 64);
  const bool ran = internal::apple_amx_gemm<ColMajor, ColMajor>(Index(64), Index(64), Index(64), a.data(), Index(64),
                                                                b.data(), Index(64), c.data(), Index(1), Index(64),
                                                                Scalar(1));
  VERIFY_IS_EQUAL(ran, internal::apple_amx::usable());
  if (ran) VERIFY_IS_APPROX(c, Mat(a.lazyProduct(b)));
  VERIFY_IS_EQUAL(internal::apple_amx::units() > 0, internal::apple_amx::usable());
  // Below the crossover the NEON kernel runs.
  VERIFY(!(internal::apple_amx_gemm<ColMajor, ColMajor>(Index(8), Index(8), Index(8), a.data(), Index(64), b.data(),
                                                         Index(64), c.data(), Index(1), Index(64), Scalar(1))));
}

// Products split over the AMX units against the same products on one thread.
template <typename Scalar>
void test_threaded() {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  static ThreadPool pool(4);
  Eigen::setGemmThreadPool(&pool);
  const Index shapes[][3] = {{512, 512, 512}, {1000, 90, 700}, {90, 1000, 700}, {2048, 64, 256}};
  for (const auto& s : shapes) {
    const Mat a = Mat::Random(s[0], s[2]), b = Mat::Random(s[2], s[1]);
    Mat serial(s[0], s[1]), threaded(s[0], s[1]);
    const int saved = Eigen::nbThreads();
    Eigen::setNbThreads(1);
    serial.noalias() = a * b;
    Eigen::setNbThreads(0);
    threaded.noalias() = a * b;
    Eigen::setNbThreads(saved > 0 ? saved : 0);
    VERIFY_IS_APPROX(threaded, serial);
  }
}

EIGEN_DECLARE_TEST(product_apple_amx) {
#ifdef EIGEN_GEMM_APPLE_AMX
  CALL_SUBTEST_1(test_dispatch<float>());
  CALL_SUBTEST_1(test_dispatch<double>());
  CALL_SUBTEST_2(test_blocking<float>());
  CALL_SUBTEST_2(test_blocking<double>());
#endif
  for (int i = 0; i < g_repeat; ++i) {
    CALL_SUBTEST_3(test_shapes<float>());
    CALL_SUBTEST_4(test_shapes<double>());
    CALL_SUBTEST_5(test_strided<float>());
    CALL_SUBTEST_5(test_strided<double>());
  }
  CALL_SUBTEST_6(test_threaded<float>());
  CALL_SUBTEST_6(test_threaded<double>());
}
