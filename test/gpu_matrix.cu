// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 The Eigen Team.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#define EIGEN_USE_GPU
#include "main.h"
#include "gpu_common.h"

// Basic coefficient-wise / elementwise matrix kernels
template <typename MatrixType>
struct matrix_square {
  EIGEN_DEVICE_FUNC void operator()(int i, const typename MatrixType::Scalar* in, typename MatrixType::Scalar* out) const {
    using namespace Eigen;
    const int offset = i * MatrixType::MaxSizeAtCompileTime;
    Map<const MatrixType> A(in + offset);
    Map<MatrixType> C(out + offset);
    C = A.array().square().matrix();
  }
};

template <typename MatrixType>
struct matrix_cwise_product {
  EIGEN_DEVICE_FUNC void operator()(int i, const typename MatrixType::Scalar* in, typename MatrixType::Scalar* out) const {
    using namespace Eigen;
    const int offset = i * 2 * MatrixType::MaxSizeAtCompileTime;
    Map<const MatrixType> A(in + offset);
    Map<const MatrixType> B(in + offset + MatrixType::MaxSizeAtCompileTime);
    Map<MatrixType> C(out + i * MatrixType::MaxSizeAtCompileTime);
    C = A.cwiseProduct(B);
  }
};

template <typename MatrixType>
struct matrix_transpose {
  EIGEN_DEVICE_FUNC void operator()(int i, const typename MatrixType::Scalar* in, typename MatrixType::Scalar* out) const {
    using namespace Eigen;
    using TransposeType = Matrix<typename MatrixType::Scalar, MatrixType::ColsAtCompileTime, MatrixType::RowsAtCompileTime>;
    const int in_offset = i * MatrixType::MaxSizeAtCompileTime;
    const int out_offset = i * TransposeType::MaxSizeAtCompileTime;
    Map<const MatrixType> A(in + in_offset);
    Map<TransposeType> C(out + out_offset);
    C = A.transpose();
  }
};

template <typename MatrixType>
struct matrix_inverse {
  EIGEN_DEVICE_FUNC void operator()(int i, const typename MatrixType::Scalar* in, typename MatrixType::Scalar* out) const {
    using namespace Eigen;
    const int offset = i * MatrixType::MaxSizeAtCompileTime;
    Map<const MatrixType> A(in + offset);
    Map<MatrixType> C(out + offset);
    C = A.inverse();
  }
};

EIGEN_DECLARE_TEST(gpu_matrix) {
  ei_test_init_gpu();

  int nthreads = 100;
  Eigen::VectorXf in, out;

#if !defined(EIGEN_GPU_COMPILE_PHASE)
  int data_size = nthreads * 512;
  in.setRandom(data_size);
  out.setConstant(data_size, -1);
#endif

  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(run_and_compare_to_gpu(matrix_square<Matrix2f>(), nthreads, in, out));
    CALL_SUBTEST_1(run_and_compare_to_gpu(matrix_square<Matrix3f>(), nthreads, in, out));
    CALL_SUBTEST_1(run_and_compare_to_gpu(matrix_square<Matrix4f>(), nthreads, in, out));
    CALL_SUBTEST_1((run_and_compare_to_gpu(matrix_square<Matrix<float, 5, 5>>(), nthreads, in, out)));

    CALL_SUBTEST_2(run_and_compare_to_gpu(matrix_cwise_product<Matrix2f>(), nthreads, in, out));
    CALL_SUBTEST_2(run_and_compare_to_gpu(matrix_cwise_product<Matrix3f>(), nthreads, in, out));
    CALL_SUBTEST_2(run_and_compare_to_gpu(matrix_cwise_product<Matrix4f>(), nthreads, in, out));
    CALL_SUBTEST_2((run_and_compare_to_gpu(matrix_cwise_product<Matrix<float, 5, 5>>(), nthreads, in, out)));

    CALL_SUBTEST_3(run_and_compare_to_gpu(matrix_transpose<Matrix2f>(), nthreads, in, out));
    CALL_SUBTEST_3(run_and_compare_to_gpu(matrix_transpose<Matrix3f>(), nthreads, in, out));
    CALL_SUBTEST_3(run_and_compare_to_gpu(matrix_transpose<Matrix4f>(), nthreads, in, out));
    CALL_SUBTEST_3((run_and_compare_to_gpu(matrix_transpose<Matrix<float, 5, 5>>(), nthreads, in, out)));
    CALL_SUBTEST_3((run_and_compare_to_gpu(matrix_transpose<Matrix<float, 3, 5>>(), nthreads, in, out)));

    CALL_SUBTEST_4(run_and_compare_to_gpu(matrix_inverse<Matrix2f>(), nthreads, in, out));
    CALL_SUBTEST_4(run_and_compare_to_gpu(matrix_inverse<Matrix3f>(), nthreads, in, out));
    CALL_SUBTEST_4(run_and_compare_to_gpu(matrix_inverse<Matrix4f>(), nthreads, in, out));
    CALL_SUBTEST_4((run_and_compare_to_gpu(matrix_inverse<Matrix<float, 5, 5>>(), nthreads, in, out)));
  }
}
