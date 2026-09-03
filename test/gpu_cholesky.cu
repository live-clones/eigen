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

// Check that Cholesky modules can be properly parsed by nvcc
#include <Eigen/Cholesky>

// Cholesky-family decompositions (LLT, LDLT, BunchKaufman) on symmetric
// positive-definite (and, for BunchKaufman, indefinite) matrices
template <typename MatrixType>
struct llt_reconstruct {
  EIGEN_DEVICE_FUNC void operator()(int i, const typename MatrixType::Scalar* in, typename MatrixType::Scalar* out) const {
    using namespace Eigen;
    const int offset = i * MatrixType::MaxSizeAtCompileTime;
    Map<const MatrixType> A_raw(in + offset);

    MatrixType A = A_raw * A_raw.transpose();
    A.diagonal().array() += typename MatrixType::Scalar(static_cast<typename NumTraits<typename MatrixType::Scalar>::Real>(MatrixType::RowsAtCompileTime));

    Map<MatrixType> C(out + offset);
    C = A.llt().reconstructedMatrix();
  }
};

template <typename MatrixType>
struct ldlt_reconstruct {
  EIGEN_DEVICE_FUNC void operator()(int i, const typename MatrixType::Scalar* in, typename MatrixType::Scalar* out) const {
    using namespace Eigen;
    const int offset = i * MatrixType::MaxSizeAtCompileTime;
    Map<const MatrixType> A_raw(in + offset);

    MatrixType A = A_raw * A_raw.transpose();
    A.diagonal().array() += typename MatrixType::Scalar(static_cast<typename NumTraits<typename MatrixType::Scalar>::Real>(MatrixType::RowsAtCompileTime));

    Map<MatrixType> C(out + offset);
    C = A.ldlt().reconstructedMatrix();
  }
};

template <typename MatrixType>
struct bunch_kaufman_reconstruct {
  EIGEN_DEVICE_FUNC void operator()(int i, const typename MatrixType::Scalar* in, typename MatrixType::Scalar* out) const {
    using namespace Eigen;
    const int offset = i * MatrixType::MaxSizeAtCompileTime;
    Map<const MatrixType> A_raw(in + offset);

    MatrixType A = A_raw + A_raw.transpose();

    Map<MatrixType> C(out + offset);
    C = A.bunchKaufman().reconstructedMatrix();
  }
};

template <typename MatrixType>
struct llt_solve {
  EIGEN_DEVICE_FUNC void operator()(int i, const typename MatrixType::Scalar* in, typename MatrixType::Scalar* out) const {
    using namespace Eigen;
    const int offset = i * MatrixType::MaxSizeAtCompileTime;
    Map<const MatrixType> A_raw(in + offset);

    MatrixType A = A_raw * A_raw.transpose();
    A.diagonal().array() += typename MatrixType::Scalar(MatrixType::RowsAtCompileTime);

    Map<MatrixType> C(out + offset);
    LLT<MatrixType> llt(A);
    C = llt.solve(MatrixType::Identity(A.rows(), A.cols()));
  }
};

EIGEN_DECLARE_TEST(gpu_cholesky) {
  ei_test_init_gpu();

  int nthreads = 100;
  Eigen::VectorXf in, out;
  Eigen::VectorXd din, dout;
  Eigen::Vector<Eigen::half, Eigen::Dynamic> hin, hout;

#if !defined(EIGEN_GPU_COMPILE_PHASE)
  int data_size = nthreads * 512;
  in.setRandom(data_size);
  out.setConstant(data_size, -1);
  din.setRandom(data_size);
  dout.setConstant(data_size, -1);
  hin.setRandom(data_size);
  hout.setConstant(data_size, static_cast<Eigen::half>(-1));
#endif

  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(run_and_compare_to_gpu(llt_reconstruct<Matrix2f>(), nthreads, in, out));
    CALL_SUBTEST_1(run_and_compare_to_gpu(llt_reconstruct<Matrix3f>(), nthreads, in, out));
    CALL_SUBTEST_1(run_and_compare_to_gpu(llt_reconstruct<Matrix4f>(), nthreads, in, out));
    CALL_SUBTEST_1((run_and_compare_to_gpu(llt_reconstruct<Matrix<float, 5, 5>>(), nthreads, in, out)));
    CALL_SUBTEST_1((run_and_compare_to_gpu(llt_reconstruct<Matrix<double, 5, 5>>(), nthreads, din, dout)));
    CALL_SUBTEST_1((run_and_compare_to_gpu(llt_reconstruct<Matrix<Eigen::half, 5, 5>>(), nthreads, hin, hout)));

    CALL_SUBTEST_2(run_and_compare_to_gpu(ldlt_reconstruct<Matrix2f>(), nthreads, in, out));
    CALL_SUBTEST_2(run_and_compare_to_gpu(ldlt_reconstruct<Matrix3f>(), nthreads, in, out));
    CALL_SUBTEST_2(run_and_compare_to_gpu(ldlt_reconstruct<Matrix4f>(), nthreads, in, out));
    CALL_SUBTEST_1((run_and_compare_to_gpu(ldlt_reconstruct<Matrix<float, 5, 5>>(), nthreads, in, out)));
    CALL_SUBTEST_1((run_and_compare_to_gpu(ldlt_reconstruct<Matrix<double, 5, 5>>(), nthreads, din, dout)));
    CALL_SUBTEST_1((run_and_compare_to_gpu(ldlt_reconstruct<Matrix<Eigen::half, 5, 5>>(), nthreads, hin, hout)));

    CALL_SUBTEST_3(run_and_compare_to_gpu(bunch_kaufman_reconstruct<Matrix2f>(), nthreads, in, out));
    CALL_SUBTEST_3(run_and_compare_to_gpu(bunch_kaufman_reconstruct<Matrix3f>(), nthreads, in, out));
    CALL_SUBTEST_3(run_and_compare_to_gpu(bunch_kaufman_reconstruct<Matrix4f>(), nthreads, in, out));
    CALL_SUBTEST_3((run_and_compare_to_gpu(bunch_kaufman_reconstruct<Matrix<float, 5, 5>>(), nthreads, in, out)));
    CALL_SUBTEST_1((run_and_compare_to_gpu(bunch_kaufman_reconstruct<Matrix<float, 5, 5>>(), nthreads, in, out)));
    CALL_SUBTEST_1((run_and_compare_to_gpu(bunch_kaufman_reconstruct<Matrix<double, 5, 5>>(), nthreads, din, dout)));
    CALL_SUBTEST_1((run_and_compare_to_gpu(bunch_kaufman_reconstruct<Matrix<Eigen::half, 5, 5>>(), nthreads, hin, hout)));

    CALL_SUBTEST_4(run_and_compare_to_gpu(llt_solve<Matrix2f>(), nthreads, in, out));
    CALL_SUBTEST_4(run_and_compare_to_gpu(llt_solve<Matrix3f>(), nthreads, in, out));
    CALL_SUBTEST_4(run_and_compare_to_gpu(llt_solve<Matrix4f>(), nthreads, in, out));
    CALL_SUBTEST_1((run_and_compare_to_gpu(llt_solve<Matrix<float, 5, 5>>(), nthreads, in, out)));
    CALL_SUBTEST_1((run_and_compare_to_gpu(llt_solve<Matrix<double, 5, 5>>(), nthreads, din, dout)));
    CALL_SUBTEST_1((run_and_compare_to_gpu(llt_solve<Matrix<Eigen::half, 5, 5>>(), nthreads, hin, hout)));
  }
}
