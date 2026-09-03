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

// Check that LU modules can be properly parsed by nvcc
#include <Eigen/LU>

// LU-family decompositions (PartialPivLU, FullPivLU) and matrix inversion
// on general (non-symmetric) invertible matrices
template <typename MatrixType>
struct partial_piv_lu_kernel {
  EIGEN_DEVICE_FUNC void operator()(int i, const typename MatrixType::Scalar* in, typename MatrixType::Scalar* out) const {
    using namespace Eigen;
    const int offset = i * MatrixType::MaxSizeAtCompileTime;
    Map<const MatrixType> A(in + offset);

    Map<MatrixType> C(out + offset);
    C = A.partialPivLu().reconstructedMatrix();
  }
};

template <typename MatrixType>
struct full_piv_lu_kernel {
  EIGEN_DEVICE_FUNC void operator()(int i, const typename MatrixType::Scalar* in, typename MatrixType::Scalar* out) const {
    using namespace Eigen;
    const int offset = i * MatrixType::MaxSizeAtCompileTime;
    Map<const MatrixType> A(in + offset);

    Map<MatrixType> C(out + offset);
    C = A.fullPivLu().reconstructedMatrix();
  }
};

template <typename MatrixType>
struct partial_piv_lu_solve_kernel {
  EIGEN_DEVICE_FUNC void operator()(int i, const typename MatrixType::Scalar* in, typename MatrixType::Scalar* out) const {
    using namespace Eigen;
    const int offset = i * MatrixType::MaxSizeAtCompileTime;
    Map<const MatrixType> A(in + offset);

    Map<MatrixType> C(out + offset);
    PartialPivLU<MatrixType> lu(A);
    C = lu.solve(MatrixType::Identity(A.rows(), A.cols()));
  }
};

template <typename MatrixType>
struct matrix_inverse_via_lu_kernel {
  EIGEN_DEVICE_FUNC void operator()(int i, const typename MatrixType::Scalar* in, typename MatrixType::Scalar* out) const {
    using namespace Eigen;
    const int offset = i * MatrixType::MaxSizeAtCompileTime;
    Map<const MatrixType> A(in + offset);

    Map<MatrixType> C(out + offset);
    C = A.inverse();
  }
};

template <typename MatrixType, typename VectorType>
void generate_invertible_inputs(int nthreads, VectorType& in) {
#if !defined(EIGEN_GPU_COMPILE_PHASE)
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;
  for (int i = 0; i < nthreads; ++i) {
    Eigen::Map<MatrixType> A(in.data() + i * MatrixType::MaxSizeAtCompileTime);
    A.setRandom();
    RealScalar shift = RealScalar(2) * RealScalar(MatrixType::RowsAtCompileTime);
    for (Index j = 0; j < A.rows(); ++j) {
      using std::real;
      A.coeffRef(j, j) += Scalar(shift);
    }
  }
#else
  EIGEN_UNUSED_VARIABLE(nthreads);
  EIGEN_UNUSED_VARIABLE(in);
#endif
}

EIGEN_DECLARE_TEST(gpu_lu) {
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
  hout.setConstant(data_size, Eigen::half(-1));
#endif

  for (int i = 0; i < g_repeat; i++) {
    generate_invertible_inputs<Matrix2f>(nthreads, in);
    CALL_SUBTEST_1(run_and_compare_to_gpu(partial_piv_lu_kernel<Matrix2f>(), nthreads, in, out));
    generate_invertible_inputs<Matrix3f>(nthreads, in);
    CALL_SUBTEST_1(run_and_compare_to_gpu(partial_piv_lu_kernel<Matrix3f>(), nthreads, in, out));
    generate_invertible_inputs<Matrix4f>(nthreads, in);
    CALL_SUBTEST_1(run_and_compare_to_gpu(partial_piv_lu_kernel<Matrix4f>(), nthreads, in, out));
    generate_invertible_inputs<Matrix<float, 5, 5>>(nthreads, in);
    CALL_SUBTEST_1((run_and_compare_to_gpu(partial_piv_lu_kernel<Matrix<float, 5, 5>>(), nthreads, in, out)));
    generate_invertible_inputs<Matrix<double, 5, 5>>(nthreads, din);
    CALL_SUBTEST_1((run_and_compare_to_gpu(partial_piv_lu_kernel<Matrix<double, 5, 5>>(), nthreads, din, dout)));
    generate_invertible_inputs<Matrix<Eigen::half, 5, 5>>(nthreads, hin);
    CALL_SUBTEST_1((run_and_compare_to_gpu(partial_piv_lu_kernel<Matrix<Eigen::half, 5, 5>>(), nthreads, hin, hout)));

    generate_invertible_inputs<Matrix2f>(nthreads, in);
    CALL_SUBTEST_2(run_and_compare_to_gpu(full_piv_lu_kernel<Matrix2f>(), nthreads, in, out));
    generate_invertible_inputs<Matrix3f>(nthreads, in);
    CALL_SUBTEST_2(run_and_compare_to_gpu(full_piv_lu_kernel<Matrix3f>(), nthreads, in, out));
    generate_invertible_inputs<Matrix4f>(nthreads, in);
    CALL_SUBTEST_2(run_and_compare_to_gpu(full_piv_lu_kernel<Matrix4f>(), nthreads, in, out));
    generate_invertible_inputs<Matrix<float, 5, 5>>(nthreads, in);
    CALL_SUBTEST_2((run_and_compare_to_gpu(full_piv_lu_kernel<Matrix<float, 5, 5>>(), nthreads, in, out)));
    generate_invertible_inputs<Matrix<double, 5, 5>>(nthreads, din);
    CALL_SUBTEST_2((run_and_compare_to_gpu(full_piv_lu_kernel<Matrix<double, 5, 5>>(), nthreads, din, dout)));
    generate_invertible_inputs<Matrix<Eigen::half, 5, 5>>(nthreads, hin);
    CALL_SUBTEST_2((run_and_compare_to_gpu(full_piv_lu_kernel<Matrix<Eigen::half, 5, 5>>(), nthreads, hin, hout)));

    generate_invertible_inputs<Matrix2f>(nthreads, in);
    CALL_SUBTEST_3(run_and_compare_to_gpu(partial_piv_lu_solve_kernel<Matrix2f>(), nthreads, in, out));
    generate_invertible_inputs<Matrix3f>(nthreads, in);
    CALL_SUBTEST_3(run_and_compare_to_gpu(partial_piv_lu_solve_kernel<Matrix3f>(), nthreads, in, out));
    generate_invertible_inputs<Matrix4f>(nthreads, in);
    CALL_SUBTEST_3(run_and_compare_to_gpu(partial_piv_lu_solve_kernel<Matrix4f>(), nthreads, in, out));
    generate_invertible_inputs<Matrix<float, 5, 5>>(nthreads, in);
    CALL_SUBTEST_3((run_and_compare_to_gpu(partial_piv_lu_solve_kernel<Matrix<float, 5, 5>>(), nthreads, in, out)));
    generate_invertible_inputs<Matrix<double, 5, 5>>(nthreads, din);
    CALL_SUBTEST_3((run_and_compare_to_gpu(partial_piv_lu_solve_kernel<Matrix<double, 5, 5>>(), nthreads, din, dout)));
    generate_invertible_inputs<Matrix<Eigen::half, 5, 5>>(nthreads, hin);
    CALL_SUBTEST_3((run_and_compare_to_gpu(partial_piv_lu_solve_kernel<Matrix<Eigen::half, 5, 5>>(), nthreads, hin, hout)));

    generate_invertible_inputs<Matrix2f>(nthreads, in);
    CALL_SUBTEST_4(run_and_compare_to_gpu(matrix_inverse_via_lu_kernel<Matrix2f>(), nthreads, in, out));
    generate_invertible_inputs<Matrix3f>(nthreads, in);
    CALL_SUBTEST_4(run_and_compare_to_gpu(matrix_inverse_via_lu_kernel<Matrix3f>(), nthreads, in, out));
    generate_invertible_inputs<Matrix4f>(nthreads, in);
    CALL_SUBTEST_4(run_and_compare_to_gpu(matrix_inverse_via_lu_kernel<Matrix4f>(), nthreads, in, out));
    generate_invertible_inputs<Matrix<float, 5, 5>>(nthreads, in);
    CALL_SUBTEST_4((run_and_compare_to_gpu(matrix_inverse_via_lu_kernel<Matrix<float, 5, 5>>(), nthreads, in, out)));
    generate_invertible_inputs<Matrix<double, 5, 5>>(nthreads, din);
    CALL_SUBTEST_4((run_and_compare_to_gpu(matrix_inverse_via_lu_kernel<Matrix<double, 5, 5>>(), nthreads, din, dout)));
    generate_invertible_inputs<Matrix<Eigen::half, 5, 5>>(nthreads, hin);
    CALL_SUBTEST_4((run_and_compare_to_gpu(matrix_inverse_via_lu_kernel<Matrix<Eigen::half, 5, 5>>(), nthreads, hin, hout)));
  }
}
