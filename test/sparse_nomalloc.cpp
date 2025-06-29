// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_NO_MALLOC

#ifndef EIGEN_SPARSE_TEST_INCLUDED_FROM_SPARSE_EXTRA
static long g_dense_op_sparse_count = 0;
#endif

#include "sparse.h"

template <typename SparseMatrixType>
void sparse_fixed_nnz_test(const SparseMatrixType&) {
  typedef typename SparseMatrixType::Scalar Scalar;

  Scalar eps = Scalar(1e-6);

  // maxNZ should not be Dynamic
  VERIFY(internal::traits<SparseMatrixType>::MaxNZ != Dynamic);

  // Create a sparse matrix with dimensions matching the reference
  SparseMatrixType m;

  // maximum number of non-zeros
  const int max_nz = internal::traits<SparseMatrixType>::MaxNZ;

  std::vector<Triplet<Scalar>> triplets(max_nz);
  // Generate non-duplicate triplets
  for (int i_count = 0; i_count < max_nz; ++i_count) {
    int i = (i_count * 2 + 1) % SparseMatrixType::RowsAtCompileTime;
    int j = (i_count * 2 + 1) / SparseMatrixType::RowsAtCompileTime;
    Scalar tmp = Scalar(i * 10 + j + 1 + 3 * i * j);
    triplets[i_count] = Triplet<Scalar>(i, j, tmp);
  }

  // Triplets must be sorted
  std::sort(triplets.begin(), triplets.end(), [&](const Triplet<Scalar>& a, const Triplet<Scalar>& b) {
    if (SparseMatrixType::IsRowMajor) {
      if (a.row() != b.row()) {
        return a.row() < b.row();
      }
      return a.col() < b.col();
    } else {
      if (a.col() != b.col()) {
        return a.col() < b.col();
      }
      return a.row() < b.row();
    }
  });
  m.assignFromSortedTriplets(triplets.begin(), triplets.end());

  typedef Matrix<Scalar, SparseMatrixType::RowsAtCompileTime, SparseMatrixType::ColsAtCompileTime,
                 SparseMatrixType::Options>
      DenseMatrixType;
  DenseMatrixType a;
  a.setZero();
  for (size_t i = 0; i < triplets.size(); ++i) {
    a(triplets[i].row(), triplets[i].col()) = triplets[i].value();
  }
  VERIFY_IS_MUCH_SMALLER_THAN((a - m).norm(), eps);
  VERIFY_IS_APPROX(m, a);
}

EIGEN_DECLARE_TEST(sparse_nomalloc) {
  g_dense_op_sparse_count = 0;  // Suppresses compiler warning
  for (int i = 0; i < g_repeat; i++) {
    // Square matrices, ColMajor,
    {
      CALL_SUBTEST_1((sparse_fixed_nnz_test(SparseMatrix<double, ColMajor, int, 4, 4, 4>())));
      CALL_SUBTEST_1((sparse_fixed_nnz_test(SparseMatrix<double, ColMajor, int, 5, 5, 4>())));
      CALL_SUBTEST_1((sparse_fixed_nnz_test(SparseMatrix<double, ColMajor, int, 10, 10, 37>())));
      CALL_SUBTEST_1((sparse_fixed_nnz_test(SparseMatrix<double, ColMajor, int, 20, 20, 123>())));
    }

    // Square matrices, RowMajor
    {
      CALL_SUBTEST_2((sparse_fixed_nnz_test(SparseMatrix<double, RowMajor, int, 3, 3, 4>())));
      CALL_SUBTEST_2((sparse_fixed_nnz_test(SparseMatrix<double, RowMajor, int, 4, 4, 4>())));
      CALL_SUBTEST_2((sparse_fixed_nnz_test(SparseMatrix<double, RowMajor, int, 10, 10, 4>())));
      CALL_SUBTEST_2((sparse_fixed_nnz_test(SparseMatrix<double, RowMajor, int, 20, 20, 13>())));
    }

    // complex numbers, 8 non-zeros
    {
      CALL_SUBTEST_3((sparse_fixed_nnz_test(SparseMatrix<std::complex<double>, RowMajor, int, 10, 10, 8>())));
      CALL_SUBTEST_3((sparse_fixed_nnz_test(SparseMatrix<std::complex<double>, RowMajor, int, 2, 16, 8>())));
      CALL_SUBTEST_3((sparse_fixed_nnz_test(SparseMatrix<std::complex<double>, RowMajor, int, 8, 3, 8>())));
      CALL_SUBTEST_3((sparse_fixed_nnz_test(SparseMatrix<std::complex<double>, RowMajor, int, 10, 2, 8>())));
      CALL_SUBTEST_3((sparse_fixed_nnz_test(SparseMatrix<std::complex<double>, ColMajor, int, 10, 10, 8>())));
      CALL_SUBTEST_3((sparse_fixed_nnz_test(SparseMatrix<std::complex<double>, ColMajor, int, 2, 16, 8>())));
      CALL_SUBTEST_3((sparse_fixed_nnz_test(SparseMatrix<std::complex<double>, ColMajor, int, 8, 3, 8>())));
      CALL_SUBTEST_3((sparse_fixed_nnz_test(SparseMatrix<std::complex<double>, ColMajor, int, 10, 2, 8>())));
    }

    // Test with non-square matrices, double
    {
      CALL_SUBTEST_4((sparse_fixed_nnz_test(SparseMatrix<double, RowMajor, int, 3, 7, 7>())));
      CALL_SUBTEST_4((sparse_fixed_nnz_test(SparseMatrix<double, RowMajor, int, 7, 9, 7>())));
      CALL_SUBTEST_4((sparse_fixed_nnz_test(SparseMatrix<double, RowMajor, int, 2, 8, 7>())));
      CALL_SUBTEST_4((sparse_fixed_nnz_test(SparseMatrix<double, RowMajor, int, 8, 3, 7>())));
      CALL_SUBTEST_4((sparse_fixed_nnz_test(SparseMatrix<double, RowMajor, int, 1, 20, 7>())));
      CALL_SUBTEST_4((sparse_fixed_nnz_test(SparseMatrix<double, ColMajor, int, 3, 7, 7>())));
      CALL_SUBTEST_4((sparse_fixed_nnz_test(SparseMatrix<double, ColMajor, int, 7, 9, 7>())));
      CALL_SUBTEST_4((sparse_fixed_nnz_test(SparseMatrix<double, ColMajor, int, 2, 8, 7>())));
      CALL_SUBTEST_4((sparse_fixed_nnz_test(SparseMatrix<double, ColMajor, int, 8, 3, 7>())));
      CALL_SUBTEST_4((sparse_fixed_nnz_test(SparseMatrix<double, ColMajor, int, 20, 1, 7>())));
    }

    // Test with non-square matrices, float
    {
      CALL_SUBTEST_5((sparse_fixed_nnz_test(SparseMatrix<float, RowMajor, int, 3, 7, 7>())));
      CALL_SUBTEST_5((sparse_fixed_nnz_test(SparseMatrix<float, RowMajor, int, 7, 9, 7>())));
      CALL_SUBTEST_5((sparse_fixed_nnz_test(SparseMatrix<float, RowMajor, int, 2, 8, 7>())));
      CALL_SUBTEST_5((sparse_fixed_nnz_test(SparseMatrix<float, RowMajor, int, 8, 3, 7>())));
      CALL_SUBTEST_5((sparse_fixed_nnz_test(SparseMatrix<float, RowMajor, int, 1, 20, 7>())));
      CALL_SUBTEST_5((sparse_fixed_nnz_test(SparseMatrix<float, ColMajor, int, 3, 7, 7>())));
      CALL_SUBTEST_5((sparse_fixed_nnz_test(SparseMatrix<float, ColMajor, int, 7, 9, 7>())));
      CALL_SUBTEST_5((sparse_fixed_nnz_test(SparseMatrix<float, ColMajor, int, 2, 8, 7>())));
      CALL_SUBTEST_5((sparse_fixed_nnz_test(SparseMatrix<float, ColMajor, int, 8, 3, 7>())));
      CALL_SUBTEST_5((sparse_fixed_nnz_test(SparseMatrix<float, ColMajor, int, 20, 1, 7>())));
    }
  }
}