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

template <typename SparseVectorType>
void sparsevector_fixed_nnz_test(const SparseVectorType&) {
  typedef typename SparseVectorType::Scalar Scalar;

  Scalar eps = Scalar(1e-6);

  // maxNZ should not be Dynamic
  VERIFY(internal::traits<SparseVectorType>::MaxNZ != Dynamic);

  // Create a sparse matrix with dimensions matching the reference
  SparseVectorType vec;
  using StorageIndex = typename SparseVectorType::StorageIndex;

  // maximum number of non-zeros
  const int max_nz = internal::traits<SparseVectorType>::MaxNZ;

  std::vector<std::pair<StorageIndex, Scalar>> pairs(max_nz);
  // Generate non-duplicate triplets
  for (int i_count = 0; i_count < max_nz; ++i_count) {
    int i = (i_count * 2 + 1);
    Scalar tmp = Scalar(i * 10 + 1);
    pairs[i_count] = std::make_pair(i, tmp);
  }
  // Pairs must be sorted
  vec.setFromSortedPairs(pairs.begin(), pairs.end());

  typedef Matrix<Scalar, SparseVectorType::RowsAtCompileTime, SparseVectorType::ColsAtCompileTime,
                 SparseVectorType::Options>
      DenseMatrixType;
  DenseMatrixType vec_full;

  const Scalar* values = vec.valuePtr();
  const StorageIndex* indices = vec.innerIndexPtr();

  vec_full.setZero();
  for (size_t i = 0; i < max_nz; ++i) {
    vec_full(indices[i]) = values[i];
  }
  VERIFY_IS_MUCH_SMALLER_THAN((vec - vec_full).norm(), eps);
  VERIFY_IS_APPROX(vec, vec_full);
}

EIGEN_DECLARE_TEST(sparse_vector_nomalloc) {
  g_dense_op_sparse_count = 0;  // Suppresses compiler warning
  for (int i = 0; i < g_repeat; i++) {
    //  ColMajor,
    {
      CALL_SUBTEST_1((sparsevector_fixed_nnz_test(SparseVector<double, ColMajor, int, 3, 1, 1>())));
      CALL_SUBTEST_1((sparsevector_fixed_nnz_test(SparseVector<double, ColMajor, int, 40, 1, 4>())));
      CALL_SUBTEST_1((sparsevector_fixed_nnz_test(SparseVector<double, ColMajor, int, 50, 1, 4>())));
      CALL_SUBTEST_1((sparsevector_fixed_nnz_test(SparseVector<double, ColMajor, int, 100, 1, 37>())));
      CALL_SUBTEST_1((sparsevector_fixed_nnz_test(SparseVector<double, ColMajor, int, 300, 1, 123>())));
    }

    // RowMajor
    {
      CALL_SUBTEST_2((sparsevector_fixed_nnz_test(SparseVector<double, RowMajor, int, 1, 30, 4>())));
      CALL_SUBTEST_2((sparsevector_fixed_nnz_test(SparseVector<double, RowMajor, int, 1, 40, 4>())));
      CALL_SUBTEST_2((sparsevector_fixed_nnz_test(SparseVector<double, RowMajor, int, 1, 100, 4>())));
      CALL_SUBTEST_2((sparsevector_fixed_nnz_test(SparseVector<double, RowMajor, int, 1, 300, 13>())));
    }

    // Complex values
    {
      CALL_SUBTEST_3((sparsevector_fixed_nnz_test(SparseVector<std::complex<double>, RowMajor, int, 1, 21, 8>())));
      CALL_SUBTEST_3((sparsevector_fixed_nnz_test(SparseVector<std::complex<double>, RowMajor, int, 1, 22, 8>())));
      CALL_SUBTEST_3((sparsevector_fixed_nnz_test(SparseVector<std::complex<double>, RowMajor, int, 1, 60, 8>())));
      CALL_SUBTEST_3((sparsevector_fixed_nnz_test(SparseVector<std::complex<double>, RowMajor, int, 1, 41, 8>())));
      CALL_SUBTEST_3((sparsevector_fixed_nnz_test(SparseVector<std::complex<double>, ColMajor, int, 21, 1, 8>())));
      CALL_SUBTEST_3((sparsevector_fixed_nnz_test(SparseVector<std::complex<double>, ColMajor, int, 22, 1, 8>())));
      CALL_SUBTEST_3((sparsevector_fixed_nnz_test(SparseVector<std::complex<double>, ColMajor, int, 60, 1, 8>())));
      CALL_SUBTEST_3((sparsevector_fixed_nnz_test(SparseVector<std::complex<double>, ColMajor, int, 41, 1, 8>())));
    }
  }
}