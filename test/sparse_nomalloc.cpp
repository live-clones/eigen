// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Verifies that fixed-size SparseMatrix operations do not allocate.
#define EIGEN_RUNTIME_NO_MALLOC

#ifndef EIGEN_SPARSE_TEST_INCLUDED_FROM_SPARSE_EXTRA
static long g_dense_op_sparse_count = 0;
#endif

#include "sparse.h"

template <typename Scalar>
struct triplet_sort {
  bool is_row_major;
  bool operator()(const Triplet<Scalar>& a, const Triplet<Scalar>& b) const {
    if (is_row_major) return (a.row() != b.row()) ? a.row() < b.row() : a.col() < b.col();
    return (a.col() != b.col()) ? a.col() < b.col() : a.row() < b.row();
  }
};

// Builds a dense reference and a sorted, duplicate-free triplet list for SparseMatrixType.
template <typename SparseMatrixType>
void make_reference(std::vector<Triplet<typename SparseMatrixType::Scalar>>& triplets,
                    Matrix<typename SparseMatrixType::Scalar, SparseMatrixType::RowsAtCompileTime,
                           SparseMatrixType::ColsAtCompileTime, SparseMatrixType::Options>& dense) {
  typedef typename SparseMatrixType::Scalar Scalar;
  const int rows = SparseMatrixType::RowsAtCompileTime;
  const int max_nz = internal::traits<SparseMatrixType>::MaxNZ;

  triplets.resize(max_nz);
  for (int k = 0; k < max_nz; ++k) {
    int i = (k * 2 + 1) % rows;
    int j = (k * 2 + 1) / rows;
    triplets[k] = Triplet<Scalar>(i, j, Scalar(i * 10 + j + 1 + 3 * i * j));
  }
  triplet_sort<Scalar> cmp{SparseMatrixType::IsRowMajor};
  std::sort(triplets.begin(), triplets.end(), cmp);

  dense.setZero();
  for (const auto& t : triplets) dense(t.row(), t.col()) = t.value();
}

// Tests assignFromSortedTriplets under EIGEN_RUNTIME_NO_MALLOC, including reuse on a second list.
template <typename SparseMatrixType>
void sparse_fixed_nnz_test(const SparseMatrixType&) {
  typedef typename SparseMatrixType::Scalar Scalar;
  const Scalar eps = Scalar(1e-6);

  VERIFY(internal::traits<SparseMatrixType>::MaxNZ != Dynamic);

  typedef Matrix<Scalar, SparseMatrixType::RowsAtCompileTime, SparseMatrixType::ColsAtCompileTime,
                 SparseMatrixType::Options>
      DenseMatrixType;

  std::vector<Triplet<Scalar>> triplets;
  DenseMatrixType dense;
  make_reference<SparseMatrixType>(triplets, dense);

  SparseMatrixType m;

  internal::set_is_malloc_allowed(false);
  m.assignFromSortedTriplets(triplets.begin(), triplets.end());
  internal::set_is_malloc_allowed(true);

  VERIFY_IS_MUCH_SMALLER_THAN((dense - m).norm(), eps);
  VERIFY_IS_APPROX(m, dense);
  VERIFY(m.isCompressed());
  VERIFY_IS_EQUAL(m.nonZeros(), Index(triplets.size()));

  // reuse: second assignment must not leak previous contents
  std::vector<Triplet<Scalar>> triplets2(triplets);
  for (auto& t : triplets2) t = Triplet<Scalar>(t.row(), t.col(), Scalar(2) * t.value());
  internal::set_is_malloc_allowed(false);
  m.assignFromSortedTriplets(triplets2.begin(), triplets2.end());
  internal::set_is_malloc_allowed(true);
  VERIFY_IS_APPROX(m, DenseMatrixType(Scalar(2) * dense));
}

// Tests that a fixed-size SparseMatrix is wrappable by Ref<>.
template <typename SparseMatrixType>
void sparse_fixed_ref_test(const SparseMatrixType&) {
  typedef typename SparseMatrixType::Scalar Scalar;
  typedef Matrix<Scalar, SparseMatrixType::RowsAtCompileTime, SparseMatrixType::ColsAtCompileTime,
                 SparseMatrixType::Options>
      DenseMatrixType;

  std::vector<Triplet<Scalar>> triplets;
  DenseMatrixType dense;
  make_reference<SparseMatrixType>(triplets, dense);

  SparseMatrixType m;
  m.assignFromSortedTriplets(triplets.begin(), triplets.end());

  Ref<const SparseMatrixType> r(m);
  VERIFY_IS_EQUAL(r.rows(), m.rows());
  VERIFY_IS_EQUAL(r.cols(), m.cols());
  VERIFY_IS_EQUAL(r.nonZeros(), m.nonZeros());
  VERIFY(r.isCompressed());
  VERIFY_IS_APPROX(r.toDense(), dense);
}

// Tests that the (rows, cols) constructor accepts only the compile-time sizes.
template <typename SparseMatrixType>
void sparse_fixed_ctor_size_match_test(const SparseMatrixType&) {
  const int Rows = SparseMatrixType::RowsAtCompileTime;
  const int Cols = SparseMatrixType::ColsAtCompileTime;
  SparseMatrixType m(Rows, Cols);
  VERIFY_IS_EQUAL(m.rows(), Rows);
  VERIFY_IS_EQUAL(m.cols(), Cols);
}

// Tests diagonal().size() on off-square fixed-size matrices.
template <typename SparseMatrixType>
void sparse_fixed_diagonal_test(const SparseMatrixType&) {
  typedef typename SparseMatrixType::Scalar Scalar;
  typedef Matrix<Scalar, SparseMatrixType::RowsAtCompileTime, SparseMatrixType::ColsAtCompileTime,
                 SparseMatrixType::Options>
      DenseMatrixType;

  std::vector<Triplet<Scalar>> triplets;
  DenseMatrixType dense;
  make_reference<SparseMatrixType>(triplets, dense);

  SparseMatrixType m;
  m.assignFromSortedTriplets(triplets.begin(), triplets.end());
  const Index expected =
      (numext::mini)(Index(SparseMatrixType::RowsAtCompileTime), Index(SparseMatrixType::ColsAtCompileTime));
  VERIFY_IS_EQUAL(Index(m.diagonal().size()), expected);
}

EIGEN_DECLARE_TEST(sparse_fixed_nnz) {
  g_dense_op_sparse_count = 0;  // suppress compiler warning
  for (int i = 0; i < g_repeat; i++) {
    // Square matrices, ColMajor
    {
      CALL_SUBTEST_1((sparse_fixed_nnz_test(SparseMatrix<double, ColMajor, int, 4, 4, 4>())));
      CALL_SUBTEST_1((sparse_fixed_nnz_test(SparseMatrix<double, ColMajor, int, 5, 5, 4>())));
      CALL_SUBTEST_1((sparse_fixed_nnz_test(SparseMatrix<double, ColMajor, int, 10, 10, 37>())));
      CALL_SUBTEST_1((sparse_fixed_nnz_test(SparseMatrix<double, ColMajor, int, 20, 20, 123>())));
      CALL_SUBTEST_1((sparse_fixed_ref_test(SparseMatrix<double, ColMajor, int, 10, 10, 37>())));
      CALL_SUBTEST_1((sparse_fixed_ctor_size_match_test(SparseMatrix<double, ColMajor, int, 4, 4, 4>())));
    }

    // Square matrices, RowMajor
    {
      CALL_SUBTEST_2((sparse_fixed_nnz_test(SparseMatrix<double, RowMajor, int, 3, 3, 4>())));
      CALL_SUBTEST_2((sparse_fixed_nnz_test(SparseMatrix<double, RowMajor, int, 4, 4, 4>())));
      CALL_SUBTEST_2((sparse_fixed_nnz_test(SparseMatrix<double, RowMajor, int, 10, 10, 4>())));
      CALL_SUBTEST_2((sparse_fixed_nnz_test(SparseMatrix<double, RowMajor, int, 20, 20, 13>())));
      CALL_SUBTEST_2((sparse_fixed_ref_test(SparseMatrix<double, RowMajor, int, 10, 10, 4>())));
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
      CALL_SUBTEST_4((sparse_fixed_diagonal_test(SparseMatrix<double, ColMajor, int, 3, 7, 7>())));
      CALL_SUBTEST_4((sparse_fixed_diagonal_test(SparseMatrix<double, RowMajor, int, 8, 3, 7>())));
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
