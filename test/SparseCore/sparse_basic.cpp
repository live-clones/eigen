// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2011 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2008 Daniel Gomez Ferro <dgomezferro@gmail.com>
// Copyright (C) 2013 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// sparse_basic split: double and float types with default StorageIndex.
// Complex types and non-default StorageIndex are in sparse_basic_extra.cpp.

#include "sparse_basic_helpers.h"

TEST(SparseBasicTest, Basic) {
  g_dense_op_sparse_count = 0;  // Suppresses compiler warning.
  for (int i = 0; i < g_repeat; i++) {
    int r = Eigen::internal::random<int>(1, 200), c = Eigen::internal::random<int>(1, 200);
    if (Eigen::internal::random<int>(0, 3) == 0) {
      r = c;  // check square matrices in 25% of tries
    }
    EIGEN_UNUSED_VARIABLE(r + c);
    sparse_basic(SparseMatrix<double>(1, 1));
    sparse_basic(SparseMatrix<double>(8, 8));
    sparse_basic(SparseMatrix<float, RowMajor>(r, c));
    sparse_basic(SparseMatrix<float, ColMajor>(r, c));
    sparse_basic(SparseMatrix<double, ColMajor>(r, c));
    sparse_basic(SparseMatrix<double, RowMajor>(r, c));
  }

  // Regression test for bug 900: (manually insert higher values here, if you have enough RAM):
  big_sparse_triplet<SparseMatrix<float, RowMajor, int>>(10000, 10000, 0.125);
  big_sparse_triplet<SparseMatrix<double, ColMajor, long int>>(10000, 10000, 0.125);

  bug1105<0>();
  sparse_sub_assign_eigenbase<0>();
  ambivector_coeff<0>();
}

// Verify that two compressed column-major sparse matrices have the same
// sparsity pattern (rows, cols, and per-column inner indices).
template <typename SparseA, typename SparseB>
void verify_same_pattern(const SparseA& got, const SparseB& expected) {
  VERIFY_IS_EQUAL(got.rows(), expected.rows());
  VERIFY_IS_EQUAL(got.cols(), expected.cols());
  for (Index j = 0; j < expected.cols(); ++j) {
    std::vector<Index> got_rows, expected_rows;
    for (typename SparseA::InnerIterator it(got, j); it; ++it) got_rows.push_back(it.row());
    for (typename SparseB::InnerIterator it(expected, j); it; ++it) expected_rows.push_back(it.row());
    std::sort(got_rows.begin(), got_rows.end());
    std::sort(expected_rows.begin(), expected_rows.end());
    VERIFY_IS_EQUAL(Index(got_rows.size()), Index(expected_rows.size()));
    for (size_t k = 0; k < got_rows.size(); ++k) VERIFY_IS_EQUAL(got_rows[k], expected_rows[k]);
  }
}

template <typename Pattern, typename SparseMatrixType>
void verify_sparsity_pattern_ref_matches(const Pattern& pattern, const SparseMatrixType& expected) {
  VERIFY_IS_EQUAL(pattern.outerSize, expected.cols());
  VERIFY_IS_EQUAL(pattern.innerSize, expected.rows());

  Index pattern_nonzeros = 0;
  for (Index j = 0; j < pattern.outerSize; ++j) {
    const Index nz = pattern.nonZeros(j);
    pattern_nonzeros += nz;
    for (Index k = pattern.outer[j], end = k + nz; k < end; ++k) {
      VERIFY(expected.coeff(pattern.inner[k], j) != typename SparseMatrixType::Scalar(0));
    }
  }
  VERIFY_IS_EQUAL(pattern_nonzeros, expected.nonZeros());
}

template <unsigned int UpLo>
void materialize_selfadjoint_pattern_random_impl() {
  typedef SparseMatrix<double, ColMajor, int> SparseMatrixType;
  typedef SparseMatrix<signed char, ColMajor, int> PatternMatrixType;
  typedef Matrix<double, Dynamic, Dynamic, ColMajor> DenseMatrixType;
  typedef Matrix<int, Dynamic, 1> VectorI;

  // Build a random sparse matrix that may have entries on both triangles; the
  // selfadjoint pattern must filter to only the requested triangle.
  const int n = internal::random<int>(8, 64);
  DenseMatrixType ref(n, n);
  SparseMatrixType a(n, n);
  initSparse<double>(0.4, ref, a, ForceNonZeroDiag);
  a.makeCompressed();

  VectorI outer, inner;
  internal::SparsityPatternRef<int> pat = internal::make_col_major_pattern_ref(a, outer, inner);
  PatternMatrixType got;
  internal::materialize_selfadjoint_pattern<UpLo>(pat, got);

  // Reference: explicit symmetrization of the requested triangle.
  SparseMatrixType expected(n, n);
  expected = a.template selfadjointView<UpLo>();
  verify_same_pattern(got, expected);
}

template <int>
void materialize_at_plus_a_pattern_basic() {
  typedef SparseMatrix<double, ColMajor, int> SparseMatrixType;
  typedef SparseMatrix<signed char, ColMajor, int> PatternMatrixType;
  typedef Matrix<int, Dynamic, 1> VectorI;

  // Asymmetric pattern: (0,0), (1,0), (2,1), (0,2), (2,2). Its A^T+A pattern
  // adds (0,1), (1,2), (2,0).
  SparseMatrixType a(3, 3);
  std::vector<Triplet<double, int>> triplets;
  triplets.emplace_back(0, 0, 1.0);
  triplets.emplace_back(1, 0, 2.0);
  triplets.emplace_back(2, 1, 3.0);
  triplets.emplace_back(0, 2, 4.0);
  triplets.emplace_back(2, 2, 5.0);
  a.setFromTriplets(triplets.begin(), triplets.end());

  VectorI outer, inner;
  internal::SparsityPatternRef<int> pat = internal::make_col_major_pattern_ref(a, outer, inner);

  PatternMatrixType got;
  internal::materialize_at_plus_a_pattern(pat, got);

  // Reference: pattern of (a + a.transpose()), keeping structural nonzeros.
  SparseMatrixType expected = a + SparseMatrixType(a.transpose());
  verify_same_pattern(got, expected);
  // Output values must be the placeholder sentinel.
  const signed char one = 1;
  for (Index j = 0; j < got.cols(); ++j) {
    for (PatternMatrixType::InnerIterator it(got, j); it; ++it) {
      VERIFY_IS_EQUAL(it.value(), one);
    }
  }
}

TEST(SparseBasicTest, MaterializeAtPlusAPatternBasic) { materialize_at_plus_a_pattern_basic<0>(); }

template <int>
void materialize_at_plus_a_pattern_random() {
  typedef SparseMatrix<double, ColMajor, int> SparseMatrixType;
  typedef SparseMatrix<signed char, ColMajor, int> PatternMatrixType;
  typedef Matrix<double, Dynamic, Dynamic, ColMajor> DenseMatrixType;
  typedef Matrix<int, Dynamic, 1> VectorI;

  const int n = internal::random<int>(8, 64);
  DenseMatrixType ref(n, n);
  SparseMatrixType a(n, n);
  initSparse<double>(0.4, ref, a, ForceNonZeroDiag);

  VectorI outer, inner;
  internal::SparsityPatternRef<int> pat = internal::make_col_major_pattern_ref(a, outer, inner);
  PatternMatrixType got;
  internal::materialize_at_plus_a_pattern(pat, got);

  SparseMatrixType expected = a + SparseMatrixType(a.transpose());
  verify_same_pattern(got, expected);
}

TEST(SparseBasicTest, MaterializeAtPlusAPatternRandom) {
  for (int i = 0; i < g_repeat; i++) {
    materialize_at_plus_a_pattern_random<0>();
  }
}

template <int>
void materialize_selfadjoint_pattern_lower_upper() {
  materialize_selfadjoint_pattern_random_impl<Lower>();
  materialize_selfadjoint_pattern_random_impl<Upper>();
}

TEST(SparseBasicTest, MaterializeSelfadjointPatternLowerUpper) {
  for (int i = 0; i < g_repeat; i++) {
    materialize_selfadjoint_pattern_lower_upper<0>();
  }
}

template <int>
void sparsity_pattern_ref_sparse_expressions() {
  typedef std::complex<double> Scalar;
  typedef SparseMatrix<Scalar, ColMajor, int> SparseMatrixType;
  typedef SparseMatrix<double, ColMajor, int> RealSparseMatrixType;
  typedef SparseMatrix<double, RowMajor, int> RowMajorSparseMatrixType;
  typedef Matrix<int, Dynamic, 1> VectorI;

  SparseMatrixType a(3, 3);
  std::vector<Triplet<Scalar, int>> triplets;
  triplets.emplace_back(0, 0, Scalar(1.0, 1.0));
  triplets.emplace_back(1, 0, Scalar(2.0, -1.0));
  triplets.emplace_back(2, 1, Scalar(3.0, 2.0));
  triplets.emplace_back(2, 2, Scalar(4.0, -3.0));
  a.setFromTriplets(triplets.begin(), triplets.end());

  PermutationMatrix<Dynamic, Dynamic, int> p(3);
  p.indices() << 1, 2, 0;

  VectorI outer, inner;
  SparseMatrixType permuted = p * a;
  const internal::SparsityPatternRef<int> permuted_pattern = internal::make_col_major_pattern_ref(p * a, outer, inner);
  verify_sparsity_pattern_ref_matches(permuted_pattern, permuted);

  RealSparseMatrixType real_part = a.real();
  const internal::SparsityPatternRef<int> real_pattern = internal::make_col_major_pattern_ref(a.real(), outer, inner);
  verify_sparsity_pattern_ref_matches(real_pattern, real_part);

  RowMajorSparseMatrixType row_major = real_part;
  const internal::SparsityPatternRef<int> row_major_pattern =
      internal::make_col_major_pattern_ref(row_major, outer, inner);
  verify_sparsity_pattern_ref_matches(row_major_pattern, real_part);
}

TEST(SparseBasicTest, SparsityPatternRefSparseExpressions) { sparsity_pattern_ref_sparse_expressions<0>(); }
