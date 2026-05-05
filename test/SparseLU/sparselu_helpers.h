// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_TEST_SPARSELU_HELPERS_H
#define EIGEN_TEST_SPARSELU_HELPERS_H

#ifdef EIGEN_DEFAULT_TO_ROW_MAJOR
#undef EIGEN_DEFAULT_TO_ROW_MAJOR
#endif

#include "sparse_solver.h"
#include <Eigen/SparseLU>

template <typename T>
void test_sparselu_T() {
  SparseLU<SparseMatrix<T, ColMajor> /*, COLAMDOrdering<int>*/> sparselu_colamd;  // COLAMDOrdering is the default
  SparseLU<SparseMatrix<T, ColMajor>, AMDOrdering<int> > sparselu_amd;
  SparseLU<SparseMatrix<T, ColMajor, long int>, NaturalOrdering<long int> > sparselu_natural;

  check_sparse_square_solving(sparselu_colamd, 300, 100000, true);
  check_sparse_square_solving(sparselu_amd, 300, 10000, true);
  check_sparse_square_solving(sparselu_natural, 300, 2000, true);

  check_sparse_square_abs_determinant(sparselu_colamd);
  check_sparse_square_abs_determinant(sparselu_amd);

  check_sparse_square_determinant(sparselu_colamd);
  check_sparse_square_determinant(sparselu_amd);
}

template <typename T>
void test_sparselu_rowmajor_compressed_input() {
  typedef SparseMatrix<T, RowMajor> RowMajorSparseMatrix;
  typedef Matrix<T, Dynamic, 1> Vector;

  Vector b(2);
  b << T(1.1), T(3.14);

  Vector expected(2);
  expected << T(1.1 - 0.0001 * 3.14), T(3.14);

  RowMajorSparseMatrix compressed(2, 2);
  compressed.insert(0, 0) = T(1.0);
  compressed.insert(0, 1) = T(0.0001);
  compressed.insert(1, 1) = T(1.0);
  compressed.makeCompressed();

  RowMajorSparseMatrix uncompressed(2, 2);
  uncompressed.insert(0, 0) = T(1.0);
  uncompressed.insert(0, 1) = T(0.0001);
  uncompressed.insert(1, 1) = T(1.0);

  SparseLU<RowMajorSparseMatrix> compressed_solver;
  compressed_solver.compute(compressed);
  VERIFY_IS_EQUAL(compressed_solver.info(), Success);
  VERIFY_IS_APPROX(compressed_solver.solve(b), expected);

  SparseLU<RowMajorSparseMatrix> two_step_solver;
  two_step_solver.analyzePattern(compressed);
  two_step_solver.factorize(compressed);
  VERIFY_IS_EQUAL(two_step_solver.info(), Success);
  VERIFY_IS_APPROX(two_step_solver.solve(b), expected);

  SparseLU<RowMajorSparseMatrix> uncompressed_solver;
  uncompressed_solver.compute(uncompressed);
  VERIFY_IS_EQUAL(uncompressed_solver.info(), Success);
  VERIFY_IS_APPROX(uncompressed_solver.solve(b), expected);
}

template <typename T>
void test_sparselu_colmajor_uncompressed_input() {
  typedef SparseMatrix<T, ColMajor> ColMajorSparseMatrix;
  typedef Matrix<T, Dynamic, 1> Vector;

  Vector b(2);
  b << T(1.1), T(3.14);

  Vector expected(2);
  expected << T(1.1 - 0.0001 * 3.14), T(3.14);

  ColMajorSparseMatrix uncompressed(2, 2);
  uncompressed.insert(0, 0) = T(1.0);
  uncompressed.insert(0, 1) = T(0.0001);
  uncompressed.insert(1, 1) = T(1.0);

  SparseLU<ColMajorSparseMatrix> uncompressed_solver;
  uncompressed_solver.compute(uncompressed);
  VERIFY_IS_EQUAL(uncompressed_solver.info(), Success);
  VERIFY_IS_APPROX(uncompressed_solver.solve(b), expected);
}

#endif  // EIGEN_TEST_SPARSELU_HELPERS_H
