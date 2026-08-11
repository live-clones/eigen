// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "sparselu_helpers.h"

TEST(SparseluTest, Float) {
  test_sparselu_T<float>();
  test_sparselu_rowmajor_compressed_input<float>();
  test_sparselu_colmajor_uncompressed_input<float>();
}

template <typename T>
void test_sparselu_clear_error_state() {
  typedef SparseMatrix<T, ColMajor> ColMajorSparseMatrix;

  ColMajorSparseMatrix singular(3, 3);  // structurally singular: no nonzeros

  ColMajorSparseMatrix non_singular(3, 3);
  non_singular.insert(0, 0) = T(1);
  non_singular.insert(1, 1) = T(1);
  non_singular.insert(2, 2) = T(1);
  non_singular.makeCompressed();

  SparseLU<ColMajorSparseMatrix> solver;
  solver.compute(singular);
  VERIFY(solver.info() != Success);
  VERIFY(!solver.lastErrorMessage().empty());

  solver.compute(non_singular);
  VERIFY_IS_EQUAL(solver.info(), Success);
  VERIFY(solver.lastErrorMessage().empty());
}

TEST(SparseluTest, TestSparseluClearErrorState) {
  test_sparselu_clear_error_state<float>();
  test_sparselu_clear_error_state<double>();
}
