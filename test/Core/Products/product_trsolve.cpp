// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// product_trsolve split: real scalar types and meta-unrollers.
// Complex types are in product_trsolve_complex.cpp.

#include "product_trsolve_helpers.h"

void trsolve_indexed_view() {
  typedef Matrix<double, Dynamic, Dynamic> MatrixX;
  typedef Matrix<double, Dynamic, 1> VectorX;

  MatrixX lhs = MatrixX::Random(8, 8);
  lhs *= 0.1;
  lhs.diagonal().array() += 1.0;

  VectorX rhs = VectorX::Random(8);
  std::vector<int> indices{0, 1, 2, 7};

  MatrixX lhs_slice = lhs(indices, indices);
  VectorX rhs_slice = rhs(indices);
  VectorX expected = lhs_slice.triangularView<Upper>().solve(rhs_slice);

  VectorX actual = lhs(indices, indices).triangularView<Upper>().solve(rhs(indices));
  VERIFY_IS_APPROX(actual, expected);

  VectorX assigned = VectorX::Random(8);
  VectorX assigned_ref = assigned;
  assigned(indices) = lhs_slice.triangularView<Upper>().solve(rhs_slice);
  assigned_ref(indices) = expected;
  VERIFY_IS_APPROX(assigned, assigned_ref);

  VectorX inplace = rhs;
  VectorX inplace_ref = rhs;
  lhs_slice.triangularView<Upper>().solveInPlace(inplace(indices));
  inplace_ref(indices) = expected;
  VERIFY_IS_APPROX(inplace, inplace_ref);
}

TEST(ProductTrsolveTest, Basic) {
  for (int i = 0; i < g_repeat; i++) {
    // matrices
    trsolve<float, Dynamic, Dynamic>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE),
                                     internal::random<int>(1, EIGEN_TEST_MAX_SIZE));
    trsolve<double, Dynamic, Dynamic>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE),
                                      internal::random<int>(1, EIGEN_TEST_MAX_SIZE));

    // vectors
    trsolve<float, Dynamic, 1>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE));
    trsolve<double, Dynamic, 1>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE));

    // meta-unrollers
    trsolve<float, 4, 1>();
    trsolve<double, 4, 1>();
    trsolve<float, 1, 1>();
    trsolve<float, 1, 2>();
    trsolve<float, 3, 1>();
  }

  // Strided solve at blocking boundaries (deterministic, outside g_repeat).
  trsolve_strided_boundary<0>();
  trsolve_indexed_view();
}
