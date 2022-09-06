// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Gael Guennebaud <g.gael@free.fr>
// Copyright (C) 2012 Kolja Brix <brix@igpm.rwth-aaachen.de>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "../../test/sparse_solver.h"
#include <Eigen/HermitianMatrix>

template <class HermitianMatrix>
void test_assignment(int size = HermitianMatrix::ColsAtCompileTime)
{
  using DenseType = typename HermitianMatrix::DenseType;

  DenseType A = DenseType::Random(size, size);
  DenseType B = A + A.transpose();

  HermitianMatrix H = B;
  DenseType C = H;
  VERIFY_IS_APPROX(C, B);
}

EIGEN_DECLARE_TEST(hermitianMatrix)
{
  CALL_SUBTEST_1(test_assignment<HermitianMatrix<double, Eigen::Dynamic>>(100));
  CALL_SUBTEST_1(test_assignment<HermitianMatrix<float, Eigen::Dynamic>>(100));
  CALL_SUBTEST_1(test_assignment<HermitianMatrix<std::complex<double>, Eigen::Dynamic>>(100));
}
