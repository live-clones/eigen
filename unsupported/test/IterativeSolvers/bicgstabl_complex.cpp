// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Gael Guennebaud <g.gael@free.fr>
// Copyright (C) 2012 Kolja Brix <brix@igpm.rwth-aaachen.de>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "sparse_solver.h"
<<<<<<<< HEAD:test/bicgstabl.cpp
#include <Eigen/IterativeLinearSolvers>
========
#include <Eigen/IterativeSolvers>
>>>>>>>> c57243a37 (Migrate Eigen test framework from custom to Google Test (gtest)):unsupported/test/IterativeSolvers/bicgstabl_complex.cpp

template <typename T>
void test_bicgstabl_T() {
  BiCGSTABL<SparseMatrix<T>, DiagonalPreconditioner<T> > bicgstabl_colmajor_diag;
  BiCGSTABL<SparseMatrix<T>, IncompleteLUT<T> > bicgstabl_colmajor_ilut;

  bicgstabl_colmajor_diag.setTolerance(NumTraits<T>::epsilon() * 20);
  bicgstabl_colmajor_ilut.setTolerance(NumTraits<T>::epsilon() * 20);

  check_sparse_square_solving(bicgstabl_colmajor_diag);
  check_sparse_square_solving(bicgstabl_colmajor_ilut);
}

<<<<<<<< HEAD:test/bicgstabl.cpp
void test_bicgstabl_solve_with_guess_restart() {
  // Regression for a solveWithGuess case where the recursive residual reached
  // the tolerance before the true residual did.
  srand(555248885);
  test_bicgstabl_T<double>();
}

EIGEN_DECLARE_TEST(bicgstabl) {
  CALL_SUBTEST_1(test_bicgstabl_T<double>());
  CALL_SUBTEST_2(test_bicgstabl_T<std::complex<double> >());
  CALL_SUBTEST_3(test_bicgstabl_solve_with_guess_restart());
}
========
TEST(BicgstablTest, Complex) { test_bicgstabl_T<std::complex<double> >(); }
>>>>>>>> c57243a37 (Migrate Eigen test framework from custom to Google Test (gtest)):unsupported/test/IterativeSolvers/bicgstabl_complex.cpp
