// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Gael Guennebaud <g.gael@free.fr>
// Copyright (C) 2012 desire Nuentsa <desire.nuentsa_wakam@inria.fr
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "sparse_solver.h"
#include <unsupported/Eigen/IterativeSolvers>

template <typename T>
void test_dgmres_T() {
  DGMRES<SparseMatrix<T>, DiagonalPreconditioner<T> > dgmres_colmajor_diag;
  DGMRES<SparseMatrix<T>, IdentityPreconditioner> dgmres_colmajor_I;
  DGMRES<SparseMatrix<T>, IncompleteLUT<T> > dgmres_colmajor_ilut;
  // GMRES<SparseMatrix<T>, SSORPreconditioner<T> >     dgmres_colmajor_ssor;

  check_sparse_square_solving(dgmres_colmajor_diag);
  //    check_sparse_square_solving(dgmres_colmajor_I)     ;
  check_sparse_square_solving(dgmres_colmajor_ilut);
  //  check_sparse_square_solving(dgmres_colmajor_ssor)     ;
}

template <typename T>
void test_dgmres_breakdown_T() {
  typedef SparseMatrix<T> Mat;
  typedef Matrix<T, 2, 1> Vec;

  // Nilpotent A with singular Hessenberg pivot on the first step.
  Mat A(2, 2);
  A.insert(0, 1) = T(1);
  A.makeCompressed();
  Vec b;
  b << T(1), T(0);

  DGMRES<Mat, IdentityPreconditioner> solver;
  solver.compute(A);
  Vec x = solver.solve(b);
  VERIFY(x.allFinite());
  VERIFY(solver.info() != Success);

  // Diagonal A with b in an eigenspace: Arnoldi converges after one step.
  Mat D(2, 2);
  D.insert(0, 0) = T(2);
  D.insert(1, 1) = T(2);
  D.makeCompressed();
  Vec d;
  d << T(2), T(2);

  DGMRES<Mat, DiagonalPreconditioner<T> > solver2;
  solver2.compute(D);
  Vec y = solver2.solve(d);
  VERIFY_IS_EQUAL(solver2.info(), Success);
  VERIFY_IS_APPROX(y, (Vec() << T(1), T(1)).finished());
}

TEST(DgmresTest, Basic) {
  test_dgmres_T<double>();
  test_dgmres_T<std::complex<double> >();
  test_dgmres_breakdown_T<double>();
  test_dgmres_breakdown_T<std::complex<double> >();
}
