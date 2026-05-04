// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2010,2012 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "eigensolver_generic.h"

TEST(EigensolverGenericTest, Extra) {
  {
    // regression test for bug 793
    MatrixXd a(3, 3);
    a << 0, 0, 1, 1, 1, 1, 1, 1e+200, 1;
    Eigen::EigenSolver<MatrixXd> eig(a);
    double scale = 1e-200;  // scale to avoid overflow during the comparisons
    VERIFY_IS_APPROX(a * eig.pseudoEigenvectors() * scale,
                     eig.pseudoEigenvectors() * eig.pseudoEigenvalueMatrix() * scale);
    VERIFY_IS_APPROX(a * eig.eigenvectors() * scale, eig.eigenvectors() * eig.eigenvalues().asDiagonal() * scale);
  }
  {
    // check a case where all eigenvalues are null.
    MatrixXd a(2, 2);
    a << 1, 1, -1, -1;
    Eigen::EigenSolver<MatrixXd> eig(a);
    VERIFY_IS_APPROX(eig.pseudoEigenvectors().squaredNorm(), 2.);
    VERIFY_IS_APPROX((a * eig.pseudoEigenvectors()).norm() + 1., 1.);
    VERIFY_IS_APPROX((eig.pseudoEigenvectors() * eig.pseudoEigenvalueMatrix()).norm() + 1., 1.);
    VERIFY_IS_APPROX((a * eig.eigenvectors()).norm() + 1., 1.);
    VERIFY_IS_APPROX((eig.eigenvectors() * eig.eigenvalues().asDiagonal()).norm() + 1., 1.);
  }

  // regression test for bug 933
  {
    {
      VectorXd coeffs(5);
      coeffs << 1, -3, -175, -225, 2250;
      MatrixXd C = make_companion(coeffs);
      EigenSolver<MatrixXd> eig(C);
      check_eigensolver_for_given_mat(eig, C);
    }
    {
      // this test is tricky because it requires high accuracy in smallest eigenvalues
      VectorXd coeffs(5);
      coeffs << 6.154671e-15, -1.003870e-10, -9.819570e-01, 3.995715e+03, 2.211511e+08;
      MatrixXd C = make_companion(coeffs);
      EigenSolver<MatrixXd> eig(C);
      check_eigensolver_for_given_mat(eig, C);
      Index n = C.rows();
      for (Index i = 0; i < n; ++i) {
        typedef std::complex<double> Complex;
        MatrixXcd ac = C.cast<Complex>();
        ac.diagonal().array() -= eig.eigenvalues()(i);
        VectorXd sv = ac.jacobiSvd().singularValues();
        // comparing to sv(0) is not enough here to catch the "bug",
        // the hard-coded 1.0 is important!
        VERIFY_IS_MUCH_SMALLER_THAN(sv(n - 1), 1.0);
      }
    }
  }
  // regression test for bug 1557
  {
    // this test is interesting because it contains zeros on the diagonal.
    MatrixXd A_bug1557(3, 3);
    A_bug1557 << 0, 0, 0, 1, 0, 0.5887907064808635127, 0, 1, 0;
    EigenSolver<MatrixXd> eig(A_bug1557);
    check_eigensolver_for_given_mat(eig, A_bug1557);
  }

  // regression test for bug 1174
  {
    Index n = 12;
    MatrixXf A_bug1174(n, n);
    A_bug1174 << 262144, 0, 0, 262144, 786432, 0, 0, 0, 0, 0, 0, 786432, 262144, 0, 0, 262144, 786432, 0, 0, 0, 0, 0, 0,
        786432, 262144, 0, 0, 262144, 786432, 0, 0, 0, 0, 0, 0, 786432, 262144, 0, 0, 262144, 786432, 0, 0, 0, 0, 0, 0,
        786432, 0, 262144, 262144, 0, 0, 262144, 262144, 262144, 262144, 262144, 262144, 0, 0, 262144, 262144, 0, 0,
        262144, 262144, 262144, 262144, 262144, 262144, 0, 0, 262144, 262144, 0, 0, 262144, 262144, 262144, 262144,
        262144, 262144, 0, 0, 262144, 262144, 0, 0, 262144, 262144, 262144, 262144, 262144, 262144, 0, 0, 262144,
        262144, 0, 0, 262144, 262144, 262144, 262144, 262144, 262144, 0, 0, 262144, 262144, 0, 0, 262144, 262144,
        262144, 262144, 262144, 262144, 0, 0, 262144, 262144, 0, 0, 262144, 262144, 262144, 262144, 262144, 262144, 0,
        0, 262144, 262144, 0, 0, 262144, 262144, 262144, 262144, 262144, 262144, 0;
    EigenSolver<MatrixXf> eig(A_bug1174);
    check_eigensolver_for_given_mat(eig, A_bug1174);
  }
}
