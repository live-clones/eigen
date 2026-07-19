// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "main.h"

template <typename Scalar, int Size, int OtherSize>
void symm(int size = Size, int othersize = OtherSize) {
  typedef Matrix<Scalar, Size, Size> MatrixType;
  typedef Matrix<Scalar, Size, OtherSize> Rhs1;
  typedef Matrix<Scalar, OtherSize, Size> Rhs2;
  enum { order = OtherSize == 1 ? 0 : RowMajor };
  typedef Matrix<Scalar, Size, OtherSize, order> Rhs3;

  Index rows = size;
  Index cols = size;

  MatrixType m1 = MatrixType::Random(rows, cols), m2 = MatrixType::Random(rows, cols), m3;

  m1 = (m1 + m1.adjoint()).eval();

  Rhs1 rhs1 = Rhs1::Random(cols, othersize), rhs12(cols, othersize), rhs13(cols, othersize);
  Rhs2 rhs2 = Rhs2::Random(othersize, rows), rhs22(othersize, rows), rhs23(othersize, rows);
  Rhs3 rhs3 = Rhs3::Random(cols, othersize), rhs32(cols, othersize), rhs33(cols, othersize);

  Scalar s1 = internal::random<Scalar>(), s2 = internal::random<Scalar>();

  m2 = m1.template triangularView<Lower>();
  m3 = m2.template selfadjointView<Lower>();
  VERIFY_IS_EQUAL(m1, m3);
  VERIFY_IS_APPROX(rhs12 = (s1 * m2).template selfadjointView<Lower>() * (s2 * rhs1), rhs13 = (s1 * m1) * (s2 * rhs1));

  VERIFY_IS_APPROX(rhs12 = (s1 * m2).transpose().template selfadjointView<Upper>() * (s2 * rhs1),
                   rhs13 = (s1 * m1.transpose()) * (s2 * rhs1));

  VERIFY_IS_APPROX(rhs12 = (s1 * m2).template selfadjointView<Lower>().transpose() * (s2 * rhs1),
                   rhs13 = (s1 * m1.transpose()) * (s2 * rhs1));

  VERIFY_IS_APPROX(rhs12 = (s1 * m2).conjugate().template selfadjointView<Lower>() * (s2 * rhs1),
                   rhs13 = (s1 * m1).conjugate() * (s2 * rhs1));

  VERIFY_IS_APPROX(rhs12 = (s1 * m2).template selfadjointView<Lower>().conjugate() * (s2 * rhs1),
                   rhs13 = (s1 * m1).conjugate() * (s2 * rhs1));

  VERIFY_IS_APPROX(rhs12 = (s1 * m2).adjoint().template selfadjointView<Upper>() * (s2 * rhs1),
                   rhs13 = (s1 * m1).adjoint() * (s2 * rhs1));

  VERIFY_IS_APPROX(rhs12 = (s1 * m2).template selfadjointView<Lower>().adjoint() * (s2 * rhs1),
                   rhs13 = (s1 * m1).adjoint() * (s2 * rhs1));

  m2 = m1.template triangularView<Upper>();
  rhs12.setRandom();
  rhs13 = rhs12;
  m3 = m2.template selfadjointView<Upper>();
  VERIFY_IS_EQUAL(m1, m3);
  VERIFY_IS_APPROX(rhs12 += (s1 * m2).template selfadjointView<Upper>() * (s2 * rhs1),
                   rhs13 += (s1 * m1) * (s2 * rhs1));

  m2 = m1.template triangularView<Lower>();
  VERIFY_IS_APPROX(rhs12 = (s1 * m2).template selfadjointView<Lower>() * (s2 * rhs2.adjoint()),
                   rhs13 = (s1 * m1) * (s2 * rhs2.adjoint()));

  m2 = m1.template triangularView<Upper>();
  VERIFY_IS_APPROX(rhs12 = (s1 * m2).template selfadjointView<Upper>() * (s2 * rhs2.adjoint()),
                   rhs13 = (s1 * m1) * (s2 * rhs2.adjoint()));

  m2 = m1.template triangularView<Upper>();
  VERIFY_IS_APPROX(rhs12 = (s1 * m2.adjoint()).template selfadjointView<Lower>() * (s2 * rhs2.adjoint()),
                   rhs13 = (s1 * m1.adjoint()) * (s2 * rhs2.adjoint()));

  // test row major = <...>
  m2 = m1.template triangularView<Lower>();
  rhs32.setRandom();
  rhs13 = rhs32;
  VERIFY_IS_APPROX(rhs32.noalias() -= (s1 * m2).template selfadjointView<Lower>() * (s2 * rhs3),
                   rhs13 -= (s1 * m1) * (s2 * rhs3));

  m2 = m1.template triangularView<Upper>();
  VERIFY_IS_APPROX(rhs32.noalias() = (s1 * m2.adjoint()).template selfadjointView<Lower>() * (s2 * rhs3).conjugate(),
                   rhs13 = (s1 * m1.adjoint()) * (s2 * rhs3).conjugate());

  m2 = m1.template triangularView<Upper>();
  rhs13 = rhs12;
  VERIFY_IS_APPROX(rhs12.noalias() += s1 * ((m2.adjoint()).template selfadjointView<Lower>() * (s2 * rhs3).conjugate()),
                   rhs13 += (s1 * m1.adjoint()) * (s2 * rhs3).conjugate());

  m2 = m1.template triangularView<Lower>();
  VERIFY_IS_APPROX(rhs22 = (rhs2) * (m2).template selfadjointView<Lower>(), rhs23 = (rhs2) * (m1));
  VERIFY_IS_APPROX(rhs22 = (s2 * rhs2) * (s1 * m2).template selfadjointView<Lower>(), rhs23 = (s2 * rhs2) * (s1 * m1));

  // destination with a non-default inner-stride
  // see bug 1741
  {
    typedef Matrix<Scalar, Dynamic, Dynamic> MatrixX;
    MatrixX buffer(2 * cols, 2 * othersize);
    Map<Rhs1, 0, Stride<Dynamic, 2> > map1(buffer.data(), cols, othersize, Stride<Dynamic, 2>(2 * rows, 2));
    buffer.setZero();
    VERIFY_IS_APPROX(map1.noalias() = (s1 * m2).template selfadjointView<Lower>() * (s2 * rhs1),
                     rhs13 = (s1 * m1) * (s2 * rhs1));

    Map<Rhs2, 0, Stride<Dynamic, 2> > map2(buffer.data(), rhs22.rows(), rhs22.cols(),
                                           Stride<Dynamic, 2>(2 * rhs22.outerStride(), 2));
    buffer.setZero();
    VERIFY_IS_APPROX(map2 = (rhs2) * (m2).template selfadjointView<Lower>(), rhs23 = (rhs2) * (m1));
  }
}

// Physical RowMajor selfadjoint operand.  symm<> above always builds a ColMajor
// operand, so the RowMajor packers -- symm_pack_lhs/symm_pack_rhs specialized on
// RowMajor, including the SME versions whose transposed regions carry the
// two-pass trailing transpose -- are otherwise never reached through the public
// API.  Both operand positions (selfadjoint on the LHS and on the RHS) and both
// stored triangles are checked against a dense reference.
template <typename Scalar>
void symm_rowmajor_selfadjoint(Index size, Index othersize) {
  typedef Matrix<Scalar, Dynamic, Dynamic, RowMajor> RowMat;
  typedef Matrix<Scalar, Dynamic, Dynamic> ColMat;

  RowMat m1 = RowMat::Random(size, size);
  m1 = (m1 + m1.adjoint()).eval();  // exactly self-adjoint
  RowMat lo = m1.template triangularView<Lower>();
  RowMat up = m1.template triangularView<Upper>();

  // Selfadjoint on the LHS: packs the RowMajor operand via symm_pack_lhs.
  ColMat rhs = ColMat::Random(size, othersize);
  ColMat ref = m1 * rhs;
  VERIFY_IS_APPROX((lo.template selfadjointView<Lower>() * rhs).eval(), ref);
  VERIFY_IS_APPROX((up.template selfadjointView<Upper>() * rhs).eval(), ref);

  // Selfadjoint on the RHS: packs the RowMajor operand via symm_pack_rhs.
  ColMat lhs = ColMat::Random(othersize, size);
  ColMat ref2 = lhs * m1;
  VERIFY_IS_APPROX((lhs * lo.template selfadjointView<Lower>()).eval(), ref2);
  VERIFY_IS_APPROX((lhs * up.template selfadjointView<Upper>()).eval(), ref2);
}

// Test symmetric products at blocking boundary sizes.
// The existing test uses random sizes; these deterministic sizes exercise
// transitions in GEBP blocking (early-return at 48, block size changes).
template <int>
void product_symm_boundary() {
  const int sizes[] = {1, 2, 3, 4, 8, 16, 47, 48, 49, 64, 96, 128};
  for (int si = 0; si < 12; ++si) {
    int n = sizes[si];

    // double, matrix RHS
    symm<double, Dynamic, Dynamic>(n, 5);
    // double, vector RHS
    symm<double, Dynamic, 1>(n);
    // float, matrix RHS
    symm<float, Dynamic, Dynamic>(n, 7);
    // complex float, matrix RHS
    symm<std::complex<float>, Dynamic, Dynamic>(n, 3);
  }

  // RowMajor selfadjoint operand.  The partial last-panel widths in this list
  // drive the RowMajor packers' transposed regions through the two-pass trailing
  // transpose for streaming vector lengths from SVL=128 (svlw=4) up to SVL=2048
  // (svlw=64): a partial width w in (svlw, 2*svlw) needs two predicated passes.
  const int sa_sizes[] = {1, 5, 7, 17, 32, 33, 39, 45, 48, 49, 55, 57, 63, 64, 65, 96};
  for (int n : sa_sizes) {
    symm_rowmajor_selfadjoint<float>(n, 7);
    symm_rowmajor_selfadjoint<float>(n, 1);
    symm_rowmajor_selfadjoint<double>(n, 4);
  }
}

EIGEN_DECLARE_TEST(product_symm) {
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1((symm<float, Dynamic, Dynamic>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE),
                                                  internal::random<int>(1, EIGEN_TEST_MAX_SIZE))));
    CALL_SUBTEST_2((symm<double, Dynamic, Dynamic>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE),
                                                   internal::random<int>(1, EIGEN_TEST_MAX_SIZE))));
    CALL_SUBTEST_3((symm<std::complex<float>, Dynamic, Dynamic>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 2),
                                                                internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 2))));
    CALL_SUBTEST_4((symm<std::complex<double>, Dynamic, Dynamic>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 2),
                                                                 internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 2))));

    CALL_SUBTEST_5((symm<float, Dynamic, 1>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE))));
    CALL_SUBTEST_6((symm<double, Dynamic, 1>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE))));
    CALL_SUBTEST_7((symm<std::complex<float>, Dynamic, 1>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE))));
    CALL_SUBTEST_8((symm<std::complex<double>, Dynamic, 1>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE))));
  }

  // Deterministic blocking boundary tests (outside g_repeat).
  CALL_SUBTEST_9(product_symm_boundary<0>());
}
