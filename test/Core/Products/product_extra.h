// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_TEST_PRODUCT_EXTRA_H
#define EIGEN_TEST_PRODUCT_EXTRA_H

#include "main.h"

template <typename MatrixType>
void product_extra(const MatrixType& m) {
  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, 1, Dynamic> RowVectorType;
  typedef Matrix<Scalar, Dynamic, 1> ColVectorType;
  typedef Matrix<Scalar, Dynamic, Dynamic, MatrixType::Flags & RowMajorBit> OtherMajorMatrixType;

  Index rows = m.rows();
  Index cols = m.cols();

  MatrixType m1 = MatrixType::Random(rows, cols), m2 = MatrixType::Random(rows, cols), m3(rows, cols),
             mzero = MatrixType::Zero(rows, cols), identity = MatrixType::Identity(rows, rows),
             square = MatrixType::Random(rows, rows), res = MatrixType::Random(rows, rows),
             square2 = MatrixType::Random(cols, cols), res2 = MatrixType::Random(cols, cols);
  RowVectorType v1 = RowVectorType::Random(rows), vrres(rows);
  ColVectorType vc2 = ColVectorType::Random(cols), vcres(cols);
  OtherMajorMatrixType tm1 = m1;

  Scalar s1 = internal::random<Scalar>(), s2 = internal::random<Scalar>(), s3 = internal::random<Scalar>();

  VERIFY_IS_APPROX(m3.noalias() = m1 * m2.adjoint(), m1 * m2.adjoint().eval());
  VERIFY_IS_APPROX(m3.noalias() = m1.adjoint() * square.adjoint(), m1.adjoint().eval() * square.adjoint().eval());
  VERIFY_IS_APPROX(m3.noalias() = m1.adjoint() * m2, m1.adjoint().eval() * m2);
  VERIFY_IS_APPROX(m3.noalias() = (s1 * m1.adjoint()) * m2, (s1 * m1.adjoint()).eval() * m2);
  VERIFY_IS_APPROX(m3.noalias() = ((s1 * m1).adjoint()) * m2, (numext::conj(s1) * m1.adjoint()).eval() * m2);
  VERIFY_IS_APPROX(m3.noalias() = (-m1.adjoint() * s1) * (s3 * m2), (-m1.adjoint() * s1).eval() * (s3 * m2).eval());
  VERIFY_IS_APPROX(m3.noalias() = (s2 * m1.adjoint() * s1) * m2, (s2 * m1.adjoint() * s1).eval() * m2);
  VERIFY_IS_APPROX(m3.noalias() = (-m1 * s2) * s1 * m2.adjoint(), (-m1 * s2).eval() * (s1 * m2.adjoint()).eval());

  // a very tricky case where a scale factor has to be automatically conjugated:
  VERIFY_IS_APPROX(m1.adjoint() * (s1 * m2).conjugate(), (m1.adjoint()).eval() * ((s1 * m2).conjugate()).eval());

  // test all possible conjugate combinations for the four matrix-vector product cases:

  VERIFY_IS_APPROX((-m1.conjugate() * s2) * (s1 * vc2), (-m1.conjugate() * s2).eval() * (s1 * vc2).eval());
  VERIFY_IS_APPROX((-m1 * s2) * (s1 * vc2.conjugate()), (-m1 * s2).eval() * (s1 * vc2.conjugate()).eval());
  VERIFY_IS_APPROX((-m1.conjugate() * s2) * (s1 * vc2.conjugate()),
                   (-m1.conjugate() * s2).eval() * (s1 * vc2.conjugate()).eval());

  VERIFY_IS_APPROX((s1 * vc2.transpose()) * (-m1.adjoint() * s2),
                   (s1 * vc2.transpose()).eval() * (-m1.adjoint() * s2).eval());
  VERIFY_IS_APPROX((s1 * vc2.adjoint()) * (-m1.transpose() * s2),
                   (s1 * vc2.adjoint()).eval() * (-m1.transpose() * s2).eval());
  VERIFY_IS_APPROX((s1 * vc2.adjoint()) * (-m1.adjoint() * s2),
                   (s1 * vc2.adjoint()).eval() * (-m1.adjoint() * s2).eval());

  VERIFY_IS_APPROX((-m1.adjoint() * s2) * (s1 * v1.transpose()),
                   (-m1.adjoint() * s2).eval() * (s1 * v1.transpose()).eval());
  VERIFY_IS_APPROX((-m1.transpose() * s2) * (s1 * v1.adjoint()),
                   (-m1.transpose() * s2).eval() * (s1 * v1.adjoint()).eval());
  VERIFY_IS_APPROX((-m1.adjoint() * s2) * (s1 * v1.adjoint()),
                   (-m1.adjoint() * s2).eval() * (s1 * v1.adjoint()).eval());

  VERIFY_IS_APPROX((s1 * v1) * (-m1.conjugate() * s2), (s1 * v1).eval() * (-m1.conjugate() * s2).eval());
  VERIFY_IS_APPROX((s1 * v1.conjugate()) * (-m1 * s2), (s1 * v1.conjugate()).eval() * (-m1 * s2).eval());
  VERIFY_IS_APPROX((s1 * v1.conjugate()) * (-m1.conjugate() * s2),
                   (s1 * v1.conjugate()).eval() * (-m1.conjugate() * s2).eval());

  VERIFY_IS_APPROX((-m1.adjoint() * s2) * (s1 * v1.adjoint()),
                   (-m1.adjoint() * s2).eval() * (s1 * v1.adjoint()).eval());

  // test the vector-matrix product with non aligned starts
  Index i = internal::random<Index>(0, m1.rows() - 2);
  Index j = internal::random<Index>(0, m1.cols() - 2);
  Index r = internal::random<Index>(1, m1.rows() - i);
  Index c = internal::random<Index>(1, m1.cols() - j);
  Index i2 = internal::random<Index>(0, m1.rows() - 1);
  Index j2 = internal::random<Index>(0, m1.cols() - 1);

  VERIFY_IS_APPROX(m1.col(j2).adjoint() * m1.block(0, j, m1.rows(), c),
                   m1.col(j2).adjoint().eval() * m1.block(0, j, m1.rows(), c).eval());
  VERIFY_IS_APPROX(m1.block(i, 0, r, m1.cols()) * m1.row(i2).adjoint(),
                   m1.block(i, 0, r, m1.cols()).eval() * m1.row(i2).adjoint().eval());

  // test negative strides
  {
    Map<MatrixType, Unaligned, Stride<Dynamic, Dynamic> > map1(&m1(rows - 1, cols - 1), rows, cols,
                                                               Stride<Dynamic, Dynamic>(-m1.outerStride(), -1));
    Map<MatrixType, Unaligned, Stride<Dynamic, Dynamic> > map2(&m2(rows - 1, cols - 1), rows, cols,
                                                               Stride<Dynamic, Dynamic>(-m2.outerStride(), -1));
    Map<RowVectorType, Unaligned, InnerStride<-1> > mapv1(&v1(v1.size() - 1), v1.size(), InnerStride<-1>(-1));
    Map<ColVectorType, Unaligned, InnerStride<-1> > mapvc2(&vc2(vc2.size() - 1), vc2.size(), InnerStride<-1>(-1));
    VERIFY_IS_APPROX(MatrixType(map1), m1.reverse());
    VERIFY_IS_APPROX(MatrixType(map2), m2.reverse());
    VERIFY_IS_APPROX(m3.noalias() = MatrixType(map1) * MatrixType(map2).adjoint(),
                     m1.reverse() * m2.reverse().adjoint());
    VERIFY_IS_APPROX(m3.noalias() = map1 * map2.adjoint(), m1.reverse() * m2.reverse().adjoint());
    VERIFY_IS_APPROX(map1 * vc2, m1.reverse() * vc2);
    VERIFY_IS_APPROX(m1 * mapvc2, m1 * mapvc2);
    VERIFY_IS_APPROX(map1.adjoint() * v1.transpose(), m1.adjoint().reverse() * v1.transpose());
    VERIFY_IS_APPROX(m1.adjoint() * mapv1.transpose(), m1.adjoint() * v1.reverse().transpose());
  }

  // regression test
  MatrixType tmp = m1 * m1.adjoint() * s1;
  VERIFY_IS_APPROX(tmp, m1 * m1.adjoint() * s1);

  // regression test for bug 1343, assignment to arrays
  Array<Scalar, Dynamic, 1> a1 = m1 * vc2;
  VERIFY_IS_APPROX(a1.matrix(), m1 * vc2);
  Array<Scalar, Dynamic, 1> a2 = s1 * (m1 * vc2);
  VERIFY_IS_APPROX(a2.matrix(), s1 * m1 * vc2);
  Array<Scalar, 1, Dynamic> a3 = v1 * m1;
  VERIFY_IS_APPROX(a3.matrix(), v1 * m1);
  Array<Scalar, Dynamic, Dynamic> a4 = m1 * m2.adjoint();
  VERIFY_IS_APPROX(a4.matrix(), m1 * m2.adjoint());
}

template <typename MatrixType>
void zero_sized_objects(const MatrixType& m) {
  typedef typename MatrixType::Scalar Scalar;
  const int PacketSize = internal::packet_traits<Scalar>::size;
  const int PacketSize1 = PacketSize > 1 ? PacketSize - 1 : 1;
  Index rows = m.rows();
  Index cols = m.cols();

  {
    MatrixType res, a(rows, 0), b(0, cols);
    VERIFY_IS_APPROX((res = a * b), MatrixType::Zero(rows, cols));
    VERIFY_IS_APPROX((res = a * a.transpose()), MatrixType::Zero(rows, rows));
    VERIFY_IS_APPROX((res = b.transpose() * b), MatrixType::Zero(cols, cols));
    VERIFY_IS_APPROX((res = b.transpose() * a.transpose()), MatrixType::Zero(cols, rows));
  }

  {
    MatrixType res, a(rows, cols), b(cols, 0);
    res = a * b;
    VERIFY(res.rows() == rows && res.cols() == 0);
    b.resize(0, rows);
    res = b * a;
    VERIFY(res.rows() == 0 && res.cols() == cols);
  }

  {
    Matrix<Scalar, PacketSize, 0> a;
    Matrix<Scalar, 0, 1> b;
    Matrix<Scalar, PacketSize, 1> res;
    VERIFY_IS_APPROX((res = a * b), MatrixType::Zero(PacketSize, 1));
    VERIFY_IS_APPROX((res = a.lazyProduct(b)), MatrixType::Zero(PacketSize, 1));
  }

  {
    Matrix<Scalar, PacketSize1, 0> a;
    Matrix<Scalar, 0, 1> b;
    Matrix<Scalar, PacketSize1, 1> res;
    VERIFY_IS_APPROX((res = a * b), MatrixType::Zero(PacketSize1, 1));
    VERIFY_IS_APPROX((res = a.lazyProduct(b)), MatrixType::Zero(PacketSize1, 1));
  }

  {
    Matrix<Scalar, PacketSize, Dynamic> a(PacketSize, 0);
    Matrix<Scalar, Dynamic, 1> b(0, 1);
    Matrix<Scalar, PacketSize, 1> res;
    VERIFY_IS_APPROX((res = a * b), MatrixType::Zero(PacketSize, 1));
    VERIFY_IS_APPROX((res = a.lazyProduct(b)), MatrixType::Zero(PacketSize, 1));
  }

  {
    Matrix<Scalar, PacketSize1, Dynamic> a(PacketSize1, 0);
    Matrix<Scalar, Dynamic, 1> b(0, 1);
    Matrix<Scalar, PacketSize1, 1> res;
    VERIFY_IS_APPROX((res = a * b), MatrixType::Zero(PacketSize1, 1));
    VERIFY_IS_APPROX((res = a.lazyProduct(b)), MatrixType::Zero(PacketSize1, 1));
  }
}

#endif  // EIGEN_TEST_PRODUCT_EXTRA_H
