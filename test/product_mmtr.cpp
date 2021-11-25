// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010-2017 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#define CHECK_MMTR(DEST, TRI, OP) {                   \
    ref3 = DEST;                                      \
    ref2 = ref1 = DEST;                               \
    DEST.template triangularView<TRI>() OP;           \
    ref1 OP;                                          \
    ref2.template triangularView<TRI>()               \
      = ref1.template triangularView<TRI>();          \
    VERIFY_IS_APPROX(DEST,ref2);                      \
    \
    DEST = ref3;                                      \
    ref3 = ref2;                                      \
    ref3.diagonal() = DEST.diagonal();                \
    DEST.template triangularView<TRI|ZeroDiag>() OP;  \
    VERIFY_IS_APPROX(DEST,ref3);                      \
  }

template<typename Scalar> void mmtr(int size)
{
  typedef Matrix<Scalar,Dynamic,Dynamic,ColMajor> MatrixColMaj;
  typedef Matrix<Scalar,Dynamic,Dynamic,RowMajor> MatrixRowMaj;

  DenseIndex othersize = internal::random<DenseIndex>(1,200);
  
  MatrixColMaj matc = MatrixColMaj::Zero(size, size);
  MatrixRowMaj matr = MatrixRowMaj::Zero(size, size);
  MatrixColMaj ref1(size, size), ref2(size, size), ref3(size,size);
  
  MatrixColMaj soc(size,othersize); soc.setRandom();
  MatrixColMaj osc(othersize,size); osc.setRandom();
  MatrixRowMaj sor(size,othersize); sor.setRandom();
  MatrixRowMaj osr(othersize,size); osr.setRandom();
  MatrixColMaj sqc(size,size); sqc.setRandom();
  MatrixRowMaj sqr(size,size); sqr.setRandom();
  
  Scalar s = internal::random<Scalar>();
  
  CHECK_MMTR(matc, Lower, = s*soc*sor.adjoint());
  CHECK_MMTR(matc, Upper, = s*(soc*soc.adjoint()));
  CHECK_MMTR(matr, Lower, = s*soc*soc.adjoint());
  CHECK_MMTR(matr, Upper, = soc*(s*sor.adjoint()));
  
  CHECK_MMTR(matc, Lower, += s*soc*soc.adjoint());
  CHECK_MMTR(matc, Upper, += s*(soc*sor.transpose()));
  CHECK_MMTR(matr, Lower, += s*sor*soc.adjoint());
  CHECK_MMTR(matr, Upper, += soc*(s*soc.adjoint()));
  
  CHECK_MMTR(matc, Lower, -= s*soc*soc.adjoint());
  CHECK_MMTR(matc, Upper, -= s*(osc.transpose()*osc.conjugate()));
  CHECK_MMTR(matr, Lower, -= s*soc*soc.adjoint());
  CHECK_MMTR(matr, Upper, -= soc*(s*soc.adjoint()));
  
  CHECK_MMTR(matc, Lower, -= s*sqr*sqc.triangularView(Upper_t{}));
  CHECK_MMTR(matc, Upper, = s*sqc*sqr.triangularView(Upper_t{}));
  CHECK_MMTR(matc, Lower, += s*sqr*sqc.triangularView(Lower_t{}));
  CHECK_MMTR(matc, Upper, = s*sqc*sqc.triangularView(Lower_t{}));
  
  CHECK_MMTR(matc, Lower, = (s*sqr).triangularView(Upper_t{})*sqc);
  CHECK_MMTR(matc, Upper, -= (s*sqc).triangularView(Upper_t{})*sqc);
  CHECK_MMTR(matc, Lower, = (s*sqr).triangularView(Lower_t{})*sqc);
  CHECK_MMTR(matc, Upper, += (s*sqc).triangularView(Lower_t{})*sqc);

  // check aliasing
  ref2 = ref1 = matc;
  ref1 = sqc.adjoint() * matc * sqc;
  ref2.triangularView(Upper_t{}) = ref1.triangularView(Upper_t{});
  matc.triangularView(Upper_t{}) = sqc.adjoint() * matc * sqc;
  VERIFY_IS_APPROX(matc, ref2);

  ref2 = ref1 = matc;
  ref1 = sqc * matc * sqc.adjoint();
  ref2.triangularView(Lower_t{}) = ref1.triangularView(Lower_t{});
  matc.triangularView(Lower_t{}) = sqc * matc * sqc.adjoint();
  VERIFY_IS_APPROX(matc, ref2);

  // destination with a non-default inner-stride
  // see bug 1741
  {
    typedef Matrix<Scalar,Dynamic,Dynamic> MatrixX;
    MatrixX buffer(2*size,2*size);
    Map<MatrixColMaj,0,Stride<Dynamic,Dynamic> > map1(buffer.data(),size,size,Stride<Dynamic,Dynamic>(2*size,2));
    buffer.setZero();
    CHECK_MMTR(map1, Lower, = s*soc*sor.adjoint());
  }
}

EIGEN_DECLARE_TEST(product_mmtr)
{
  for(int i = 0; i < g_repeat ; i++)
  {
    CALL_SUBTEST_1((mmtr<float>(internal::random<int>(1,EIGEN_TEST_MAX_SIZE))));
    CALL_SUBTEST_2((mmtr<double>(internal::random<int>(1,EIGEN_TEST_MAX_SIZE))));
    CALL_SUBTEST_3((mmtr<std::complex<float> >(internal::random<int>(1,EIGEN_TEST_MAX_SIZE/2))));
    CALL_SUBTEST_4((mmtr<std::complex<double> >(internal::random<int>(1,EIGEN_TEST_MAX_SIZE/2))));
  }
}
