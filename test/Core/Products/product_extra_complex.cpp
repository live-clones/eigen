// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "product_extra.h"

template <typename T>
EIGEN_DONT_INLINE Index test_compute_block_size(Index m, Index n, Index k) {
  Index mc(m), nc(n), kc(k);
  internal::computeProductBlockingSizes<T, T>(kc, mc, nc);
  return kc + mc + nc;
}

template <typename T>
Index compute_block_size_complex() {
  Index ret = 0;
  ret += test_compute_block_size<T>(0, 1, 1);
  ret += test_compute_block_size<T>(1, 0, 1);
  ret += test_compute_block_size<T>(1, 1, 0);
  {
    Index m = 200, n = 200, k = 200;
    Index mc = m, nc = n, kc = k;
    internal::computeProductBlockingSizes<T, T>(kc, mc, nc);
    VERIFY(kc > 0 && kc <= k);
    VERIFY(mc > 0 && mc <= m);
    VERIFY(nc > 0 && nc <= n);
  }
  return ret;
}

// Test complex GEMV with all conjugation combinations at sizes that
// exercise full, half, and quarter packet code paths.
// The GEMV kernels in GeneralMatrixVector.h use conj_helper at three
// packet levels. The existing product_extra tests cover conjugation
// but only at random sizes, never systematically at packet boundaries.
template <int>
void gemv_complex_conjugate() {
  typedef std::complex<float> Scf;
  typedef std::complex<double> Scd;
  const Index PS_f = internal::packet_traits<Scf>::size;
  const Index PS_d = internal::packet_traits<Scd>::size;

  // Sizes chosen to exercise packet boundaries for both float and double.
  const Index sizes[] = {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33};

  for (int si = 0; si < 14; ++si) {
    Index m = sizes[si];
    // Test complex<float> GEMV with all conjugation combos.
    {
      typedef Matrix<Scf, Dynamic, Dynamic> Mat;
      typedef Matrix<Scf, Dynamic, 1> Vec;
      Mat A = Mat::Random(m, m);
      Vec v = Vec::Random(m);
      Vec res(m);

      // A * v (no conjugation)
      res.noalias() = A * v;
      VERIFY_IS_APPROX(res, (A.eval() * v.eval()).eval());

      // A.conjugate() * v
      res.noalias() = A.conjugate() * v;
      VERIFY_IS_APPROX(res, (A.conjugate().eval() * v.eval()).eval());

      // A * v.conjugate()
      res.noalias() = A * v.conjugate();
      VERIFY_IS_APPROX(res, (A.eval() * v.conjugate().eval()).eval());

      // A.conjugate() * v.conjugate()
      res.noalias() = A.conjugate() * v.conjugate();
      VERIFY_IS_APPROX(res, (A.conjugate().eval() * v.conjugate().eval()).eval());

      // A.adjoint() * v (transpose + conjugate of lhs)
      Vec res2(m);
      res2.noalias() = A.adjoint() * v;
      VERIFY_IS_APPROX(res2, (A.adjoint().eval() * v.eval()).eval());

      // Row-major complex GEMV
      typedef Matrix<Scf, Dynamic, Dynamic, RowMajor> RMat;
      RMat B = A;
      res.noalias() = B * v;
      VERIFY_IS_APPROX(res, (A.eval() * v.eval()).eval());

      res.noalias() = B.conjugate() * v;
      VERIFY_IS_APPROX(res, (A.conjugate().eval() * v.eval()).eval());
    }

    // Test complex<double> GEMV with conjugation.
    {
      typedef Matrix<Scd, Dynamic, Dynamic> Mat;
      typedef Matrix<Scd, Dynamic, 1> Vec;
      Mat A = Mat::Random(m, m);
      Vec v = Vec::Random(m);
      Vec res(m);

      res.noalias() = A.conjugate() * v;
      VERIFY_IS_APPROX(res, (A.conjugate().eval() * v.eval()).eval());

      res.noalias() = A * v.conjugate();
      VERIFY_IS_APPROX(res, (A.eval() * v.conjugate().eval()).eval());

      // Non-square: wide matrix x vector (exercises different cols path).
      Mat C = Mat::Random(m, m + 3);
      Vec w = Vec::Random(m + 3);
      Vec res3(m);
      res3.noalias() = C.conjugate() * w;
      VERIFY_IS_APPROX(res3, (C.conjugate().eval() * w.eval()).eval());
    }
  }
  (void)PS_f;
  (void)PS_d;
}

// =============================================================================
// Tests for product_extra (complex types)
// =============================================================================
TEST(ProductExtraComplexTest, Basic) {
  for (int i = 0; i < g_repeat; i++) {
    product_extra(MatrixXcf(internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 2),
                            internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 2)));
    product_extra(MatrixXcd(internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 2),
                            internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 2)));
  }
  compute_block_size_complex<std::complex<double> >();
  gemv_complex_conjugate<0>();
}
