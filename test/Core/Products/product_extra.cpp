// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "product_extra.h"

// Regression test for bug reported at http://forum.kde.org/viewtopic.php?f=74&t=96947
void mat_mat_scalar_scalar_product() {
  Eigen::Matrix2Xd dNdxy(2, 3);
  dNdxy << -0.5, 0.5, 0, -0.3, 0, 0.3;
  double det = 6.0, wt = 0.5;
  VERIFY_IS_APPROX(dNdxy.transpose() * dNdxy * det * wt, det * wt * dNdxy.transpose() * dNdxy);
}

template <int>
void bug_127() {
  // Bug 127
  //
  // a product of the form lhs*rhs with
  //
  // lhs:
  // rows = 1, cols = 4
  // RowsAtCompileTime = 1, ColsAtCompileTime = -1
  // MaxRowsAtCompileTime = 1, MaxColsAtCompileTime = 5
  //
  // rhs:
  // rows = 4, cols = 0
  // RowsAtCompileTime = -1, ColsAtCompileTime = -1
  // MaxRowsAtCompileTime = 5, MaxColsAtCompileTime = 1
  //
  // was failing on a runtime assertion, because it had been mis-compiled as a dot product because Product.h was using
  // the max-sizes to detect size 1 indicating vectors, and that didn't account for 0-sized object with max-size 1.

  Matrix<float, 1, Dynamic, RowMajor, 1, 5> a(1, 4);
  Matrix<float, Dynamic, Dynamic, ColMajor, 5, 1> b(4, 0);
  a* b;
}

template <int>
void bug_817() {
  ArrayXXf B = ArrayXXf::Random(10, 10), C;
  VectorXf x = VectorXf::Random(10);
  C = (x.transpose() * B.matrix());
  B = (x.transpose() * B.matrix());
  VERIFY_IS_APPROX(B, C);
}

template <int>
void triangular_product_assignment_size_mismatch() {
  MatrixXd m1 = MatrixXd::Random(5, 5);
  MatrixXd m2(6, 6);
  VERIFY_RAISES_ASSERT(m2.triangularView<Lower>() = m1 * m1);
}

template <int>
void unaligned_objects() {
  // Regression test for the bug reported here:
  // http://forum.kde.org/viewtopic.php?f=74&t=107541
  // Recall the matrix*vector kernel avoid unaligned loads by loading two packets and then reassemble then.
  // There was a mistake in the computation of the valid range for fully unaligned objects: in some rare cases,
  // memory was read outside the allocated matrix memory. Though the values were not used, this might raise segfault.
  for (int m = 450; m < 460; ++m) {
    for (int n = 8; n < 12; ++n) {
      MatrixXf M(m, n);
      VectorXf v1(n), r1(500);
      RowVectorXf v2(m), r2(16);

      M.setRandom();
      v1.setRandom();
      v2.setRandom();
      for (int o = 0; o < 4; ++o) {
        r1.segment(o, m).noalias() = M * v1;
        VERIFY_IS_APPROX(r1.segment(o, m), M * MatrixXf(v1));
        r2.segment(o, n).noalias() = v2 * M;
        VERIFY_IS_APPROX(r2.segment(o, n), MatrixXf(v2) * M);
      }
    }
  }
}

template <typename T>
EIGEN_DONT_INLINE Index test_compute_block_size(Index m, Index n, Index k) {
  Index mc(m), nc(n), kc(k);
  internal::computeProductBlockingSizes<T, T>(kc, mc, nc);
  return kc + mc + nc;
}

template <typename T>
Index compute_block_size() {
  Index ret = 0;
  // Zero-sized inputs: verify they compile and don't crash.
  ret += test_compute_block_size<T>(0, 1, 1);
  ret += test_compute_block_size<T>(1, 0, 1);
  ret += test_compute_block_size<T>(1, 1, 0);
  ret += test_compute_block_size<T>(0, 0, 1);
  ret += test_compute_block_size<T>(0, 1, 0);
  ret += test_compute_block_size<T>(1, 0, 0);
  ret += test_compute_block_size<T>(0, 0, 0);

  // Sanity checks: blocking sizes must be positive and not exceed the original.
  {
    Index m = 200, n = 200, k = 200;
    Index mc = m, nc = n, kc = k;
    internal::computeProductBlockingSizes<T, T>(kc, mc, nc);
    VERIFY(kc > 0 && kc <= k);
    VERIFY(mc > 0 && mc <= m);
    VERIFY(nc > 0 && nc <= n);
  }
  // With EIGEN_DEBUG_SMALL_PRODUCT_BLOCKS (l1=9KB, l2=32KB, l3=512KB),
  // large sizes must be actually blocked (not returned as-is).
  {
    Index m = 500, n = 500, k = 500;
    Index mc = m, nc = n, kc = k;
    internal::computeProductBlockingSizes<T, T>(kc, mc, nc);
    VERIFY(kc < k);
  }

  return ret;
}

// Verify correctness of GEMM at sizes that require multiple blocking passes
// under EIGEN_DEBUG_SMALL_PRODUCT_BLOCKS (l1=9KB, l2=32KB, l3=512KB).
// The blocking early-return threshold is max(k,m,n) < 48, so sizes >= 48
// trigger actual multi-pass blocking with these tiny cache sizes.
// Verifies GEMM against column-by-column GEMV (a different code path).
template <int>
void test_small_block_correctness() {
  const int sizes[] = {48, 64, 96, 128, 200};
  for (int si = 0; si < 5; ++si) {
    int n = sizes[si];
    MatrixXd A = MatrixXd::Random(n, n);
    MatrixXd B = MatrixXd::Random(n, n);
    MatrixXd C(n, n);
    C.noalias() = A * B;
    MatrixXd Cref(n, n);
    for (int j = 0; j < n; ++j) Cref.col(j) = A * B.col(j);
    VERIFY_IS_APPROX(C, Cref);
  }
  // Non-square: exercise different blocking in m, n, k dimensions.
  {
    MatrixXd A = MatrixXd::Random(200, 64);
    MatrixXd B = MatrixXd::Random(64, 300);
    MatrixXd C(200, 300);
    C.noalias() = A * B;
    MatrixXd Cref(200, 300);
    for (int j = 0; j < 300; ++j) Cref.col(j) = A * B.col(j);
    VERIFY_IS_APPROX(C, Cref);
  }
}

template <typename>
void aliasing_with_resize() {
  Index m = internal::random<Index>(10, 50);
  Index n = internal::random<Index>(10, 50);
  MatrixXd A, B, C(m, n), D(m, m);
  VectorXd a, b, c(n);
  C.setRandom();
  D.setRandom();
  c.setRandom();
  double s = internal::random<double>(1, 10);

  A = C;
  B = A * A.transpose();
  A = A * A.transpose();
  VERIFY_IS_APPROX(A, B);

  A = C;
  B = (A * A.transpose()) / s;
  A = (A * A.transpose()) / s;
  VERIFY_IS_APPROX(A, B);

  A = C;
  B = (A * A.transpose()) + D;
  A = (A * A.transpose()) + D;
  VERIFY_IS_APPROX(A, B);

  A = C;
  B = D + (A * A.transpose());
  A = D + (A * A.transpose());
  VERIFY_IS_APPROX(A, B);

  A = C;
  B = s * (A * A.transpose());
  A = s * (A * A.transpose());
  VERIFY_IS_APPROX(A, B);

  A = C;
  a = c;
  b = (A * a) / s;
  a = (A * a) / s;
  VERIFY_IS_APPROX(a, b);
}

template <int>
void bug_1308() {
  int n = 10;
  MatrixXd r(n, n);
  VectorXd v = VectorXd::Random(n);
  r = v * RowVectorXd::Ones(n);
  VERIFY_IS_APPROX(r, v.rowwise().replicate(n));
  r = VectorXd::Ones(n) * v.transpose();
  VERIFY_IS_APPROX(r, v.rowwise().replicate(n).transpose());

  Matrix4d ones44 = Matrix4d::Ones();
  Matrix4d m44 = Matrix4d::Ones() * Matrix4d::Ones();
  VERIFY_IS_APPROX(m44, Matrix4d::Constant(4));
  VERIFY_IS_APPROX(m44.noalias() = ones44 * Matrix4d::Ones(), Matrix4d::Constant(4));
  VERIFY_IS_APPROX(m44.noalias() = ones44.transpose() * Matrix4d::Ones(), Matrix4d::Constant(4));
  VERIFY_IS_APPROX(m44.noalias() = Matrix4d::Ones() * ones44, Matrix4d::Constant(4));
  VERIFY_IS_APPROX(m44.noalias() = Matrix4d::Ones() * ones44.transpose(), Matrix4d::Constant(4));

  typedef Matrix<double, 4, 4, RowMajor> RMatrix4d;
  RMatrix4d r44 = Matrix4d::Ones() * Matrix4d::Ones();
  VERIFY_IS_APPROX(r44, Matrix4d::Constant(4));
  VERIFY_IS_APPROX(r44.noalias() = ones44 * Matrix4d::Ones(), Matrix4d::Constant(4));
  VERIFY_IS_APPROX(r44.noalias() = ones44.transpose() * Matrix4d::Ones(), Matrix4d::Constant(4));
  VERIFY_IS_APPROX(r44.noalias() = Matrix4d::Ones() * ones44, Matrix4d::Constant(4));
  VERIFY_IS_APPROX(r44.noalias() = Matrix4d::Ones() * ones44.transpose(), Matrix4d::Constant(4));
  VERIFY_IS_APPROX(r44.noalias() = ones44 * RMatrix4d::Ones(), Matrix4d::Constant(4));
  VERIFY_IS_APPROX(r44.noalias() = ones44.transpose() * RMatrix4d::Ones(), Matrix4d::Constant(4));
  VERIFY_IS_APPROX(r44.noalias() = RMatrix4d::Ones() * ones44, Matrix4d::Constant(4));
  VERIFY_IS_APPROX(r44.noalias() = RMatrix4d::Ones() * ones44.transpose(), Matrix4d::Constant(4));

  //   RowVector4d r4;
  m44.setOnes();
  r44.setZero();
  VERIFY_IS_APPROX(r44.noalias() += m44.row(0).transpose() * RowVector4d::Ones(), ones44);
  r44.setZero();
  VERIFY_IS_APPROX(r44.noalias() += m44.col(0) * RowVector4d::Ones(), ones44);
  r44.setZero();
  VERIFY_IS_APPROX(r44.noalias() += Vector4d::Ones() * m44.row(0), ones44);
  r44.setZero();
  VERIFY_IS_APPROX(r44.noalias() += Vector4d::Ones() * m44.col(0).transpose(), ones44);
}

// Regression test for issue #3059: GEBP asm register constraints fail
// for custom (non-vectorizable) scalar types. Type T has a non-trivial
// destructor (making sizeof(T) > sizeof(double)), while type U is a
// simple wrapper. Both must compile and produce correct products.
namespace issue_3059 {

class Ptr {
 public:
  ~Ptr() {}
  double* m_ptr = nullptr;
};

class T {
 public:
  T() = default;
  T(double v) : m_value(v) {}

  friend T operator*(const T& a, const T& b) { return T(a.m_value * b.m_value); }
  T& operator*=(const T& o) {
    m_value *= o.m_value;
    return *this;
  }
  friend T operator/(const T& a, const T& b) { return T(a.m_value / b.m_value); }
  T& operator/=(const T& o) {
    m_value /= o.m_value;
    return *this;
  }
  friend T operator+(const T& a, const T& b) { return T(a.m_value + b.m_value); }
  T& operator+=(const T& o) {
    m_value += o.m_value;
    return *this;
  }
  friend T operator-(const T& a, const T& b) { return T(a.m_value - b.m_value); }
  T& operator-=(const T& o) {
    m_value -= o.m_value;
    return *this;
  }
  friend T operator-(const T& a) { return T(-a.m_value); }

  bool operator==(const T& o) const { return m_value == o.m_value; }
  bool operator<(const T& o) const { return m_value < o.m_value; }
  bool operator<=(const T& o) const { return m_value <= o.m_value; }
  bool operator>(const T& o) const { return m_value > o.m_value; }
  bool operator>=(const T& o) const { return m_value >= o.m_value; }
  bool operator!=(const T& o) const { return m_value != o.m_value; }

  double value() const { return m_value; }

 private:
  double m_value = 0.0;
  Ptr m_ptr;  // Makes sizeof(T) > sizeof(double)
};

T sqrt(const T& x) { return T(std::sqrt(x.value())); }
T abs(const T& x) { return T(std::abs(x.value())); }
T abs2(const T& x) { return T(x.value() * x.value()); }

class U {
 public:
  U() = default;
  U(double v) : m_value(v) {}

  friend U operator*(const U& a, const U& b) { return U(a.m_value * b.m_value); }
  U& operator*=(const U& o) {
    m_value *= o.m_value;
    return *this;
  }
  friend U operator/(const U& a, const U& b) { return U(a.m_value / b.m_value); }
  U& operator/=(const U& o) {
    m_value /= o.m_value;
    return *this;
  }
  friend U operator+(const U& a, const U& b) { return U(a.m_value + b.m_value); }
  U& operator+=(const U& o) {
    m_value += o.m_value;
    return *this;
  }
  friend U operator-(const U& a, const U& b) { return U(a.m_value - b.m_value); }
  U& operator-=(const U& o) {
    m_value -= o.m_value;
    return *this;
  }
  friend U operator-(const U& a) { return U(-a.m_value); }

  bool operator==(const U& o) const { return m_value == o.m_value; }
  bool operator<(const U& o) const { return m_value < o.m_value; }
  bool operator<=(const U& o) const { return m_value <= o.m_value; }
  bool operator>(const U& o) const { return m_value > o.m_value; }
  bool operator>=(const U& o) const { return m_value >= o.m_value; }
  bool operator!=(const U& o) const { return m_value != o.m_value; }

  double value() const { return m_value; }

 private:
  double m_value = 0.0;
};

U sqrt(const U& x) { return U(std::sqrt(x.value())); }
U abs(const U& x) { return U(std::abs(x.value())); }
U abs2(const U& x) { return U(x.value() * x.value()); }

}  // namespace issue_3059

namespace Eigen {

template <>
struct NumTraits<issue_3059::T> : NumTraits<double> {
  using Real = issue_3059::T;
  using NonInteger = issue_3059::T;
  using Nested = issue_3059::T;
  enum { IsComplex = 0, RequireInitialization = 1 };
};

template <>
struct NumTraits<issue_3059::U> : NumTraits<double> {
  using Real = issue_3059::U;
  using NonInteger = issue_3059::U;
  using Nested = issue_3059::U;
  enum { IsComplex = 0, RequireInitialization = 0 };
};

}  // namespace Eigen

template <int>
void product_custom_scalar_types() {
  using namespace issue_3059;
  // Type T: has non-trivial destructor, sizeof(T) > sizeof(double)
  {
    Matrix<T, Dynamic, Dynamic> A(4, 4), B(4, 4), C(4, 4);
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j) {
        A(i, j) = T(static_cast<double>(i + 1));
        B(i, j) = T(static_cast<double>(j + 1));
      }
    C.noalias() = A * B;
    // A*B: C(i,j) = sum_k (i+1)*(k+1) * ... no, A(i,k)=(i+1), B(k,j)=(j+1)
    // so C(i,j) = sum_k (i+1)*(j+1) = 4*(i+1)*(j+1)
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j) VERIFY(C(i, j) == T(4.0 * (i + 1) * (j + 1)));
  }
  // Type U: simple wrapper, sizeof(U) == sizeof(double)
  {
    Matrix<U, Dynamic, Dynamic> A(4, 4), B(4, 4), C(4, 4);
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j) {
        A(i, j) = U(static_cast<double>(i + 1));
        B(i, j) = U(static_cast<double>(j + 1));
      }
    C.noalias() = A * B;
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j) VERIFY(C(i, j) == U(4.0 * (i + 1) * (j + 1)));
  }
  // Larger matrices to exercise GEBP blocking.
  {
    const int n = 33;
    Matrix<U, Dynamic, Dynamic> A(n, n), B(n, n), C(n, n);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) {
        A(i, j) = U(static_cast<double>((i * 7 + j * 3) % 13));
        B(i, j) = U(static_cast<double>((i * 5 + j * 11) % 17));
      }
    C.noalias() = A * B;
    // Verify against explicit triple loop.
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) {
        double sum = 0;
        for (int k = 0; k < n; ++k) sum += A(i, k).value() * B(k, j).value();
        VERIFY(C(i, j) == U(sum));
      }
  }
}

// =============================================================================
// Tests for product_extra (real types)
// =============================================================================
TEST(ProductExtraTest, Basic) {
  for (int i = 0; i < g_repeat; i++) {
    product_extra(
        MatrixXf(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
    product_extra(
        MatrixXd(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
    mat_mat_scalar_scalar_product();
    zero_sized_objects(
        MatrixXf(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
  }
  bug_127<0>();
  bug_817<0>();
  bug_1308<0>();
  unaligned_objects<0>();
  compute_block_size<float>();
  compute_block_size<double>();
  aliasing_with_resize<void>();
  product_custom_scalar_types<0>();
  test_small_block_correctness<0>();
}
