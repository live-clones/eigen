// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"

#include <type_traits>
#include <vector>

using Eigen::placeholders::all;

// ---------------------------------------------------------------------------
// No-materialization static asserts
//
// IvcType<Span<int,N>, Dynamic> must equal Span<int,N> itself (not
// Matrix<int,N,1>), proving the index buffer is never eval()-led.
// ---------------------------------------------------------------------------
static_assert(std::is_same<Eigen::internal::IvcType<Eigen::Span<int, 4>, Eigen::Dynamic>, Eigen::Span<int, 4>>::value,
              "Static-extent Span must not be materialised in IndexedView");

static_assert(std::is_same<Eigen::internal::IvcType<Eigen::Span<int, Eigen::Dynamic>, Eigen::Dynamic>,
                           Eigen::Span<int, Eigen::Dynamic>>::value,
              "Dynamic-extent Span must not be materialised in IndexedView");

static_assert(!Eigen::internal::is_eigen_index_expression<Eigen::Span<int, 4>>::value,
              "is_eigen_index_expression must be false for Span");

// ---------------------------------------------------------------------------
// Compile-time size traits
// ---------------------------------------------------------------------------
static_assert(Eigen::internal::array_size<Eigen::Span<float, 4>>::value == 4,
              "Static-extent Span must report size at compile time");
static_assert(Eigen::internal::array_size<Eigen::Span<int, Eigen::Dynamic>>::value == Eigen::Dynamic,
              "Dynamic-extent Span must report Dynamic size");

// LvalueBit set for mutable Span, clear for const Span
static_assert((Eigen::internal::traits<Eigen::Span<float, 4>>::Flags & Eigen::LvalueBit) != 0,
              "Mutable Span must have LvalueBit");
static_assert((Eigen::internal::traits<Eigen::Span<const float, 4>>::Flags & Eigen::LvalueBit) == 0,
              "Const Span must not have LvalueBit");

// DirectAccessBit always set
static_assert((Eigen::internal::traits<Eigen::Span<float, 4>>::Flags & Eigen::DirectAccessBit) != 0,
              "Span must have DirectAccessBit");

// PlainObject redirection: Span::PlainObject must equal Span (not Matrix)
static_assert(std::is_same<Eigen::Span<int, 4>::PlainObject, Eigen::Span<int, 4>>::value,
              "Span::PlainObject must be Span itself");

// ---------------------------------------------------------------------------
// Runtime tests
// ---------------------------------------------------------------------------

void test_dynamic_span() {
  const Index n = 8;
  VectorXf storage = VectorXf::LinSpaced(n, 1.f, float(n));

  Eigen::Span<float> s(storage.data(), n);
  VERIFY_IS_EQUAL(s.size(), n);
  VERIFY_IS_EQUAL(s.rows(), n);
  VERIFY_IS_EQUAL(s.cols(), Index(1));
  VERIFY_IS_EQUAL(s.data(), storage.data());

  // Read: use as Eigen expression
  VectorXf copy = s;
  VERIFY_IS_APPROX(copy, storage);

  // Write: assign an Eigen expression into the span
  s = VectorXf::Ones(n);
  VERIFY_IS_APPROX(storage, VectorXf::Ones(n));

  // In-place arithmetic
  Eigen::Span<float> s2(storage.data(), n);
  s2 += VectorXf::Constant(n, 2.f);
  VERIFY_IS_APPROX(storage, VectorXf::Constant(n, 3.f));
}

void test_static_span() {
  float buf[4] = {1.f, 2.f, 3.f, 4.f};

  // Pointer-only constructor (deduces N=4 at compile time)
  Eigen::Span<float, 4> s(buf);
  VERIFY_IS_EQUAL(s.size(), Index(4));
  VERIFY_IS_EQUAL(int(Eigen::Span<float, 4>::SizeAtCompileTime), 4);

  // Read
  VectorXf v = s;
  typedef Matrix<float, 4, 1> Vec4f;
  VERIFY_IS_APPROX(v, Map<Vec4f>(buf));

  // Write
  s = VectorXf::Zero(4);
  for (int i = 0; i < 4; ++i) VERIFY_IS_EQUAL(buf[i], 0.f);

  // Pointer+size constructor also valid for static span (runtime-asserts size == N)
  float arr[4] = {5.f, 6.f, 7.f, 8.f};
  Eigen::Span<float, 4> s2(arr, 4);
  VERIFY_IS_EQUAL(s2[0], 5.f);
  VERIFY_IS_EQUAL(s2[3], 8.f);
}

void test_const_span() {
  VectorXf v = VectorXf::LinSpaced(6, 0.f, 5.f);
  Eigen::Span<const float> cs(v.data(), v.size());

  VERIFY_IS_EQUAL(cs.size(), v.size());
  VERIFY_IS_EQUAL(cs.data(), v.data());
  VERIFY_IS_APPROX(VectorXf(cs), v);

  // Span<const T> constructed from mutable pointer
  Eigen::Span<const float, 6> cs2(v.data());
  VERIFY_IS_APPROX(VectorXf(cs2), v);
}

void test_indexedview_no_copy() {
  // Gather: use Span<int> as row-index list in operator().
  // The static_asserts at the top prove no copy takes place; here we verify
  // functional correctness.
  const Index n = 6;
  MatrixXf A = MatrixXf::Zero(n, n);
  for (Index i = 0; i < n; ++i)
    for (Index j = 0; j < n; ++j) A(i, j) = float(i * 10 + j);

  int idx_arr[3] = {2, 0, 4};
  Eigen::Span<int, 3> row_span(idx_arr, 3);

  auto gathered = A(row_span, all);

  for (Index k = 0; k < 3; ++k)
    for (Index j = 0; j < n; ++j) VERIFY_IS_EQUAL(gathered(k, j), A(idx_arr[k], j));

  // Scatter-write
  MatrixXf B = MatrixXf::Zero(n, n);
  Eigen::Span<int> dyn_span(idx_arr, 3);
  B(dyn_span, all) = A(dyn_span, all);

  for (Index k = 0; k < 3; ++k)
    for (Index j = 0; j < n; ++j) VERIFY_IS_EQUAL(B(idx_arr[k], j), A(idx_arr[k], j));

  // Rows not in the index should remain 0
  VERIFY_IS_EQUAL(B(1, 0), 0.f);
  VERIFY_IS_EQUAL(B(3, 0), 0.f);
  VERIFY_IS_EQUAL(B(5, 0), 0.f);
}

void test_indexedview_dynamic_no_copy() {
  const Index n = 8;
  VectorXf v = VectorXf::LinSpaced(n, 0.f, float(n - 1));

  std::vector<int> idx_vec = {7, 3, 1};
  Eigen::Span<int> idx_span(idx_vec.data(), Index(idx_vec.size()));

  VectorXf gathered = v(idx_span);
  for (Index k = 0; k < Index(idx_vec.size()); ++k) VERIFY_IS_EQUAL(gathered(k), v(idx_vec[std::size_t(k)]));

  // Element scatter-write
  VectorXf w = VectorXf::Zero(n);
  w(idx_span).noalias() = gathered;
  for (Index k = 0; k < Index(idx_vec.size()); ++k)
    VERIFY_IS_EQUAL(w(idx_vec[std::size_t(k)]), v(idx_vec[std::size_t(k)]));
  // Every element not in the index must remain zero
  std::vector<bool> selected(std::size_t(n), false);
  for (int idx : idx_vec) selected[std::size_t(idx)] = true;
  for (Index i = 0; i < n; ++i)
    if (!selected[std::size_t(i)]) VERIFY_IS_EQUAL(w(i), 0.f);
}

void test_arithmetic() {
  float a_buf[4] = {1.f, 2.f, 3.f, 4.f};
  float b_buf[4] = {4.f, 3.f, 2.f, 1.f};

  Eigen::Span<float, 4> a(a_buf), b(b_buf);

  VectorXf sum = a + b;
  VERIFY_IS_APPROX(sum, VectorXf::Constant(4, 5.f));

  // Coefficient-wise product via .cwiseProduct()
  float dot = a.dot(b);
  VERIFY_IS_APPROX(dot, 20.f);
}

void test_inner_outer_stride() {
  float buf[4] = {1.f, 2.f, 3.f, 4.f};
  Eigen::Span<float, 4> s(buf);

  VERIFY_IS_EQUAL(s.innerStride(), Index(1));
  VERIFY_IS_EQUAL(s.outerStride(), Index(4));

  Eigen::Span<float> ds(buf, 4);
  VERIFY_IS_EQUAL(ds.innerStride(), Index(1));
  VERIFY_IS_EQUAL(ds.outerStride(), Index(4));
}

#ifdef EIGEN_HAS_STD_SPAN
void test_std_span_interop() {
  float buf[6] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
  float* const ptr = buf;

  // Dynamic std::span -> dynamic Eigen::Span
  std::span<float> std_dyn(buf, 6);
  Eigen::Span<float> e_dyn(std_dyn);
  VERIFY_IS_EQUAL(e_dyn.size(), Index(6));
  VERIFY_IS_EQUAL(e_dyn.data(), ptr);

  // Static std::span -> static Eigen::Span
  std::span<float, 6> std_static(buf);
  Eigen::Span<float, 6> e_static(std_static);
  VERIFY_IS_EQUAL(e_static.size(), Index(6));

  // const variant
  std::span<const float> std_const(buf, 6);
  Eigen::Span<const float> e_const(std_const);
  VERIFY_IS_EQUAL(e_const.size(), Index(6));
  VERIFY_IS_EQUAL(e_const.data(), static_cast<const float*>(ptr));
}
#endif

EIGEN_DECLARE_TEST(span) {
  CALL_SUBTEST_1(test_dynamic_span());
  CALL_SUBTEST_2(test_static_span());
  CALL_SUBTEST_3(test_const_span());
  CALL_SUBTEST_4(test_indexedview_no_copy());
  CALL_SUBTEST_5(test_indexedview_dynamic_no_copy());
  CALL_SUBTEST_6(test_arithmetic());
  CALL_SUBTEST_7(test_inner_outer_stride());
#ifdef EIGEN_HAS_STD_SPAN
  CALL_SUBTEST_8(test_std_span_interop());
#endif
}
