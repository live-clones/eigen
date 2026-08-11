// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2015 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_TEST_REDUX_HELPERS_H
#define EIGEN_TEST_REDUX_HELPERS_H

#define TEST_ENABLE_TEMPORARY_TRACKING
#define EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD 8
// ^^ see bug 1449

#include "main.h"

template <typename MatrixType>
void matrixRedux(const MatrixType& m) {
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;

  Index rows = m.rows();
  Index cols = m.cols();

  MatrixType m1 = MatrixType::Random(rows, cols);

  // The entries of m1 are uniformly distributed in [-1,1), so m1.prod() is very small. This may lead to test
  // failures if we underflow into denormals. Thus, we scale so that entries are close to 1.
  MatrixType m1_for_prod = MatrixType::Ones(rows, cols) + RealScalar(0.2) * m1;

  Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime> m2(rows, rows);
  m2.setRandom();
  // Prevent overflows for integer types.
  if (Eigen::NumTraits<Scalar>::IsInteger) {
    Scalar kMaxVal = Scalar(8);
    m1.array() = m1.array() - kMaxVal * (m1.array() / kMaxVal);
    m2.array() = m2.array() - kMaxVal * (m2.array() / kMaxVal);
  }

  VERIFY_IS_EQUAL(MatrixType::Zero(rows, cols).sum(), Scalar(0));
  Scalar sizeAsScalar = internal::cast<Index, Scalar>(rows * cols);
  VERIFY_IS_APPROX(MatrixType::Ones(rows, cols).sum(), sizeAsScalar);
  Scalar s(0), p(1), minc(numext::real(m1.coeff(0))), maxc(numext::real(m1.coeff(0)));
  for (int j = 0; j < cols; j++)
    for (int i = 0; i < rows; i++) {
      s += m1(i, j);
      p *= m1_for_prod(i, j);
      minc = (std::min)(numext::real(minc), numext::real(m1(i, j)));
      maxc = (std::max)(numext::real(maxc), numext::real(m1(i, j)));
    }
  const Scalar mean = s / Scalar(RealScalar(rows * cols));

  VERIFY_IS_APPROX(m1.sum(), s);
  VERIFY_IS_APPROX(m1.mean(), mean);
  VERIFY_IS_APPROX(m1_for_prod.prod(), p);
  VERIFY_IS_APPROX(m1.real().minCoeff(), numext::real(minc));
  VERIFY_IS_APPROX(m1.real().maxCoeff(), numext::real(maxc));

  // test that partial reduction works if nested expressions is forced to evaluate early
  VERIFY_IS_APPROX((m1.matrix() * m1.matrix().transpose()).cwiseProduct(m2.matrix()).rowwise().sum().sum(),
                   (m1.matrix() * m1.matrix().transpose()).eval().cwiseProduct(m2.matrix()).rowwise().sum().sum());

  // test slice vectorization assuming assign is ok
  Index r0 = internal::random<Index>(0, rows - 1);
  Index c0 = internal::random<Index>(0, cols - 1);
  Index r1 = internal::random<Index>(r0 + 1, rows) - r0;
  Index c1 = internal::random<Index>(c0 + 1, cols) - c0;
  VERIFY_IS_APPROX(m1.block(r0, c0, r1, c1).sum(), m1.block(r0, c0, r1, c1).eval().sum());
  VERIFY_IS_APPROX(m1.block(r0, c0, r1, c1).mean(), m1.block(r0, c0, r1, c1).eval().mean());
  VERIFY_IS_APPROX(m1_for_prod.block(r0, c0, r1, c1).prod(), m1_for_prod.block(r0, c0, r1, c1).eval().prod());
  VERIFY_IS_APPROX(m1.block(r0, c0, r1, c1).real().minCoeff(), m1.block(r0, c0, r1, c1).real().eval().minCoeff());
  VERIFY_IS_APPROX(m1.block(r0, c0, r1, c1).real().maxCoeff(), m1.block(r0, c0, r1, c1).real().eval().maxCoeff());

  // regression for bug 1090
  constexpr int R1 = MatrixType::RowsAtCompileTime >= 2 ? MatrixType::RowsAtCompileTime / 2 : 6;
  constexpr int C1 = MatrixType::ColsAtCompileTime >= 2 ? MatrixType::ColsAtCompileTime / 2 : 6;
  if (R1 <= rows - r0 && C1 <= cols - c0) {
    VERIFY_IS_APPROX((m1.template block<R1, C1>(r0, c0).sum()), m1.block(r0, c0, R1, C1).sum());
  }

  // test empty objects
  VERIFY_IS_APPROX(m1.block(r0, c0, 0, 0).sum(), Scalar(0));
  VERIFY_IS_APPROX(m1.block(r0, c0, 0, 0).prod(), Scalar(1));

  // test nesting complex expression
  VERIFY_EVALUATION_COUNT((m1.matrix() * m1.matrix().transpose()).sum(),
                          (MatrixType::IsVectorAtCompileTime && MatrixType::SizeAtCompileTime != 1 ? 0 : 1));
  VERIFY_EVALUATION_COUNT(((m1.matrix() * m1.matrix().transpose()) + m2).sum(),
                          (MatrixType::IsVectorAtCompileTime && MatrixType::SizeAtCompileTime != 1 ? 0 : 1));
}

template <typename VectorType>
void vectorRedux(const VectorType& w) {
  using std::abs;
  typedef typename VectorType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  Index size = w.size();

  VectorType v = VectorType::Random(size);
  VectorType v_for_prod = VectorType::Ones(size) + Scalar(0.2) * v;  // see comment above declaration of m1_for_prod
  if (Eigen::NumTraits<Scalar>::IsInteger) {
    Scalar kMaxVal = Scalar(8);
    v.array() = v.array() - kMaxVal * (v.array() / kMaxVal);
    v_for_prod = VectorType::Ones(size) + Scalar(0.2) * v;
  }

  for (int i = 1; i < size; i++) {
    Scalar s(0), p(1);
    RealScalar minc(numext::real(v.coeff(0))), maxc(numext::real(v.coeff(0)));
    for (int j = 0; j < i; j++) {
      s += v[j];
      p *= v_for_prod[j];
      minc = (std::min)(minc, numext::real(v[j]));
      maxc = (std::max)(maxc, numext::real(v[j]));
    }
    VERIFY_IS_MUCH_SMALLER_THAN(abs(s - v.head(i).sum()), Scalar(1));
    VERIFY_IS_APPROX(p, v_for_prod.head(i).prod());
    VERIFY_IS_APPROX(minc, v.real().head(i).minCoeff());
    VERIFY_IS_APPROX(maxc, v.real().head(i).maxCoeff());
  }

  for (int i = 0; i < size - 1; i++) {
    Scalar s(0), p(1);
    RealScalar minc(numext::real(v.coeff(i))), maxc(numext::real(v.coeff(i)));
    for (int j = i; j < size; j++) {
      s += v[j];
      p *= v_for_prod[j];
      minc = (std::min)(minc, numext::real(v[j]));
      maxc = (std::max)(maxc, numext::real(v[j]));
    }
    VERIFY_IS_MUCH_SMALLER_THAN(abs(s - v.tail(size - i).sum()), Scalar(1));
    VERIFY_IS_APPROX(p, v_for_prod.tail(size - i).prod());
    VERIFY_IS_APPROX(minc, v.real().tail(size - i).minCoeff());
    VERIFY_IS_APPROX(maxc, v.real().tail(size - i).maxCoeff());
  }

  for (int i = 0; i < size / 2; i++) {
    Scalar s(0), p(1);
    RealScalar minc(numext::real(v.coeff(i))), maxc(numext::real(v.coeff(i)));
    for (int j = i; j < size - i; j++) {
      s += v[j];
      p *= v_for_prod[j];
      minc = (std::min)(minc, numext::real(v[j]));
      maxc = (std::max)(maxc, numext::real(v[j]));
    }
    VERIFY_IS_MUCH_SMALLER_THAN(abs(s - v.segment(i, size - 2 * i).sum()), Scalar(1));
    VERIFY_IS_APPROX(p, v_for_prod.segment(i, size - 2 * i).prod());
    VERIFY_IS_APPROX(minc, v.real().segment(i, size - 2 * i).minCoeff());
    VERIFY_IS_APPROX(maxc, v.real().segment(i, size - 2 * i).maxCoeff());
  }

  // test empty objects
  VERIFY_IS_APPROX(v.head(0).sum(), Scalar(0));
  VERIFY_IS_APPROX(v.tail(0).prod(), Scalar(1));
  VERIFY_RAISES_ASSERT(v.head(0).mean());
  VERIFY_RAISES_ASSERT(v.head(0).minCoeff());
  VERIFY_RAISES_ASSERT(v.head(0).maxCoeff());
}

void boolRedux(Index rows, Index cols) {
  // Test boolean reductions: all(), any(), count()
  typedef Array<bool, Dynamic, Dynamic> BoolArray;

  // All-true
  BoolArray all_true = BoolArray::Constant(rows, cols, true);
  VERIFY(all_true.all());
  VERIFY(all_true.any());
  VERIFY_IS_EQUAL(all_true.count(), rows * cols);

  // All-false
  BoolArray all_false = BoolArray::Constant(rows, cols, false);
  if (rows > 0 && cols > 0) {
    VERIFY(!all_false.all());
    VERIFY(!all_false.any());
  }
  VERIFY_IS_EQUAL(all_false.count(), Index(0));

  // Mixed: set a checkerboard pattern
  BoolArray mixed(rows, cols);
  Index expected_count = 0;
  for (Index j = 0; j < cols; ++j)
    for (Index i = 0; i < rows; ++i) {
      mixed(i, j) = ((i + j) % 2 == 0);
      if (mixed(i, j)) expected_count++;
    }
  VERIFY_IS_EQUAL(mixed.count(), expected_count);
  if (rows > 0 && cols > 0) {
    VERIFY(mixed.any());
    VERIFY(mixed.all() == (expected_count == rows * cols));
  }

  // Partial reductions
  if (rows > 0 && cols > 0) {
    auto col_counts = mixed.colwise().count();
    for (Index k = 0; k < cols; ++k) VERIFY_IS_EQUAL(col_counts(k), mixed.col(k).count());
    auto row_counts = mixed.rowwise().count();
    for (Index k = 0; k < rows; ++k) VERIFY_IS_EQUAL(row_counts(k), mixed.row(k).count());
  }
}

// Test reductions at sizes that hit vectorization boundaries in Redux.h:
// LinearVectorizedTraversal with 2-way unrolled packet loop, scalar pre/post loops.
template <typename Scalar>
void redux_vec_boundary() {
  const Index PS = internal::packet_traits<Scalar>::size;
  // Critical sizes: around packet multiples and at 2-way unroll boundaries
  const Index sizes[] = {1,      PS - 1,     PS,         PS + 1, 2 * PS - 1, 2 * PS, 2 * PS + 1,
                         3 * PS, 3 * PS + 1, 4 * PS - 1, 4 * PS, 4 * PS + 1, 8 * PS, 8 * PS + 1};
  for (int si = 0; si < 14; ++si) {
    const Index n = sizes[si];
    if (n <= 0) continue;
    typedef Matrix<Scalar, Dynamic, 1> Vec;
    Vec v = Vec::Random(n);
    // For prod, use values near 1 to avoid underflow (float) or overflow (int).
    Vec v_for_prod = Vec::Ones(n) + Scalar(typename NumTraits<Scalar>::Real(0.2)) * v;
    // Reference: scalar loops
    Scalar ref_sum(0), ref_prod(1);
    typename NumTraits<Scalar>::Real ref_min = numext::real(v(0)), ref_max = numext::real(v(0));
    for (Index k = 0; k < n; ++k) {
      ref_sum += v(k);
      ref_prod *= v_for_prod(k);
      ref_min = (std::min)(ref_min, numext::real(v(k)));
      ref_max = (std::max)(ref_max, numext::real(v(k)));
    }
    VERIFY_IS_APPROX(v.sum(), ref_sum);
    VERIFY_IS_APPROX(v_for_prod.prod(), ref_prod);
    VERIFY_IS_APPROX(v.real().minCoeff(), ref_min);
    VERIFY_IS_APPROX(v.real().maxCoeff(), ref_max);
  }
}

// Test reductions on strided (non-contiguous) mapped data.
// This exercises SliceVectorizedTraversal or DefaultTraversal in Redux.h
// depending on stride and packet size.
template <typename Scalar>
void redux_strided() {
  const Index n = 64;
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  Vec data = Vec::Random(2 * n);
  // Map with inner stride of 2 — every other element
  Map<Vec, 0, InnerStride<2>> strided(data.data(), n);
  Scalar ref_sum(0);
  typename NumTraits<Scalar>::Real ref_min = numext::real(strided(0)), ref_max = numext::real(strided(0));
  for (Index k = 0; k < n; ++k) {
    ref_sum += strided(k);
    ref_min = (std::min)(ref_min, numext::real(strided(k)));
    ref_max = (std::max)(ref_max, numext::real(strided(k)));
  }
  VERIFY_IS_APPROX(strided.sum(), ref_sum);
  VERIFY_IS_APPROX(strided.real().minCoeff(), ref_min);
  VERIFY_IS_APPROX(strided.real().maxCoeff(), ref_max);

  // Also test reduction on a non-contiguous matrix block (SliceVectorizedTraversal)
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;
  Mat m = Mat::Random(16, 16);
  for (Index bsz = 1; bsz <= 8; bsz *= 2) {
    Scalar block_sum(0);
    for (Index j = 0; j < bsz; ++j)
      for (Index i = 0; i < bsz; ++i) block_sum += m(1 + i, 1 + j);
    VERIFY_IS_APPROX(m.block(1, 1, bsz, bsz).sum(), block_sum);
  }
}

struct keep_first_op {
  template <typename Scalar>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar operator()(const Scalar& a, const Scalar& /*b*/) const {
    return a;
  }
};

struct keep_last_op {
  template <typename Scalar>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar operator()(const Scalar& /*a*/, const Scalar& b) const {
    return b;
  }
};

template <typename Scalar>
struct marked_commutative_sum_op {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar operator()(const Scalar& a, const Scalar& b) const { return a + b; }
};

template <typename Scalar>
void redux_operand_order() {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;
  // Sizes straddle the small-size fallback, the ordered-tree cutoff, and its ragged tail.
  const Index sizes[] = {1, 2, 5, 7, 8, 9, 15, 16, 17, 23, 24, 31, 32, 33, 64, 129, 191, 192, 193, 250};
  keep_first_op first;
  keep_last_op last;
  for (int si = 0; si < 20; ++si) {
    const Index n = sizes[si];
    // Distinct values so that any reordering is observable.
    Vec v(n);
    for (Index i = 0; i < n; ++i) v.coeffRef(i) = Scalar(i + 1);
    // LinearTraversal: operands run 0 .. n-1.
    VERIFY_IS_EQUAL(v.redux(first), v.coeff(0));
    VERIFY_IS_EQUAL(v.redux(last), v.coeff(n - 1));

    // DefaultTraversal: a non-inner-panel block drops LinearAccessBit, and is traversed
    // outer-then-inner, so the operands run from (0,0) to (innerSize-1, outerSize-1).
    Mat m(n + 1, n + 1);
    for (Index c = 0; c < n + 1; ++c)
      for (Index r = 0; r < n + 1; ++r) m.coeffRef(r, c) = Scalar(c * (n + 1) + r + 1);
    Block<Mat, Dynamic, Dynamic, false> b(m, 1, 1, n, n);
    VERIFY_IS_EQUAL(b.redux(first), b.coeff(0, 0));
    VERIFY_IS_EQUAL(b.redux(last), b.coeff(n - 1, n - 1));
  }
}
template <typename Scalar>
void redux_commutative() {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;
  const Index sizes[] = {1, 2, 7, 8, 9, 15, 16, 17, 24, 31, 32, 33, 63, 64, 65, 129, 192, 250};
  marked_commutative_sum_op<Scalar> op;
  for (int si = 0; si < 18; ++si) {
    const Index n = sizes[si];
    Vec v(n);
    Scalar vref(0);
    for (Index i = 0; i < n; ++i) {
      v.coeffRef(i) = Scalar(i % 9 + 1);
      vref = vref + v.coeff(i);
    }
    VERIFY_IS_EQUAL(v.redux(op), vref);

    Mat m(n + 1, n + 1);
    m.setZero();
    Scalar bref(0);
    for (Index c = 1; c < n + 1; ++c)
      for (Index r = 1; r < n + 1; ++r) {
        m.coeffRef(r, c) = Scalar((r * 3 + c) % 7 + 1);
        bref = bref + m.coeff(r, c);
      }
    Block<Mat, Dynamic, Dynamic, false> b(m, 1, 1, n, n);
    VERIFY_IS_EQUAL(b.redux(op), bref);
  }
}
template <typename Scalar>
void redux_minmax_nan() {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;
  const Scalar kNaN = std::numeric_limits<Scalar>::quiet_NaN();
  const Index sizes[] = {2, 3, 7, 8, 9, 15, 16, 17, 31, 33, 64, 129, 250};
  for (int si = 0; si < 13; ++si) {
    const Index n = sizes[si];

    // Strided vector view: LinearTraversal without packet access.
    Vec backing(2 * n);
    backing.setZero();
    Map<Vec, 0, InnerStride<2>> v(backing.data(), n, InnerStride<2>(2));
    for (Index i = 0; i < n; ++i) v.coeffRef(i) = Scalar((i * 7) % 13) - Scalar(6);
    const Scalar refMin = v.minCoeff();  // reference from NaN-free data
    const Scalar refMax = v.maxCoeff();

    // NaN-free: all modes must agree with the reference.
    VERIFY_IS_EQUAL((v.template minCoeff<PropagateNaN>()), refMin);
    VERIFY_IS_EQUAL((v.template maxCoeff<PropagateNaN>()), refMax);
    VERIFY_IS_EQUAL((v.template minCoeff<PropagateNumbers>()), refMin);
    VERIFY_IS_EQUAL((v.template maxCoeff<PropagateNumbers>()), refMax);

    const Index nanPositions[] = {0, 1, 7, 8, n / 2, n - 9, n - 2, n - 1};
    for (int pi = 0; pi < 8; ++pi) {
      const Index p = nanPositions[pi];
      if (p < 0 || p >= n) continue;
      const Scalar saved = v.coeff(p);
      v.coeffRef(p) = kNaN;
      // Reference over the remaining numbers, computed serially.
      Scalar numMin = NumTraits<Scalar>::highest(), numMax = NumTraits<Scalar>::lowest();
      bool allNaN = true;
      for (Index i = 0; i < n; ++i) {
        if ((numext::isnan)(v.coeff(i))) continue;
        allNaN = false;
        numMin = numext::mini(numMin, v.coeff(i));
        numMax = numext::maxi(numMax, v.coeff(i));
      }
      VERIFY((numext::isnan)(v.template minCoeff<PropagateNaN>()));
      VERIFY((numext::isnan)(v.template maxCoeff<PropagateNaN>()));
      if (!allNaN) {
        VERIFY_IS_EQUAL((v.template minCoeff<PropagateNumbers>()), numMin);
        VERIFY_IS_EQUAL((v.template maxCoeff<PropagateNumbers>()), numMax);
      }
      v.coeffRef(p) = saved;
    }

    // Two NaNs in different lanes, and every coefficient NaN.
    if (n >= 2) {
      const Scalar s0 = v.coeff(0), s1 = v.coeff(n - 1);
      v.coeffRef(0) = kNaN;
      v.coeffRef(n - 1) = kNaN;
      VERIFY((numext::isnan)(v.template minCoeff<PropagateNaN>()));
      if (n > 2) {
        Scalar numMin = NumTraits<Scalar>::highest();
        for (Index i = 1; i < n - 1; ++i) numMin = numext::mini(numMin, v.coeff(i));
        VERIFY_IS_EQUAL((v.template minCoeff<PropagateNumbers>()), numMin);
      }
      v.coeffRef(0) = s0;
      v.coeffRef(n - 1) = s1;
    }
    v.setConstant(kNaN);
    VERIFY((numext::isnan)(v.template minCoeff<PropagateNaN>()));
    VERIFY((numext::isnan)(v.template minCoeff<PropagateNumbers>()));
    VERIFY((numext::isnan)(v.template maxCoeff<PropagateNaN>()));
    VERIFY((numext::isnan)(v.template maxCoeff<PropagateNumbers>()));

    // Strided matrix view: DefaultTraversal without packet access. NaN in the interior.
    if (n >= 3) {
      Mat mbacking(2 * n, n);
      mbacking.setZero();
      Map<Mat, 0, Stride<Dynamic, 2>> m(mbacking.data(), n, n, Stride<Dynamic, 2>(2 * n, 2));
      for (Index c = 0; c < n; ++c)
        for (Index r = 0; r < n; ++r) m.coeffRef(r, c) = Scalar((r * 5 + c * 3) % 11) - Scalar(5);
      const Scalar mrefMin = m.minCoeff();
      VERIFY_IS_EQUAL((m.template minCoeff<PropagateNaN>()), mrefMin);
      VERIFY_IS_EQUAL((m.template minCoeff<PropagateNumbers>()), mrefMin);
      m.coeffRef(n / 2, n / 2) = kNaN;
      Scalar numMin = NumTraits<Scalar>::highest(), numMax = NumTraits<Scalar>::lowest();
      for (Index c = 0; c < n; ++c)
        for (Index r = 0; r < n; ++r) {
          if ((numext::isnan)(m.coeff(r, c))) continue;
          numMin = numext::mini(numMin, m.coeff(r, c));
          numMax = numext::maxi(numMax, m.coeff(r, c));
        }
      VERIFY((numext::isnan)(m.template minCoeff<PropagateNaN>()));
      VERIFY((numext::isnan)(m.template maxCoeff<PropagateNaN>()));
      VERIFY_IS_EQUAL((m.template minCoeff<PropagateNumbers>()), numMin);
      VERIFY_IS_EQUAL((m.template maxCoeff<PropagateNumbers>()), numMax);
    }
  }
}
// A custom scalar whose comparison ignores its tag, so equivalent values are observably
// distinct. The generic std::min/std::max keep the first operand of a tie, and custom scalars
// are excluded from the min/max commutativity opt-in, so minCoeff()/maxCoeff() must return the
// first extremum in traversal order.
struct TaggedScalar {
  double v;
  int tag;
  TaggedScalar() : v(0), tag(0) {}
  TaggedScalar(double v_, int tag_) : v(v_), tag(tag_) {}
  bool operator<(const TaggedScalar& other) const { return v < other.v; }
};

namespace Eigen {
template <>
struct NumTraits<TaggedScalar> : GenericNumTraits<TaggedScalar> {};
}  // namespace Eigen

void redux_custom_scalar_min_ties() {
  typedef Matrix<TaggedScalar, Dynamic, 1> Vec;
  STATIC_CHECK(
      !(internal::functor_is_commutative<internal::scalar_min_op<TaggedScalar, TaggedScalar, PropagateFast>>::value));
  // Sizes on both sides of the ordered-tree cutoff, extrema in different unroll regions.
  const Index sizes[] = {10, 33, 250, 1000};
  for (int si = 0; si < 4; ++si) {
    const Index n = sizes[si];
    Vec x(n);
    for (Index i = 0; i < n; ++i) x.coeffRef(i) = TaggedScalar(double(i % 17), int(i));
    // Two equal minima (v == -1): the first in traversal order must win.
    x.coeffRef(2) = TaggedScalar(-1.0, 2);
    x.coeffRef(n - 2) = TaggedScalar(-1.0, int(n - 2));
    VERIFY_IS_EQUAL(x.minCoeff().tag, 2);
    // Two equal maxima (v == 100).
    x.coeffRef(3) = TaggedScalar(100.0, 3);
    x.coeffRef(n - 3) = TaggedScalar(100.0, int(n - 3));
    VERIFY_IS_EQUAL(x.maxCoeff().tag, 3);
    // All coefficients equivalent: the very first must win.
    Vec y(n);
    for (Index i = 0; i < n; ++i) y.coeffRef(i) = TaggedScalar(5.0, int(i));
    VERIFY_IS_EQUAL(y.minCoeff().tag, 0);
    VERIFY_IS_EQUAL(y.maxCoeff().tag, 0);
  }
}
template <typename Scalar>
void redux_runtime_contiguous() {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  const Index sizes[] = {1, 2, 7, 8, 9, 16, 17, 64, 255, 256, 257};
  for (int si = 0; si < 11; ++si) {
    const Index n = sizes[si];
    Vec data = Vec::Random(2 * n);
    Vec data_for_prod = Vec::Ones(2 * n) + Scalar(RealScalar(0.2)) * data;
    Scalar rs(0), rp(1);
    RealScalar rmin = numext::real(data(0)), rmax = numext::real(data(0)), rsqn(0);
    for (Index k = 0; k < n; ++k) {
      rs += data(k);
      rp *= data_for_prod(k);
      rmin = (std::min)(rmin, numext::real(data(k)));
      rmax = (std::max)(rmax, numext::real(data(k)));
      rsqn += numext::abs2(data(k));
    }

    // (a) dynamic-inner-stride Map with runtime stride 1 -> fast path.
    Map<Vec, 0, InnerStride<Dynamic>> m(data.data(), n, InnerStride<Dynamic>(1));
    VERIFY_IS_APPROX(m.sum(), rs);
    VERIFY_IS_APPROX(m.mean(), rs / Scalar(RealScalar(n)));
    VERIFY_IS_APPROX(m.real().minCoeff(), rmin);
    VERIFY_IS_APPROX(m.real().maxCoeff(), rmax);
    VERIFY_IS_APPROX(m.squaredNorm(), rsqn);  // squaredNorm/norm fast path (squared_norm_impl)
    VERIFY_IS_APPROX(m.norm(), numext::sqrt(rsqn));
    Map<Vec, 0, InnerStride<Dynamic>> mp(data_for_prod.data(), n, InnerStride<Dynamic>(1));
    VERIFY_IS_APPROX(mp.prod(), rp);

    // (a') an *aligned* dynamic-inner-stride Map with runtime stride 1 exercises the aligned-load
    // fast path: the runtime-contiguous Map inherits evaluator<Derived>::Alignment, so an aligned
    // source must reduce correctly and must not trip the Map alignment assertion. The data buffer
    // comes from an Eigen Vec, which is allocated AlignedMax-aligned.
    Map<Vec, AlignedMax, InnerStride<Dynamic>> ma(data.data(), n, InnerStride<Dynamic>(1));
    VERIFY_IS_APPROX(ma.sum(), rs);
    VERIFY_IS_APPROX(ma.real().minCoeff(), rmin);
    VERIFY_IS_APPROX(ma.real().maxCoeff(), rmax);
    VERIFY_IS_APPROX(ma.squaredNorm(), rsqn);
    VERIFY_IS_APPROX(ma.norm(), numext::sqrt(rsqn));

    // (b) a row of a 1xN dynamic matrix is contiguous at runtime (inner stride == 1).
    Mat r = Mat::Random(1, n);
    Scalar row_sum(0);
    RealScalar row_sqn(0);
    for (Index j = 0; j < n; ++j) {
      row_sum += r(0, j);
      row_sqn += numext::abs2(r(0, j));
    }
    VERIFY_IS_APPROX(r.row(0).sum(), row_sum);
    VERIFY_IS_APPROX(r.row(0).squaredNorm(), row_sqn);

    // (c) fully-packed dynamic-stride matrix Ref -> fast path; must match the dense result.
    if (n >= 2) {
      Mat M = Mat::Random(n, 3);
      Ref<Mat, 0, Stride<Dynamic, Dynamic>> mref(M);
      VERIFY_IS_APPROX(mref.sum(), M.sum());
      VERIFY_IS_APPROX(mref.real().minCoeff(), M.real().minCoeff());
    }

    // (d) fallback: a genuinely strided (stride 2) dynamic Map must still reduce correctly.
    Map<Vec, 0, InnerStride<Dynamic>> m2(data.data(), n, InnerStride<Dynamic>(2));
    Scalar rs2(0);
    RealScalar rsqn2(0);
    for (Index k = 0; k < n; ++k) {
      rs2 += data(2 * k);
      rsqn2 += numext::abs2(data(2 * k));
    }
    VERIFY_IS_APPROX(m2.sum(), rs2);
    VERIFY_IS_APPROX(m2.squaredNorm(), rsqn2);
  }
}

#endif  // EIGEN_TEST_REDUX_HELPERS_H
