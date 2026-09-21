// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0
#include "main.h"
#include <Eigen/Core>

#ifndef EIGEN_VECTORIZE_SME
#error "vector_sme requires an SME2 target"
#endif

template <typename Lhs, typename Rhs>
void sme_check_dot(const Lhs& lhs, const Rhs& rhs) {
  using Scalar = typename Lhs::Scalar;
  long double expected = 0, magnitude = 0;
  for (Index i = 0; i < lhs.size(); ++i) {
    const long double term = static_cast<long double>(lhs[i]) * static_cast<long double>(rhs[i]);
    expected += term;
    magnitude += numext::abs(term);
  }
  // gamma_(2*n+4) bounds multiplication and accumulation in both the kernel and reference.
  const long double rounding =
      (2 * static_cast<long double>(lhs.size()) + 4) * static_cast<long double>(NumTraits<Scalar>::epsilon());
  const long double bound = rounding / (1 - rounding) * magnitude;
  const Scalar actual = lhs.dot(rhs);
  VERIFY(rounding < 1 && (numext::isfinite)(bound));
  VERIFY((numext::isfinite)(actual));
  VERIFY(numext::abs(static_cast<long double>(actual) - expected) <= bound);
}

template <typename Scalar>
void sme_vectors() {
  using Vec = Vector<Scalar, Dynamic>;
  using Row = Matrix<Scalar, 1, Dynamic>;
  using Strided = Map<Vec, 0, InnerStride<Dynamic>>;
  STATIC_CHECK((internal::sme_dot_supported<Vec, Row>::value));
  STATIC_CHECK((internal::sme_dot_supported<Strided, Vec>::value));
  STATIC_CHECK((!internal::sme_dot_supported<decltype(std::declval<Vec>() + std::declval<Vec>()), Vec>::value));
  STATIC_CHECK((!internal::sme_vector_scalar<std::complex<Scalar>>::value));
  STATIC_CHECK((!internal::sme_vector_scalar<int>::value));

  std::vector<Index> sizes;
  for (Index n = 0; n <= 260; ++n) {
    sizes.push_back(n);
    sizes.push_back(4096 + n);
  }
  for (Index n : {511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096, 4097, 16385}) sizes.push_back(n);
  for (Index n : sizes) {
    Vec x = Vec::Ones(n), storage = Vec::Ones(n + 2);
    VERIFY_IS_EQUAL(x.dot(storage.head(n)), Scalar(n));
    x.setRandom();
    storage.setRandom();
    const Vec before = storage;
    auto y = storage.segment(1, n);
    sme_check_dot(x, y);
    sme_check_dot(x.transpose(), y);
    for (int side = 0; side < 2; ++side) {
      storage = before;
      if (side == 0)
        y += Scalar(0.75) * x;
      else
        y += x * Scalar(0.75);
      for (Index i = 0; i < n; ++i) {
        const long double expected = static_cast<long double>(before[i + 1]) + 0.75L * static_cast<long double>(x[i]);
        const long double magnitude =
            numext::abs(static_cast<long double>(before[i + 1])) + 0.75L * numext::abs(static_cast<long double>(x[i]));
        VERIFY(numext::abs(static_cast<long double>(y[i]) - expected) <=
               4 * static_cast<long double>(NumTraits<Scalar>::epsilon()) * magnitude);
      }
      VERIFY_IS_EQUAL(storage[0], before[0]);
      VERIFY_IS_EQUAL(storage[n + 1], before[n + 1]);
    }
    Vec alias = x;
    alias += Scalar(0.75) * alias;
    for (Index i = 0; i < n; ++i) {
      const long double expected = static_cast<long double>(x[i]) * 1.75L;
      VERIFY(numext::abs(static_cast<long double>(alias[i]) - expected) <=
             4 * static_cast<long double>(NumTraits<Scalar>::epsilon()) * numext::abs(expected));
    }
    Strided contiguous(x.data(), n, InnerStride<Dynamic>(1));
    sme_check_dot(contiguous, y);
  }
  Vec x = Vec::Random(16386), y = Vec::Random(8193), expected = x;
  Strided strided(x.data(), y.size(), InnerStride<Dynamic>(2));
  Strided negative(x.data() + x.size() - 2, y.size(), InnerStride<Dynamic>(-2));
  sme_check_dot(strided, y);
  sme_check_dot(negative, y);
  for (Index i = 0; i < y.size(); ++i) expected[2 * i] += Scalar(0.75) * y[i];
  strided += Scalar(0.75) * y;
  VERIFY_IS_APPROX(x, expected);
  sme_check_dot(-y, strided);
  sme_check_dot(y + y, strided);
  sme_check_dot(y.reverse(), strided);
  VERIFY_RAISES_ASSERT(y.dot(x));

  for (int pattern = 0; pattern < 3; ++pattern) {
    Vec a = Vec::Random(65537), b = Vec::Random(65537);
    for (Index i = 0; i < a.size(); ++i) {
      if (pattern == 0) b[i] = (i & 1) ? -a[i] : a[i];
      if (pattern == 1) a[i] = numext::ldexp(a[i], int(i % 81) - 40);
      if (pattern == 2) {
        a[i] = Scalar(1);
        b[i] = i % 3 == 0   ? Scalar(1) / NumTraits<Scalar>::epsilon()
               : i % 3 == 1 ? Scalar(1)
                            : -Scalar(1) / NumTraits<Scalar>::epsilon();
      }
    }
    sme_check_dot(a, b);
  }
  for (Index n : {524287, 524288, 524289, 1048575, 1048576, 1048577}) {
    Vec a = Vec::Ones(n), b = Vec::Ones(n);
    VERIFY_IS_EQUAL(a.dot(b), Scalar(n));
    b += Scalar(0.75) * a;
    VERIFY((b.array() == Scalar(1.75)).all());
  }
}

template <typename Scalar>
void sme_vector_cache_sizes() {
  using Vec = Vector<Scalar, Dynamic>;
  const std::ptrdiff_t old_l1 = l1CacheSize(), old_l2 = l2CacheSize(), old_l3 = l3CacheSize();
  const Index n = 4097;
  const std::ptrdiff_t bytes = n * sizeof(Scalar);
  const Vec x = Vec::Ones(n);
  Vec y(n);
  for (std::ptrdiff_t l2 : {bytes, bytes - 1, std::ptrdiff_t(0), std::ptrdiff_t(-1), bytes + 1}) {
    setCpuCacheSizes(old_l1, l2, old_l3);
    VERIFY_IS_EQUAL(internal::sme_vector_fits_budget<Scalar>(n), l2 >= bytes);
    VERIFY(!internal::sme_vector_fits_budget<Scalar>((std::numeric_limits<Index>::max)()));
    VERIFY_IS_EQUAL(x.dot(x), Scalar(n));
    y.setOnes();
    y += Scalar(0.75) * x;
    VERIFY((y.array() == Scalar(1.75)).all());
    y.setOnes();
    y += x * Scalar(0.75);
    VERIFY((y.array() == Scalar(1.75)).all());
  }
  setCpuCacheSizes(old_l1, old_l2, old_l3);
}

template <typename Scalar>
void sme_vector_scalar_factors() {
  using Vec = Vector<Scalar, Dynamic>;
  const Index n = 8193;
  Vec x = Vec::Constant(n, Scalar(0.25)), y = Vec::Zero(n);
  y += x.cwiseProduct(x);
  VERIFY((y.array() == Scalar(0.0625)).all());
  y.setZero();
  y += Vec::Constant(n, Scalar(0.75)).cwiseProduct(x);
  VERIFY((y.array() == Scalar(0.1875)).all());

  // Folding the two scale factors would overflow even though every output is finite.
  const Scalar large = (std::numeric_limits<Scalar>::max)() / Scalar(2);
  y.setZero();
  y += large * (Scalar(4) * x);
  VERIFY((y.array() == large).all());
  y.setZero();
  y += (x * Scalar(4)) * large;
  VERIFY((y.array() == large).all());

  Array<Scalar, Dynamic, 1> a = x.array(), b = Array<Scalar, Dynamic, 1>::Zero(n);
  b += Scalar(0.75) * a;
  VERIFY((b == Scalar(0.1875)).all());
  b.setZero();
  b += a * Scalar(0.75);
  VERIFY((b == Scalar(0.1875)).all());
}

template <typename Scalar>
void sme_matrix_vectors() {
  using Vec = Vector<Scalar, Dynamic>;
  using Mat = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;
  for (Index rows : {127, 128, 129, 255, 256, 257, 513}) {
    for (Index cols : {3, 4, 5, 17, 65, 129}) {
      for (Index padding : {0, 3}) {
        Mat storage = Mat::Random(rows + padding, cols);
        auto a = storage.topRows(rows);
        Vec x = Vec::Random(cols), original = Vec::Random(rows), result(rows);
        for (Scalar alpha : {Scalar(0), Scalar(0.75), Scalar(-1)}) {
          result = original;
          result.noalias() += alpha * a * x;
          for (Index i = 0; i < rows; ++i) {
            long double expected = static_cast<long double>(original[i]), magnitude = numext::abs(expected);
            for (Index j = 0; j < cols; ++j) {
              const long double term =
                  static_cast<long double>(alpha) * static_cast<long double>(a(i, j)) * static_cast<long double>(x[j]);
              expected += term;
              magnitude += numext::abs(term);
            }
            const long double bound =
                8 * (cols + 1) * static_cast<long double>(NumTraits<Scalar>::epsilon()) * magnitude;
            VERIFY((numext::isfinite)(bound));
            VERIFY(numext::abs(static_cast<long double>(result[i]) - expected) <= bound);
          }
        }
        Vec strided_storage = Vec::Random(2 * rows), untouched = strided_storage;
        Map<Vec, 0, InnerStride<2>> strided(strided_storage.data(), rows);
        Vec reference = strided;
        reference.noalias() += a * x;
        strided.noalias() += a * x;
        VERIFY_IS_APPROX(strided, reference);
        for (Index i = 0; i < rows; ++i) VERIFY_IS_EQUAL(strided_storage[2 * i + 1], untouched[2 * i + 1]);
      }
    }
  }
}

template <typename Scalar>
void sme_vector_special_values() {
  using Vec = Vector<Scalar, Dynamic>;
  using Mat = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;
  const Scalar inf = NumTraits<Scalar>::infinity(), nan = NumTraits<Scalar>::quiet_NaN();
  const Scalar tiny = (std::numeric_limits<Scalar>::denorm_min)();
  for (Index pos : {0, 7, 31, 63, 127, 255, 256, 4096}) {
    Vec x = Vec::Zero(4097), y = Vec::Ones(4097), result(4097);
    for (Scalar special : {inf, -inf, nan, tiny, -tiny, Scalar(-0.0)}) {
      x[pos] = special;
      const Scalar actual = x.dot(y);
      if ((numext::isnan)(special))
        VERIFY((numext::isnan)(actual));
      else
        VERIFY_IS_EQUAL(actual, special);
      result.setZero();
      result += Scalar(1) * x;
      for (Index i = 0; i < result.size(); ++i) {
        const Scalar expected = x[i] + Scalar(0);
        if ((numext::isnan)(expected))
          VERIFY((numext::isnan)(result[i]));
        else {
          VERIFY_IS_EQUAL(result[i], expected);
          VERIFY_IS_EQUAL(std::signbit(result[i]), std::signbit(expected));
        }
      }
    }
  }
  for (Scalar special : {inf, -inf, nan, tiny, -tiny}) {
    Mat a = Mat::Zero(257, 4);
    Vec x = Vec::Ones(4), y = Vec::Zero(257);
    a(0, 0) = a(256, 3) = special;
    y.noalias() += a * x;
    for (Index i = 0; i < y.size(); ++i) {
      const Scalar expected = i == 0 || i == 256 ? special : Scalar(0);
      if ((numext::isnan)(expected))
        VERIFY((numext::isnan)(y[i]));
      else
        VERIFY_IS_EQUAL(y[i], expected);
    }
    y.setOnes();
    y.noalias() += Scalar(0) * a * x;
    VERIFY((y.array() == Scalar(1)).all());
  }
  Vec x = Vec::Zero(4097), y = Vec::Zero(4097);
  x[0] = inf;
  VERIFY((numext::isnan)(x.dot(y)));
}

EIGEN_DECLARE_TEST(vector_sme) {
  CALL_SUBTEST_1(sme_vector_cache_sizes<float>());
  CALL_SUBTEST_1(sme_vector_scalar_factors<float>());
  CALL_SUBTEST_1(sme_vectors<float>());
  CALL_SUBTEST_1(sme_matrix_vectors<float>());
  CALL_SUBTEST_1(sme_vector_special_values<float>());
#ifdef EIGEN_VECTORIZE_SME_F64F64
  CALL_SUBTEST_2(sme_vector_cache_sizes<double>());
  CALL_SUBTEST_2(sme_vector_scalar_factors<double>());
  CALL_SUBTEST_2(sme_vectors<double>());
  CALL_SUBTEST_2(sme_matrix_vectors<double>());
  CALL_SUBTEST_2(sme_vector_special_values<double>());
#endif
}
