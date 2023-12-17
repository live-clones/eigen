// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cstdlib>
#include "main.h"

template <typename Scalar>
Scalar check_in_range(Scalar x, Scalar y) {
  Scalar r = internal::random<Scalar>(x, y);
  VERIFY(r >= x);
  if (y >= x) {
    VERIFY(r <= y);
  }
  return r;
}

template <typename Scalar>
void check_all_in_range(Scalar x, Scalar y) {
  Array<int, 1, Dynamic> mask(y - x + 1);
  mask.fill(0);
  long n = (y - x + 1) * 32;
  for (long k = 0; k < n; ++k) {
    mask(check_in_range(x, y) - x)++;
  }
  for (Index i = 0; i < mask.size(); ++i)
    if (mask(i) == 0) std::cout << "WARNING: value " << x + i << " not reached." << std::endl;
  VERIFY((mask > 0).all());
}

template <typename Scalar, typename EnableIf = void>
class HistogramHelper {
 public:
  HistogramHelper(int nbins) : HistogramHelper(Scalar(-1), Scalar(1), nbins) {}
  HistogramHelper(Scalar lower, Scalar upper, int nbins)
      : lower_{lower}, upper_{upper}, num_bins_{nbins}, bin_width_{(upper - lower) / Scalar(nbins)} {}
  int bin(Scalar v) { return std::min<int>((v - lower_) / bin_width_, num_bins_ - 1); }

  double uniform_bin_probability(int bin) {
    double range = static_cast<double>(upper_) - static_cast<double>(lower_);
    if (bin < num_bins_ - 1) {
      return static_cast<double>(bin_width_) / range;
    }
    return static_cast<double>(upper_ - (lower_ + Scalar(bin) * bin_width_)) / range;
  }

 private:
  Scalar lower_;
  Scalar upper_;
  int num_bins_;
  Scalar bin_width_;
};

template <typename Scalar>
class HistogramHelper<Scalar, std::enable_if_t<Eigen::NumTraits<Scalar>::IsInteger>> {
 public:
  using RangeType = typename Eigen::internal::make_unsigned<Scalar>::type;
  HistogramHelper(int nbins)
      : HistogramHelper(Eigen::NumTraits<Scalar>::lowest(), Eigen::NumTraits<Scalar>::highest(), nbins) {}
  HistogramHelper(Scalar lower, Scalar upper, int nbins)
      : lower_{lower}, upper_{upper}, num_bins_{nbins}, bin_width_{bin_width(lower, upper, nbins)} {}

  int bin(Scalar v) { return static_cast<int>(RangeType(v - lower_) / bin_width_); }

  double uniform_bin_probability(int bin) {
    // Avoid overflow in computing range.
    double range = static_cast<double>(RangeType(upper_ - lower_)) + 1.0;
    if (bin < num_bins_ - 1) {
      return static_cast<double>(bin_width_) / range;
    }
    return static_cast<double>(upper_ - (lower_ + bin * bin_width_) + 1) / range;
  }

 private:
  static constexpr Scalar bin_width(Scalar lower, Scalar upper, int nbins) {
    // Avoid overflow in computing the full range.
    return RangeType(upper - nbins - lower + 1) / nbins + 1;
  }

  Scalar lower_;
  Scalar upper_;
  int num_bins_;
  Scalar bin_width_;
};

template <typename Scalar>
void check_histogram(Scalar x, Scalar y, int bins) {
  Eigen::VectorXd hist = Eigen::VectorXd::Zero(bins);
  HistogramHelper<Scalar> hist_helper(x, y, bins);
  int64_t n = static_cast<int64_t>(bins) * 10000;  // Approx 10000 per bin.
  for (int64_t k = 0; k < n; ++k) {
    Scalar r = check_in_range(x, y);
    int bin = hist_helper.bin(r);
    hist(bin)++;
  }
  // Normalize bins by probability.
  for (int i = 0; i < bins; ++i) {
    hist(i) = hist(i) / n / hist_helper.uniform_bin_probability(i);
  }
  VERIFY(((hist.array() - 1.0).abs() < 0.05).all());
}

template <typename Scalar>
void check_histogram(int bins) {
  Eigen::VectorXd hist = Eigen::VectorXd::Zero(bins);
  HistogramHelper<Scalar> hist_helper(bins);
  int64_t n = static_cast<int64_t>(bins) * 10000;  // Approx 10000 per bin.
  for (int64_t k = 0; k < n; ++k) {
    Scalar r = Eigen::internal::random<Scalar>();
    int bin = hist_helper.bin(r);
    hist(bin)++;
  }
  // Normalize bins by probability.
  for (int i = 0; i < bins; ++i) {
    hist(i) = hist(i) / n / hist_helper.uniform_bin_probability(i);
  }
  VERIFY(((hist.array() - 1.0).abs() < 0.05).all());
}

EIGEN_DECLARE_TEST(rand) {
  long long_ref = NumTraits<long>::highest() / 10;
  // the minimum guarantees that these conversions are safe
  signed char char_offset = static_cast<signed char>((std::min)(g_repeat, 64));
  signed short short_offset = static_cast<signed short>((std::min)(g_repeat, 8000));
  EIGEN_UNUSED_VARIABLE(long_ref);
  EIGEN_UNUSED_VARIABLE(char_offset);
  EIGEN_UNUSED_VARIABLE(short_offset);

  for (int i = 0; i < g_repeat * 10000; i++) {
    CALL_SUBTEST_1(check_in_range<float>(10.0f, 11.0f));
    CALL_SUBTEST_1(check_in_range<float>(1.24234523f, 1.24234523f));
    CALL_SUBTEST_1(check_in_range<float>(-1.0f, 1.0f));
    CALL_SUBTEST_1(check_in_range<float>(-1432.2352f, -1432.2352f));

    CALL_SUBTEST_2(check_in_range<double>(10.0, 11.0));
    CALL_SUBTEST_2(check_in_range<double>(1.24234523, 1.24234523));
    CALL_SUBTEST_2(check_in_range<double>(-1.0, 1.0));
    CALL_SUBTEST_2(check_in_range<double>(-1432.2352, -1432.2352));

    CALL_SUBTEST_2(check_in_range<long double>(10.0L, 11.0L));
    CALL_SUBTEST_2(check_in_range<long double>(1.24234523L, 1.24234523L));
    CALL_SUBTEST_2(check_in_range<long double>(-1.0L, 1.0L));
    CALL_SUBTEST_2(check_in_range<long double>(-1432.2352L, -1432.2352L));

    CALL_SUBTEST_3(check_in_range<half>(half(10.0f), half(11.0f)));
    CALL_SUBTEST_3(check_in_range<half>(half(1.24234523f), half(1.24234523f)));
    CALL_SUBTEST_3(check_in_range<half>(half(-1.0f), half(1.0f)));
    CALL_SUBTEST_3(check_in_range<half>(half(-1432.2352f), half(-1432.2352f)));

    CALL_SUBTEST_4(check_in_range<bfloat16>(bfloat16(10.0f), bfloat16(11.0f)));
    CALL_SUBTEST_4(check_in_range<bfloat16>(bfloat16(1.24234523f), bfloat16(1.24234523f)));
    CALL_SUBTEST_4(check_in_range<bfloat16>(bfloat16(-1.0f), bfloat16(1.0f)));
    CALL_SUBTEST_4(check_in_range<bfloat16>(bfloat16(-1432.2352f), bfloat16(-1432.2352f)));

    CALL_SUBTEST_5(check_in_range<int>(0, -1));
    CALL_SUBTEST_5(check_in_range<short>(0, -1));
    CALL_SUBTEST_5(check_in_range<long>(0, -1));
    CALL_SUBTEST_5(check_in_range<int>(-673456, 673456));
    CALL_SUBTEST_5(check_in_range<int>(-RAND_MAX + 10, RAND_MAX - 10));
    CALL_SUBTEST_5(check_in_range<short>(-24345, 24345));
    CALL_SUBTEST_5(check_in_range<long>(-long_ref, long_ref));
  }

  CALL_SUBTEST_6(check_all_in_range<signed char>(11, 11));
  CALL_SUBTEST_6(check_all_in_range<signed char>(11, 11 + char_offset));
  CALL_SUBTEST_6(check_all_in_range<signed char>(-5, 5));
  CALL_SUBTEST_6(check_all_in_range<signed char>(-11 - char_offset, -11));
  CALL_SUBTEST_6(check_all_in_range<signed char>(-126, -126 + char_offset));
  CALL_SUBTEST_6(check_all_in_range<signed char>(126 - char_offset, 126));
  CALL_SUBTEST_6(check_all_in_range<signed char>(-126, 126));

  CALL_SUBTEST_7(check_all_in_range<short>(11, 11));
  CALL_SUBTEST_7(check_all_in_range<short>(11, 11 + short_offset));
  CALL_SUBTEST_7(check_all_in_range<short>(-5, 5));
  CALL_SUBTEST_7(check_all_in_range<short>(-11 - short_offset, -11));
  CALL_SUBTEST_7(check_all_in_range<short>(-24345, -24345 + short_offset));
  CALL_SUBTEST_7(check_all_in_range<short>(24345, 24345 + short_offset));

  CALL_SUBTEST_8(check_all_in_range<int>(11, 11));
  CALL_SUBTEST_8(check_all_in_range<int>(11, 11 + g_repeat));
  CALL_SUBTEST_8(check_all_in_range<int>(-5, 5));
  CALL_SUBTEST_8(check_all_in_range<int>(-11 - g_repeat, -11));
  CALL_SUBTEST_8(check_all_in_range<int>(-673456, -673456 + g_repeat));
  CALL_SUBTEST_8(check_all_in_range<int>(673456, 673456 + g_repeat));

  CALL_SUBTEST_9(check_all_in_range<long>(11, 11));
  CALL_SUBTEST_9(check_all_in_range<long>(11, 11 + g_repeat));
  CALL_SUBTEST_9(check_all_in_range<long>(-5, 5));
  CALL_SUBTEST_9(check_all_in_range<long>(-11 - g_repeat, -11));
  CALL_SUBTEST_9(check_all_in_range<long>(-long_ref, -long_ref + g_repeat));
  CALL_SUBTEST_9(check_all_in_range<long>(long_ref, long_ref + g_repeat));

  CALL_SUBTEST_10(check_histogram<int>(-5, 5, 11));
  int bins = 100;
  EIGEN_UNUSED_VARIABLE(bins)
  CALL_SUBTEST_10(check_histogram<int>(-3333, -3333 + bins * (3333 / bins) - 1, bins));
  bins = 1000;
  CALL_SUBTEST_10(check_histogram<int>(-RAND_MAX + 10, -RAND_MAX + 10 + bins * (RAND_MAX / bins) - 1, bins));
  CALL_SUBTEST_10(
      check_histogram<int>(-RAND_MAX + 10, -int64_t(RAND_MAX) + 10 + bins * (2 * int64_t(RAND_MAX) / bins) - 1, bins));

  CALL_SUBTEST_11(
      check_histogram<uint8_t>(Eigen::NumTraits<uint8_t>::lowest(), Eigen::NumTraits<uint8_t>::highest(), /*bins=*/16));
  CALL_SUBTEST_11(check_histogram<uint16_t>(Eigen::NumTraits<uint16_t>::lowest(), Eigen::NumTraits<uint16_t>::highest(),
                                            /*bins=*/1024));
  CALL_SUBTEST_11(check_histogram<uint32_t>(Eigen::NumTraits<uint32_t>::lowest(), Eigen::NumTraits<uint32_t>::highest(),
                                            /*bins=*/1024));
  CALL_SUBTEST_11(check_histogram<uint64_t>(Eigen::NumTraits<uint64_t>::lowest(), Eigen::NumTraits<uint64_t>::highest(),
                                            /*bins=*/1024));

  CALL_SUBTEST_12(
      check_histogram<int8_t>(Eigen::NumTraits<int8_t>::lowest(), Eigen::NumTraits<int8_t>::highest(), /*bins=*/16));
  CALL_SUBTEST_12(check_histogram<int16_t>(Eigen::NumTraits<int16_t>::lowest(), Eigen::NumTraits<int16_t>::highest(),
                                           /*bins=*/1024));
  CALL_SUBTEST_12(check_histogram<int32_t>(Eigen::NumTraits<int32_t>::lowest(), Eigen::NumTraits<int32_t>::highest(),
                                           /*bins=*/1024));
  CALL_SUBTEST_12(check_histogram<int64_t>(Eigen::NumTraits<int64_t>::lowest(), Eigen::NumTraits<int64_t>::highest(),
                                           /*bins=*/1024));

  CALL_SUBTEST_13(check_histogram<uint8_t>(/*bins=*/16));
  CALL_SUBTEST_13(check_histogram<uint16_t>(/*bins=*/1024));
  CALL_SUBTEST_13(check_histogram<uint32_t>(/*bins=*/1024));
  CALL_SUBTEST_13(check_histogram<uint64_t>(/*bins=*/1024));

  CALL_SUBTEST_14(check_histogram<int8_t>(/*bins=*/16));
  CALL_SUBTEST_14(check_histogram<int16_t>(/*bins=*/1024));
  CALL_SUBTEST_14(check_histogram<int32_t>(/*bins=*/1024));
  CALL_SUBTEST_14(check_histogram<int64_t>(/*bins=*/1024));

  CALL_SUBTEST_15(check_histogram<float>(-10.0f, 10.0f, /*bins=*/1024));
  CALL_SUBTEST_15(check_histogram<double>(-10.0, 10.0, /*bins=*/1024));
  CALL_SUBTEST_15(check_histogram<long double>(-10.0L, 10.0L, /*bins=*/1024));
  CALL_SUBTEST_15(check_histogram<half>(half(-10.0f), half(10.0f), /*bins=*/256));
  CALL_SUBTEST_15(check_histogram<bfloat16>(bfloat16(-10.0f), bfloat16(10.0f), /*bins=*/32));

  CALL_SUBTEST_16(check_histogram<float>(/*bins=*/1024));
  CALL_SUBTEST_16(check_histogram<double>(/*bins=*/1024));
  CALL_SUBTEST_16(check_histogram<long double>(/*bins=*/1024));
  CALL_SUBTEST_16(check_histogram<half>(/*bins=*/256));
  CALL_SUBTEST_16(check_histogram<bfloat16>(/*bins=*/32));
}
