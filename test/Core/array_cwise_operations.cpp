// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// array_cwise split: generic operations, comparisons, and min/max for float types.

#include "array_cwise_helpers.h"

EIGEN_DECLARE_TEST(array_cwise_operations) {
  for (int i = 0; i < g_repeat; i++) {
    array_generic(Array<float, 1, 1>());
    array_generic(Array22f());
    array_generic(
        ArrayXXf(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
  }
  for (int i = 0; i < g_repeat; i++) {
    comparisons(Array<float, 1, 1>());
    comparisons(Array22f());
    comparisons(ArrayXXf(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
  }
  for (int i = 0; i < g_repeat; i++) {
    min_max(Array<float, 1, 1>());
    min_max(Array22f());
    min_max(ArrayXXf(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
  }

  VERIFY((internal::is_same<internal::global_math_functions_filtering_base<int>::type, int>::value));
  VERIFY((internal::is_same<internal::global_math_functions_filtering_base<float>::type, float>::value));
  VERIFY((internal::is_same<internal::global_math_functions_filtering_base<Array2i>::type, ArrayBase<Array2i>>::value));
  typedef CwiseUnaryOp<internal::scalar_abs_op<double>, ArrayXd> Xpr;
  VERIFY((internal::is_same<internal::global_math_functions_filtering_base<Xpr>::type, ArrayBase<Xpr>>::value));
}
