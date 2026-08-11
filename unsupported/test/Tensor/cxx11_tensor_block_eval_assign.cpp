// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "cxx11_tensor_block_eval.h"

TEST(TensorBlockEvalTest, Assign) {
  // clang-format off
  CALL_SUBTESTS_DIMS_LAYOUTS_TYPES(7, test_assign_to_tensor);
  CALL_SUBTESTS_DIMS_LAYOUTS_TYPES(7, test_assign_to_tensor_reshape);
  CALL_SUBTESTS_DIMS_LAYOUTS_TYPES(7, test_assign_to_tensor_chipping);
  CALL_SUBTESTS_DIMS_LAYOUTS_TYPES(8, test_assign_to_tensor_slice);
  CALL_SUBTESTS_DIMS_LAYOUTS_TYPES(8, test_assign_to_tensor_shuffle);
  // clang-format on
}
