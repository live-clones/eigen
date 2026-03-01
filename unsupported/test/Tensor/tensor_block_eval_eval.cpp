// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "tensor_block_eval.h"

TEST(TensorBlockEvalTest, Eval) {
  // clang-format off
  CALL_SUBTESTS_DIMS_LAYOUTS_TYPES(1, test_eval_tensor_block);
  CALL_SUBTESTS_DIMS_LAYOUTS_TYPES(1, test_eval_tensor_binary_expr_block);
  CALL_SUBTESTS_DIMS_LAYOUTS(1, test_eval_tensor_unary_expr_block);
  CALL_SUBTESTS_DIMS_LAYOUTS(2, test_eval_tensor_binary_with_unary_expr_block);
  CALL_SUBTESTS_DIMS_LAYOUTS_TYPES(2, test_eval_tensor_broadcast);
  CALL_SUBTESTS_DIMS_LAYOUTS_TYPES(2, test_eval_tensor_reshape);
  CALL_SUBTESTS_DIMS_LAYOUTS_TYPES(3, test_eval_tensor_cast);
  CALL_SUBTESTS_DIMS_LAYOUTS_TYPES(3, test_eval_tensor_select);
  CALL_SUBTESTS_DIMS_LAYOUTS_TYPES(3, test_eval_tensor_padding);
  CALL_SUBTESTS_DIMS_LAYOUTS_TYPES(4, test_eval_tensor_chipping);
  CALL_SUBTESTS_DIMS_LAYOUTS_TYPES(4, test_eval_tensor_generator);
  CALL_SUBTESTS_DIMS_LAYOUTS_TYPES(4, test_eval_tensor_reverse);
  CALL_SUBTESTS_DIMS_LAYOUTS_TYPES(5, test_eval_tensor_slice);
  CALL_SUBTESTS_DIMS_LAYOUTS_TYPES(5, test_eval_tensor_shuffle);

  CALL_SUBTESTS_LAYOUTS_TYPES(6, test_eval_tensor_reshape_with_bcast);
  CALL_SUBTESTS_LAYOUTS_TYPES(6, test_eval_tensor_forced_eval);
  CALL_SUBTESTS_LAYOUTS_TYPES(6, test_eval_tensor_chipping_of_bcast);
  // clang-format on
}
