// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/

// jacobisvd split: verify_inputs, verify_assert for fixed-size types.

#include "jacobisvd_helpers.h"

EIGEN_DECLARE_TEST(jacobisvd_verify) {
  (jacobisvd_verify_inputs<Matrix4d>());
  (jacobisvd_verify_inputs(Matrix<float, 5, Dynamic>(5, 6)));
  (jacobisvd_verify_inputs<Matrix<std::complex<double>, 7, 5>>());

  (jacobisvd_verify_assert<Matrix3f>());
  (jacobisvd_verify_assert<Matrix4d>());
  (jacobisvd_verify_assert<Matrix<float, 10, 12>>());
  (jacobisvd_verify_assert<Matrix<float, 12, 10>>());
  (jacobisvd_verify_assert<MatrixXf>(MatrixXf(10, 12)));
  (jacobisvd_verify_assert<MatrixXcd>(MatrixXcd(7, 5)));
}
