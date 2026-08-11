// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/
// SPDX-License-Identifier: MPL-2.0

// bdcsvd split: assert checks and option-enum regression.

#include "bdcsvd_helpers.h"

TEST(BDCSVDAssertTest, Basic) {
  (bdcsvd_asserts<Matrix3f>());
  (bdcsvd_asserts<Matrix4d>());
  (bdcsvd_asserts<Matrix<float, 10, 7>>());
  (bdcsvd_asserts<Matrix<float, 7, 10>>());
  (bdcsvd_asserts<Matrix<std::complex<double>, 6, 9>>());
}

void bdcsvd_mixed_option_enum_regression() {
  using NoQrFullSVD = BDCSVD<MatrixXd, NoQRPreconditioner | ComputeFullU | ComputeFullV>;
  using ReversedMixedSVD = BDCSVD<MatrixXd, ComputeThinU | DisableQRDecomposition | ComputeFullV>;

  STATIC_CHECK((int(NoQrFullSVD::QRDecomposition) == int(NoQRPreconditioner)));
  STATIC_CHECK((NoQrFullSVD::ComputationOptions == (ComputeFullU | ComputeFullV)));

  STATIC_CHECK((int(ReversedMixedSVD::QRDecomposition) == int(DisableQRDecomposition)));
  STATIC_CHECK((ReversedMixedSVD::ComputationOptions == (ComputeThinU | ComputeFullV)));
}

TEST(BDCSVDAssertTest, MixedOptionEnumRegression) { bdcsvd_mixed_option_enum_regression(); }
