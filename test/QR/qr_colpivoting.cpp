// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// qr_colpivoting split: float types, fixed-size, Kahan matrix, assert tests.
// Double/complex types and stress tests are in qr_colpivoting_double.cpp.

#include "qr_colpivoting_helpers.h"

TEST(QRColpivotingTest, Basic) {
  for (int i = 0; i < g_repeat; i++) {
    qr<MatrixXf>();
    qr_fixedsize<Matrix<float, 3, 5>, 4>();
    qr_fixedsize<Matrix<double, 6, 2>, 3>();
    qr_fixedsize<Matrix<double, 1, 1>, 1>();
  }

  for (int i = 0; i < g_repeat; i++) {
    qr_invertible<MatrixXf>();
    qr_invertible<MatrixXcf>();
  }

  qr_verify_assert<Matrix3f>();
  qr_verify_assert<MatrixXf>();
  qr_verify_assert<MatrixXcf>();

  ColPivHouseholderQR<MatrixXf>(10, 20);

  qr_kahan_matrix<MatrixXf>();
}

// Stress tests for rank detection threshold robustness and efficiency.
TEST(QRColpivotingTest, RankDetectionStress) { qr_rank_detection_stress<MatrixXf>(); }

TEST(QRColpivotingTest, ThresholdEfficiency) { qr_threshold_efficiency<MatrixXf>(); }

TEST(QRColpivotingTest, RankGap) { qr_rank_gap_test<MatrixXf>(); }
