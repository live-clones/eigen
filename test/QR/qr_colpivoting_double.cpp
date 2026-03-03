// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// qr_colpivoting split: double/complex types and stress tests — split from
// qr_colpivoting.cpp to reduce per-TU memory usage during compilation.

#include "qr_colpivoting_helpers.h"

TEST(QRColpivotingDoubleTest, Basic) {
  for (int i = 0; i < g_repeat; i++) {
    qr<MatrixXd>();
    qr<MatrixXcd>();
  }

  for (int i = 0; i < g_repeat; i++) {
    qr_invertible<MatrixXd>();
    qr_invertible<MatrixXcd>();
  }

  qr_verify_assert<Matrix3d>();
  qr_verify_assert<MatrixXd>();
  qr_verify_assert<MatrixXcd>();

  qr_kahan_matrix<MatrixXd>();
}

TEST(QRColpivotingDoubleTest, RankDetectionStress) { qr_rank_detection_stress<MatrixXd>(); }

TEST(QRColpivotingDoubleTest, ThresholdEfficiency) { qr_threshold_efficiency<MatrixXd>(); }

TEST(QRColpivotingDoubleTest, RankGap) { qr_rank_gap_test<MatrixXd>(); }
