// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// geo_transformations split: float Affine RowMajor + product/associativity tests.

#include "geo_transformations_helpers.h"

TEST(TransformationsProjectiveMiscTest, Basic) {
  for (int i = 0; i < g_repeat; i++) {
    transformations<float, Affine, RowMajor | AutoAlign>();
    non_projective_only<float, Affine, RowMajor>();

    transform_products<double, 3, RowMajor | AutoAlign>();
    transform_products<float, 2, AutoAlign>();

    transform_associativity<double, 2, ColMajor>(Rotation2D<double>(internal::random<double>() * double(EIGEN_PI)));
    transform_associativity<double, 3, ColMajor>(Quaterniond::UnitRandom());
  }
}
