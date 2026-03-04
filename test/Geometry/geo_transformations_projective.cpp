// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// geo_transformations split: Projective, RowMajor, associativity, products.
// Affine and AffineCompact modes are in geo_transformations.cpp.

#include "geo_transformations_helpers.h"

TEST(TransformationsProjectiveTest, Basic) {
  for (int i = 0; i < g_repeat; i++) {
    transformations<double, Projective, AutoAlign>();
    transformations<double, Projective, DontAlign>();

    transformations<float, Affine, RowMajor | AutoAlign>();
    non_projective_only<float, Affine, RowMajor>();

    transformations<double, AffineCompact, RowMajor | AutoAlign>();
    non_projective_only<double, AffineCompact, RowMajor>();

    transformations<double, Projective, RowMajor | AutoAlign>();
    transformations<double, Projective, RowMajor | DontAlign>();

    transform_products<double, 3, RowMajor | AutoAlign>();
    transform_products<float, 2, AutoAlign>();

    transform_associativity<double, 2, ColMajor>(Rotation2D<double>(internal::random<double>() * double(EIGEN_PI)));
    transform_associativity<double, 3, ColMajor>(Quaterniond::UnitRandom());
  }
}
