// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// geo_transformations split: Affine, AffineCompact, and no-scale modes.
// Projective, RowMajor, associativity, and products are in
// geo_transformations_projective.cpp.

#include "geo_transformations_helpers.h"

TEST(TransformationsAffineTest, Basic) {
  for (int i = 0; i < g_repeat; i++) {
    transformations<double, Affine, AutoAlign>();
    non_projective_only<double, Affine, AutoAlign>();
    transformations_computed_scaling_continuity<double, Affine, AutoAlign>();

    transformations<float, AffineCompact, AutoAlign>();
    non_projective_only<float, AffineCompact, AutoAlign>();
    transform_alignment<float>();
    transform_alignment<double>();

    transformations_no_scale<double, Affine, AutoAlign>();
    transformations_no_scale<double, Isometry, AutoAlign>();
  }
}
