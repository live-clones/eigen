// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// geo_transformations split: Projective and RowMajor variants for double.

#include "geo_transformations_helpers.h"

TEST(TransformationsProjectiveTest, Basic) {
  for (int i = 0; i < g_repeat; i++) {
    transformations<double, Projective, AutoAlign>();
    transformations<double, Projective, DontAlign>();

    transformations<double, AffineCompact, RowMajor | AutoAlign>();
    non_projective_only<double, AffineCompact, RowMajor>();

    transformations<double, Projective, RowMajor | AutoAlign>();
    transformations<double, Projective, RowMajor | DontAlign>();
  }
}
