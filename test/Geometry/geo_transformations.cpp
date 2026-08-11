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
#include <Eigen/Eigenvalues>

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

template <int>
void permutation_transpose_product_issue_1322() {
  Affine3d transform;
  transform.setIdentity();

  Matrix3d m = Vector3d(1.0, 2.0, 3.0).asDiagonal();
  SelfAdjointEigenSolver<Matrix3d> solver(m);
  const PermutationMatrix<3, 3> p(Vector3i::LinSpaced(0, 2).reverse());

  const Vector3d expected = p.transpose().operator*(solver.eigenvalues());
  const Vector3d result = p.transpose() * solver.eigenvalues();
  VERIFY_IS_APPROX(result, expected);
}

TEST(TransformationsAffineTest, PermutationTransposeProductIssue1322) {
  for (int i = 0; i < g_repeat; i++) {
    permutation_transpose_product_issue_1322<0>();
  }
}
