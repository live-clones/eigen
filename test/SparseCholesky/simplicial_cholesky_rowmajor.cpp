// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Gael Guennebaud <g.gael@free.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// simplicial_cholesky split: RowMajor tests.

#include "simplicial_cholesky_helpers.h"

// =============================================================================
// Config struct + typed test suite for SimplicialCholesky RowMajor
// =============================================================================
template <typename T_, typename I__>
struct SimplicialCholeskyRowmajorConfig {
  using Scalar = T_;
  using Index = I__;
};

template <typename T>
class SimplicialCholeskyRowmajorTest : public ::testing::Test {};

using SimplicialCholeskyRowmajorTypes = ::testing::Types<SimplicialCholeskyRowmajorConfig<double, int>,
                                                         SimplicialCholeskyRowmajorConfig<std::complex<double>, int>,
                                                         SimplicialCholeskyRowmajorConfig<double, long int>>;
TYPED_TEST_SUITE(SimplicialCholeskyRowmajorTest, SimplicialCholeskyRowmajorTypes);

TYPED_TEST(SimplicialCholeskyRowmajorTest, Basic) {
  test_simplicial_cholesky_T<typename TypeParam::Scalar, typename TypeParam::Index, RowMajor>();
}
