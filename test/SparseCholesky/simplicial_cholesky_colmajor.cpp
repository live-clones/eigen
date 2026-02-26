// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Gael Guennebaud <g.gael@free.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// simplicial_cholesky split: ColMajor tests.

#include "simplicial_cholesky_helpers.h"

// =============================================================================
// Config struct + typed test suite for SimplicialCholesky ColMajor
// =============================================================================
template <typename T_, typename I__>
struct SimplicialCholeskyColmajorConfig {
  using Scalar = T_;
  using Index = I__;
};

template <typename T>
class SimplicialCholeskyColmajorTest : public ::testing::Test {};

using SimplicialCholeskyColmajorTypes = ::testing::Types<SimplicialCholeskyColmajorConfig<double, int>,
                                                         SimplicialCholeskyColmajorConfig<std::complex<double>, int>,
                                                         SimplicialCholeskyColmajorConfig<double, long int>>;
TYPED_TEST_SUITE(SimplicialCholeskyColmajorTest, SimplicialCholeskyColmajorTypes);

TYPED_TEST(SimplicialCholeskyColmajorTest, Basic) {
  test_simplicial_cholesky_T<typename TypeParam::Scalar, typename TypeParam::Index, ColMajor>();
}
