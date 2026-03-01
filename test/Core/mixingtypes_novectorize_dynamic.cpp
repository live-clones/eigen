// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2015 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_DONT_VECTORIZE
#define EIGEN_DONT_VECTORIZE
#endif

#include "mixingtypes_helpers.h"

// Dynamic-size instantiation — split from mixingtypes_novectorize.cpp to
// reduce per-TU memory usage under ASAN+UBSAN.
TEST(MixingTypesNoVectorizeDynamicTest, Basic) {
  g_called = false;  // Silence -Wunneeded-internal-declaration.
  for (int i = 0; i < g_repeat; i++) {
    mixingtypes<Dynamic>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE));
  }
}
