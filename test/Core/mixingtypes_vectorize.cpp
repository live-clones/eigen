// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2015 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// ignore double-promotion diagnostic for clang and gcc:
#if defined(__clang__)
#if (__clang_major__ * 100 + __clang_minor__) >= 308
#pragma clang diagnostic ignored "-Wdouble-promotion"
#endif
#elif defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#endif

#include "mixingtypes_helpers.h"

EIGEN_DECLARE_TEST(mixingtypes_vectorize) {
  g_called = false;  // Silence -Wunneeded-internal-declaration.
  for (int i = 0; i < g_repeat; i++) {
    mixingtypes<3>();
    mixingtypes<4>();
    mixingtypes<Dynamic>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE));
  }
}
