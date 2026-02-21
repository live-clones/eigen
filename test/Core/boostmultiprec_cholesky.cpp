// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// boostmultiprec split: cholesky and stream output tests.

#include "boostmultiprec_helpers.h"

// =============================================================================
// Tests for boostmultiprec_cholesky
// =============================================================================
TEST(BoostMultiprecCholeskyTest, Basic) {
  std::cout << "NumTraits<Real>::epsilon()         = " << NumTraits<Real>::epsilon() << std::endl;
  std::cout << "NumTraits<Real>::dummy_precision() = " << NumTraits<Real>::dummy_precision() << std::endl;
  std::cout << "NumTraits<Real>::lowest()          = " << NumTraits<Real>::lowest() << std::endl;
  std::cout << "NumTraits<Real>::highest()         = " << NumTraits<Real>::highest() << std::endl;
  std::cout << "NumTraits<Real>::digits10()        = " << NumTraits<Real>::digits10() << std::endl;
  std::cout << "NumTraits<Real>::max_digits10()    = " << NumTraits<Real>::max_digits10() << std::endl;

  // check stream output
  {
    Mat A(10, 10);
    A.setRandom();
    std::stringstream ss;
    ss << A;
  }
  {
    MatC A(10, 10);
    A.setRandom();
    std::stringstream ss;
    ss << A;
  }

  for (int i = 0; i < g_repeat; i++) {
    int s = internal::random<int>(1, EIGEN_TEST_MAX_SIZE);
    boostmp_cholesky(Mat(s, s));
    TEST_SET_BUT_UNUSED_VARIABLE(s)
  }
}
