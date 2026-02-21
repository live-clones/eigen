// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2015 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "redux_helpers.h"

EIGEN_DECLARE_TEST(redux_vector) {
  // the max size cannot be too large, otherwise reduxion operations obviously generate large errors.
  int maxsize = (std::min)(100, EIGEN_TEST_MAX_SIZE);
  TEST_SET_BUT_UNUSED_VARIABLE(maxsize);
  for (int i = 0; i < g_repeat; i++) {
    int size = internal::random<int>(1, maxsize);
    EIGEN_UNUSED_VARIABLE(size);
    vectorRedux(Vector4f());
    vectorRedux(Array4f());
    vectorRedux(VectorXf(size));
    vectorRedux(ArrayXf(size));
    vectorRedux(VectorXd(size));
    vectorRedux(ArrayXd(size));
    vectorRedux(VectorXi(size));
    vectorRedux(ArrayXi(size));
    vectorRedux(VectorX<int64_t>(size));
    vectorRedux(ArrayX<int64_t>(size));
  }
}
