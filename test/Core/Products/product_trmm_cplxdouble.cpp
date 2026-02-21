// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// product_trmm split: complex<double> scalar type.

#include "product_trmm_helpers.h"

EIGEN_DECLARE_TEST(product_trmm_cplxdouble) {
  for (int i = 0; i < g_repeat; i++) {
    CALL_ALL(std::complex<double>);
  }
}
