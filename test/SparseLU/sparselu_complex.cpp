// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// sparselu split: complex scalar types.

#include "sparselu_helpers.h"

TEST(SparseluTest, Complex) {
  test_sparselu_T<std::complex<float> >();
  test_sparselu_T<std::complex<double> >();
}
