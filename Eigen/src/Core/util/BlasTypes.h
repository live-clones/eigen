// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_BLAS_TYPES_H
#define EIGEN_BLAS_TYPES_H

/* Scalar and index types of the external BLAS/LAPACK interface. These are used pervasively by the BLAS and LAPACKE
 * backends, so they live in a backend-neutral header rather than in MKL_support.h. BlasIndex follows the integer
 * width selected there (MKL_INT for MKL, EIGEN_BLAS_INT otherwise), so this header must be included after
 * MKL_support.h has set those up; MKL_support.h includes it for that reason.
 */

// IWYU pragma: private
#include "../InternalHeaderCheck.h"

namespace Eigen {

typedef std::complex<double> dcomplex;
typedef std::complex<float> scomplex;

#if defined(EIGEN_USE_MKL)
typedef MKL_INT BlasIndex;
// Plain static_assert (not EIGEN_STATIC_ASSERT): like the LAPACKE/BLAS cross-check it must not be suppressible.
#if defined(EIGEN_64BIT_BLAS)
static_assert(sizeof(MKL_INT) == 8,
              "EIGEN_64BIT_BLAS is defined but MKL_INT is 32-bit. Define MKL_ILP64 and link the MKL *_ilp64 "
              "libraries, or undefine EIGEN_64BIT_BLAS.");
#else
static_assert(sizeof(MKL_INT) == 4,
              "MKL_INT is 64-bit but EIGEN_64BIT_BLAS is not defined. Define EIGEN_64BIT_BLAS to match MKL_ILP64, or "
              "link the MKL *_lp64 libraries.");
#endif
#else
typedef EIGEN_BLAS_INT BlasIndex;
#endif

}  // end namespace Eigen

#endif  // EIGEN_BLAS_TYPES_H
