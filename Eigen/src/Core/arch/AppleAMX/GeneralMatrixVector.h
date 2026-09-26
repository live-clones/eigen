// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_APPLE_AMX_GENERALMATRIXVECTOR_H
#define EIGEN_APPLE_AMX_GENERALMATRIXVECTOR_H

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

// NEON crossover in bytes: rows sizeof(Scalar) >= 384 and rows cols sizeof(Scalar) >= 2^15.
#ifndef EIGEN_APPLE_AMX_GEMV_MIN_ROW_BYTES
#define EIGEN_APPLE_AMX_GEMV_MIN_ROW_BYTES 384
#endif
#ifndef EIGEN_APPLE_AMX_GEMV_MIN_BYTES
#define EIGEN_APPLE_AMX_GEMV_MIN_BYTES (double(1 << 15))
#endif

namespace Eigen {
namespace internal {
namespace apple_amx {

// y += alpha A x, A column-major, rows % lanes == 0: Z row z holds y[i0 + z lanes, +lanes), Y copies of alpha x[j].
// Prefetch: the columns 16 ahead into L2, which pays only for a matrix that streams from memory.
template <typename T, bool Prefetch, typename RhsMapper>
EIGEN_DONT_INLINE void gemv_kernel(Index rows, Index cols, const T* A, Index lda, const RhsMapper& x, T* y, T alpha) {
  constexpr Index kLanes = 64 / sizeof(T), kPass = 64 * kLanes, kChunk = 256;
  constexpr uint64_t kVector = uint64_t(1) << 63;
  alignas(256) T broadcast[kChunk * kLanes];
  eigen_internal_assert(rows % kLanes == 0);
  set();
  for (Index j0 = 0; j0 < cols; j0 += kChunk) {
    const Index nj = numext::mini(kChunk, cols - j0);
    for (Index j = 0; j < nj; ++j) {
      const T v = alpha * x(j0 + j, 0);
      for (Index l = 0; l < kLanes; ++l) broadcast[j * kLanes + l] = v;
    }
    for (Index i0 = 0; i0 < rows; i0 += kPass) {
      const int nz = int(numext::mini(kPass, rows - i0) / kLanes);
      for (int z = 0; z < nz; ++z) ldz(zrow(y + i0 + z * kLanes, z));
      for (Index j = 0; j < nj; j += 8) {
        ldy(xy(broadcast + j * kLanes, 0, kQuad));
        ldy(xy(broadcast + (j + 4) * kLanes, 4, kQuad));
        const int nc = int(numext::mini(Index(8), nj - j));
        for (int c = 0; c < nc; ++c) {
          const T* column = A + (j0 + j + c) * lda + i0;
          if (Prefetch && j0 + j + c + 16 < cols) prefetch_l2(column + 16 * lda, nz * 64);
          // Four-register loads need 128-byte alignment; i0 keeps the column's alignment.
          const bool quad = (reinterpret_cast<uintptr_t>(column) & 127) == 0;
          for (int z = 0; z < nz; z += 8) {
            const int n = nz - z < 8 ? nz - z : 8;
            if (quad && n == 8) {
              ldx(xy(column + z * kLanes, 0, kQuad));
              ldx(xy(column + (z + 4) * kLanes, 4, kQuad));
            } else {
              for (int u = 0; u < n; ++u) ldx(xy(column + (z + u) * kLanes, u));
            }
            for (int u = 0; u < n; ++u) fma<T>(kVector | fma_op(z + u, 64 * u, 64 * c));
          }
        }
      }
      for (int z = 0; z < nz; ++z) stz(zrow(y + i0 + z * kLanes, z));
    }
  }
  clr();
}

template <typename T, typename RhsMapper>
void gemv(Index rows, Index cols, const T* A, Index lda, const RhsMapper& x, T* y, T alpha) {
  if (double(rows) * double(cols) * sizeof(T) >= double(16 << 20))
    gemv_kernel<T, true>(rows, cols, A, lda, x, y, alpha);
  else
    gemv_kernel<T, false>(rows, cols, A, lda, x, y, alpha);
}

}  // namespace apple_amx

// Column-major real GEMV on AMX past the crossover; the last rows % lanes rows and other products keep the generic one.
#define EIGEN_APPLE_AMX_GEMV_SPECIALIZATION(Scalar)                                                                   \
  template <typename Index, bool ConjugateLhs, bool ConjugateRhs>                                                     \
  struct general_matrix_vector_product<Index, Scalar, const_blas_data_mapper<Scalar, Index, ColMajor>, ColMajor,      \
                                       ConjugateLhs, Scalar, const_blas_data_mapper<Scalar, Index, RowMajor>,         \
                                       ConjugateRhs, Specialized> {                                                   \
    using LhsMapper = const_blas_data_mapper<Scalar, Index, ColMajor>;                                                \
    using RhsMapper = const_blas_data_mapper<Scalar, Index, RowMajor>;                                                \
    using Generic = general_matrix_vector_product<Index, Scalar, LhsMapper, ColMajor, ConjugateLhs, Scalar,          \
                                                  RhsMapper, ConjugateRhs, BuiltIn>;                                  \
    static EIGEN_STRONG_INLINE void run(Index rows, Index cols, const LhsMapper& lhs, const RhsMapper& rhs,          \
                                        Scalar* res, Index resIncr, Scalar alpha) {                                   \
      constexpr Index kLanes = 64 / sizeof(Scalar);                                                                   \
      if (resIncr != 1 || rows * Index(sizeof(Scalar)) < EIGEN_APPLE_AMX_GEMV_MIN_ROW_BYTES ||                        \
          double(rows) * double(cols) * sizeof(Scalar) < EIGEN_APPLE_AMX_GEMV_MIN_BYTES || !apple_amx::usable())     \
        return Generic::run(rows, cols, lhs, rhs, res, resIncr, alpha);                                               \
      /* BLAS contract: alpha == 0 leaves the result unchanged. */                                                    \
      if (numext::is_exactly_zero(alpha)) return;                                                                     \
      const Index main_rows = rows / kLanes * kLanes;                                                                 \
      apple_amx::gemv(main_rows, cols, lhs.data(), lhs.stride(), rhs, res, alpha);                                    \
      if (main_rows < rows)                                                                                           \
        Generic::run(rows - main_rows, cols, lhs.getSubMapper(main_rows, 0), rhs, res + main_rows, 1, alpha);         \
    }                                                                                                                 \
  };
EIGEN_APPLE_AMX_GEMV_SPECIALIZATION(float)
EIGEN_APPLE_AMX_GEMV_SPECIALIZATION(double)
#undef EIGEN_APPLE_AMX_GEMV_SPECIALIZATION

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_APPLE_AMX_GENERALMATRIXVECTOR_H
