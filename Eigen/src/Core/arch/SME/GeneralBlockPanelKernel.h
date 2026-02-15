// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.

#ifndef EIGEN_SME_GENERALBLOCKPANELKERNEL_H
#define EIGEN_SME_GENERALBLOCKPANELKERNEL_H

#include <arm_sme.h>
#include <arm_sve.h>

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {
namespace internal {

// Define fixed VL for M4 Pro (512 bits = 64 bytes = 16 floats)
#define EIGEN_SME_VL_FLOATS 16

template <bool ConjLhs_, bool ConjRhs_, int PacketSize_>
class gebp_traits<float, float, ConjLhs_, ConjRhs_, Architecture::Target, PacketSize_>
    : public gebp_traits<float, float, ConjLhs_, ConjRhs_, Architecture::Generic, PacketSize_> {
  using Base = gebp_traits<float, float, ConjLhs_, ConjRhs_, Architecture::Generic, PacketSize_>;

 public:
  enum {
    mr = EIGEN_SME_VL_FLOATS,
    nr = EIGEN_SME_VL_FLOATS,  // Optimized for 1-way ZA tiling (16 columns)
    LhsProgress = mr,
    RhsProgress = nr
  };
};

template <typename DataMapper, typename Scalar, typename Index, int mr, int nr>
__attribute__((noinline)) __arm_new("za") __arm_locally_streaming void run_sme_gemm(
    const DataMapper& res, const Scalar* blockA, const Scalar* blockB, Index rows, Index depth, Index cols,
    Scalar alpha, Index strideA, Index strideB, Index offsetA, Index offsetB) {
  // Naive SME Implementation with Predicates for safety
  // Assumes VL=16 (mr=16, nr=16)

  for (Index i = 0; i < rows; i += mr) {
    for (Index j = 0; j < cols; j += nr) {
      // Generate predicates for boundary handling
      svbool_t pg_rows = svwhilelt_b32(static_cast<int64_t>(i), static_cast<int64_t>(rows));
      svbool_t pg_cols = svwhilelt_b32(static_cast<int64_t>(j), static_cast<int64_t>(cols));

      // Clear Accumulator Tile (ZA0)
      svzero_za();

      const Scalar* pA = blockA + offsetA + i * depth;
      const Scalar* pB = blockB + offsetB + j * depth;

      Index k = 0;
      // Unroll 4x to reduce loop overhead
      for (; k + 4 <= depth; k += 4) {
        svfloat32_t va0 = svld1_f32(pg_rows, pA);
        svfloat32_t vb0 = svld1_f32(pg_cols, pB);

        svfloat32_t va1 = svld1_f32(pg_rows, pA + mr);
        svfloat32_t vb1 = svld1_f32(pg_cols, pB + nr);

        svfloat32_t va2 = svld1_f32(pg_rows, pA + 2 * mr);
        svfloat32_t vb2 = svld1_f32(pg_cols, pB + 2 * nr);

        svfloat32_t va3 = svld1_f32(pg_rows, pA + 3 * mr);
        svfloat32_t vb3 = svld1_f32(pg_cols, pB + 3 * nr);

        svmopa_za32_f32_m(0, pg_rows, pg_cols, va0, vb0);
        svmopa_za32_f32_m(0, pg_rows, pg_cols, va1, vb1);
        svmopa_za32_f32_m(0, pg_rows, pg_cols, va2, vb2);
        svmopa_za32_f32_m(0, pg_rows, pg_cols, va3, vb3);

        pA += 4 * mr;
        pB += 4 * nr;
      }

      for (; k < depth; ++k) {
        svfloat32_t va = svld1_f32(pg_rows, pA);
        pA += mr;
        svfloat32_t vb = svld1_f32(pg_cols, pB);
        pB += nr;
        svmopa_za32_f32_m(0, pg_rows, pg_cols, va, vb);
      }

      // Store Result (ZA0) back to C
      // Iterate over rows of the tile (0..15)
      for (int r = 0; r < mr; ++r) {
        // If this row index is beyond the matrix rows, stop
        if (i + r >= rows) break;

        // Read vector from ZA.
        // This vector corresponds to C(i+r, j : j+nr)
        svfloat32_t inactive = svundef_f32();
        svfloat32_t vres = svread_hor_za32_f32_m(inactive, pg_cols, 0, r);

        if (alpha != 1.0f) vres = svmul_x(pg_cols, vres, alpha);

        // Extract to temp buffer to do scatter update
        EIGEN_ALIGN64 Scalar tmp[EIGEN_SME_VL_FLOATS];
        svst1_f32(pg_cols, tmp, vres);

        for (int c_idx = 0; c_idx < nr; ++c_idx) {
          if (j + c_idx < cols) {
            res(i + r, j + c_idx) += tmp[c_idx];
          }
        }
      }
    }
  }
}

template <typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
struct gebp_kernel<float, float, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs> {
  typedef float Scalar;

  EIGEN_DONT_INLINE void operator()(const DataMapper& res, const Scalar* blockA, const Scalar* blockB, Index rows,
                                    Index depth, Index cols, Scalar alpha, Index strideA = -1, Index strideB = -1,
                                    Index offsetA = 0, Index offsetB = 0) {
    run_sme_gemm<DataMapper, Scalar, Index, mr, nr>(res, blockA, blockB, rows, depth, cols, alpha, strideA, strideB,
                                                    offsetA, offsetB);
  }
};

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_SME_GENERALBLOCKPANELKERNEL_H
