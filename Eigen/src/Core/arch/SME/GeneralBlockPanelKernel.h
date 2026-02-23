// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SME_GENERALBLOCKPANELKERNEL_H
#define EIGEN_SME_GENERALBLOCKPANELKERNEL_H

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

#include <arm_sme.h>

namespace Eigen {
namespace internal {

// Streaming vector length in floats, derived from the compile-time SVE VL.
// EIGEN_ARM64_SVE_VL is set in ConfigureVectorization.h from __ARM_FEATURE_SVE_BITS.
static constexpr int kSmeVlFloats = EIGEN_ARM64_SVE_VL / 32;

// Micro-kernel dimensions.
//   mr  = SVL32     = 16     (LHS panel width, fills one Z register)
//   nr  = 4 * SVL32 = 64     (RHS super-panel width, 4 sub-panels of VL)
//   The micro-kernel loads 1 LHS vector + 4 RHS vectors per depth step,
//   accumulating into all 4 ZA tiles simultaneously -> 16x64 output tile.
static constexpr int kSmeMr = kSmeVlFloats;  // 16
static constexpr int kSmeNr = kSmeVlFloats;  // 16 -- one sub-panel width
static constexpr int kSmeNr4 = 4 * kSmeNr;   // 64 -- full super-panel width

/*****************************************************************************
 * gebp_traits specialization for SME  (float x float)
 *
 * Overrides mr and nr so that:
 *   - gemm_pack_lhs receives Pack1 = mr = 16, creating uniform VL-wide
 *     LHS panels (no 3*PS/2*PS/PS sub-panel hierarchy)
 *   - gemm_pack_rhs receives nr = 64, creating 4xVL-wide RHS super-panels
 *   - mc is rounded to a multiple of 16, nc to a multiple of 64
 *   - Cache blocking (kc, mc, nc) is recomputed accordingly
 *
 * We also provide custom gemm_pack_lhs and gemm_pack_rhs specializations
 * for float, so both ColMajor and RowMajor source matrices produce an
 * identical, simple packed format that the SME kernel consumes.
 *****************************************************************************/

template <>
class gebp_traits<float, float, false, false, Architecture::Target, GEBPPacketFull>
    : public gebp_traits<float, float, false, false, Architecture::Target, GEBPPacketHalf> {
 public:
  // The base class provides all the standard typedefs (LhsPacket, etc.)
  // We only override the register-block sizes.
  enum {
    mr = kSmeMr,  // 16 -- LHS panel width
    nr = kSmeNr4  // 64 -- RHS super-panel width (4 x VL)
  };
};

/*****************************************************************************
 * gemm_pack_lhs specialization for SME  (float, ColMajor)
 *
 * Packs the LHS matrix into uniform panels of width = mr = 16.
 * Each depth step k writes mr contiguous floats (one full SVE register).
 * Tail rows (< mr) are NOT zero-padded -- the kernel uses predicated
 * loads (svwhilelt) to handle the shorter panel.
 *
 * Layout for `rows` source rows and `depth` columns:
 *
 *   Panel 0 (rows 0..mr-1):    [row0_k0 .. row15_k0] [row0_k1 .. row15_k1] ...
 *   Panel 1 (rows mr..2*mr-1): [row16_k0 ..row31_k0] [row16_k1 ..row31_k1] ...
 *   ...
 *   Last panel (tail, not zero-padded)
 *
 * When PanelMode == true, each panel is strided by `stride` (instead of
 * depth) and offset by `offset * mr`.
 *
 * This layout is deliberately kept byte-identical to what a future SME
 * streaming packer (using LD1W into ZA + ST1W from ZA vertical slices)
 * would produce, so the kernel does not need to change.
 *****************************************************************************/

template <typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
struct gemm_pack_lhs<float, Index, DataMapper, Pack1, Pack2, Packet, ColMajor, Conjugate, PanelMode> {
  typedef float Scalar;

  EIGEN_DONT_INLINE void operator()(Scalar* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride = 0,
                                    Index offset = 0) {
    constexpr int MR = kSmeMr;  // 16
    const Index peeled_rows = (rows / MR) * MR;

    // Full panels of width MR.
    for (Index i = 0; i < peeled_rows; i += MR) {
      // In PanelMode the panels are spaced `stride` apart with `offset`
      // controlling where within each strided panel writing begins.
      Scalar* dst = PanelMode ? blockA + (i / MR) * MR * stride + offset * MR : blockA + i * depth;

      for (Index k = 0; k < depth; ++k) {
        for (Index r = 0; r < MR; ++r) {
          dst[k * MR + r] = lhs(i + r, k);
        }
      }
    }

    // Tail panel: rows < MR.  No zero-padding -- write exactly `tail`
    // elements per depth step (the buffer is sized rows*depth, not
    // rounded up).  The kernel uses predicated loads (svwhilelt) to
    // handle the shorter panel.
    if (peeled_rows < rows) {
      const Index tail = rows - peeled_rows;

      Scalar* dst =
          PanelMode ? blockA + (peeled_rows / MR) * MR * stride + offset * tail : blockA + peeled_rows * depth;

      for (Index k = 0; k < depth; ++k) {
        for (Index r = 0; r < tail; ++r) dst[k * tail + r] = lhs(peeled_rows + r, k);
      }
    }
  }
};

// RowMajor LHS packer -- the DataMapper abstracts storage order via
// operator()(row, col), so the packing logic is identical to ColMajor.
template <typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
struct gemm_pack_lhs<float, Index, DataMapper, Pack1, Pack2, Packet, RowMajor, Conjugate, PanelMode>
    : gemm_pack_lhs<float, Index, DataMapper, Pack1, Pack2, Packet, ColMajor, Conjugate, PanelMode> {};

/*****************************************************************************
 * gemm_pack_rhs specialization for SME  (float, ColMajor)
 *
 * Packs the RHS matrix into super-panels of width = nr = 64.
 * Each depth step k writes nr contiguous floats (4 x VL).
 * Tail columns (< nr) are NOT zero-padded.
 *
 * Layout (each super-panel holds 4 sub-panels of 16 columns):
 *   Super-panel 0 (cols 0..63):   [col0_k0..col63_k0] [col0_k1..col63_k1] ...
 *   Super-panel 1 (cols 64..127): [col64_k0..col127_k0] ...
 *   ...
 *   Last panel (tail < nr, packed at tail width)
 *****************************************************************************/

template <typename Index, typename DataMapper, int nr_, bool Conjugate, bool PanelMode>
struct gemm_pack_rhs<float, Index, DataMapper, nr_, ColMajor, Conjugate, PanelMode> {
  typedef float Scalar;

  EIGEN_DONT_INLINE void operator()(Scalar* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride = 0,
                                    Index offset = 0) {
    constexpr int NR = kSmeNr4;  // 64
    const Index peeled_cols = (cols / NR) * NR;

    // Full super-panels of width NR = 64.
    for (Index j = 0; j < peeled_cols; j += NR) {
      Scalar* dst = PanelMode ? blockB + (j / NR) * NR * stride + offset * NR : blockB + j * depth;

      for (Index k = 0; k < depth; ++k) {
        for (Index c = 0; c < NR; ++c) {
          dst[k * NR + c] = rhs(k, j + c);
        }
      }
    }

    // Tail panel: cols < NR.  No zero-padding.
    if (peeled_cols < cols) {
      const Index tail = cols - peeled_cols;

      Scalar* dst =
          PanelMode ? blockB + (peeled_cols / NR) * NR * stride + offset * tail : blockB + peeled_cols * depth;

      for (Index k = 0; k < depth; ++k) {
        for (Index c = 0; c < tail; ++c) dst[k * tail + c] = rhs(k, peeled_cols + c);
      }
    }
  }
};

// RowMajor RHS packer -- the DataMapper abstracts storage order via
// operator()(row, col), so the packing logic is identical to ColMajor.
template <typename Index, typename DataMapper, int nr_, bool Conjugate, bool PanelMode>
struct gemm_pack_rhs<float, Index, DataMapper, nr_, RowMajor, Conjugate, PanelMode>
    : gemm_pack_rhs<float, Index, DataMapper, nr_, ColMajor, Conjugate, PanelMode> {};

/*****************************************************************************
 * sme_store_za_tile -- Store one ZA.S tile back to C with alpha scaling.
 *
 * Reads `cw` columns from the specified ZA tile, scales by alpha, and
 * accumulates into the output matrix C at position (row_start, col_start).
 * `pw` active rows are controlled by a predicate.
 *
 * TileId is a template parameter because ACLE requires the ZA tile
 * index to be a compile-time constant in svread_*_za32 intrinsics.
 *****************************************************************************/

template <int TileId, typename Scalar, typename Index>
EIGEN_ALWAYS_INLINE void sme_store_za_tile(Scalar* EIGEN_RESTRICT C, Index C_stride_row, Index C_stride_col,
                                           Scalar alpha, Index row_start, int pw, Index col_start,
                                           int cw) __arm_streaming __arm_inout("za") {
  const svbool_t pg_m = svwhilelt_b32((uint32_t)0, (uint32_t)pw);
  const svbool_t pg_n = svwhilelt_b32((uint32_t)0, (uint32_t)cw);
  const svfloat32_t valpha = svdup_f32(alpha);

  if (C_stride_row == 1) {
    // ColMajor output: read vertical slices (columns of the tile).
    for (int ci = 0; ci < cw; ++ci) {
      svfloat32_t vres = svread_ver_za32_f32_m(svdup_f32(0.f), pg_m, TileId, (uint32_t)ci);
      vres = svmul_f32_x(pg_m, vres, valpha);
      Scalar* pC = C + row_start + (col_start + ci) * C_stride_col;
      svfloat32_t vc = svld1_f32(pg_m, pC);
      vc = svadd_f32_x(pg_m, vc, vres);
      svst1_f32(pg_m, pC, vc);
    }
  } else if (C_stride_col == 1) {
    // RowMajor output: read horizontal slices (rows of the tile).
    for (int ri = 0; ri < pw; ++ri) {
      svfloat32_t vres = svread_hor_za32_f32_m(svdup_f32(0.f), pg_n, TileId, (uint32_t)ri);
      vres = svmul_f32_x(pg_n, vres, valpha);
      Scalar* pC = C + (row_start + ri) * C_stride_row + col_start;
      svfloat32_t vc = svld1_f32(pg_n, pC);
      vc = svadd_f32_x(pg_n, vc, vres);
      svst1_f32(pg_n, pC, vc);
    }
  } else {
    // General stride: scalar fallback via temporary.
    for (int ri = 0; ri < pw; ++ri) {
      svfloat32_t vres = svread_hor_za32_f32_m(svdup_f32(0.f), pg_n, TileId, (uint32_t)ri);
      vres = svmul_f32_x(pg_n, vres, valpha);
      EIGEN_ALIGN64 Scalar tmp[kSmeVlFloats];
      svst1_f32(pg_n, tmp, vres);
      for (int ci = 0; ci < cw; ++ci) {
        C[(row_start + ri) * C_stride_row + (col_start + ci) * C_stride_col] += tmp[ci];
      }
    }
  }
}

/*****************************************************************************
 * sme_process_4tiles -- Interleaved 4-tile micro-kernel (pw x 64 output)
 *
 * This is the performance-critical inner kernel.  For each depth step:
 *   1. Load ONE LHS vector (16 floats, reused across all 4 tiles)
 *   2. Load FOUR consecutive RHS vectors via svld1_f32_x4 (64 contiguous
 *      floats in a single multi-vector load for peak memory bandwidth)
 *   3. Issue 4 FMOPA instructions, one per ZA tile
 *
 * All 4 ZA tiles accumulate in parallel throughout the depth loop,
 * then all 4 are stored to C afterwards.  This maximises LHS reuse
 * and provides 4-way ILP for the FMOPA pipeline.
 *
 * The RHS data layout (from the NR=64 packer) is:
 *   [col0..col15 | col16..col31 | col32..col47 | col48..col63]  x depth
 * i.e., 64 contiguous floats per depth step -- used by svld1_f32_x4.
 *****************************************************************************/

template <typename Scalar, typename Index>
EIGEN_ALWAYS_INLINE void sme_process_4tiles(Scalar* EIGEN_RESTRICT C, Index C_stride_row, Index C_stride_col,
                                            const Scalar* EIGEN_RESTRICT blA, const Scalar* EIGEN_RESTRICT blB,
                                            Index depth, Scalar alpha, Index row_start, int pw,
                                            Index col_start) __arm_streaming __arm_inout("za") {
  constexpr int NR = kSmeNr;    // 16  (sub-panel width)
  constexpr int NR4 = kSmeNr4;  // 64  (super-panel width)

  const svbool_t pg_m = svwhilelt_b32((uint32_t)0, (uint32_t)pw);
  // pg_all is always-true: correct here because each of the 4 RHS sub-panels
  // is exactly VL-wide (16 floats), so all lanes participate in FMOPA.
  const svbool_t pg_all = svptrue_b32();
  const svcount_t pn_all = svptrue_c32();

  // Zero all 4 .S tiles at once (mask 0xFF = all 8 .D tiles).
  svzero_za();

  for (Index k = 0; k < depth; ++k) {
    // One LHS load, reused across all 4 FMOPA instructions.
    svfloat32_t va = svld1_f32(pg_m, &blA[k * pw]);

    // Multi-vector x4 load: 64 contiguous floats in one instruction.
    const Scalar* blBk = &blB[k * NR4];
    svfloat32x4_t vb = svld1_f32_x4(pn_all, blBk);

    // Outer-product accumulate into all 4 tiles.
    svmopa_za32_f32_m(0, pg_m, pg_all, va, svget4_f32(vb, 0));
    svmopa_za32_f32_m(1, pg_m, pg_all, va, svget4_f32(vb, 1));
    svmopa_za32_f32_m(2, pg_m, pg_all, va, svget4_f32(vb, 2));
    svmopa_za32_f32_m(3, pg_m, pg_all, va, svget4_f32(vb, 3));
  }

  // Store all 4 tiles to C.
  sme_store_za_tile<0>(C, C_stride_row, C_stride_col, alpha, row_start, pw, col_start + 0 * NR, NR);
  sme_store_za_tile<1>(C, C_stride_row, C_stride_col, alpha, row_start, pw, col_start + 1 * NR, NR);
  sme_store_za_tile<2>(C, C_stride_row, C_stride_col, alpha, row_start, pw, col_start + 2 * NR, NR);
  sme_store_za_tile<3>(C, C_stride_row, C_stride_col, alpha, row_start, pw, col_start + 3 * NR, NR);
}

/*****************************************************************************
 * sme_process_microblock -- Single-tile micro-kernel for tail columns.
 *
 * Processes one micro-block of size pw x cw x depth using a single ZA tile.
 * `blB_stride` is the distance (in elements) between consecutive depth
 * steps in blB -- this may differ from `cw` when the micro-block is a
 * sub-group within a wider tail panel.
 *
 * TileId is a template parameter because ACLE requires the ZA tile
 * index to be a compile-time constant.
 *
 * Called from within a __arm_locally_streaming __arm_new("za") function.
 *****************************************************************************/

template <int TileId, typename Scalar, typename Index>
EIGEN_ALWAYS_INLINE void sme_process_microblock(Scalar* EIGEN_RESTRICT C, Index C_stride_row, Index C_stride_col,
                                                const Scalar* EIGEN_RESTRICT blA, const Scalar* EIGEN_RESTRICT blB,
                                                int blB_stride, Index depth, Scalar alpha, Index row_start, int pw,
                                                Index col_start, int cw) __arm_streaming __arm_inout("za") {
  const svbool_t pg_m = svwhilelt_b32((uint32_t)0, (uint32_t)pw);
  const svbool_t pg_n = svwhilelt_b32((uint32_t)0, (uint32_t)cw);

  // Zero the target ZA tile.
  // The ZERO instruction's 8-bit mask operates on .D-tile granularity
  // (8 tiles for SVL=512).  Each .S tile spans two .D tiles:
  //   ZA0.S -> mask = 0x11,  ZA1.S -> 0x22,  ZA2.S -> 0x44,  ZA3.S -> 0x88
  svzero_mask_za(0x11 << TileId);

  for (Index k = 0; k < depth; ++k) {
    svfloat32_t va = svld1_f32(pg_m, &blA[k * pw]);
    svfloat32_t vb = svld1_f32(pg_n, &blB[k * blB_stride]);
    svmopa_za32_f32_m(TileId, pg_m, pg_n, va, vb);
  }

  sme_store_za_tile<TileId>(C, C_stride_row, C_stride_col, alpha, row_start, pw, col_start, cw);
}

/*****************************************************************************
 * RHS column iteration for one LHS panel.
 *
 * For full 64-column super-panels: uses sme_process_4tiles (interleaved
 * 4-tile depth loop with LHS reuse -- the fast path).
 *
 * For tail columns (< 64): decomposes into VL-wide sub-groups and
 * processes each with sme_process_microblock.  The sub-groups share
 * the tail panel's packing stride (which may differ from their width).
 *****************************************************************************/

template <typename Scalar, typename Index>
EIGEN_ALWAYS_INLINE void sme_process_all_cols(Scalar* EIGEN_RESTRICT C, Index C_stride_row, Index C_stride_col,
                                              const Scalar* EIGEN_RESTRICT blA, const Scalar* EIGEN_RESTRICT blockB,
                                              Index depth, Index cols, Scalar alpha, Index strideB, Index offsetB,
                                              Index row_start, int pw) __arm_streaming __arm_inout("za") {
  constexpr int NR = kSmeNr;    // 16  (sub-panel width)
  constexpr int NR4 = kSmeNr4;  // 64  (super-panel width)

  const Index peeled_cols = (cols / NR4) * NR4;

  // --- Full 64-column super-panels: interleaved 4-tile depth loop ---
  for (Index j = 0; j < peeled_cols; j += NR4) {
    sme_process_4tiles(C, C_stride_row, C_stride_col, blA, &blockB[j * strideB + offsetB * NR4], depth, alpha,
                       row_start, pw, j);
  }

  // --- Tail columns (< 64) ---
  if (peeled_cols < cols) {
    const int tail = static_cast<int>(cols - peeled_cols);
    // The packer wrote `tail` floats per depth step (no zero-padding),
    // so the stride between k-steps within this tail panel is `tail`.
    const Scalar* blB_tail = &blockB[peeled_cols * strideB + offsetB * tail];

    // Decompose tail into full VL-wide (16) sub-groups + remainder.
    // At most 3 full sub-groups (tail < 64 = 4x16), so tile IDs 0-3 suffice.
    const int n_full_sub = tail / NR;
    const int sub_tail = tail % NR;

    // Process each VL-wide sub-group with its own ZA tile.
    // Each sub-group runs a separate depth loop (unlike sme_process_4tiles
    // which fuses all 4 into one loop with LHS reuse).  This is simpler
    // and acceptable for the tail path (at most 3 sub-groups).
    // Each sub-group needs a compile-time TileId, so we dispatch via if/else.
    if (n_full_sub >= 1) {
      sme_process_microblock<0>(C, C_stride_row, C_stride_col, blA, blB_tail + 0 * NR, tail, depth, alpha, row_start,
                                pw, peeled_cols + 0 * NR, NR);
    }
    if (n_full_sub >= 2) {
      sme_process_microblock<1>(C, C_stride_row, C_stride_col, blA, blB_tail + 1 * NR, tail, depth, alpha, row_start,
                                pw, peeled_cols + 1 * NR, NR);
    }
    if (n_full_sub >= 3) {
      sme_process_microblock<2>(C, C_stride_row, C_stride_col, blA, blB_tail + 2 * NR, tail, depth, alpha, row_start,
                                pw, peeled_cols + 2 * NR, NR);
    }

    if (sub_tail > 0) {
      // The remainder sub-group uses the tile ID after the last full sub-group.
      switch (n_full_sub) {
        case 0:
          sme_process_microblock<0>(C, C_stride_row, C_stride_col, blA, blB_tail + 0 * NR, tail, depth, alpha,
                                    row_start, pw, peeled_cols + 0 * NR, sub_tail);
          break;
        case 1:
          sme_process_microblock<1>(C, C_stride_row, C_stride_col, blA, blB_tail + 1 * NR, tail, depth, alpha,
                                    row_start, pw, peeled_cols + 1 * NR, sub_tail);
          break;
        case 2:
          sme_process_microblock<2>(C, C_stride_row, C_stride_col, blA, blB_tail + 2 * NR, tail, depth, alpha,
                                    row_start, pw, peeled_cols + 2 * NR, sub_tail);
          break;
        default:
          sme_process_microblock<3>(C, C_stride_row, C_stride_col, blA, blB_tail + 3 * NR, tail, depth, alpha,
                                    row_start, pw, peeled_cols + 3 * NR, sub_tail);
          break;
      }
    }
  }
}

/*****************************************************************************
 * Top-level streaming entry point.
 *
 * Iterates over LHS panels (all uniform width = MR = 16, plus a tail),
 * dispatching sme_process_all_cols for each.
 *
 * NOTE: For very small matrices (< ~32 in all dimensions), the cost of
 * entering/exiting streaming SVE mode may exceed the FMOPA speedup.
 * A future improvement could bypass SME for such cases, but this requires
 * runtime dispatch before packing (since SME and NEON use different pack formats).
 *****************************************************************************/

template <typename Scalar, typename Index>
__attribute__((noinline)) __arm_locally_streaming __arm_new("za") void sme_gebp_impl(
    Scalar* C, Index C_stride_row, Index C_stride_col, const Scalar* blockA, const Scalar* blockB, Index rows,
    Index depth, Index cols, Scalar alpha, Index strideA, Index strideB, Index offsetA, Index offsetB) {
  constexpr int MR = kSmeMr;  // 16

  const Index peeled_rows = (rows / MR) * MR;

  // --- Full LHS panels (pw = MR = 16) ---
  for (Index i = 0; i < peeled_rows; i += MR) {
    const Scalar* blA = blockA + i * strideA + offsetA * MR;
    sme_process_all_cols(C, C_stride_row, C_stride_col, blA, blockB, depth, cols, alpha, strideB, offsetB, i, MR);
  }

  // --- Tail LHS panel (pw < MR) ---
  if (peeled_rows < rows) {
    const int tail = static_cast<int>(rows - peeled_rows);
    const Scalar* blA = blockA + peeled_rows * strideA + offsetA * tail;
    sme_process_all_cols(C, C_stride_row, C_stride_col, blA, blockB, depth, cols, alpha, strideB, offsetB, peeled_rows,
                         tail);
  }
}

/*****************************************************************************
 * gebp_kernel specialization for SME  (float x float)
 *
 * Extracts DataMapper into raw pointer + strides (outside streaming mode),
 * then delegates to sme_gebp_impl.
 *****************************************************************************/

template <typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
struct gebp_kernel<float, float, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs> {
  typedef float Scalar;
  typedef float ResScalar;

  EIGEN_DONT_INLINE void operator()(const DataMapper& res, const Scalar* blockA, const Scalar* blockB, Index rows,
                                    Index depth, Index cols, ResScalar alpha, Index strideA = -1, Index strideB = -1,
                                    Index offsetA = 0, Index offsetB = 0) {
    if (strideA == -1) strideA = depth;
    if (strideB == -1) strideB = depth;

    if (rows <= 0 || cols <= 0 || depth <= 0) return;

    // --- Extract DataMapper into raw pointer + strides (non-streaming) ---
    Scalar* C_base = const_cast<Scalar*>(&res(0, 0));
    const Index C_stride_row = &res(1, 0) - &res(0, 0);
    const Index C_stride_col = &res(0, 1) - &res(0, 0);

    // --- Delegate to streaming implementation ---
    sme_gebp_impl<Scalar, Index>(C_base, C_stride_row, C_stride_col, blockA, blockB, rows, depth, cols, alpha, strideA,
                                 strideB, offsetA, offsetB);
  }
};

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_SME_GENERALBLOCKPANELKERNEL_H
