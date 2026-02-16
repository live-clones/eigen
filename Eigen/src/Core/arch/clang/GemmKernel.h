// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2025 Rasmus Munk Larsen <rmlarsen@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Custom GEMM kernel for the clang generic vector backend.
//
// The clang generic backend uses fixed 64-byte vector types without
// half/quarter packet variants (half = full). This breaks the
// primary gebp_kernel which relies on progressive packet-size
// reduction for remainder handling and the "swapped product" path.
//
// Instead of adding half/quarter packet types, we provide complete
// gebp_kernel replacements that use simple scalar fallbacks for
// remainder rows, avoiding all half/quarter packet dependencies.

#ifndef EIGEN_CORE_ARCH_CLANG_GEMM_KERNEL_H
#define EIGEN_CORE_ARCH_CLANG_GEMM_KERNEL_H

namespace Eigen {
namespace internal {

// ============================================================
// gebp_traits specializations for Architecture::GenericVec
// ============================================================

// Same-type: float x float
template <bool ConjLhs_, bool ConjRhs_, int PacketSize_>
class gebp_traits<float, float, ConjLhs_, ConjRhs_, Architecture::GenericVec, PacketSize_>
    : public gebp_traits<float, float, ConjLhs_, ConjRhs_, Architecture::Generic, PacketSize_> {
  using Base = gebp_traits<float, float, ConjLhs_, ConjRhs_, Architecture::Generic, PacketSize_>;

 public:
  enum { nr = Base::Vectorizable ? 8 : 4 };
};

// Same-type: double x double
template <bool ConjLhs_, bool ConjRhs_, int PacketSize_>
class gebp_traits<double, double, ConjLhs_, ConjRhs_, Architecture::GenericVec, PacketSize_>
    : public gebp_traits<double, double, ConjLhs_, ConjRhs_, Architecture::Generic, PacketSize_> {
  using Base = gebp_traits<double, double, ConjLhs_, ConjRhs_, Architecture::Generic, PacketSize_>;

 public:
  enum { nr = Base::Vectorizable ? 8 : 4 };
};

// Mixed: real x complex
template <typename RealScalar, bool ConjRhs_, int PacketSize_>
class gebp_traits<RealScalar, std::complex<RealScalar>, false, ConjRhs_, Architecture::GenericVec, PacketSize_>
    : public gebp_traits<RealScalar, std::complex<RealScalar>, false, ConjRhs_, Architecture::Generic, PacketSize_> {
  using Base = gebp_traits<RealScalar, std::complex<RealScalar>, false, ConjRhs_, Architecture::Generic, PacketSize_>;

 public:
  enum { nr = Base::Vectorizable ? 8 : 4 };
};

// Mixed: complex x real
template <typename RealScalar, bool ConjLhs_, int PacketSize_>
class gebp_traits<std::complex<RealScalar>, RealScalar, ConjLhs_, false, Architecture::GenericVec, PacketSize_>
    : public gebp_traits<std::complex<RealScalar>, RealScalar, ConjLhs_, false, Architecture::Generic, PacketSize_> {
  using Base = gebp_traits<std::complex<RealScalar>, RealScalar, ConjLhs_, false, Architecture::Generic, PacketSize_>;

 public:
  enum { nr = Base::Vectorizable ? 8 : 4 };
};

#ifndef EIGEN_DONT_VECTORIZE

// ============================================================
// Custom gemm_pack_lhs for real*complex case
// ============================================================
//
// For real*complex, the LHS contains real values but is packed using
// a real packet (Packet16f/Packet8d) whose size is twice the complex
// packet size. This means Pack1 = Pack2 = CplxPacketSize < PacketSize,
// causing the standard pack_lhs to create variable-sized groups via
// its "last_lhs_progress" path. Our kernel expects fixed groups of
// LhsProgress (= CplxPacketSize) rows, so we provide custom pack_lhs
// that packs in consistent fixed-size groups.

template <typename Scalar, typename Index, typename DataMapper, int GroupSize, typename Packet, int StorageOrder,
          bool Conjugate, bool PanelMode>
EIGEN_DONT_INLINE void pack_lhs_fixed_groups(Scalar* blockA, const DataMapper& lhs, Index depth, Index rows,
                                             Index stride, Index offset) {
  eigen_assert(((!PanelMode) && stride == 0 && offset == 0) || (PanelMode && stride >= depth && offset <= stride));
  conj_if<NumTraits<Scalar>::IsComplex && Conjugate> cj;
  Index count = 0;
  const Index peeled_mc = (rows / GroupSize) * GroupSize;

  // Pack groups of GroupSize rows
  for (Index i = 0; i < peeled_mc; i += GroupSize) {
    if (PanelMode) count += GroupSize * offset;
    for (Index k = 0; k < depth; k++)
      for (int w = 0; w < GroupSize; w++) blockA[count++] = cj(lhs(i + w, k));
    if (PanelMode) count += GroupSize * (stride - offset - depth);
  }

  // Pack remaining rows one at a time
  for (Index i = peeled_mc; i < rows; i++) {
    if (PanelMode) count += offset;
    for (Index k = 0; k < depth; k++) blockA[count++] = cj(lhs(i, k));
    if (PanelMode) count += (stride - offset - depth);
  }
}

// float real*complex: Pack1=8, Pack2=8, Packet=Packet16f (PacketSize=16)
template <typename Index, typename DataMapper, bool Conjugate, bool PanelMode>
struct gemm_pack_lhs<float, Index, DataMapper, 8, 8, Packet16f, ColMajor, Conjugate, PanelMode> {
  EIGEN_DONT_INLINE void operator()(float* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride = 0,
                                    Index offset = 0) {
    pack_lhs_fixed_groups<float, Index, DataMapper, 8, Packet16f, ColMajor, Conjugate, PanelMode>(blockA, lhs, depth,
                                                                                                  rows, stride, offset);
  }
};
template <typename Index, typename DataMapper, bool Conjugate, bool PanelMode>
struct gemm_pack_lhs<float, Index, DataMapper, 8, 8, Packet16f, RowMajor, Conjugate, PanelMode> {
  EIGEN_DONT_INLINE void operator()(float* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride = 0,
                                    Index offset = 0) {
    pack_lhs_fixed_groups<float, Index, DataMapper, 8, Packet16f, RowMajor, Conjugate, PanelMode>(blockA, lhs, depth,
                                                                                                  rows, stride, offset);
  }
};

// double real*complex: Pack1=4, Pack2=4, Packet=Packet8d (PacketSize=8)
template <typename Index, typename DataMapper, bool Conjugate, bool PanelMode>
struct gemm_pack_lhs<double, Index, DataMapper, 4, 4, Packet8d, ColMajor, Conjugate, PanelMode> {
  EIGEN_DONT_INLINE void operator()(double* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride = 0,
                                    Index offset = 0) {
    pack_lhs_fixed_groups<double, Index, DataMapper, 4, Packet8d, ColMajor, Conjugate, PanelMode>(blockA, lhs, depth,
                                                                                                  rows, stride, offset);
  }
};
template <typename Index, typename DataMapper, bool Conjugate, bool PanelMode>
struct gemm_pack_lhs<double, Index, DataMapper, 4, 4, Packet8d, RowMajor, Conjugate, PanelMode> {
  EIGEN_DONT_INLINE void operator()(double* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride = 0,
                                    Index offset = 0) {
    pack_lhs_fixed_groups<double, Index, DataMapper, 4, Packet8d, RowMajor, Conjugate, PanelMode>(blockA, lhs, depth,
                                                                                                  rows, stride, offset);
  }
};

// ============================================================
// gemm_pack_rhs specializations for nr=8
// ============================================================

// ColMajor packing for nr=8
template <typename Scalar, typename Index, typename DataMapper, bool Conjugate, bool PanelMode>
struct gemm_pack_rhs<Scalar, Index, DataMapper, 8, ColMajor, Conjugate, PanelMode> {
  typedef typename DataMapper::LinearMapper LinearMapper;
  EIGEN_DONT_INLINE void operator()(Scalar* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride = 0,
                                    Index offset = 0);
};

template <typename Scalar, typename Index, typename DataMapper, bool Conjugate, bool PanelMode>
EIGEN_DONT_INLINE void gemm_pack_rhs<Scalar, Index, DataMapper, 8, ColMajor, Conjugate, PanelMode>::operator()(
    Scalar* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride, Index offset) {
  EIGEN_ASM_COMMENT("EIGEN PRODUCT PACK RHS COLMAJOR nr=8");
  eigen_assert(((!PanelMode) && stride == 0 && offset == 0) || (PanelMode && stride >= depth && offset <= stride));
  conj_if<NumTraits<Scalar>::IsComplex && Conjugate> cj;
  Index packet_cols8 = (cols / 8) * 8;
  Index packet_cols4 = (cols / 4) * 4;
  Index count = 0;

  // Pack 8-column blocks
  for (Index j2 = 0; j2 < packet_cols8; j2 += 8) {
    if (PanelMode) count += 8 * offset;
    const LinearMapper dm0 = rhs.getLinearMapper(0, j2 + 0);
    const LinearMapper dm1 = rhs.getLinearMapper(0, j2 + 1);
    const LinearMapper dm2 = rhs.getLinearMapper(0, j2 + 2);
    const LinearMapper dm3 = rhs.getLinearMapper(0, j2 + 3);
    const LinearMapper dm4 = rhs.getLinearMapper(0, j2 + 4);
    const LinearMapper dm5 = rhs.getLinearMapper(0, j2 + 5);
    const LinearMapper dm6 = rhs.getLinearMapper(0, j2 + 6);
    const LinearMapper dm7 = rhs.getLinearMapper(0, j2 + 7);
    for (Index k = 0; k < depth; k++) {
      blockB[count + 0] = cj(dm0(k));
      blockB[count + 1] = cj(dm1(k));
      blockB[count + 2] = cj(dm2(k));
      blockB[count + 3] = cj(dm3(k));
      blockB[count + 4] = cj(dm4(k));
      blockB[count + 5] = cj(dm5(k));
      blockB[count + 6] = cj(dm6(k));
      blockB[count + 7] = cj(dm7(k));
      count += 8;
    }
    if (PanelMode) count += 8 * (stride - offset - depth);
  }

  // Pack 4-column blocks
  for (Index j2 = packet_cols8; j2 < packet_cols4; j2 += 4) {
    if (PanelMode) count += 4 * offset;
    const LinearMapper dm0 = rhs.getLinearMapper(0, j2 + 0);
    const LinearMapper dm1 = rhs.getLinearMapper(0, j2 + 1);
    const LinearMapper dm2 = rhs.getLinearMapper(0, j2 + 2);
    const LinearMapper dm3 = rhs.getLinearMapper(0, j2 + 3);
    for (Index k = 0; k < depth; k++) {
      blockB[count + 0] = cj(dm0(k));
      blockB[count + 1] = cj(dm1(k));
      blockB[count + 2] = cj(dm2(k));
      blockB[count + 3] = cj(dm3(k));
      count += 4;
    }
    if (PanelMode) count += 4 * (stride - offset - depth);
  }

  // Pack remaining columns one at a time
  for (Index j2 = packet_cols4; j2 < cols; ++j2) {
    if (PanelMode) count += offset;
    const LinearMapper dm0 = rhs.getLinearMapper(0, j2);
    for (Index k = 0; k < depth; k++) {
      blockB[count] = cj(dm0(k));
      count += 1;
    }
    if (PanelMode) count += (stride - offset - depth);
  }
}

// RowMajor packing for nr=8
template <typename Scalar, typename Index, typename DataMapper, bool Conjugate, bool PanelMode>
struct gemm_pack_rhs<Scalar, Index, DataMapper, 8, RowMajor, Conjugate, PanelMode> {
  typedef typename DataMapper::LinearMapper LinearMapper;
  EIGEN_DONT_INLINE void operator()(Scalar* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride = 0,
                                    Index offset = 0) {
    EIGEN_ASM_COMMENT("EIGEN PRODUCT PACK RHS ROWMAJOR nr=8");
    eigen_assert(((!PanelMode) && stride == 0 && offset == 0) || (PanelMode && stride >= depth && offset <= stride));
    conj_if<NumTraits<Scalar>::IsComplex && Conjugate> cj;
    Index packet_cols8 = (cols / 8) * 8;
    Index packet_cols4 = (cols / 4) * 4;
    Index count = 0;

    for (Index j2 = 0; j2 < packet_cols8; j2 += 8) {
      if (PanelMode) count += 8 * offset;
      for (Index k = 0; k < depth; k++) {
        const LinearMapper dm0 = rhs.getLinearMapper(k, j2);
        blockB[count + 0] = cj(dm0(0));
        blockB[count + 1] = cj(dm0(1));
        blockB[count + 2] = cj(dm0(2));
        blockB[count + 3] = cj(dm0(3));
        blockB[count + 4] = cj(dm0(4));
        blockB[count + 5] = cj(dm0(5));
        blockB[count + 6] = cj(dm0(6));
        blockB[count + 7] = cj(dm0(7));
        count += 8;
      }
      if (PanelMode) count += 8 * (stride - offset - depth);
    }

    for (Index j2 = packet_cols8; j2 < packet_cols4; j2 += 4) {
      if (PanelMode) count += 4 * offset;
      for (Index k = 0; k < depth; k++) {
        const LinearMapper dm0 = rhs.getLinearMapper(k, j2);
        blockB[count + 0] = cj(dm0(0));
        blockB[count + 1] = cj(dm0(1));
        blockB[count + 2] = cj(dm0(2));
        blockB[count + 3] = cj(dm0(3));
        count += 4;
      }
      if (PanelMode) count += 4 * (stride - offset - depth);
    }

    for (Index j2 = packet_cols4; j2 < cols; ++j2) {
      if (PanelMode) count += offset;
      for (Index k = 0; k < depth; k++) {
        blockB[count] = cj(rhs(k, j2));
        count += 1;
      }
      if (PanelMode) count += stride - offset - depth;
    }
  }
};

// ============================================================
// gebp_kernel specialization: same-type (Scalar x Scalar), nr=8
// ============================================================

template <typename Scalar, typename Index, typename DataMapper, int mr, bool ConjugateLhs, bool ConjugateRhs>
struct gebp_kernel<Scalar, Scalar, Index, DataMapper, mr, 8, ConjugateLhs, ConjugateRhs> {
  EIGEN_DONT_INLINE void operator()(const DataMapper& res, const Scalar* blockA, const Scalar* blockB, Index rows,
                                    Index depth, Index cols, Scalar alpha, Index strideA = -1, Index strideB = -1,
                                    Index offsetA = 0, Index offsetB = 0);
};

template <typename Scalar, typename Index, typename DataMapper, int mr, bool ConjugateLhs, bool ConjugateRhs>
EIGEN_DONT_INLINE void gebp_kernel<Scalar, Scalar, Index, DataMapper, mr, 8, ConjugateLhs, ConjugateRhs>::operator()(
    const DataMapper& res, const Scalar* blockA, const Scalar* blockB, Index rows, Index depth, Index cols,
    Scalar alpha, Index strideA, Index strideB, Index offsetA, Index offsetB) {
  typedef typename packet_traits<Scalar>::type Packet;
  constexpr Index PacketSize = packet_traits<Scalar>::size;
  typedef typename DataMapper::LinearMapper LinearMapper;

  if (strideA == -1) strideA = depth;
  if (strideB == -1) strideB = depth;

  const Index packet_cols8 = (cols / 8) * 8;
  const Index packet_cols4 = (cols / 4) * 4;

  // Row peeling: match gemm_pack_lhs block sizes.
  const Index peeled_mc3 = mr >= 3 * PacketSize ? (rows / (3 * PacketSize)) * (3 * PacketSize) : 0;
  const Index peeled_mc2 =
      mr >= 2 * PacketSize ? peeled_mc3 + ((rows - peeled_mc3) / (2 * PacketSize)) * (2 * PacketSize) : 0;
  const Index peeled_mc1 = peeled_mc2 + ((rows - peeled_mc2) / PacketSize) * PacketSize;

  // Helper: process MrP packet-rows x NrC columns.
  // blockA layout: MrP*PacketSize values per depth step.
  // blockB layout: NrC values per depth step.
  auto process_block = [&](Index i, Index j, int MrP, int NrC) {
    constexpr int MaxMr = 3;
    constexpr int MaxNr = 8;
    const Scalar* blA = &blockA[i * strideA + offsetA * (MrP * PacketSize)];
    const Scalar* blB = &blockB[j * strideB + offsetB * NrC];
    Packet C[MaxMr * MaxNr];
    for (int n = 0; n < MrP * NrC; ++n) C[n] = pset1<Packet>(Scalar(0));

    for (Index k = 0; k < depth; ++k) {
      Packet A[MaxMr];
      for (int p = 0; p < MrP; ++p) A[p] = ploadu<Packet>(&blA[k * MrP * PacketSize + p * PacketSize]);
      for (int c = 0; c < NrC; ++c) {
        Packet B = pset1<Packet>(blB[k * NrC + c]);
        for (int p = 0; p < MrP; ++p) C[c + p * NrC] = pmadd(A[p], B, C[c + p * NrC]);
      }
    }

    Packet alphav = pset1<Packet>(alpha);
    for (int c = 0; c < NrC; ++c) {
      LinearMapper r = res.getLinearMapper(i, j + c);
      for (int p = 0; p < MrP; ++p) {
        Packet R = r.template loadPacket<Packet>(p * PacketSize);
        r.storePacket(p * PacketSize, pmadd(C[c + p * NrC], alphav, R));
      }
    }
  };

  // Helper: process a single scalar row.
  auto process_row = [&](Index i, Index j, int NrC) {
    const Scalar* blA = &blockA[i * strideA + offsetA];
    const Scalar* blB = &blockB[j * strideB + offsetB * NrC];
    Scalar C[8] = {};

    for (Index k = 0; k < depth; ++k) {
      Scalar A0 = blA[k];
      for (int c = 0; c < NrC; ++c) C[c] += A0 * blB[k * NrC + c];
    }
    for (int c = 0; c < NrC; ++c) res(i, j + c) += alpha * C[c];
  };

  // --- 3-packet rows ---
  for (Index i = 0; i < peeled_mc3; i += 3 * PacketSize) {
    for (Index j = 0; j < packet_cols8; j += 8) process_block(i, j, 3, 8);
    for (Index j = packet_cols8; j < packet_cols4; j += 4) process_block(i, j, 3, 4);
    for (Index j = packet_cols4; j < cols; j++) process_block(i, j, 3, 1);
  }

  // --- 2-packet rows ---
  for (Index i = peeled_mc3; i < peeled_mc2; i += 2 * PacketSize) {
    for (Index j = 0; j < packet_cols8; j += 8) process_block(i, j, 2, 8);
    for (Index j = packet_cols8; j < packet_cols4; j += 4) process_block(i, j, 2, 4);
    for (Index j = packet_cols4; j < cols; j++) process_block(i, j, 2, 1);
  }

  // --- 1-packet rows ---
  for (Index i = peeled_mc2; i < peeled_mc1; i += PacketSize) {
    for (Index j = 0; j < packet_cols8; j += 8) process_block(i, j, 1, 8);
    for (Index j = packet_cols8; j < packet_cols4; j += 4) process_block(i, j, 1, 4);
    for (Index j = packet_cols4; j < cols; j++) process_block(i, j, 1, 1);
  }

  // --- Remaining scalar rows ---
  for (Index i = peeled_mc1; i < rows; ++i) {
    for (Index j = 0; j < packet_cols8; j += 8) process_row(i, j, 8);
    for (Index j = packet_cols8; j < packet_cols4; j += 4) process_row(i, j, 4);
    for (Index j = packet_cols4; j < cols; j++) process_row(i, j, 1);
  }
}

// ============================================================
// gebp_kernel specialization: real x complex, nr=8
// ============================================================
// For real*complex: blockA contains reals, blockB contains complex values.
// LhsProgress = CplxPacketSize rows are processed at a time using
// ploaddup to duplicate each real for re/im components.

template <typename RealScalar, typename Index, typename DataMapper, int mr, bool ConjugateLhs, bool ConjugateRhs>
struct gebp_kernel<RealScalar, std::complex<RealScalar>, Index, DataMapper, mr, 8, ConjugateLhs, ConjugateRhs> {
  typedef std::complex<RealScalar> CplxScalar;
  EIGEN_DONT_INLINE void operator()(const DataMapper& res, const RealScalar* blockA, const CplxScalar* blockB,
                                    Index rows, Index depth, Index cols, CplxScalar alpha, Index strideA = -1,
                                    Index strideB = -1, Index offsetA = 0, Index offsetB = 0);
};

template <typename RealScalar, typename Index, typename DataMapper, int mr, bool ConjugateLhs, bool ConjugateRhs>
EIGEN_DONT_INLINE void
gebp_kernel<RealScalar, std::complex<RealScalar>, Index, DataMapper, mr, 8, ConjugateLhs, ConjugateRhs>::operator()(
    const DataMapper& res, const RealScalar* blockA, const std::complex<RealScalar>* blockB, Index rows, Index depth,
    Index cols, std::complex<RealScalar> alpha, Index strideA, Index strideB, Index offsetA, Index offsetB) {
  typedef typename packet_traits<RealScalar>::type RealPacket;
  typedef typename packet_traits<CplxScalar>::type CplxPacket;
  constexpr Index CplxPacketSize = packet_traits<CplxScalar>::size;
  // LhsProgress = ResPacketSize = CplxPacketSize (from base traits).
  // pack_lhs packs CplxPacketSize reals per depth step.
  constexpr Index LhsProgress = CplxPacketSize;
  typedef typename DataMapper::LinearMapper LinearMapper;

  if (strideA == -1) strideA = depth;
  if (strideB == -1) strideB = depth;

  const Index packet_cols8 = (cols / 8) * 8;
  const Index packet_cols4 = (cols / 4) * 4;
  const Index peeled_mc1 = (rows / LhsProgress) * LhsProgress;

  // Vectorized block: process LhsProgress rows x NrC complex columns.
  // blockA: LhsProgress reals per depth step (from ploaddup-style packing).
  // blockB: NrC complex values per depth step.
  auto process_vec = [&](Index i, Index j, int NrC) {
    const RealScalar* blA = &blockA[i * strideA + offsetA * LhsProgress];
    const CplxScalar* blB = &blockB[j * strideB + offsetB * NrC];

    // Accumulators: one CplxPacket per column.
    // CplxPacketSize complex values = 2*CplxPacketSize reals per packet.
    CplxPacket C[8];
    for (int c = 0; c < NrC; ++c) C[c] = pset1<CplxPacket>(CplxScalar(0));

    for (Index k = 0; k < depth; ++k) {
      // Load LhsProgress reals and duplicate each for re/im: ploaddup gives
      // (a0,a0,a1,a1,...) matching the (re,im,re,im,...) layout of complex packets.
      RealPacket A = ploaddup<RealPacket>(&blA[k * LhsProgress]);
      for (int c = 0; c < NrC; ++c) {
        CplxPacket B = pset1<CplxPacket>(blB[k * NrC + c]);
        // Element-wise: A * B.v + C[c].v (real packet arithmetic on interleaved complex)
        C[c].v = pmadd(A, B.v, C[c].v);
      }
    }

    // Apply alpha and conjugation, store results.
    // conj_helper handles: alpha * conj_if(ConjRhs, C[c]) + R
    CplxPacket alphav = pset1<CplxPacket>(alpha);
    conj_helper<CplxPacket, CplxPacket, false, ConjugateRhs> cjr;
    for (int c = 0; c < NrC; ++c) {
      LinearMapper r = res.getLinearMapper(i, j + c);
      CplxPacket R = r.template loadPacket<CplxPacket>(0);
      r.storePacket(0, cjr.pmadd(alphav, C[c], R));
    }
  };

  // Scalar row: one row at a time.
  auto process_row = [&](Index i, Index j, int NrC) {
    const RealScalar* blA = &blockA[i * strideA + offsetA];
    const CplxScalar* blB = &blockB[j * strideB + offsetB * NrC];
    conj_helper<RealScalar, CplxScalar, ConjugateLhs, ConjugateRhs> cj;
    CplxScalar C[8] = {};
    for (Index k = 0; k < depth; ++k) {
      RealScalar A0 = blA[k];
      for (int c = 0; c < NrC; ++c) C[c] += cj.pmul(A0, blB[k * NrC + c]);
    }
    for (int c = 0; c < NrC; ++c) res(i, j + c) += alpha * C[c];
  };

  // Vectorized rows
  for (Index i = 0; i < peeled_mc1; i += LhsProgress) {
    for (Index j = 0; j < packet_cols8; j += 8) process_vec(i, j, 8);
    for (Index j = packet_cols8; j < packet_cols4; j += 4) process_vec(i, j, 4);
    for (Index j = packet_cols4; j < cols; j++) process_vec(i, j, 1);
  }

  // Remaining scalar rows
  for (Index i = peeled_mc1; i < rows; ++i) {
    for (Index j = 0; j < packet_cols8; j += 8) process_row(i, j, 8);
    for (Index j = packet_cols8; j < packet_cols4; j += 4) process_row(i, j, 4);
    for (Index j = packet_cols4; j < cols; j++) process_row(i, j, 1);
  }
}

// ============================================================
// gebp_kernel specialization: complex x real, nr=8
// ============================================================
// For complex*real: blockA contains complex values, blockB contains reals.
// LhsProgress = CplxPacketSize rows processed at a time.
// complex(re,im) * b = complex(re*b, im*b)

template <typename RealScalar, typename Index, typename DataMapper, int mr, bool ConjugateLhs, bool ConjugateRhs>
struct gebp_kernel<std::complex<RealScalar>, RealScalar, Index, DataMapper, mr, 8, ConjugateLhs, ConjugateRhs> {
  typedef std::complex<RealScalar> CplxScalar;
  EIGEN_DONT_INLINE void operator()(const DataMapper& res, const CplxScalar* blockA, const RealScalar* blockB,
                                    Index rows, Index depth, Index cols, CplxScalar alpha, Index strideA = -1,
                                    Index strideB = -1, Index offsetA = 0, Index offsetB = 0);
};

template <typename RealScalar, typename Index, typename DataMapper, int mr, bool ConjugateLhs, bool ConjugateRhs>
EIGEN_DONT_INLINE void
gebp_kernel<std::complex<RealScalar>, RealScalar, Index, DataMapper, mr, 8, ConjugateLhs, ConjugateRhs>::operator()(
    const DataMapper& res, const std::complex<RealScalar>* blockA, const RealScalar* blockB, Index rows, Index depth,
    Index cols, std::complex<RealScalar> alpha, Index strideA, Index strideB, Index offsetA, Index offsetB) {
  typedef typename packet_traits<RealScalar>::type RealPacket;
  typedef typename packet_traits<CplxScalar>::type CplxPacket;
  constexpr Index CplxPacketSize = packet_traits<CplxScalar>::size;
  // LhsProgress = LhsPacketSize = CplxPacketSize (from base traits).
  constexpr Index LhsProgress = CplxPacketSize;
  typedef typename DataMapper::LinearMapper LinearMapper;

  if (strideA == -1) strideA = depth;
  if (strideB == -1) strideB = depth;

  const Index packet_cols8 = (cols / 8) * 8;
  const Index packet_cols4 = (cols / 4) * 4;
  const Index peeled_mc1 = (rows / LhsProgress) * LhsProgress;

  // Vectorized block: LhsProgress complex rows x NrC real columns.
  // blockA: LhsProgress complex values per depth step.
  // blockB: NrC real values per depth step.
  auto process_vec = [&](Index i, Index j, int NrC) {
    const CplxScalar* blA = &blockA[i * strideA + offsetA * LhsProgress];
    const RealScalar* blB = &blockB[j * strideB + offsetB * NrC];

    CplxPacket C[8];
    for (int c = 0; c < NrC; ++c) C[c] = pset1<CplxPacket>(CplxScalar(0));

    for (Index k = 0; k < depth; ++k) {
      // Load LhsProgress complex values as a CplxPacket.
      CplxPacket A = ploadu<CplxPacket>(&blA[k * LhsProgress]);
      for (int c = 0; c < NrC; ++c) {
        // Broadcast real RHS value, then element-wise FMA on interleaved representation.
        RealPacket B = pset1<RealPacket>(blB[k * NrC + c]);
        C[c].v = pmadd(A.v, B, C[c].v);
      }
    }

    // Apply conjugation and alpha, store results.
    // conj_helper handles: conj_if(ConjLhs, C[c]) * alpha + R
    CplxPacket alphav = pset1<CplxPacket>(alpha);
    conj_helper<CplxPacket, CplxPacket, ConjugateLhs, false> cjl;
    for (int c = 0; c < NrC; ++c) {
      LinearMapper r = res.getLinearMapper(i, j + c);
      CplxPacket R = r.template loadPacket<CplxPacket>(0);
      r.storePacket(0, cjl.pmadd(C[c], alphav, R));
    }
  };

  // Scalar row
  auto process_row = [&](Index i, Index j, int NrC) {
    const CplxScalar* blA = &blockA[i * strideA + offsetA];
    const RealScalar* blB = &blockB[j * strideB + offsetB * NrC];
    conj_helper<CplxScalar, RealScalar, ConjugateLhs, ConjugateRhs> cj;
    CplxScalar C[8] = {};
    for (Index k = 0; k < depth; ++k) {
      CplxScalar A0 = blA[k];
      for (int c = 0; c < NrC; ++c) C[c] += cj.pmul(A0, blB[k * NrC + c]);
    }
    for (int c = 0; c < NrC; ++c) res(i, j + c) += alpha * C[c];
  };

  // Vectorized rows
  for (Index i = 0; i < peeled_mc1; i += LhsProgress) {
    for (Index j = 0; j < packet_cols8; j += 8) process_vec(i, j, 8);
    for (Index j = packet_cols8; j < packet_cols4; j += 4) process_vec(i, j, 4);
    for (Index j = packet_cols4; j < cols; j++) process_vec(i, j, 1);
  }

  // Remaining scalar rows
  for (Index i = peeled_mc1; i < rows; ++i) {
    for (Index j = 0; j < packet_cols8; j += 8) process_row(i, j, 8);
    for (Index j = packet_cols8; j < packet_cols4; j += 4) process_row(i, j, 4);
    for (Index j = packet_cols4; j < cols; j++) process_row(i, j, 1);
  }
}

#endif  // !EIGEN_DONT_VECTORIZE

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_CORE_ARCH_CLANG_GEMM_KERNEL_H
