// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2011-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2011-2012 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_ASSIGN_EVALUATOR_H
#define EIGEN_ASSIGN_EVALUATOR_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

// This implementation is based on Assign.h

namespace internal {

/***************************************************************************
 * Part 1 : the logic deciding a strategy for traversal and unrolling       *
 ***************************************************************************/

// copy_using_evaluator_traits is based on assign_traits

template <typename DstEvaluator, typename SrcEvaluator, typename AssignFunc, int MaxPacketSize = -1>
struct copy_using_evaluator_traits {
  using Src = typename SrcEvaluator::XprType;
  using Dst = typename DstEvaluator::XprType;
  using DstScalar = typename Dst::Scalar;

  static constexpr int DstFlags = DstEvaluator::Flags, SrcFlags = SrcEvaluator::Flags;

 public:
  static constexpr bool DstHasDirectAccess = bool(DstFlags & DirectAccessBit),
                        DstIsRowMajor = bool(DstFlags & RowMajorBit),
                        DstIsVectorAtCompileTime = Dst::IsVectorAtCompileTime,
                        SrcIsRowMajor = bool(SrcFlags & RowMajorBit);
  static constexpr int DstAlignment = DstEvaluator::Alignment, SrcAlignment = SrcEvaluator::Alignment,
                       JointAlignment = plain_enum_min(DstAlignment, SrcAlignment);

  // private:
  static constexpr int RowsAtCompileTime = size_prefer_fixed(Src::RowsAtCompileTime, Dst::RowsAtCompileTime),
                       ColsAtCompileTime = size_prefer_fixed(Src::ColsAtCompileTime, Dst::ColsAtCompileTime),
                       // SizeAtCompileTime = size_prefer_fixed(Src::SizeAtCompileTime, Dst::SizeAtCompileTime),
      SizeAtCompileTime = size_prefer_fixed(Src::SizeAtCompileTime, Dst::SizeAtCompileTime),
                       MaxRowsAtCompileTime =
                           min_size_prefer_fixed(Src::MaxRowsAtCompileTime, Dst::MaxRowsAtCompileTime),
                       MaxColsAtCompileTime =
                           min_size_prefer_fixed(Src::MaxColsAtCompileTime, Dst::MaxColsAtCompileTime),
                       MaxSizeAtCompileTime =
                           min_size_prefer_fixed(Src::MaxSizeAtCompileTime, Dst::MaxSizeAtCompileTime),
                       InnerSizeAtCompileTime = DstIsVectorAtCompileTime
                                                    ? SizeAtCompileTime
                                                    : (DstIsRowMajor ? ColsAtCompileTime : RowsAtCompileTime),
                       MaxInnerSizeAtCompileTime = DstIsVectorAtCompileTime
                                                       ? MaxSizeAtCompileTime
                                                       : (DstIsRowMajor ? MaxColsAtCompileTime : MaxRowsAtCompileTime),
                       RestrictedInnerSize = min_size_prefer_fixed(InnerSizeAtCompileTime, MaxPacketSize),
                       RestrictedLinearSize = min_size_prefer_fixed(SizeAtCompileTime, MaxPacketSize),
                       OuterStride = outer_stride_at_compile_time<Dst>::ret;

  // TODO distinguish between linear traversal and inner-traversals
  using LinearPacketType = typename find_best_packet<DstScalar, RestrictedLinearSize>::type;
  using InnerPacketType = typename find_best_packet<DstScalar, RestrictedInnerSize>::type;

  static constexpr int LinearPacketSize = unpacket_traits<LinearPacketType>::size,
                       InnerPacketSize = unpacket_traits<InnerPacketType>::size;

 public:
  static constexpr int LinearRequiredAlignment = unpacket_traits<LinearPacketType>::alignment,
                       InnerRequiredAlignment = unpacket_traits<InnerPacketType>::alignment;

 private:
  static constexpr bool StorageOrdersAgree = DstIsRowMajor == SrcIsRowMajor,
                        MightVectorize = StorageOrdersAgree && bool(DstFlags & SrcFlags & ActualPacketAccessBit) &&
                                         bool(functor_traits<AssignFunc>::PacketAccess),
                        InnerAlignmentOk = EIGEN_UNALIGNED_VECTORIZE || (JointAlignment >= InnerRequiredAlignment),
                        MayInnerVectorize = MightVectorize && (InnerSizeAtCompileTime != Dynamic) &&
                                            (InnerSizeAtCompileTime % InnerPacketSize == 0) &&
                                            (OuterStride != Dynamic) && (OuterStride % InnerPacketSize == 0) &&
                                            InnerAlignmentOk,
                        MayLinearize = StorageOrdersAgree && bool(DstFlags & SrcFlags & LinearAccessBit),
                        LinearAlignmentOk = EIGEN_UNALIGNED_VECTORIZE || (DstAlignment >= LinearRequiredAlignment),
                        MayLinearVectorize =
                            MightVectorize && MayLinearize && DstHasDirectAccess &&
                            (LinearAlignmentOk ||
                             (MaxSizeAtCompileTime == Dynamic ? true : MaxSizeAtCompileTime >= LinearPacketSize)),
                        /* If the destination isn't aligned, we have to do runtime checks and we don't unroll, so it's
                           only good for large enough sizes. slice vectorization can be slow, so we only want it if the
                           slices are big, which is indicated by MaxInnerSizeAtCompileTime rather than
                           InnerSizeAtCompileTime, think of the case of a dynamic block in a fixed-size matrix. However,
                           with EIGEN_UNALIGNED_VECTORIZE and unrolling, slice vectorization is still worth it */
      MaySliceVectorize =
          MightVectorize && DstHasDirectAccess &&
          (MaxInnerSizeAtCompileTime == Dynamic ||
           MaxInnerSizeAtCompileTime >= (EIGEN_UNALIGNED_VECTORIZE ? InnerPacketSize : (3 * InnerPacketSize)));

 public:
  // If compile-size is zero, traversing will fail at compile-time.
  static constexpr int Traversal = SizeAtCompileTime == 0 ? AllAtOnceTraversal
                                   : MayLinearVectorize && (LinearPacketSize > InnerPacketSize)
                                       ? LinearVectorizedTraversal
                                   : MayInnerVectorize  ? InnerVectorizedTraversal
                                   : MayLinearVectorize ? LinearVectorizedTraversal
                                   : MaySliceVectorize  ? SliceVectorizedTraversal
                                   : MayLinearize       ? LinearTraversal
                                                        : DefaultTraversal;
  static constexpr bool Vectorized = (Traversal == InnerVectorizedTraversal) ||
                                     (Traversal == LinearVectorizedTraversal) ||
                                     (Traversal == SliceVectorizedTraversal);

  using PacketType = std::conditional_t<Traversal == LinearVectorizedTraversal, LinearPacketType, InnerPacketType>;

 private:
  static constexpr int ActualPacketSize = Vectorized ? unpacket_traits<PacketType>::size : 1,
                       UnrollingLimit = EIGEN_UNROLLING_LIMIT * ActualPacketSize,
                       CoeffReadCost = int(DstEvaluator::CoeffReadCost) + int(SrcEvaluator::CoeffReadCost);
  static constexpr bool MayUnrollCompletely =
                            (SizeAtCompileTime != Dynamic) && ((SizeAtCompileTime * CoeffReadCost) <= UnrollingLimit),
                        MayUnrollInner = (InnerSizeAtCompileTime != Dynamic) &&
                                         ((InnerSizeAtCompileTime * CoeffReadCost) <= UnrollingLimit);

 public:
  static constexpr int Unrolling =
      (Traversal == InnerVectorizedTraversal || Traversal == DefaultTraversal)
          ? (MayUnrollCompletely ? CompleteUnrolling
             : MayUnrollInner    ? InnerUnrolling
                                 : NoUnrolling)
      : Traversal == LinearVectorizedTraversal
          ? (MayUnrollCompletely && LinearAlignmentOk ? CompleteUnrolling : NoUnrolling)
      : Traversal == LinearTraversal ? (MayUnrollCompletely ? CompleteUnrolling : NoUnrolling)
#if EIGEN_UNALIGNED_VECTORIZE
      : Traversal == SliceVectorizedTraversal ? (MayUnrollInner ? InnerUnrolling : NoUnrolling)
#endif
                                              : NoUnrolling;

#ifdef EIGEN_DEBUG_ASSIGN
  static void debug() {
    std::cerr << "DstXpr: " << typeid(typename DstEvaluator::XprType).name() << std::endl;
    std::cerr << "SrcXpr: " << typeid(typename SrcEvaluator::XprType).name() << std::endl;
    std::cerr.setf(std::ios::hex, std::ios::basefield);
    std::cerr << "DstFlags"
              << " = " << DstFlags << " (" << demangle_flags(DstFlags) << " )" << std::endl;
    std::cerr << "SrcFlags"
              << " = " << SrcFlags << " (" << demangle_flags(SrcFlags) << " )" << std::endl;
    std::cerr.unsetf(std::ios::hex);
    EIGEN_DEBUG_VAR(DstAlignment)
    EIGEN_DEBUG_VAR(SrcAlignment)
    EIGEN_DEBUG_VAR(LinearRequiredAlignment)
    EIGEN_DEBUG_VAR(InnerRequiredAlignment)
    EIGEN_DEBUG_VAR(JointAlignment)
    EIGEN_DEBUG_VAR(InnerSizeAtCompileTime)
    EIGEN_DEBUG_VAR(MaxInnerSizeAtCompileTime)
    EIGEN_DEBUG_VAR(LinearPacketSize)
    EIGEN_DEBUG_VAR(InnerPacketSize)
    EIGEN_DEBUG_VAR(ActualPacketSize)
    EIGEN_DEBUG_VAR(StorageOrdersAgree)
    EIGEN_DEBUG_VAR(MightVectorize)
    EIGEN_DEBUG_VAR(MayLinearize)
    EIGEN_DEBUG_VAR(MayInnerVectorize)
    EIGEN_DEBUG_VAR(MayLinearVectorize)
    EIGEN_DEBUG_VAR(MaySliceVectorize)
    std::cerr << "Traversal"
              << " = " << Traversal << " (" << demangle_traversal(Traversal) << ")" << std::endl;
    EIGEN_DEBUG_VAR(SrcEvaluator::CoeffReadCost)
    EIGEN_DEBUG_VAR(DstEvaluator::CoeffReadCost)
    EIGEN_DEBUG_VAR(Dst::SizeAtCompileTime)
    EIGEN_DEBUG_VAR(UnrollingLimit)
    EIGEN_DEBUG_VAR(MayUnrollCompletely)
    EIGEN_DEBUG_VAR(MayUnrollInner)
    std::cerr << "Unrolling"
              << " = " << Unrolling << " (" << demangle_unrolling(Unrolling) << ")" << std::endl;
    std::cerr << std::endl;
  }
#endif
};

/***************************************************************************
 * Part 2 : meta-unrollers
 ***************************************************************************/

/************************
*** Default traversal ***
************************/

template <typename Kernel, int Index_, int Stop>
struct copy_using_evaluator_DefaultTraversal_CompleteUnrolling {
  using Traits = typename Kernel::AssignmentTraits;
  static constexpr int Outer = Index_ / Traits::InnerSizeAtCompileTime, Inner = Index_ % Traits::InnerSizeAtCompileTime;

  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void run(Kernel& kernel) {
    kernel.assignCoeffByOuterInner(Outer, Inner);
    copy_using_evaluator_DefaultTraversal_CompleteUnrolling<Kernel, Index_ + 1, Stop>::run(kernel);
  }
};

template <typename Kernel, int Stop>
struct copy_using_evaluator_DefaultTraversal_CompleteUnrolling<Kernel, Stop, Stop> {
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE EIGEN_CONSTEXPR void run(Kernel&) {}
};

template <typename Kernel, int Index_, int Stop>
struct copy_using_evaluator_DefaultTraversal_InnerUnrolling {
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void run(Kernel& kernel, Index outer) {
    kernel.assignCoeffByOuterInner(outer, Index_);
    copy_using_evaluator_DefaultTraversal_InnerUnrolling<Kernel, Index_ + 1, Stop>::run(kernel, outer);
  }
};

template <typename Kernel, int Stop>
struct copy_using_evaluator_DefaultTraversal_InnerUnrolling<Kernel, Stop, Stop> {
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void run(Kernel&, Index) {}
};

/***********************
*** Linear traversal ***
***********************/

template <typename Kernel, int Index_, int Stop>
struct copy_using_evaluator_LinearTraversal_CompleteUnrolling {
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void run(Kernel& kernel) {
    kernel.assignCoeff(Index_);
    copy_using_evaluator_LinearTraversal_CompleteUnrolling<Kernel, Index_ + 1, Stop>::run(kernel);
  }
};

template <typename Kernel, int Stop>
struct copy_using_evaluator_LinearTraversal_CompleteUnrolling<Kernel, Stop, Stop> {
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void run(Kernel&) {}
};

/**************************
*** Inner vectorization ***
**************************/

template <typename Kernel, int Index_, int Stop>
struct copy_using_evaluator_innervec_CompleteUnrolling {
  using Traits = typename Kernel::AssignmentTraits;
  using PacketType = typename Kernel::PacketType;
  static constexpr int Outer = Index_ / Traits::InnerSizeAtCompileTime, Inner = Index_ % Traits::InnerSizeAtCompileTime,
                       SrcAlignment = Traits::SrcAlignment, DstAlignment = Traits::DstAlignment,
                       NextIndex = Index_ + unpacket_traits<PacketType>::size;

  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void run(Kernel& kernel) {
    kernel.template assignPacketByOuterInner<DstAlignment, SrcAlignment, PacketType>(Outer, Inner);
    copy_using_evaluator_innervec_CompleteUnrolling<Kernel, NextIndex, Stop>::run(kernel);
  }
};

template <typename Kernel, int Stop>
struct copy_using_evaluator_innervec_CompleteUnrolling<Kernel, Stop, Stop> {
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE EIGEN_CONSTEXPR void run(Kernel&) {}
};

template <typename Kernel, int Index_, int Stop, int SrcAlignment, int DstAlignment>
struct copy_using_evaluator_innervec_InnerUnrolling {
  using PacketType = typename Kernel::PacketType;
  static constexpr int NextIndex = Index_ + unpacket_traits<PacketType>::size;

  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void run(Kernel& kernel, Index outer) {
    kernel.template assignPacketByOuterInner<DstAlignment, SrcAlignment, PacketType>(outer, Index_);
    copy_using_evaluator_innervec_InnerUnrolling<Kernel, NextIndex, Stop, SrcAlignment, DstAlignment>::run(kernel,
                                                                                                           outer);
  }
};

template <typename Kernel, int Stop, int SrcAlignment, int DstAlignment>
struct copy_using_evaluator_innervec_InnerUnrolling<Kernel, Stop, Stop, SrcAlignment, DstAlignment> {
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void run(Kernel&, Index) {}
};

/***************************************************************************
 * Part 3 : implementation of all cases
 ***************************************************************************/

// dense_assignment_loop is based on assign_impl

template <typename Kernel, int Traversal = Kernel::AssignmentTraits::Traversal,
          int Unrolling = Kernel::AssignmentTraits::Unrolling>
struct dense_assignment_loop;

/************************
***** Special Cases *****
************************/

// Zero-sized assignment is a no-op.
template <typename Kernel, int Unrolling>
struct dense_assignment_loop<Kernel, AllAtOnceTraversal, Unrolling> {
  using Traits = typename Kernel::AssignmentTraits;
  EIGEN_DEVICE_FUNC static void EIGEN_STRONG_INLINE EIGEN_CONSTEXPR run(Kernel& /*kernel*/) {
    EIGEN_STATIC_ASSERT(int(Traits::SizeAtCompileTime) == 0, EIGEN_INTERNAL_ERROR_PLEASE_FILE_A_BUG_REPORT)
  }
};

/************************
*** Default traversal ***
************************/

template <typename Kernel>
struct dense_assignment_loop<Kernel, DefaultTraversal, NoUnrolling> {
  EIGEN_DEVICE_FUNC static void EIGEN_STRONG_INLINE run(Kernel& kernel) {
    for (Index outer = 0; outer < kernel.outerSize(); ++outer) {
      for (Index inner = 0; inner < kernel.innerSize(); ++inner) {
        kernel.assignCoeffByOuterInner(outer, inner);
      }
    }
  }
};

template <typename Kernel>
struct dense_assignment_loop<Kernel, DefaultTraversal, CompleteUnrolling> {
  using Traits = typename Kernel::AssignmentTraits;

  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void run(Kernel& kernel) {
    copy_using_evaluator_DefaultTraversal_CompleteUnrolling<Kernel, 0, Traits::SizeAtCompileTime>::run(kernel);
  }
};

template <typename Kernel>
struct dense_assignment_loop<Kernel, DefaultTraversal, InnerUnrolling> {
  using Traits = typename Kernel::AssignmentTraits;

  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void run(Kernel& kernel) {
    const Index outerSize = kernel.outerSize();
    for (Index outer = 0; outer < outerSize; ++outer)
      copy_using_evaluator_DefaultTraversal_InnerUnrolling<Kernel, 0, Traits::InnerSizeAtCompileTime>::run(kernel,
                                                                                                           outer);
  }
};

/***************************
*** Linear vectorization ***
***************************/

// The goal of unaligned_dense_assignment_loop is simply to factorize the handling
// of the non vectorizable beginning and ending parts

template <bool IsAligned = false>
struct unaligned_dense_assignment_loop {
  // if IsAligned = true, then do nothing
  template <typename Kernel>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE EIGEN_CONSTEXPR void run(Kernel&, Index, Index) {}
};

template <>
struct unaligned_dense_assignment_loop<false> {
  // MSVC must not inline this functions. If it does, it fails to optimize the
  // packet access path.
  // FIXME check which version exhibits this issue
#if EIGEN_COMP_MSVC
  template <typename Kernel>
  static EIGEN_DONT_INLINE void run(Kernel& kernel, Index start, Index end)
#else
  template <typename Kernel>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE EIGEN_CONSTEXPR void run(Kernel& kernel, Index start, Index end)
#endif
  {
    for (Index index = start; index < end; ++index) kernel.assignCoeff(index);
  }
};

template <typename Kernel, int Index_, int Stop>
struct copy_using_evaluator_linearvec_CompleteUnrolling {
  using Traits = typename Kernel::AssignmentTraits;
  using PacketType = typename Kernel::PacketType;
  static constexpr int SrcAlignment = Traits::SrcAlignment, DstAlignment = Traits::DstAlignment,
                       NextIndex = Index_ + unpacket_traits<PacketType>::size;

  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void run(Kernel& kernel) {
    kernel.template assignPacket<DstAlignment, SrcAlignment, PacketType>(Index_);
    copy_using_evaluator_linearvec_CompleteUnrolling<Kernel, NextIndex, Stop>::run(kernel);
  }
};

template <typename Kernel, int Stop>
struct copy_using_evaluator_linearvec_CompleteUnrolling<Kernel, Stop, Stop> {
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE EIGEN_CONSTEXPR void run(Kernel&) {}
};

template <typename Kernel>
struct dense_assignment_loop<Kernel, LinearVectorizedTraversal, NoUnrolling> {
  using Scalar = typename Kernel::Scalar;
  using PacketType = typename Kernel::PacketType;

  static constexpr int RequestedAlignment = Kernel::AssignmentTraits::LinearRequiredAlignment,
                       PacketSize = unpacket_traits<PacketType>::size,
                       DstAlignment = packet_traits<Scalar>::AlignedOnScalar ? RequestedAlignment
                                                                             : Kernel::AssignmentTraits::DstAlignment,
                       SrcAlignment = Kernel::AssignmentTraits::JointAlignment;
  static constexpr bool DstIsAligned = Kernel::AssignmentTraits::DstAlignment >= RequestedAlignment;

  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE EIGEN_CONSTEXPR void run(Kernel& kernel) {
    const Index size = kernel.size();
    const Index alignedStart =
        DstIsAligned ? 0 : internal::first_aligned<RequestedAlignment>(kernel.dstDataPtr(), size);
    const Index alignedEnd = alignedStart + numext::round_down(size - alignedStart, PacketSize);

    unaligned_dense_assignment_loop<DstIsAligned != 0>::run(kernel, 0, alignedStart);

    for (Index index = alignedStart; index < alignedEnd; index += PacketSize)
      kernel.template assignPacket<DstAlignment, SrcAlignment, PacketType>(index);

    unaligned_dense_assignment_loop<>::run(kernel, alignedEnd, size);
  }
};

template <typename Kernel>
struct dense_assignment_loop<Kernel, LinearVectorizedTraversal, CompleteUnrolling> {
  using Traits = typename Kernel::AssignmentTraits;
  using PacketType = typename Kernel::PacketType;
  static constexpr int Size = Traits::SizeAtCompileTime, PacketSize = unpacket_traits<PacketType>::size,
                       AlignedSize = numext::round_down(Size, PacketSize);

  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE EIGEN_CONSTEXPR void run(Kernel& kernel) {
    copy_using_evaluator_linearvec_CompleteUnrolling<Kernel, 0, AlignedSize>::run(kernel);
    copy_using_evaluator_LinearTraversal_CompleteUnrolling<Kernel, AlignedSize, Size>::run(kernel);
  }
};

/**************************
*** Inner vectorization ***
**************************/

template <typename Kernel>
struct dense_assignment_loop<Kernel, InnerVectorizedTraversal, NoUnrolling> {
  using Traits = typename Kernel::AssignmentTraits;
  using PacketType = typename Kernel::PacketType;
  static constexpr int SrcAlignment = Traits::SrcAlignment, DstAlignment = Traits::DstAlignment,
                       PacketSize = unpacket_traits<PacketType>::size;

  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE EIGEN_CONSTEXPR void run(Kernel& kernel) {
    const Index innerSize = kernel.innerSize();
    const Index outerSize = kernel.outerSize();
    for (Index outer = 0; outer < outerSize; ++outer)
      for (Index inner = 0; inner < innerSize; inner += PacketSize)
        kernel.template assignPacketByOuterInner<DstAlignment, SrcAlignment, PacketType>(outer, inner);
  }
};

template <typename Kernel>
struct dense_assignment_loop<Kernel, InnerVectorizedTraversal, CompleteUnrolling> {
  using Traits = typename Kernel::AssignmentTraits;
  static constexpr int Size = Traits::SizeAtCompileTime;
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void run(Kernel& kernel) {
    copy_using_evaluator_innervec_CompleteUnrolling<Kernel, 0, Size>::run(kernel);
  }
};

template <typename Kernel>
struct dense_assignment_loop<Kernel, InnerVectorizedTraversal, InnerUnrolling> {
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void run(Kernel& kernel) {
    using Traits = typename Kernel::AssignmentTraits;
    const Index outerSize = kernel.outerSize();
    for (Index outer = 0; outer < outerSize; ++outer)
      copy_using_evaluator_innervec_InnerUnrolling<Kernel, 0, Traits::InnerSizeAtCompileTime, Traits::SrcAlignment,
                                                   Traits::DstAlignment>::run(kernel, outer);
  }
};

/***********************
*** Linear traversal ***
***********************/

template <typename Kernel>
struct dense_assignment_loop<Kernel, LinearTraversal, NoUnrolling> {
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE EIGEN_CONSTEXPR void run(Kernel& kernel) {
    const Index size = kernel.size();
    for (Index i = 0; i < size; ++i) kernel.assignCoeff(i);
  }
};

template <typename Kernel>
struct dense_assignment_loop<Kernel, LinearTraversal, CompleteUnrolling> {
  using Traits = typename Kernel::AssignmentTraits;
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE EIGEN_CONSTEXPR void run(Kernel& kernel) {
    copy_using_evaluator_LinearTraversal_CompleteUnrolling<Kernel, 0, Traits::SizeAtCompileTime>::run(kernel);
  }
};

/**************************
*** Slice vectorization ***
***************************/

template <typename Kernel>
struct dense_assignment_loop<Kernel, SliceVectorizedTraversal, NoUnrolling> {
  using Scalar = typename Kernel::Scalar;
  using Traits = typename Kernel::AssignmentTraits;
  using PacketType = typename Kernel::PacketType;
  static constexpr bool Alignable = packet_traits<Scalar>::AlignedOnScalar || Traits::DstAlignment >= sizeof(Scalar);
  static constexpr int PacketSize = unpacket_traits<PacketType>::size,
                       RequestedAlignment = Traits::InnerRequiredAlignment,
                       DstAlignment = Alignable ? RequestedAlignment : Traits::DstAlignment;
  static constexpr bool DstIsAligned = Traits::DstAlignment >= RequestedAlignment;

  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE EIGEN_CONSTEXPR void run(Kernel& kernel) {
    const Scalar* dst_ptr = kernel.dstDataPtr();
    if ((!DstIsAligned) && (std::uintptr_t(dst_ptr) % sizeof(Scalar)) > 0) {
      // the pointer is not aligned-on scalar, so alignment is not possible
      return dense_assignment_loop<Kernel, DefaultTraversal, NoUnrolling>::run(kernel);
    }
    const Index packetAlignedMask = PacketSize - 1;
    const Index innerSize = kernel.innerSize();
    const Index outerSize = kernel.outerSize();
    const Index alignedStep = Alignable ? (PacketSize - kernel.outerStride() % PacketSize) & packetAlignedMask : 0;
    Index alignedStart =
        ((!Alignable) || bool(DstIsAligned)) ? 0 : internal::first_aligned<RequestedAlignment>(dst_ptr, innerSize);

    for (Index outer = 0; outer < outerSize; ++outer) {
      const Index alignedEnd = alignedStart + ((innerSize - alignedStart) & ~packetAlignedMask);
      // do the non-vectorizable part of the assignment
      for (Index inner = 0; inner < alignedStart; ++inner) kernel.assignCoeffByOuterInner(outer, inner);

      // do the vectorizable part of the assignment
      for (Index inner = alignedStart; inner < alignedEnd; inner += PacketSize)
        kernel.template assignPacketByOuterInner<DstAlignment, Unaligned, PacketType>(outer, inner);

      // do the non-vectorizable part of the assignment
      for (Index inner = alignedEnd; inner < innerSize; ++inner) kernel.assignCoeffByOuterInner(outer, inner);

      alignedStart = numext::mini((alignedStart + alignedStep) % PacketSize, innerSize);
    }
  }
};

#if EIGEN_UNALIGNED_VECTORIZE
template <typename Kernel>
struct dense_assignment_loop<Kernel, SliceVectorizedTraversal, InnerUnrolling> {
  using Traits = typename Kernel::AssignmentTraits;
  using PacketType = typename Kernel::PacketType;
  static constexpr int InnerSize = Traits::InnerSizeAtCompileTime, PacketSize = unpacket_traits<PacketType>::size,
                       VectorizableSize = numext::round_down(InnerSize, PacketSize);

  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE EIGEN_CONSTEXPR void run(Kernel& kernel) {
    for (Index outer = 0; outer < kernel.outerSize(); ++outer) {
      copy_using_evaluator_innervec_InnerUnrolling<Kernel, 0, VectorizableSize, 0, 0>::run(kernel, outer);
      copy_using_evaluator_DefaultTraversal_InnerUnrolling<Kernel, VectorizableSize, InnerSize>::run(kernel, outer);
    }
  }
};
#endif

/***************************************************************************
 * Part 4 : Generic dense assignment kernel
 ***************************************************************************/

// This class generalize the assignment of a coefficient (or packet) from one dense evaluator
// to another dense writable evaluator.
// It is parametrized by the two evaluators, and the actual assignment functor.
// This abstraction level permits to keep the evaluation loops as simple and as generic as possible.
// One can customize the assignment using this generic dense_assignment_kernel with different
// functors, or by completely overloading it, by-passing a functor.
template <typename DstEvaluatorTypeT, typename SrcEvaluatorTypeT, typename Functor, int Version = Specialized>
class generic_dense_assignment_kernel {
 protected:
  typedef typename DstEvaluatorTypeT::XprType DstXprType;
  typedef typename SrcEvaluatorTypeT::XprType SrcXprType;

 public:
  typedef DstEvaluatorTypeT DstEvaluatorType;
  typedef SrcEvaluatorTypeT SrcEvaluatorType;
  typedef typename DstEvaluatorType::Scalar Scalar;
  typedef copy_using_evaluator_traits<DstEvaluatorTypeT, SrcEvaluatorTypeT, Functor> AssignmentTraits;
  typedef typename AssignmentTraits::PacketType PacketType;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE generic_dense_assignment_kernel(DstEvaluatorType& dst,
                                                                        const SrcEvaluatorType& src,
                                                                        const Functor& func, DstXprType& dstExpr)
      : m_dst(dst), m_src(src), m_functor(func), m_dstExpr(dstExpr) {
#ifdef EIGEN_DEBUG_ASSIGN
    AssignmentTraits::debug();
#endif
  }

  EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR Index size() const EIGEN_NOEXCEPT { return m_dstExpr.size(); }
  EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR Index innerSize() const EIGEN_NOEXCEPT { return m_dstExpr.innerSize(); }
  EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR Index outerSize() const EIGEN_NOEXCEPT { return m_dstExpr.outerSize(); }
  EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR Index rows() const EIGEN_NOEXCEPT { return m_dstExpr.rows(); }
  EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR Index cols() const EIGEN_NOEXCEPT { return m_dstExpr.cols(); }
  EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR Index outerStride() const EIGEN_NOEXCEPT { return m_dstExpr.outerStride(); }

  EIGEN_DEVICE_FUNC DstEvaluatorType& dstEvaluator() EIGEN_NOEXCEPT { return m_dst; }
  EIGEN_DEVICE_FUNC const SrcEvaluatorType& srcEvaluator() const EIGEN_NOEXCEPT { return m_src; }

  /// Assign src(row,col) to dst(row,col) through the assignment functor.
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void assignCoeff(Index row, Index col) {
    m_functor.assignCoeff(m_dst.coeffRef(row, col), m_src.coeff(row, col));
  }

  /// \sa assignCoeff(Index,Index)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void assignCoeff(Index index) {
    m_functor.assignCoeff(m_dst.coeffRef(index), m_src.coeff(index));
  }

  /// \sa assignCoeff(Index,Index)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void assignCoeffByOuterInner(Index outer, Index inner) {
    Index row = rowIndexByOuterInner(outer, inner);
    Index col = colIndexByOuterInner(outer, inner);
    assignCoeff(row, col);
  }

  template <int StoreMode, int LoadMode, typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void assignPacket(Index row, Index col) {
    m_functor.template assignPacket<StoreMode>(&m_dst.coeffRef(row, col),
                                               m_src.template packet<LoadMode, Packet>(row, col));
  }

  template <int StoreMode, int LoadMode, typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void assignPacket(Index index) {
    m_functor.template assignPacket<StoreMode>(&m_dst.coeffRef(index), m_src.template packet<LoadMode, Packet>(index));
  }

  template <int StoreMode, int LoadMode, typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void assignPacketByOuterInner(Index outer, Index inner) {
    Index row = rowIndexByOuterInner(outer, inner);
    Index col = colIndexByOuterInner(outer, inner);
    assignPacket<StoreMode, LoadMode, Packet>(row, col);
  }

  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Index rowIndexByOuterInner(Index outer, Index inner) {
    typedef typename DstEvaluatorType::ExpressionTraits Traits;
    return int(Traits::RowsAtCompileTime) == 1          ? 0
           : int(Traits::ColsAtCompileTime) == 1        ? inner
           : int(DstEvaluatorType::Flags) & RowMajorBit ? outer
                                                        : inner;
  }

  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Index colIndexByOuterInner(Index outer, Index inner) {
    typedef typename DstEvaluatorType::ExpressionTraits Traits;
    return int(Traits::ColsAtCompileTime) == 1          ? 0
           : int(Traits::RowsAtCompileTime) == 1        ? inner
           : int(DstEvaluatorType::Flags) & RowMajorBit ? inner
                                                        : outer;
  }

  EIGEN_DEVICE_FUNC const Scalar* dstDataPtr() const { return m_dstExpr.data(); }

 protected:
  DstEvaluatorType& m_dst;
  const SrcEvaluatorType& m_src;
  const Functor& m_functor;
  // TODO find a way to avoid the needs of the original expression
  DstXprType& m_dstExpr;
};

// Special kernel used when computing small products whose operands have dynamic dimensions.  It ensures that the
// PacketSize used is no larger than 4, thereby increasing the chance that vectorized instructions will be used
// when computing the product.

template <typename DstEvaluatorTypeT, typename SrcEvaluatorTypeT, typename Functor>
class restricted_packet_dense_assignment_kernel
    : public generic_dense_assignment_kernel<DstEvaluatorTypeT, SrcEvaluatorTypeT, Functor, BuiltIn> {
 protected:
  typedef generic_dense_assignment_kernel<DstEvaluatorTypeT, SrcEvaluatorTypeT, Functor, BuiltIn> Base;

 public:
  typedef typename Base::Scalar Scalar;
  typedef typename Base::DstXprType DstXprType;
  typedef copy_using_evaluator_traits<DstEvaluatorTypeT, SrcEvaluatorTypeT, Functor, 4> AssignmentTraits;
  typedef typename AssignmentTraits::PacketType PacketType;

  EIGEN_DEVICE_FUNC restricted_packet_dense_assignment_kernel(DstEvaluatorTypeT& dst, const SrcEvaluatorTypeT& src,
                                                              const Functor& func, DstXprType& dstExpr)
      : Base(dst, src, func, dstExpr) {}
};

/***************************************************************************
 * Part 5 : Entry point for dense rectangular assignment
 ***************************************************************************/

template <typename DstXprType, typename SrcXprType, typename Functor>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void resize_if_allowed(DstXprType& dst, const SrcXprType& src,
                                                             const Functor& /*func*/) {
  EIGEN_ONLY_USED_FOR_DEBUG(dst);
  EIGEN_ONLY_USED_FOR_DEBUG(src);
  eigen_assert(dst.rows() == src.rows() && dst.cols() == src.cols());
}

template <typename DstXprType, typename SrcXprType, typename T1, typename T2>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void resize_if_allowed(DstXprType& dst, const SrcXprType& src,
                                                             const internal::assign_op<T1, T2>& /*func*/) {
  Index dstRows = src.rows();
  Index dstCols = src.cols();
  if (((dst.rows() != dstRows) || (dst.cols() != dstCols))) dst.resize(dstRows, dstCols);
  eigen_assert(dst.rows() == dstRows && dst.cols() == dstCols);
}

template <typename DstXprType, typename SrcXprType, typename Functor>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE EIGEN_CONSTEXPR void call_dense_assignment_loop(DstXprType& dst,
                                                                                      const SrcXprType& src,
                                                                                      const Functor& func) {
  typedef evaluator<DstXprType> DstEvaluatorType;
  typedef evaluator<SrcXprType> SrcEvaluatorType;

  SrcEvaluatorType srcEvaluator(src);

  // NOTE To properly handle A = (A*A.transpose())/s with A rectangular,
  // we need to resize the destination after the source evaluator has been created.
  resize_if_allowed(dst, src, func);

  DstEvaluatorType dstEvaluator(dst);

  typedef generic_dense_assignment_kernel<DstEvaluatorType, SrcEvaluatorType, Functor> Kernel;
  Kernel kernel(dstEvaluator, srcEvaluator, func, dst.const_cast_derived());

  dense_assignment_loop<Kernel>::run(kernel);
}

template <typename DstXprType, typename SrcXprType>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void call_dense_assignment_loop(DstXprType& dst, const SrcXprType& src) {
  call_dense_assignment_loop(dst, src, internal::assign_op<typename DstXprType::Scalar, typename SrcXprType::Scalar>());
}

/***************************************************************************
 * Part 6 : Generic assignment
 ***************************************************************************/

// Based on the respective shapes of the destination and source,
// the class AssignmentKind determine the kind of assignment mechanism.
// AssignmentKind must define a Kind typedef.
template <typename DstShape, typename SrcShape>
struct AssignmentKind;

// Assignment kind defined in this file:
struct Dense2Dense {};
struct EigenBase2EigenBase {};

template <typename, typename>
struct AssignmentKind {
  typedef EigenBase2EigenBase Kind;
};
template <>
struct AssignmentKind<DenseShape, DenseShape> {
  typedef Dense2Dense Kind;
};

// This is the main assignment class
template <typename DstXprType, typename SrcXprType, typename Functor,
          typename Kind = typename AssignmentKind<typename evaluator_traits<DstXprType>::Shape,
                                                  typename evaluator_traits<SrcXprType>::Shape>::Kind,
          typename EnableIf = void>
struct Assignment;

// The only purpose of this call_assignment() function is to deal with noalias() / "assume-aliasing" and automatic
// transposition. Indeed, I (Gael) think that this concept of "assume-aliasing" was a mistake, and it makes thing quite
// complicated. So this intermediate function removes everything related to "assume-aliasing" such that Assignment does
// not has to bother about these annoying details.

template <typename Dst, typename Src>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void call_assignment(Dst& dst, const Src& src) {
  call_assignment(dst, src, internal::assign_op<typename Dst::Scalar, typename Src::Scalar>());
}
template <typename Dst, typename Src>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void call_assignment(const Dst& dst, const Src& src) {
  call_assignment(dst, src, internal::assign_op<typename Dst::Scalar, typename Src::Scalar>());
}

// Deal with "assume-aliasing"
template <typename Dst, typename Src, typename Func>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE EIGEN_CONSTEXPR void call_assignment(
    Dst& dst, const Src& src, const Func& func, std::enable_if_t<evaluator_assume_aliasing<Src>::value, void*> = 0) {
  typename plain_matrix_type<Src>::type tmp(src);
  call_assignment_no_alias(dst, tmp, func);
}

template <typename Dst, typename Src, typename Func>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void call_assignment(
    Dst& dst, const Src& src, const Func& func, std::enable_if_t<!evaluator_assume_aliasing<Src>::value, void*> = 0) {
  call_assignment_no_alias(dst, src, func);
}

// by-pass "assume-aliasing"
// When there is no aliasing, we require that 'dst' has been properly resized
template <typename Dst, template <typename> class StorageBase, typename Src, typename Func>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE EIGEN_CONSTEXPR void call_assignment(NoAlias<Dst, StorageBase>& dst,
                                                                           const Src& src, const Func& func) {
  call_assignment_no_alias(dst.expression(), src, func);
}

template <typename Dst, typename Src, typename Func>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE EIGEN_CONSTEXPR void call_assignment_no_alias(Dst& dst, const Src& src,
                                                                                    const Func& func) {
  enum {
    NeedToTranspose = ((int(Dst::RowsAtCompileTime) == 1 && int(Src::ColsAtCompileTime) == 1) ||
                       (int(Dst::ColsAtCompileTime) == 1 && int(Src::RowsAtCompileTime) == 1)) &&
                      int(Dst::SizeAtCompileTime) != 1
  };

  typedef std::conditional_t<NeedToTranspose, Transpose<Dst>, Dst> ActualDstTypeCleaned;
  typedef std::conditional_t<NeedToTranspose, Transpose<Dst>, Dst&> ActualDstType;
  ActualDstType actualDst(dst);

  // TODO check whether this is the right place to perform these checks:
  EIGEN_STATIC_ASSERT_LVALUE(Dst)
  EIGEN_STATIC_ASSERT_SAME_MATRIX_SIZE(ActualDstTypeCleaned, Src)
  EIGEN_CHECK_BINARY_COMPATIBILIY(Func, typename ActualDstTypeCleaned::Scalar, typename Src::Scalar);

  Assignment<ActualDstTypeCleaned, Src, Func>::run(actualDst, src, func);
}

template <typename Dst, typename Src, typename Func>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void call_restricted_packet_assignment_no_alias(Dst& dst, const Src& src,
                                                                                      const Func& func) {
  typedef evaluator<Dst> DstEvaluatorType;
  typedef evaluator<Src> SrcEvaluatorType;
  typedef restricted_packet_dense_assignment_kernel<DstEvaluatorType, SrcEvaluatorType, Func> Kernel;

  EIGEN_STATIC_ASSERT_LVALUE(Dst)
  EIGEN_CHECK_BINARY_COMPATIBILIY(Func, typename Dst::Scalar, typename Src::Scalar);

  SrcEvaluatorType srcEvaluator(src);
  resize_if_allowed(dst, src, func);

  DstEvaluatorType dstEvaluator(dst);
  Kernel kernel(dstEvaluator, srcEvaluator, func, dst.const_cast_derived());

  dense_assignment_loop<Kernel>::run(kernel);
}

template <typename Dst, typename Src>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE EIGEN_CONSTEXPR void call_assignment_no_alias(Dst& dst, const Src& src) {
  call_assignment_no_alias(dst, src, internal::assign_op<typename Dst::Scalar, typename Src::Scalar>());
}

template <typename Dst, typename Src, typename Func>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE EIGEN_CONSTEXPR void call_assignment_no_alias_no_transpose(Dst& dst,
                                                                                                 const Src& src,
                                                                                                 const Func& func) {
  // TODO check whether this is the right place to perform these checks:
  EIGEN_STATIC_ASSERT_LVALUE(Dst)
  EIGEN_STATIC_ASSERT_SAME_MATRIX_SIZE(Dst, Src)
  EIGEN_CHECK_BINARY_COMPATIBILIY(Func, typename Dst::Scalar, typename Src::Scalar);

  Assignment<Dst, Src, Func>::run(dst, src, func);
}
template <typename Dst, typename Src>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE EIGEN_CONSTEXPR void call_assignment_no_alias_no_transpose(Dst& dst,
                                                                                                 const Src& src) {
  call_assignment_no_alias_no_transpose(dst, src, internal::assign_op<typename Dst::Scalar, typename Src::Scalar>());
}

// forward declaration
template <typename Dst, typename Src>
EIGEN_DEVICE_FUNC void check_for_aliasing(const Dst& dst, const Src& src);

// Generic Dense to Dense assignment
// Note that the last template argument "Weak" is needed to make it possible to perform
// both partial specialization+SFINAE without ambiguous specialization
template <typename DstXprType, typename SrcXprType, typename Functor, typename Weak>
struct Assignment<DstXprType, SrcXprType, Functor, Dense2Dense, Weak> {
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void run(DstXprType& dst, const SrcXprType& src, const Functor& func) {
#ifndef EIGEN_NO_DEBUG
    internal::check_for_aliasing(dst, src);
#endif

    call_dense_assignment_loop(dst, src, func);
  }
};

template <typename DstXprType, typename SrcPlainObject, typename Weak>
struct Assignment<DstXprType, CwiseNullaryOp<scalar_constant_op<typename DstXprType::Scalar>, SrcPlainObject>,
                  assign_op<typename DstXprType::Scalar, typename DstXprType::Scalar>, Dense2Dense, Weak> {
  using Scalar = typename DstXprType::Scalar;
  using NullaryOp = scalar_constant_op<Scalar>;
  using SrcXprType = CwiseNullaryOp<NullaryOp, SrcPlainObject>;
  using Functor = assign_op<Scalar, Scalar>;
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void run(DstXprType& dst, const SrcXprType& src,
                                                        const Functor& /*func*/) {
    eigen_fill_impl<DstXprType>::run(dst, src);
  }
};

template <typename DstXprType, typename SrcPlainObject, typename Weak>
struct Assignment<DstXprType, CwiseNullaryOp<scalar_zero_op<typename DstXprType::Scalar>, SrcPlainObject>,
                  assign_op<typename DstXprType::Scalar, typename DstXprType::Scalar>, Dense2Dense, Weak> {
  using Scalar = typename DstXprType::Scalar;
  using NullaryOp = scalar_zero_op<Scalar>;
  using SrcXprType = CwiseNullaryOp<NullaryOp, SrcPlainObject>;
  using Functor = assign_op<Scalar, Scalar>;
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void run(DstXprType& dst, const SrcXprType& src,
                                                        const Functor& /*func*/) {
    eigen_zero_impl<DstXprType>::run(dst, src);
  }
};

// Generic assignment through evalTo.
// TODO: not sure we have to keep that one, but it helps porting current code to new evaluator mechanism.
// Note that the last template argument "Weak" is needed to make it possible to perform
// both partial specialization+SFINAE without ambiguous specialization
template <typename DstXprType, typename SrcXprType, typename Functor, typename Weak>
struct Assignment<DstXprType, SrcXprType, Functor, EigenBase2EigenBase, Weak> {
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void run(
      DstXprType& dst, const SrcXprType& src,
      const internal::assign_op<typename DstXprType::Scalar, typename SrcXprType::Scalar>& /*func*/) {
    Index dstRows = src.rows();
    Index dstCols = src.cols();
    if ((dst.rows() != dstRows) || (dst.cols() != dstCols)) dst.resize(dstRows, dstCols);

    eigen_assert(dst.rows() == src.rows() && dst.cols() == src.cols());
    src.evalTo(dst);
  }

  // NOTE The following two functions are templated to avoid their instantiation if not needed
  //      This is needed because some expressions supports evalTo only and/or have 'void' as scalar type.
  template <typename SrcScalarType>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void run(
      DstXprType& dst, const SrcXprType& src,
      const internal::add_assign_op<typename DstXprType::Scalar, SrcScalarType>& /*func*/) {
    Index dstRows = src.rows();
    Index dstCols = src.cols();
    if ((dst.rows() != dstRows) || (dst.cols() != dstCols)) dst.resize(dstRows, dstCols);

    eigen_assert(dst.rows() == src.rows() && dst.cols() == src.cols());
    src.addTo(dst);
  }

  template <typename SrcScalarType>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void run(
      DstXprType& dst, const SrcXprType& src,
      const internal::sub_assign_op<typename DstXprType::Scalar, SrcScalarType>& /*func*/) {
    Index dstRows = src.rows();
    Index dstCols = src.cols();
    if ((dst.rows() != dstRows) || (dst.cols() != dstCols)) dst.resize(dstRows, dstCols);

    eigen_assert(dst.rows() == src.rows() && dst.cols() == src.cols());
    src.subTo(dst);
  }
};

}  // namespace internal

}  // end namespace Eigen

#endif  // EIGEN_ASSIGN_EVALUATOR_H
