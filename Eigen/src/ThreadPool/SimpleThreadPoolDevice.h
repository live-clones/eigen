// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2023 Charlie Schlosser <cs.schlosser@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SIMPLE_THREAD_POOL_DEVICE_H
#define EIGEN_SIMPLE_THREAD_POOL_DEVICE_H

namespace Eigen {

struct SimpleThreadPoolDevice {
  using Task = std::function<void()>;
  using LinearFunctor = std::function<void(Index)>;

  SimpleThreadPoolDevice(ThreadPool& pool, Index threadCostThreshold = (1 << 15)) : m_pool(pool) {
    setThreadCostThreshold(threadCostThreshold);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void parallelFor(Index begin, Index end, Index incr, LinearFunctor f,
                                                        Barrier& barrier, Index maxDepth, Index depth) {
    if (depth < maxDepth) {
      Index size = end - begin;
      Index mid = begin + (size / 2);
      mid &= -incr;  // soooooo fancy!
      Task left = [this, begin, mid, incr, f, &barrier, maxDepth, depth]() {
        parallelFor(begin, mid, incr, f, barrier, maxDepth, depth + 1);
      };
      Task right = [this, mid, end, incr, f, &barrier, maxDepth, depth]() {
        parallelFor(mid, end, incr, f, barrier, maxDepth, depth + 1);
      };
      pool().Schedule(left);
      pool().Schedule(right);
    } else {
      for (Index i = begin; i + incr <= end; i += incr) f(i);
      barrier.Notify();
    }
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void initParallelFor(Index begin, Index end, Index incr, LinearFunctor f,
                                                            Index cost) {
    eigen_assert((incr & (incr - 1)) == 0 && "this method specialized for power-of-two increments");
    Index size = end - begin;
    eigen_assert(size % incr == 0 && "size should be a multiple of incr");
    Index numOps = size / incr;
    Index totalCost = numOps * cost;
    Index idealThreads = numext::div_ceil(totalCost, threadCostThreshold());
    Index actualThreads = numext::mini(idealThreads, numThreads());
    Index maxDepth = numext::log2(actualThreads);
    // use log2_ceil to ensure available resources are used
    if ((actualThreads & (actualThreads - 1)) != 0) maxDepth++;
    uint32_t numNotifies = 1 << maxDepth;
    Barrier barrier(numNotifies);
    parallelFor(begin, end, incr, f, barrier, maxDepth, 0);
    barrier.Wait();
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE ThreadPool& pool() { return m_pool; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index numThreads() const { return static_cast<Index>(m_pool.NumThreads()); }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index threadCostThreshold() const { return m_threadCostThreshold; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void setThreadCostThreshold(Index cost) {
    eigen_assert(cost > 0);
    m_threadCostThreshold = cost;
  }
  ThreadPool& m_pool;
  Index m_threadCostThreshold;
};

namespace internal {

template <typename Kernel>
struct cost_helper {
  using SrcEvaluatorType = typename Kernel::SrcEvaluatorType;
  using DstEvaluatorType = typename Kernel::DstEvaluatorType;
  using SrcXprType = typename SrcEvaluatorType::XprType;
  using DstXprType = typename DstEvaluatorType::XprType;
  enum : Index {
    ScalarCost = functor_cost<SrcXprType>::ScalarCost + functor_cost<DstXprType>::ScalarCost,
    VectorCost = functor_cost<SrcXprType>::VectorCost + functor_cost<DstXprType>::VectorCost
  };
};

template <typename Kernel>
struct dense_assignment_loop_with_device<Kernel, SimpleThreadPoolDevice, DefaultTraversal, NoUnrolling> {
  using LinearFunctor = std::function<void(Index)>;
  enum : Index { XprEvaluatorCost = cost_helper<Kernel>::ScalarCost };
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void run(Kernel& kernel, SimpleThreadPoolDevice& device) {
    const Index innerSize = kernel.innerSize();
    const Index outerSize = kernel.outerSize();
    LinearFunctor functor = [&, innerSize](Index outer) {
      for (Index inner = 0; inner < innerSize; inner++) {
        kernel.assignCoeffByOuterInner(outer, inner);
      }
    };
    device.initParallelFor(0, outerSize, 1, functor, XprEvaluatorCost * innerSize);
  }
};

template <typename Kernel>
struct dense_assignment_loop_with_device<Kernel, SimpleThreadPoolDevice, DefaultTraversal, InnerUnrolling> {
  typedef typename Kernel::DstEvaluatorType::XprType DstXprType;
  using LinearFunctor = std::function<void(Index)>;
  enum : Index { XprEvaluatorCost = cost_helper<Kernel>::ScalarCost, InnerSize = DstXprType::InnerSizeAtCompileTime };
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void run(Kernel& kernel, SimpleThreadPoolDevice& device) {
    const Index outerSize = kernel.outerSize();
    LinearFunctor functor = [&](Index outer) {
      copy_using_evaluator_DefaultTraversal_InnerUnrolling<Kernel, 0, InnerSize>::run(kernel, outer);
    };
    device.initParallelFor(0, outerSize, 1, functor, XprEvaluatorCost * InnerSize);
  }
};

template <typename Kernel>
struct dense_assignment_loop_with_device<Kernel, SimpleThreadPoolDevice, InnerVectorizedTraversal, NoUnrolling> {
  using LinearFunctor = std::function<void(Index)>;
  using PacketType = typename Kernel::PacketType;
  enum : Index {
    XprEvaluatorCost = cost_helper<Kernel>::VectorCost,
    PacketSize = unpacket_traits<PacketType>::size,
    SrcAlignment = Kernel::AssignmentTraits::SrcAlignment,
    DstAlignment = Kernel::AssignmentTraits::DstAlignment
  };
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE EIGEN_CONSTEXPR void run(Kernel& kernel, SimpleThreadPoolDevice& device) {
    const Index innerSize = kernel.innerSize();
    const Index outerSize = kernel.outerSize();
    LinearFunctor functor = [&, innerSize](Index outer) {
      for (Index inner = 0; inner < innerSize; inner += PacketSize)
        kernel.template assignPacketByOuterInner<DstAlignment, SrcAlignment, PacketType>(outer, inner);
    };
    device.initParallelFor(0, outerSize, 1, functor, XprEvaluatorCost * innerSize);
  }
};

template <typename Kernel>
struct dense_assignment_loop_with_device<Kernel, SimpleThreadPoolDevice, SliceVectorizedTraversal, NoUnrolling> {
  using LinearFunctor = std::function<void(Index)>;
  using Scalar = typename Kernel::Scalar;
  using PacketType = typename Kernel::PacketType;
  enum : Index {
    XprEvaluatorCost = cost_helper<Kernel>::VectorCost,
    PacketSize = unpacket_traits<PacketType>::size,
    RequestedAlignment = Kernel::AssignmentTraits::InnerRequiredAlignment,
    Alignable = packet_traits<Scalar>::AlignedOnScalar || Kernel::AssignmentTraits::DstAlignment >= sizeof(Scalar),
    DstIsAligned = Kernel::AssignmentTraits::DstAlignment >= RequestedAlignment,
    DstAlignment = Alignable ? RequestedAlignment : Kernel::AssignmentTraits::DstAlignment,
    PacketAlignedMask = PacketSize - 1
  };
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE EIGEN_CONSTEXPR void run(Kernel& kernel,
                                                                        SimpleThreadPoolDevice& device) {
    const Scalar* dstDataPtr = kernel.dstDataPtr();
    if ((!bool(DstIsAligned)) && (std::uintptr_t(dstDataPtr) % sizeof(Scalar)) > 0) {
      // the pointer is not aligned-on scalar, so alignment is not possible
      return dense_assignment_loop<Kernel, DefaultTraversal, NoUnrolling>::run(kernel);
    }
    const Index innerSize = kernel.innerSize();
    const Index outerSize = kernel.outerSize();
    const Index outerStride = kernel.outerStride();

    LinearFunctor functor = [=, &kernel](Index outer) {
      const Scalar* dstAligned = dstDataPtr + (outer * outerStride);
      const Index alignedStart = internal::first_aligned<DstAlignment>(dstAligned, innerSize);
      const Index alignedEnd = alignedStart + ((innerSize - alignedStart) & ~PacketAlignedMask);

      // do the non-vectorizable part of the assignment
      for (Index inner = 0; inner < alignedStart; ++inner) kernel.assignCoeffByOuterInner(outer, inner);

      // do the vectorizable part of the assignment
      for (Index inner = alignedStart; inner < alignedEnd; inner += PacketSize)
        kernel.template assignPacketByOuterInner<DstAlignment, Unaligned, PacketType>(outer, inner);

      // do the non-vectorizable part of the assignment
      for (Index inner = alignedEnd; inner < innerSize; ++inner) kernel.assignCoeffByOuterInner(outer, inner);
    };

    device.initParallelFor(0, outerSize, 1, functor,
                           XprEvaluatorCost * innerSize);  // not strictly correct as some evaulations are scalar-wise
  }
};

template <typename Kernel>
struct dense_assignment_loop_with_device<Kernel, SimpleThreadPoolDevice, LinearTraversal, NoUnrolling> {
  using LinearFunctor = std::function<void(Index)>;
  enum : Index { XprEvaluatorCost = cost_helper<Kernel>::ScalarCost };
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE EIGEN_CONSTEXPR void run(Kernel& kernel,
                                                                        SimpleThreadPoolDevice& device) {
    LinearFunctor functor = [&](Index index) { kernel.assignCoeff(index); };
    device.initParallelFor(0, kernel.size(), 1, functor, XprEvaluatorCost);
  }
};

template <typename Kernel>
struct dense_assignment_loop_with_device<Kernel, SimpleThreadPoolDevice, LinearVectorizedTraversal, NoUnrolling> {
  using Scalar = typedef typename Kernel::Scalar;
  using PacketType = typedef typename Kernel::PacketType;
  using LinearFunctor = std::function<void(Index)>;
  enum : Index {
    XprEvaluatorCost = cost_helper<Kernel>::VectorCost,
    RequestedAlignment = Kernel::AssignmentTraits::LinearRequiredAlignment,
    PacketSize = unpacket_traits<PacketType>::size,
    DstIsAligned = Kernel::AssignmentTraits::DstAlignment >= RequestedAlignment,
    DstAlignment = packet_traits<Scalar>::AlignedOnScalar ? RequestedAlignment : Kernel::AssignmentTraits::DstAlignment,
    SrcAlignment = Kernel::AssignmentTraits::JointAlignment
  };

  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE EIGEN_CONSTEXPR void run(Kernel& kernel,
                                                                        SimpleThreadPoolDevice& device) {
    const Index size = kernel.size();
    const Index alignedStart =
        DstIsAligned ? 0 : internal::first_aligned<RequestedAlignment>(kernel.dstDataPtr(), size);
    const Index alignedEnd = alignedStart + ((size - alignedStart) / PacketSize) * PacketSize;

    unaligned_dense_assignment_loop<DstIsAligned != 0>::run(kernel, 0, alignedStart);

    LinearFunctor vectorIterFunctor = [&](Index index) {
      kernel.template assignPacket<DstAlignment, SrcAlignment, PacketType>(index);
    };
    device.initParallelFor(alignedStart, alignedEnd, PacketSize, vectorIterFunctor, XprEvaluatorCost);

    unaligned_dense_assignment_loop<>::run(kernel, alignedEnd, size);
  }
};

}  // namespace internal

}  // namespace Eigen

#endif  // EIGEN_SIMPLE_THREAD_POOL_DEVICE_H
