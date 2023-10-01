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
  SimpleThreadPoolDevice(ThreadPool& pool) : m_pool(pool) {}

  static constexpr Index ThreadCost = 1 << 15;  // tuneable -- prefer power of two so the division is cheaper

  using Task = std::function<void()>;
  using LinearFunctor = std::function<void(Index)>;

  void parallelFor(Index begin, Index end, Index incr, LinearFunctor f, Barrier& barrier, Index maxDepth, Index depth) {
    eigen_assert((incr & (incr - 1)) == 0 && "this method specialized for power-of-two increments");
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

  void initParallelFor(Index begin, Index end, Index incr, LinearFunctor f, Index cost) {
    Index numOps = (end - begin) / incr;
    Index totalCost = numOps * cost;
    Index idealThreads = numext::div_ceil(totalCost, ThreadCost);
    Index actualThreads = numext::mini(idealThreads, numThreads());
    Index maxDepth = numext::log2(actualThreads);
    // use log2_ceil to ensure available resources are used
    if ((actualThreads & (actualThreads - 1)) != 0) maxDepth++;
    uint32_t numNotifies = 1 << static_cast<uint32_t>(maxDepth);
    Barrier barrier(numNotifies);
    parallelFor(begin, end, incr, f, barrier, maxDepth, 0);
    barrier.Wait();
  }

  ThreadPool& pool() { return m_pool; }
  Index numThreads() const { return static_cast<Index>(m_pool.NumThreads()); }

  ThreadPool& m_pool;
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
  using OuterInnerFunctor = std::function<void(Index, Index)>;
  static constexpr Index XprEvaluatorCost = cost_helper<Kernel>::ScalarCost;
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void run(Kernel& kernel, SimpleThreadPoolDevice& device) {
    LinearFunctor functor = [&](Index outer) {
      for (Index inner = 0; inner < kernel.innerSize(); ++inner) kernel.assignCoeffByOuterInner(outer, inner);
    };
    device.initParallelFor(0, kernel.outerSize(), 1, functor, XprEvaluatorCost * kernel.innerSize());
  }
};

template <typename Kernel>
struct dense_assignment_loop_with_device<Kernel, SimpleThreadPoolDevice, LinearTraversal, NoUnrolling> {
  using LinearFunctor = std::function<void(Index)>;
  static constexpr Index XprEvaluatorCost = cost_helper<Kernel>::ScalarCost;
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
  using Task = std::function<void()>;
  using LinearFunctor = std::function<void(Index)>;
  static constexpr Index XprEvaluatorCost = cost_helper<Kernel>::VectorCost;
  enum {
    requestedAlignment = Kernel::AssignmentTraits::LinearRequiredAlignment,
    packetSize = unpacket_traits<PacketType>::size,
    dstIsAligned = int(Kernel::AssignmentTraits::DstAlignment) >= int(requestedAlignment),
    dstAlignment =
        packet_traits<Scalar>::AlignedOnScalar ? int(requestedAlignment) : int(Kernel::AssignmentTraits::DstAlignment),
    srcAlignment = Kernel::AssignmentTraits::JointAlignment
  };

  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE EIGEN_CONSTEXPR void run(Kernel& kernel,
                                                                        SimpleThreadPoolDevice& device) {
    const Index size = kernel.size();
    const Index alignedStart =
        dstIsAligned ? 0 : internal::first_aligned<requestedAlignment>(kernel.dstDataPtr(), size);
    const Index alignedEnd = alignedStart + ((size - alignedStart) / packetSize) * packetSize;

    Barrier barrier(dstIsAligned ? 2 : 3);

    Task headLoopFunctor = [&, alignedStart]() {
      unaligned_dense_assignment_loop<dstIsAligned != 0>::run(kernel, 0, alignedStart);
      barrier.Notify();
    };

    Task vectorLoopFunctor = [&, alignedStart, alignedEnd]() {
      LinearFunctor vectorIterFunctor = [&](Index index) {
        kernel.template assignPacket<dstAlignment, srcAlignment, PacketType>(index);
      };
      device.initParallelFor(alignedStart, alignedEnd, packetSize, vectorIterFunctor, XprEvaluatorCost);
      barrier.Notify();
    };

    Task tailLoopFunctor = [&, alignedEnd, size]() {
      unaligned_dense_assignment_loop<>::run(kernel, alignedEnd, size);
      barrier.Notify();
    };

    if (!dstIsAligned) device.pool().Schedule(headLoopFunctor);
    device.pool().Schedule(vectorLoopFunctor);
    device.pool().Schedule(tailLoopFunctor);
    barrier.Wait();
  }
};

}  // namespace internal

}  // namespace Eigen

#endif  // EIGEN_SIMPLE_THREAD_POOL_DEVICE_H
