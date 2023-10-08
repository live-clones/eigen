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

// SimpleThreadPoolDevice provides an easy-to-understand Device for parallelizing Eigen Core expressions with Threadpool
// expressions are recursively bifurcated until the cost is less than the prescribed threshold for delegating the task to a thread

struct SimpleThreadPoolDevice {
  using Task = std::function<void()>;
  SimpleThreadPoolDevice(ThreadPool& pool, float threadCostThreshold = 500'000) : m_pool(pool) {
    setThreadCostThreshold(threadCostThreshold);
  }

  template <typename UnaryFunctor>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void parallelFor(Index begin, Index end, Index stride, UnaryFunctor f,
                                                         Barrier& barrier, int depth, int maxDepth, int scheduleLimit) {
    eigen_assert((stride & (stride - 1)) == 0 && "this function assumes stride is a power of two");
    eigen_assert((end - begin) % stride == 0 && "this function assumes size is a multiple of stride");
    const Index strideMask = -stride;
    if (depth < maxDepth) {
      Index mid = begin + (((end - begin) / 2) & strideMask);
      Task left = [this, begin, mid, stride, f, &barrier, maxDepth, depth, scheduleLimit]() {
        parallelFor(begin, mid, stride, f, barrier, depth + 1, maxDepth, scheduleLimit);
      };
      Task right = [this, mid, end, stride, f, &barrier, maxDepth, depth, scheduleLimit]() {
        parallelFor(mid, end, stride, f, barrier, depth + 1, maxDepth, scheduleLimit);
      };
      pool().ScheduleWithHint(left, 0, scheduleLimit);
      pool().ScheduleWithHint(right, 0, scheduleLimit);
    } else {
      for (Index i = begin; i + stride <= end; i += stride) f(i);
      barrier.Notify();
    }
  }
  template <typename UnaryFunctor>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void initParallelFor(Index begin, Index end, Index stride, UnaryFunctor f,
                                                             float cost) {
    Index numOps = (end - begin) / stride;
    float totalCost = static_cast<float>(numOps) * cost;
    int actualThreads = numThreads();
    if (numext::isfinite(totalCost)) {
      float idealThreads = totalCost / threadCostThreshold();
      idealThreads = numext::maxi(idealThreads, 1.0f);
      actualThreads = numext::mini(actualThreads, static_cast<int>(idealThreads));
    }
    int numTasks = numext::log2(actualThreads);
    // round up the number of tasks so that all the allocated threads are in use
    if ((actualThreads & (actualThreads - 1)) != 0) numTasks++;
    Barrier barrier(1 << numTasks);
    parallelFor(begin, end, stride, f, barrier, 0, numTasks, actualThreads);
    barrier.Wait();
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE ThreadPool& pool() { return m_pool; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int numThreads() const { return m_pool.NumThreads(); }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float threadCostThreshold() const { return m_threadCostThreshold; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void setThreadCostThreshold(float cost) {
    eigen_assert(cost > 0 && numext::isfinite(cost) && "cost must be positive and finite");
    m_threadCostThreshold = cost;
  }
  ThreadPool& m_pool;
  float m_threadCostThreshold;
};

// specializations of assignment loops for SimpleThreadPoolDevice

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
  using UnaryFunctor = std::function<void(Index)>;
  enum : Index { XprEvaluationCost = cost_helper<Kernel>::ScalarCost };
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void run(Kernel& kernel, SimpleThreadPoolDevice& device) {
    const Index innerSize = kernel.innerSize();
    const Index outerSize = kernel.outerSize();
    UnaryFunctor functor = [=, &kernel](Index outer) {
      for (Index inner = 0; inner < innerSize; inner++) {
        kernel.assignCoeffByOuterInner(outer, inner);
      }
    };
    float cost = static_cast<float>(XprEvaluationCost) * static_cast<float>(innerSize);
    device.initParallelFor(0, outerSize, 1, functor, cost);
  }
};

template <typename Kernel>
struct dense_assignment_loop_with_device<Kernel, SimpleThreadPoolDevice, DefaultTraversal, InnerUnrolling> {
  using DstXprType = typename Kernel::DstEvaluatorType::XprType;
  using UnaryFunctor = std::function<void(Index)>;
  enum : Index { XprEvaluationCost = cost_helper<Kernel>::ScalarCost, InnerSize = DstXprType::InnerSizeAtCompileTime };
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void run(Kernel& kernel, SimpleThreadPoolDevice& device) {
    const Index outerSize = kernel.outerSize();
    UnaryFunctor functor = [=, &kernel](Index outer) {
      copy_using_evaluator_DefaultTraversal_InnerUnrolling<Kernel, 0, InnerSize>::run(kernel, outer);
    };
    constexpr float Cost = static_cast<float>(XprEvaluationCost) * static_cast<float>(InnerSize);
    device.initParallelFor(0, outerSize, 1, functor, Cost);
  }
};

template <typename Kernel>
struct dense_assignment_loop_with_device<Kernel, SimpleThreadPoolDevice, InnerVectorizedTraversal, NoUnrolling> {
  using UnaryFunctor = std::function<void(Index)>;
  using PacketType = typename Kernel::PacketType;
  enum : Index {
    XprEvaluationCost = cost_helper<Kernel>::VectorCost,
    PacketSize = unpacket_traits<PacketType>::size,
    SrcAlignment = Kernel::AssignmentTraits::SrcAlignment,
    DstAlignment = Kernel::AssignmentTraits::DstAlignment
  };
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE EIGEN_CONSTEXPR void run(Kernel& kernel,
                                                                        SimpleThreadPoolDevice& device) {
    const Index innerSize = kernel.innerSize();
    const Index outerSize = kernel.outerSize();
    UnaryFunctor functor = [=, &kernel](Index outer) {
      for (Index inner = 0; inner < innerSize; inner += PacketSize)
        kernel.template assignPacketByOuterInner<DstAlignment, SrcAlignment, PacketType>(outer, inner);
    };
    float cost = static_cast<float>(XprEvaluationCost) * static_cast<float>(innerSize);
    device.initParallelFor(0, outerSize, 1, functor, cost);
  }
};

template <typename Kernel>
struct dense_assignment_loop_with_device<Kernel, SimpleThreadPoolDevice, SliceVectorizedTraversal, NoUnrolling> {
  using UnaryFunctor = std::function<void(Index)>;
  using Scalar = typename Kernel::Scalar;
  using PacketType = typename Kernel::PacketType;
  enum : Index {
    XprEvaluationCost = cost_helper<Kernel>::VectorCost,
    PacketSize = unpacket_traits<PacketType>::size,
    StrideMask = -PacketSize
  };
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE EIGEN_CONSTEXPR void run(Kernel& kernel,
                                                                        SimpleThreadPoolDevice& device) {
    const Index innerSize = kernel.innerSize();
    const Index packetAccessEnd = innerSize & StrideMask;
    const Index outerSize = kernel.outerSize();

    // prefer unaligned packet ops for simplicity
    UnaryFunctor functor = [=, &kernel](Index outer) {
      for (Index inner = 0; inner < packetAccessEnd; inner += PacketSize)
        kernel.template assignPacketByOuterInner<Unaligned, Unaligned, PacketType>(outer, inner);
      for (Index inner = packetAccessEnd; inner < innerSize; ++inner) kernel.assignCoeffByOuterInner(outer, inner);
    };

    float cost = static_cast<float>(XprEvaluationCost) * static_cast<float>(innerSize);
    device.initParallelFor(0, outerSize, 1, functor, cost);
  };
};

template <typename Kernel>
struct dense_assignment_loop_with_device<Kernel, SimpleThreadPoolDevice, LinearTraversal, NoUnrolling> {
  using UnaryFunctor = std::function<void(Index)>;
  enum : Index { XprEvaluationCost = cost_helper<Kernel>::ScalarCost };
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE EIGEN_CONSTEXPR void run(Kernel& kernel,
                                                                        SimpleThreadPoolDevice& device) {
    const Index size = kernel.size();
    UnaryFunctor functor = [=, &kernel](Index index) { kernel.assignCoeff(index); };
    device.initParallelFor(0, size, 1, functor, static_cast<float>(XprEvaluationCost));
  }
};

template <typename Kernel>
struct dense_assignment_loop_with_device<Kernel, SimpleThreadPoolDevice, LinearVectorizedTraversal, NoUnrolling> {
  using Scalar = typename Kernel::Scalar;
  using PacketType = typename Kernel::PacketType;
  using LinearFunctor = std::function<void(Index)>;
  enum : Index {
    XprEvaluationCost = cost_helper<Kernel>::VectorCost,
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

    auto vectorIterFunctor = [&kernel](Index index) { kernel.template assignPacket<DstAlignment, SrcAlignment, PacketType>(index); };
    device.initParallelFor(alignedStart, alignedEnd, PacketSize, vectorIterFunctor, static_cast<float>(XprEvaluationCost));

    unaligned_dense_assignment_loop<>::run(kernel, alignedEnd, size);
  }
};

}  // namespace internal

}  // namespace Eigen

#endif  // EIGEN_SIMPLE_THREAD_POOL_DEVICE_H
