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

// SimpleThreadPoolDevice provides an easy-to-understand Device for parallelizing Eigen Core expressions with
// Threadpool. Expressions are recursively bifurcated until the evaluation cost is less than the threshold for
// delegating the task to a thread.

struct SimpleThreadPoolDevice {
  using Task = std::function<void()>;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE SimpleThreadPoolDevice(ThreadPool& pool, float threadCostThreshold = 500'000)
      : m_pool(pool) {
    setThreadCostThreshold(threadCostThreshold);
  }

  template <int Stride>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void analyzeCost(Index size, float cost, int& maxDepth,
                                                         int& actualThreads) const {
    eigen_assert(cost >= 0.0f && "cost must be non-negative");
    Index numOps = size / Stride;
    float totalCost = static_cast<float>(numOps) * cost;
    actualThreads = numOps < numThreads() ? numOps : numThreads();
    if (numext::isfinite(totalCost)) {
      float idealThreads = totalCost / threadCostThreshold();
      idealThreads = numext::maxi(idealThreads, 1.0f);
      actualThreads = numext::mini(actualThreads, static_cast<int>(idealThreads));
    }
    // use the ceiling of log2 to ensure that threads are fully utilized
    maxDepth = numext::log2(actualThreads);
    if ((actualThreads & (actualThreads - 1)) != 0) maxDepth++;
  }

  template <typename UnaryFunctor, int Stride>
  EIGEN_DEVICE_FUNC EIGEN_DONT_INLINE void parallelFor(Index begin, Index end, UnaryFunctor f, Barrier& barrier,
                                                       int depth, int maxDepth, int actualThreads) {
    EIGEN_STATIC_ASSERT((Stride & (Stride - 1)) == 0, "this function assumes stride is a power of two");
    constexpr Index StrideMask = -Stride;
    while (depth < maxDepth) {
      depth++;
      Index size = end - begin;
      eigen_assert(size % Stride == 0 && "this function assumes size is a multiple of Stride");
      Index mid = begin + ((size / 2) & StrideMask);
      Task right = [=, this, &barrier]() {
        parallelFor<UnaryFunctor, Stride>(mid, end, f, barrier, depth, maxDepth, actualThreads);
      };
      pool().ScheduleWithHint(std::move(right), 0, actualThreads);
      end = mid;
    }
    for (Index i = begin; i < end; i += Stride) f(i);
    barrier.Notify();
  }

  template <typename UnaryFunctor, int Stride>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void initParallelFor(Index begin, Index end, UnaryFunctor f, float cost) {
    Index size = end - begin;
    int maxDepth = 0, actualThreads = 0;
    analyzeCost<Stride>(size, cost, maxDepth, actualThreads);
    Barrier barrier(1 << maxDepth);
    parallelFor<UnaryFunctor, Stride>(begin, end, std::move(f), barrier, 0, maxDepth, actualThreads);
    barrier.Wait();
  }

  template <typename BinaryFunctor, int Stride>
  EIGEN_DEVICE_FUNC EIGEN_DONT_INLINE void parallelFor(Index outerBegin, Index outerEnd, Index innerBegin,
                                                       Index innerEnd, BinaryFunctor f, Barrier& barrier, int depth,
                                                       int maxDepth, int actualThreads) {
    EIGEN_STATIC_ASSERT((Stride & (Stride - 1)) == 0, "this function assumes stride is a power of two");
    constexpr Index StrideMask = -Stride;
    while (depth < maxDepth) {
      depth++;
      Index outerSize = outerEnd - outerBegin;
      if (outerSize > 1) {
        Index mid = outerBegin + (outerSize / 2);
        Task right = [=, this, &barrier]() {
          parallelFor<BinaryFunctor, Stride>(mid, outerEnd, innerBegin, innerEnd, f, barrier, depth, maxDepth,
                                             actualThreads);
        };
        pool().ScheduleWithHint(std::move(right), 0, actualThreads);
        outerEnd = mid;
      } else {
        Index innerSize = innerEnd - innerBegin;
        eigen_assert(innerSize % Stride == 0 && "this function assumes innerSize is a multiple of Stride");
        Index mid = innerBegin + ((innerSize / 2) & StrideMask);
        Task right = [=, this, &barrier]() {
          parallelFor<BinaryFunctor, Stride>(outerBegin, outerEnd, mid, innerEnd, f, barrier, depth, maxDepth,
                                             actualThreads);
        };
        pool().ScheduleWithHint(std::move(right), 0, actualThreads);
        innerEnd = mid;
      }
    }
    for (Index outer = outerBegin; outer < outerEnd; outer++)
      for (Index inner = innerBegin; inner < innerEnd; inner += Stride) f(outer, inner);
    barrier.Notify();
  }

  template <typename BinaryFunctor, int Stride>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void initParallelFor(Index outerBegin, Index outerEnd, Index innerBegin,
                                                             Index innerEnd, BinaryFunctor f, float cost) {
    Index outerSize = outerEnd - outerBegin;
    Index innerSize = innerEnd - innerBegin;
    Index size = outerSize * innerSize;
    int maxDepth = 0, actualThreads = 0;
    analyzeCost<Stride>(size, cost, maxDepth, actualThreads);
    Barrier barrier(1 << maxDepth);
    parallelFor<BinaryFunctor, Stride>(outerBegin, outerEnd, innerBegin, innerEnd, std::move(f), barrier, 0, maxDepth,
                                       actualThreads);
    barrier.Wait();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE ThreadPool& pool() { return m_pool; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int numThreads() const { return m_pool.NumThreads(); }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float threadCostThreshold() const { return m_threadCostThreshold; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void setThreadCostThreshold(float cost) {
    eigen_assert(cost >= 0.0f && "cost must be non-negative");
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
  enum : Index { XprEvaluationCost = cost_helper<Kernel>::ScalarCost };
  struct AssignmentFunctor {
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE AssignmentFunctor(Kernel& kernel) : m_kernel(kernel) {}
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void operator()(Index outer, Index inner) {
      m_kernel.assignCoeffByOuterInner(outer, inner);
    }
    Kernel& m_kernel;
  };
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void run(Kernel& kernel, SimpleThreadPoolDevice& device) {
    const Index innerSize = kernel.innerSize();
    const Index outerSize = kernel.outerSize();
    constexpr float cost = static_cast<float>(XprEvaluationCost);
    AssignmentFunctor functor(kernel);
    device.template initParallelFor<AssignmentFunctor, 1>(0, outerSize, 0, innerSize, std::move(functor), cost);
  }
};

template <typename Kernel>
struct dense_assignment_loop_with_device<Kernel, SimpleThreadPoolDevice, DefaultTraversal, InnerUnrolling> {
  using DstXprType = typename Kernel::DstEvaluatorType::XprType;
  enum : Index { XprEvaluationCost = cost_helper<Kernel>::ScalarCost, InnerSize = DstXprType::InnerSizeAtCompileTime };
  struct AssignmentFunctor {
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE AssignmentFunctor(Kernel& kernel) : m_kernel(kernel) {}
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void operator()(Index outer) {
      copy_using_evaluator_DefaultTraversal_InnerUnrolling<Kernel, 0, InnerSize>::run(m_kernel, outer);
    }
    Kernel& m_kernel;
  };
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void run(Kernel& kernel, SimpleThreadPoolDevice& device) {
    const Index outerSize = kernel.outerSize();
    AssignmentFunctor functor(kernel);
    constexpr float cost = static_cast<float>(XprEvaluationCost) * static_cast<float>(InnerSize);
    device.template initParallelFor<AssignmentFunctor, 1>(0, outerSize, std::move(functor), cost);
  }
};

template <typename Kernel>
struct dense_assignment_loop_with_device<Kernel, SimpleThreadPoolDevice, InnerVectorizedTraversal, NoUnrolling> {
  using PacketType = typename Kernel::PacketType;
  enum : Index {
    XprEvaluationCost = cost_helper<Kernel>::VectorCost,
    PacketSize = unpacket_traits<PacketType>::size,
    SrcAlignment = Kernel::AssignmentTraits::SrcAlignment,
    DstAlignment = Kernel::AssignmentTraits::DstAlignment
  };
  struct AssignmentFunctor {
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE AssignmentFunctor(Kernel& kernel) : m_kernel(kernel) {}
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void operator()(Index outer, Index inner) {
      m_kernel.template assignPacketByOuterInner<Unaligned, Unaligned, PacketType>(outer, inner);
    }
    Kernel& m_kernel;
  };
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void run(Kernel& kernel, SimpleThreadPoolDevice& device) {
    const Index innerSize = kernel.innerSize();
    const Index outerSize = kernel.outerSize();
    float cost = static_cast<float>(XprEvaluationCost) * static_cast<float>(innerSize);
    AssignmentFunctor functor(kernel);
    device.template initParallelFor<AssignmentFunctor, PacketSize>(0, outerSize, 0, innerSize, std::move(functor),
                                                                   cost);
  }
};

template <typename Kernel>
struct dense_assignment_loop_with_device<Kernel, SimpleThreadPoolDevice, SliceVectorizedTraversal, NoUnrolling> {
  using Scalar = typename Kernel::Scalar;
  using PacketType = typename Kernel::PacketType;
  enum : Index {
    XprVectorEvaluationCost = cost_helper<Kernel>::VectorCost,
    XprScalarEvaluationCost = cost_helper<Kernel>::ScalarCost,
    PacketSize = unpacket_traits<PacketType>::size,
    StrideMask = -PacketSize
  };
  struct AssignmentFunctor {
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE AssignmentFunctor(Kernel& kernel) : m_kernel(kernel) {}
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void operator()(Index outer) {
      const Index innerSize = m_kernel.innerSize();
      const Index packetAccessSize = innerSize & StrideMask;
      for (Index inner = 0; inner < packetAccessSize; inner += PacketSize)
        m_kernel.template assignPacketByOuterInner<Unaligned, Unaligned, PacketType>(outer, inner);
      for (Index inner = packetAccessSize; inner < innerSize; inner++) m_kernel.assignCoeffByOuterInner(outer, inner);
    }
    Kernel& m_kernel;
  };
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void run(Kernel& kernel, SimpleThreadPoolDevice& device) {
    const Index outerSize = kernel.outerSize();
    const Index innerSize = kernel.innerSize();
    const Index packetAccessSize = innerSize & StrideMask;
    float cost = static_cast<float>(XprVectorEvaluationCost) * static_cast<float>(packetAccessSize / PacketSize) +
                 static_cast<float>(XprScalarEvaluationCost) * static_cast<float>(innerSize - packetAccessSize);
    AssignmentFunctor functor(kernel);
    device.template initParallelFor<AssignmentFunctor, 1>(0, outerSize, std::move(functor), cost);
  };
};

template <typename Kernel>
struct dense_assignment_loop_with_device<Kernel, SimpleThreadPoolDevice, LinearTraversal, NoUnrolling> {
  enum : Index { XprEvaluationCost = cost_helper<Kernel>::ScalarCost };
  struct AssignmentFunctor {
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE AssignmentFunctor(Kernel& kernel) : m_kernel(kernel) {}
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void operator()(Index index) { m_kernel.assignCoeff(index); }
    Kernel& m_kernel;
  };
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void run(Kernel& kernel, SimpleThreadPoolDevice& device) {
    const Index size = kernel.size();
    constexpr float cost = static_cast<float>(XprEvaluationCost);
    AssignmentFunctor functor(kernel);
    device.template initParallelFor<AssignmentFunctor, 1>(0, size, std::move(functor), cost);
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
  struct AssignmentFunctor {
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE AssignmentFunctor(Kernel& kernel) : m_kernel(kernel) {}
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void operator()(Index index) {
      m_kernel.template assignPacket<DstAlignment, SrcAlignment, PacketType>(index);
    }
    Kernel& m_kernel;
  };
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void run(Kernel& kernel, SimpleThreadPoolDevice& device) {
    const Index size = kernel.size();
    const Index alignedStart =
        DstIsAligned ? 0 : internal::first_aligned<RequestedAlignment>(kernel.dstDataPtr(), size);
    const Index alignedEnd = alignedStart + ((size - alignedStart) / PacketSize) * PacketSize;

    unaligned_dense_assignment_loop<DstIsAligned != 0>::run(kernel, 0, alignedStart);

    constexpr float cost = static_cast<float>(XprEvaluationCost);
    AssignmentFunctor functor(kernel);
    device.template initParallelFor<AssignmentFunctor, PacketSize>(alignedStart, alignedEnd, std::move(functor), cost);

    unaligned_dense_assignment_loop<>::run(kernel, alignedEnd, size);
  }
};

}  // namespace internal

}  // namespace Eigen

#endif  // EIGEN_SIMPLE_THREAD_POOL_DEVICE_H
