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
// Threadpool. Expressions are recursively split evenly until the evaluation cost is less than the threshold for
// delegating the task to a thread. 

//                a
//               / \
//              /   \
//             /     \
//            /       \
//           /         \
//          /           \
//         /             \
//        a               e
//       / \             / \
//      /   \           /   \
//     /     \         /     \
//    a       c       e       g
//   / \     / \     / \     / \
//  /   \   /   \   /   \   /   \
// a     b c     d e     f g     h

// Each thread descends the binary tree to the left, delegates the right task to a new thread, and continues to the
// left. This ensures that work is evenly distributed to the thread pool as quickly as possible and minimizes the number
// of threads creating during the evaluation. Consider an expression that is divided into 8 chunks. The
// primary thread 'a' creates tasks 'c' and 'b', and executes its portion of the expression at the bottom of the tree.
// Likewise, task 'e' creates tasks 'g' and 'f', and executes its portion of the expression. 

struct SimpleThreadPoolDevice {
  using Task = std::function<void()>;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE SimpleThreadPoolDevice(ThreadPool& pool, float threadCostThreshold = 2e-7f)
      : m_pool(pool) {
    setCostFactor(threadCostThreshold);
  }

  template <int PacketSize>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE unsigned int calculateLevels(Index size, float cost) const {
    eigen_assert(cost >= 0.0f && "cost must be non-negative");
    Index numOps = size / PacketSize;
    float totalCost = static_cast<float>(numOps) * cost;
    unsigned int actualThreads = numOps < static_cast<Index>(numThreads()) ? static_cast<unsigned int>(numOps) : numThreads();
    if (numext::isfinite(totalCost)) {
      float idealThreads = totalCost * costFactor();
      idealThreads = numext::maxi(idealThreads, 1.0f);
      actualThreads = numext::mini(actualThreads, static_cast<unsigned int>(idealThreads));
    }
    // use the ceiling of log2 to ensure that threads are fully utilized
    unsigned int maxLevel = numext::log2(actualThreads);
    if ((actualThreads & (actualThreads - 1)) != 0) maxLevel++;
    return maxLevel;
  }

  template <typename UnaryFunctor, int PacketSize>
  EIGEN_DEVICE_FUNC EIGEN_DONT_INLINE void parallelForImpl(Index begin, Index end, const UnaryFunctor& f,
                                                           Barrier& barrier, unsigned int level) {
    EIGEN_STATIC_ASSERT((PacketSize & (PacketSize - 1)) == 0, "this function assumes PacketSize is a power of two");
    constexpr Index PacketSizeMask = -PacketSize;
    while (level > 0) {
      level--;
      Index size = end - begin;
      eigen_assert(size % PacketSize == 0 && "this function assumes size is a multiple of PacketSize");
      Index mid = begin + ((size >> 1) & PacketSizeMask);
      Task right = [=, this, &f, &barrier]() { parallelForImpl<UnaryFunctor, PacketSize>(mid, end, f, barrier, level); };
      pool().Schedule(std::move(right));
      end = mid;
    }
    for (Index i = begin; i < end; i += PacketSize) f(i);
    barrier.Notify();
  }

  template <typename UnaryFunctor, int PacketSize>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void parallelFor(Index begin, Index end, const UnaryFunctor& f, float cost) {
    Index size = end - begin;
    unsigned int maxLevel = calculateLevels<PacketSize>(size, cost);
    Barrier barrier(1 << maxLevel);
    parallelForImpl<UnaryFunctor, PacketSize>(begin, end, f, barrier, maxLevel);
    barrier.Wait();
  }

  template <typename BinaryFunctor, int PacketSize>
  EIGEN_DEVICE_FUNC EIGEN_DONT_INLINE void parallelForImpl(Index outerBegin, Index outerEnd, Index innerBegin,
                                                           Index innerEnd, const BinaryFunctor& f, Barrier& barrier,
                                                           unsigned int level) {
    EIGEN_STATIC_ASSERT((PacketSize & (PacketSize - 1)) == 0, "this function assumes PacketSize is a power of two");
    constexpr Index PacketSizeMask = -PacketSize;
    while (level > 0) {
      level--;
      Index outerSize = outerEnd - outerBegin;
      if (outerSize > 1) {
        Index outerMid = outerBegin + (outerSize >> 1);
        Task right = [=, this, &f, &barrier]() {
          parallelForImpl<BinaryFunctor, PacketSize>(outerMid, outerEnd, innerBegin, innerEnd, f, barrier, level);
        };
        pool().Schedule(std::move(right));
        outerEnd = outerMid;
      } else {
        Index innerSize = innerEnd - innerBegin;
        eigen_assert(innerSize % PacketSize == 0 && "this function assumes innerSize is a multiple of PacketSize");
        Index innerMid = innerBegin + ((innerSize >> 1) & PacketSizeMask);
        Task right = [=, this, &f, &barrier]() {
          parallelForImpl<BinaryFunctor, PacketSize>(outerBegin, outerEnd, innerMid, innerEnd, f, barrier, level);
        };
        pool().Schedule(std::move(right));
        innerEnd = innerMid;
      }
    }
    for (Index outer = outerBegin; outer < outerEnd; outer++)
      for (Index inner = innerBegin; inner < innerEnd; inner += PacketSize) f(outer, inner);
    barrier.Notify();
  }

  template <typename BinaryFunctor, int PacketSize>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void parallelFor(Index outerBegin, Index outerEnd, Index innerBegin,
                                                         Index innerEnd, const BinaryFunctor& f, float cost) {
    Index outerSize = outerEnd - outerBegin;
    Index innerSize = innerEnd - innerBegin;
    Index size = outerSize * innerSize;
    unsigned int maxLevel = calculateLevels<PacketSize>(size, cost);
    Barrier barrier(1 << maxLevel);
    parallelForImpl<BinaryFunctor, PacketSize>(outerBegin, outerEnd, innerBegin, innerEnd, f, barrier, maxLevel);
    barrier.Wait();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE ThreadPool& pool() { return m_pool; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE unsigned int numThreads() const {
    return static_cast<unsigned int>(m_pool.NumThreads());
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float costFactor() const { return m_costFactor; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void setCostFactor(float costFactor) {
    eigen_assert(costFactor >= 0.0f && "costFactor must be non-negative");
    m_costFactor = costFactor;
  }
  ThreadPool& m_pool;
  // costFactor is the inverse of cost of delegating a task to a thread
  // the inverse is used to avoid a floating point division
  float m_costFactor;
};

// specialization of coefficient-wise assignment loops for SimpleThreadPoolDevice

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
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void operator()(Index outer, Index inner) const {
      m_kernel.assignCoeffByOuterInner(outer, inner);
    }
    Kernel& m_kernel;
  };
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void run(Kernel& kernel, SimpleThreadPoolDevice& device) {
    const Index innerSize = kernel.innerSize();
    const Index outerSize = kernel.outerSize();
    constexpr float cost = static_cast<float>(XprEvaluationCost);
    AssignmentFunctor functor(kernel);
    device.template parallelFor<AssignmentFunctor, 1>(0, outerSize, 0, innerSize, functor, cost);
  }
};

template <typename Kernel>
struct dense_assignment_loop_with_device<Kernel, SimpleThreadPoolDevice, DefaultTraversal, InnerUnrolling> {
  using DstXprType = typename Kernel::DstEvaluatorType::XprType;
  enum : Index { XprEvaluationCost = cost_helper<Kernel>::ScalarCost, InnerSize = DstXprType::InnerSizeAtCompileTime };
  struct AssignmentFunctor {
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE AssignmentFunctor(Kernel& kernel) : m_kernel(kernel) {}
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void operator()(Index outer) const {
      copy_using_evaluator_DefaultTraversal_InnerUnrolling<Kernel, 0, InnerSize>::run(m_kernel, outer);
    }
    Kernel& m_kernel;
  };
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void run(Kernel& kernel, SimpleThreadPoolDevice& device) {
    const Index outerSize = kernel.outerSize();
    AssignmentFunctor functor(kernel);
    constexpr float cost = static_cast<float>(XprEvaluationCost) * static_cast<float>(InnerSize);
    device.template parallelFor<AssignmentFunctor, 1>(0, outerSize, functor, cost);
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
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void operator()(Index outer, Index inner) const {
      m_kernel.template assignPacketByOuterInner<Unaligned, Unaligned, PacketType>(outer, inner);
    }
    Kernel& m_kernel;
  };
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void run(Kernel& kernel, SimpleThreadPoolDevice& device) {
    const Index innerSize = kernel.innerSize();
    const Index outerSize = kernel.outerSize();
    const float cost = static_cast<float>(XprEvaluationCost) * static_cast<float>(innerSize);
    AssignmentFunctor functor(kernel);
    device.template parallelFor<AssignmentFunctor, PacketSize>(0, outerSize, 0, innerSize, functor, cost);
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
    PacketSizeMask = -PacketSize
  };
  struct AssignmentFunctor {
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE AssignmentFunctor(Kernel& kernel) : m_kernel(kernel) {}
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void operator()(Index outer) const {
      const Index innerSize = m_kernel.innerSize();
      const Index packetAccessSize = innerSize & PacketSizeMask;
      for (Index inner = 0; inner < packetAccessSize; inner += PacketSize)
        m_kernel.template assignPacketByOuterInner<Unaligned, Unaligned, PacketType>(outer, inner);
      for (Index inner = packetAccessSize; inner < innerSize; inner++) m_kernel.assignCoeffByOuterInner(outer, inner);
    }
    Kernel& m_kernel;
  };
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void run(Kernel& kernel, SimpleThreadPoolDevice& device) {
    const Index outerSize = kernel.outerSize();
    const Index innerSize = kernel.innerSize();
    const Index packetAccessSize = innerSize & PacketSizeMask;
    const float cost = static_cast<float>(XprVectorEvaluationCost) * static_cast<float>(packetAccessSize / PacketSize) +
                       static_cast<float>(XprScalarEvaluationCost) * static_cast<float>(innerSize - packetAccessSize);
    AssignmentFunctor functor(kernel);
    device.template parallelFor<AssignmentFunctor, 1>(0, outerSize, functor, cost);
  };
};

template <typename Kernel>
struct dense_assignment_loop_with_device<Kernel, SimpleThreadPoolDevice, LinearTraversal, NoUnrolling> {
  enum : Index { XprEvaluationCost = cost_helper<Kernel>::ScalarCost };
  struct AssignmentFunctor {
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE AssignmentFunctor(Kernel& kernel) : m_kernel(kernel) {}
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void operator()(Index index) const { m_kernel.assignCoeff(index); }
    Kernel& m_kernel;
  };
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void run(Kernel& kernel, SimpleThreadPoolDevice& device) {
    const Index size = kernel.size();
    constexpr float cost = static_cast<float>(XprEvaluationCost);
    AssignmentFunctor functor(kernel);
    device.template parallelFor<AssignmentFunctor, 1>(0, size, functor, cost);
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
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void operator()(Index index) const {
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
    device.template parallelFor<AssignmentFunctor, PacketSize>(alignedStart, alignedEnd, functor, cost);

    unaligned_dense_assignment_loop<>::run(kernel, alignedEnd, size);
  }
};

}  // namespace internal

}  // namespace Eigen

#endif  // EIGEN_SIMPLE_THREAD_POOL_DEVICE_H
