// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2023 Not Sober <cs.schlosser@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SIMPLE_THREAD_POOL_DEVICE_H
#define EIGEN_SIMPLE_THREAD_POOL_DEVICE_H

namespace Eigen {

struct SimpleThreadPoolDevice {
  SimpleThreadPoolDevice(ThreadPool& pool) : m_pool(pool) {}

  using Task = std::function<void()>;
  using LinearFunctor = std::function<void(Index)>;

  void parallelFor(Index begin, Index end, Index incr, LinearFunctor f, Barrier& barrier, Index maxDepth, Index depth) {
    eigen_assert((incr & (incr - 1)) == 0 && "this method specialized for power-of-two increments");
    if (depth < maxDepth)
    {
      Index size = end - begin;
      Index mid = begin + (size / 2);
      mid &= -incr; // soooooo fancy!
      Task left =  [this, begin, mid, incr, f, &barrier, maxDepth, depth]() { parallelFor(begin, mid, incr, f, barrier, maxDepth, depth + 1); };
      Task right = [this,   mid, end, incr, f, &barrier, maxDepth, depth]() { parallelFor(  mid, end, incr, f, barrier, maxDepth, depth + 1); };
      pool().Schedule(left);
      pool().Schedule(right);
    }
    else
    {
      for (Index i = begin; i + incr <= end; i += incr) f(i);
      barrier.Notify();
    }
  }

  void initParallelFor(Index begin, Index end, Index incr, LinearFunctor f) {
    Index minSize = 50; // for testing
    Index size = end - begin;
    Index maxDepth = size <= minSize ? 0 : numext::log2(size) - numext::log2(minSize);
    uint32_t numNotifies = 1 << static_cast<uint32_t>(maxDepth);
    Barrier barrier(numNotifies);
    parallelFor(begin, end, incr, f, barrier, maxDepth, 0);
    barrier.Wait();
  }
  ThreadPool& pool() { return m_pool; }

  ThreadPool& m_pool;
};

namespace internal
{

template <typename Kernel>
struct dense_assignment_loop_with_device<Kernel, SimpleThreadPoolDevice, LinearVectorizedTraversal, NoUnrolling> {
  typedef typename Kernel::Scalar Scalar;
  typedef typename Kernel::PacketType PacketType;
  enum {
    requestedAlignment = Kernel::AssignmentTraits::LinearRequiredAlignment,
    packetSize = unpacket_traits<PacketType>::size,
    dstIsAligned = int(Kernel::AssignmentTraits::DstAlignment) >= int(requestedAlignment),
    dstAlignment =
        packet_traits<Scalar>::AlignedOnScalar ? int(requestedAlignment) : int(Kernel::AssignmentTraits::DstAlignment),
    srcAlignment = Kernel::AssignmentTraits::JointAlignment
  };

  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE EIGEN_CONSTEXPR void run(Kernel& kernel,
                                                                        SimpleThreadPoolDevice& device) {
    const Index size = kernel.size();
    const Index alignedStart =
        dstIsAligned ? 0 : internal::first_aligned<requestedAlignment>(kernel.dstDataPtr(), size);
    const Index alignedEnd = alignedStart + ((size - alignedStart) / packetSize) * packetSize;

    Barrier barrier(dstIsAligned ? 2 : 3);

    std::function<void()> headLoopFunctor = [&, alignedStart]() {
      unaligned_dense_assignment_loop<dstIsAligned != 0>::run(kernel, 0, alignedStart);
      barrier.Notify();
    };

    std::function<void()> vectorLoopFunctor = [&, alignedStart, alignedEnd]() {
      std::function<void(Index)> vectorIterFunctor = [&](Index index) {
        kernel.template assignPacket<dstAlignment, srcAlignment, PacketType>(index);
      };
      device.initParallelFor(alignedStart, alignedEnd, packetSize, vectorIterFunctor);
      barrier.Notify();
    };

    std::function<void()> tailLoopFunctor = [&, alignedEnd, size]() {
      unaligned_dense_assignment_loop<>::run(kernel, alignedEnd, size);
      barrier.Notify();
    };

    if (!dstIsAligned) device.pool().Schedule(headLoopFunctor);
    device.pool().Schedule(vectorLoopFunctor);
    device.pool().Schedule(tailLoopFunctor);
    barrier.Wait();
  }
};

}

}  // namespace Eigen

#endif  // EIGEN_SIMPLE_THREAD_POOL_DEVICE_H
