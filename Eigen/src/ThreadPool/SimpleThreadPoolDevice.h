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

  void parallelFor(Index begin, Index end, Index incr, std::function<void(Index)> f, Index minJobs) {
    Index numJobs = end - begin;
    if (numJobs > minJobs) {
      Index mid = begin + (numJobs / 2);
      mid = incr * (mid / incr); // mid & (-incr)

      std::function<void()> fLeft =  [this, begin, mid, incr, f, minJobs]() { parallelFor(begin, mid, incr, f, minJobs); };
      std::function<void()> fRight = [this,   mid, end, incr, f, minJobs]() { parallelFor(  mid, end, incr, f, minJobs); };

      pool().Schedule(fLeft);
      pool().Schedule(fRight);
    }
    else
    {
      for (Index index = begin; index + incr <= end; index += incr) {
        f(index);
      }
    }
  }

  void initParallelFor(Index begin, Index end, Index incr, std::function<void(Index)> f) {
    Index minJobs = 50;// for testing
    parallelFor(begin, end, incr, f, minJobs);
  }
  ThreadPool& pool() { return m_pool; }

  ThreadPool& m_pool;
};

namespace internal
{

// specialize what you want
// Linear Vector Traversal -- the workhorse of Eigen assignments
template <typename Kernel>
struct dense_assignment_loop_with_device<Kernel, SimpleThreadPoolDevice, LinearVectorizedTraversal, NoUnrolling> {
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE EIGEN_CONSTEXPR void run(Kernel& kernel,
                                                                        SimpleThreadPoolDevice& device) {
    const Index size = kernel.size();
    typedef typename Kernel::Scalar Scalar;
    typedef typename Kernel::PacketType PacketType;
    enum {
      requestedAlignment = Kernel::AssignmentTraits::LinearRequiredAlignment,
      packetSize = unpacket_traits<PacketType>::size,
      dstIsAligned = int(Kernel::AssignmentTraits::DstAlignment) >= int(requestedAlignment),
      dstAlignment = packet_traits<Scalar>::AlignedOnScalar ? int(requestedAlignment)
                                                            : int(Kernel::AssignmentTraits::DstAlignment),
      srcAlignment = Kernel::AssignmentTraits::JointAlignment
    };
    const Index alignedStart =
        dstIsAligned ? 0 : internal::first_aligned<requestedAlignment>(kernel.dstDataPtr(), size);
    const Index alignedEnd = alignedStart + ((size - alignedStart) / packetSize) * packetSize;

    // should we thread this loop?
    unaligned_dense_assignment_loop<dstIsAligned != 0>::run(kernel, 0, alignedStart);

    
    // for (Index index = alignedStart; index < alignedEnd; index += packetSize)
    //   kernel.template assignPacket<dstAlignment, srcAlignment, PacketType>(index);

    std::function<void(Index)> f = [&kernel](Index index) {
      kernel.template assignPacket<dstAlignment, srcAlignment, PacketType>(index);
    };

    device.initParallelFor(alignedStart, alignedEnd, packetSize, f);

    // should we thread this loop?
    unaligned_dense_assignment_loop<>::run(kernel, alignedEnd, size);
  }
};

}

}  // namespace Eigen

#endif  // EIGEN_SIMPLE_THREAD_POOL_DEVICE_H
