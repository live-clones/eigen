// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2023 Rasmus Munk Larsen <rmlarsen@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#if defined(EIGEN_USE_THREADS)
#undef EIGEN_USE_THREADS
#endif

#define EIGEN_USE_THREADS

#include "main.h"

namespace Eigen
{
namespace internal {
// conveniently control vectorization logic
template <typename Scalar, bool Vectorize>
struct scalar_dummy_op {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar operator()(const Scalar& a) const { return a; }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& a) const {
    return a;
  }
};
template <typename Scalar, bool Vectorize>
struct functor_traits<scalar_dummy_op<Scalar, Vectorize> > {
  enum { Cost = 1'000'000, PacketAccess = Vectorize && packet_traits<Scalar>::Vectorizable };
};
}  // namespace internal
}

template<typename PlainObject>
void test_threaded_assignment(const PlainObject&, Index rows = PlainObject::RowsAtCompileTime, Index cols = PlainObject::ColsAtCompileTime) {
  using Scalar = typename PlainObject::Scalar;
  using VectorizationOff = internal::scalar_dummy_op<Scalar, false>;
  using VectorizationOn = internal::scalar_dummy_op<Scalar, true>;

  int threads = 4;
  ThreadPool pool(threads);
  CoreThreadPoolDevice threadPoolDevice(pool);

  PlainObject dst(rows, cols), ref(rows, cols), rhs(rows, cols);
  rhs.setRandom();
  const auto rhs_xpr = rhs.cwiseMax(Scalar(1)).cwiseAbs2();

  // linear access
  ref = rhs_xpr.unaryExpr(VectorizationOff());
  dst.device(threadPoolDevice) = rhs_xpr.unaryExpr(VectorizationOff());
  VERIFY_IS_CWISE_EQUAL(ref, dst);

  ref = rhs_xpr.unaryExpr(VectorizationOn());
  dst.device(threadPoolDevice) = rhs_xpr.unaryExpr(VectorizationOn());
  VERIFY_IS_CWISE_EQUAL(ref, dst);

  // outer-inner access
  Index blockRows = numext::maxi(Index(1), rows - 1);
  Index blockCols = numext::maxi(Index(1), cols - 1);

  ref.setZero();
  dst.setZero();
  ref.bottomRightCorner(blockRows, blockCols) =
      rhs_xpr.bottomRightCorner(blockRows, blockCols).unaryExpr(VectorizationOff());
  dst.bottomRightCorner(blockRows, blockCols).device(threadPoolDevice) =
      rhs_xpr.bottomRightCorner(blockRows, blockCols).unaryExpr(VectorizationOff());
  VERIFY_IS_CWISE_EQUAL(ref, dst);

  ref.setZero();
  dst.setZero();
  ref.bottomRightCorner(blockRows, blockCols) =
      rhs_xpr.bottomRightCorner(blockRows, blockCols).unaryExpr(VectorizationOn());
  dst.bottomRightCorner(blockRows, blockCols).device(threadPoolDevice) =
      rhs_xpr.bottomRightCorner(blockRows, blockCols).unaryExpr(VectorizationOn());
  VERIFY_IS_CWISE_EQUAL(ref, dst);
}

EIGEN_DECLARE_TEST(test) {
  CALL_SUBTEST(test_threaded_assignment(MatrixXd(), 123, 123));
  CALL_SUBTEST(test_threaded_assignment(Matrix<float,16,16>()));
}
