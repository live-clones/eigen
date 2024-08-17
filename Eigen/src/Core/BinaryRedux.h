// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2024 Charlie Schlosser <cs.schlosser@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_BINARY_REDUX_H
#define EIGEN_BINARY_REDUX_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

template <typename Lhs, typename Rhs, bool VectorXpr = Lhs::IsVectorAtCompileTime && Rhs::IsVectorAtCompileTime>
struct binary_redux_assert {
  EIGEN_STATIC_ASSERT_SAME_MATRIX_SIZE(Lhs, Rhs)
  static void run() {}
};
template <typename Lhs, typename Rhs>
struct binary_redux_assert<Lhs,Rhs,true> {
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Lhs)
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Rhs)
  EIGEN_STATIC_ASSERT_SAME_VECTOR_SIZE(Lhs, Rhs)
  static void run() {}
};

template <typename Lhs, typename Rhs>
struct binary_redux_traits {
  static constexpr int LhsFlags = evaluator<Lhs>::Flags, RhsFlags = evaluator<Rhs>::Flags;

  static constexpr bool VectorXpr = Lhs::IsVectorAtCompileTime && Rhs::IsVectorAtCompileTime,
                        IsRowMajor = Lhs::IsRowMajor,
                        StorageOrdersAgree = (LhsFlags & RowMajorBit) == (RhsFlags & RowMajorBit),
                        LinearAccess = VectorXpr || (StorageOrdersAgree && (LhsFlags & RhsFlags & LinearAccessBit)),
                        MaybePacketAccess = bool(LhsFlags & RhsFlags & PacketAccessBit);

  static constexpr int LhsRowsAtCompileTime = Lhs::RowsAtCompileTime, RhsRowsAtCompileTime = Rhs::RowsAtCompileTime,
                       LhsColsAtCompileTime = Lhs::ColsAtCompileTime, RhsColsAtCompileTime = Rhs::ColsAtCompileTime,
                       LhsSizeAtCompileTime = Lhs::SizeAtCompileTime, RhsSizeAtCompileTime = Rhs::SizeAtCompileTime,
                       RowsAtCompileTime = min_size_prefer_fixed(LhsRowsAtCompileTime, RhsRowsAtCompileTime),
                       ColsAtCompileTime = min_size_prefer_fixed(LhsColsAtCompileTime, RhsColsAtCompileTime),
                       SizeAtCompileTime = min_size_prefer_fixed(LhsSizeAtCompileTime, RhsSizeAtCompileTime),
                       OuterSizeAtCompileTime = VectorXpr    ? 1
                                                : IsRowMajor ? RowsAtCompileTime
                                                             : ColsAtCompileTime,
                       InnerSizeAtCompileTime = VectorXpr    ? SizeAtCompileTime
                                                : IsRowMajor ? ColsAtCompileTime
                                                             : RowsAtCompileTime;
};

template <typename Func, typename Lhs, typename Rhs>
struct binary_redux_evaluator {
  using Scalar = typename Func::result_type;
  using PacketType = typename packet_traits<Scalar>::type;
  using Traits = binary_redux_traits<Lhs, Rhs>;

  static constexpr int OuterSizeAtCompileTime = Traits::OuterSizeAtCompileTime,
                       InnerSizeAtCompileTime = Traits::InnerSizeAtCompileTime,
                       PacketSize = unpacket_traits<PacketType>::size;

  static constexpr bool VectorXpr = Traits::VectorXpr, IsRowMajor = Traits::IsRowMajor,
                        LinearAccess = Traits::LinearAccess,
                        PacketAccess = Traits::MaybePacketAccess && Func::PacketAccess,
                        UseAlignedMode = LinearAccess || (InnerSizeAtCompileTime % PacketSize) == 0;

  static constexpr int LhsAlignment = UseAlignedMode ? int(evaluator<Lhs>::Alignment) : int(Unaligned),
                       RhsAlignment = UseAlignedMode ? int(evaluator<Rhs>::Alignment) : int(Unaligned);

  static constexpr TraversalType PreferredTraversal =
      PacketAccess ? (LinearAccess ? LinearVectorizedTraversal : SliceVectorizedTraversal)
                   : (LinearAccess ? LinearTraversal : DefaultTraversal);

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE explicit binary_redux_evaluator(const Lhs& lhs, const Rhs& rhs,
                                                                        Func func = Func())
      : m_func(func),
        m_lhs(lhs),
        m_rhs(rhs),
        m_outerSize(VectorXpr ? 1 : lhs.outerSize()),
        m_innerSize(VectorXpr ? lhs.size() : lhs.innerSize()) {
    binary_redux_assert<Lhs, Rhs>::run();
    eigen_assert(checkSizes(lhs, rhs) && "Incompatible dimensions");
  }

  Index outerSize() const { return m_outerSize.value(); }
  Index innerSize() const { return m_innerSize.value(); }
  Index size() const { return m_outerSize.value() * m_innerSize.value(); }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar initialize() const { return m_func.initialize(); }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar coeffByOuterInner(const Scalar& value, Index outer, Index inner) const {
    Index row = IsRowMajor ? outer : inner;
    Index col = IsRowMajor ? inner : outer;
    return m_func(value, m_lhs.coeff(row, col), m_rhs.coeff(row, col));
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar coeff(const Scalar& value, Index index) const {
    return m_func(value, m_lhs.coeff(index), m_rhs.coeff(index));
  }

  template <typename PacketType, int LhsMode = LhsAlignment, int RhsMode = RhsAlignment>
  EIGEN_STRONG_INLINE PacketType packetByOuterInner(PacketType value, Index outer, Index inner) const {
    Index row = IsRowMajor ? outer : inner;
    Index col = IsRowMajor ? inner : outer;
    return m_func.packetOp(value, m_lhs.template packet<LhsMode, PacketType>(row, col),
                           m_rhs.template packet<RhsMode, PacketType>(row, col));
  }

  template <typename PacketType, int LhsMode = LhsAlignment, int RhsMode = RhsAlignment>
  EIGEN_STRONG_INLINE PacketType packet(PacketType value, Index index) const {
    return m_func.packetOp(value, m_lhs.template packet<LhsMode, PacketType>(index),
                           m_rhs.template packet<RhsMode, PacketType>(index));
  }

  template <typename PacketType>
  EIGEN_STRONG_INLINE Scalar predux(PacketType packet) const {
    return m_func.preduxOp(packet);
  }

  template <typename PacketType>
  EIGEN_STRONG_INLINE Scalar predux(const PacketType& packetAccum, const Scalar& scalarAccum) const {
    return m_func.preduxOp(packetAccum, scalarAccum);
  }

  const Func m_func;
  const evaluator<Lhs> m_lhs;
  const evaluator<Rhs> m_rhs;
  const variable_if_dynamic<Index, OuterSizeAtCompileTime> m_outerSize;
  const variable_if_dynamic<Index, InnerSizeAtCompileTime> m_innerSize;

 private:
  static bool checkSizes(const Lhs& lhs, const Rhs& rhs) {
    if (VectorXpr)
      return lhs.size() == rhs.size();
    else
      return (lhs.rows() == rhs.rows()) && (lhs.cols() == rhs.cols());
  }
};

template <typename BinaryEvaluator, TraversalType Traversal = BinaryEvaluator::PreferredTraversal>
struct binary_redux_impl;

template <typename BinaryEvaluator>
struct binary_redux_impl<BinaryEvaluator, DefaultTraversal> {
  using Scalar = typename BinaryEvaluator::Scalar;
  static Scalar run(const BinaryEvaluator& eval) {
    const Index outerSize = eval.outerSize();
    const Index innerSize = eval.innerSize();
    Scalar scalarAccum = eval.initialize();
    for (Index j = 0; j < outerSize; j++) {
      for (Index i = 0; i < innerSize; i++) {
        scalarAccum = eval.coeffByOuterInner(scalarAccum, j, i);
      }
    }
    return scalarAccum;
  };
};

template <typename BinaryEvaluator>
struct binary_redux_impl<BinaryEvaluator, LinearTraversal> {
  using Scalar = typename BinaryEvaluator::Scalar;
  static Scalar run(const BinaryEvaluator& eval) {
    const Index size = eval.size();
    Scalar scalarAccum = eval.initialize();
    for (Index k = 0; k < size; k++) {
      scalarAccum = eval.coeff(scalarAccum, k);
    }
    return scalarAccum;
  };
};

template <typename BinaryEvaluator>
struct binary_redux_impl<BinaryEvaluator, LinearVectorizedTraversal> {
  using Scalar = typename BinaryEvaluator::Scalar;
  using Packet = typename BinaryEvaluator::PacketType;
  static constexpr int kPacketSize = unpacket_traits<Packet>::size;
  static Scalar run(const BinaryEvaluator& eval) {
    const Index size = eval.size();
    const Index packetEnd = numext::round_down(size, kPacketSize);
    Packet packetAccum = pset1<Packet>(eval.initialize());
    for (Index k = 0; k < packetEnd; k += kPacketSize) {
      packetAccum = eval.packet(packetAccum, k);
    }
    Scalar scalarAccum = eval.predux(packetAccum);
    for (Index k = packetEnd; k < size; k++) {
      scalarAccum = eval.coeff(scalarAccum, k);
    }
    return scalarAccum;
  };
};

template <typename BinaryEvaluator>
struct binary_redux_impl<BinaryEvaluator, SliceVectorizedTraversal> {
  using Scalar = typename BinaryEvaluator::Scalar;
  using Packet = typename BinaryEvaluator::PacketType;
  static constexpr int kPacketSize = unpacket_traits<Packet>::size;
  static Scalar run(const BinaryEvaluator& eval) {
    const Index outerSize = eval.outerSize();
    const Index innerSize = eval.innerSize();
    const Index packetEnd = numext::round_down(innerSize, kPacketSize);
    Scalar scalarAccum = eval.initialize();
    Packet packetAccum = pset1<Packet>(scalarAccum);
    for (Index j = 0; j < outerSize; j++) {
      for (Index i = 0; i < packetEnd; i += kPacketSize) {
        packetAccum = eval.packet(packetAccum, j, i);
      }
      for (Index i = packetEnd; i < innerSize; i++) {
        scalarAccum = eval.coeff(scalarAccum, j, i);
      }
    }
    scalarAccum = eval.predux(packetAccum, scalarAccum);
    return scalarAccum;
  };
};
}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_BINARY_REDUX_H
