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
struct binary_redux_traits {
  // non-vector expression: operation has strict dimensional requirements
  static constexpr int LhsFlags = evaluator<Lhs>::Flags, RhsFlags = evaluator<Rhs>::Flags,
                       LhsRowsAtCompileTime = Lhs::RowsAtCompileTime, RhsRowsAtCompileTime = Rhs::RowsAtCompileTime,
                       LhsColsAtCompileTime = Lhs::ColsAtCompileTime, RhsColsAtCompileTime = Rhs::ColsAtCompileTime,
                       LhsSizeAtCompileTime = Lhs::SizeAtCompileTime, RhsSizeAtCompileTime = Rhs::SizeAtCompileTime;

  static constexpr bool IsRowMajor = Lhs::IsRowMajor,
                        StorageOrdersAgree = (LhsFlags & RowMajorBit) == (RhsFlags & RowMajorBit),
                        LinearAccess = StorageOrdersAgree && (LhsFlags & RhsFlags & LinearAccessBit),
                        MaybePacketAccess = (LhsFlags & RhsFlags & PacketAccessBit);

  static constexpr int RowsAtCompileTime = min_size_prefer_fixed(LhsRowsAtCompileTime, RhsRowsAtCompileTime),
                       ColsAtCompileTime = min_size_prefer_fixed(LhsColsAtCompileTime, RhsColsAtCompileTime),
                       OuterSizeAtCompileTime = IsRowMajor ? RowsAtCompileTime : ColsAtCompileTime,
                       InnerSizeAtCompileTime = IsRowMajor ? ColsAtCompileTime : RowsAtCompileTime,
                       SizeAtCompileTime = size_at_compile_time(OuterSizeAtCompileTime, InnerSizeAtCompileTime);
};

template <typename Lhs, typename Rhs>
struct binary_redux_traits<Lhs, Rhs, true> {
  // vector expression: operation is allowed, provided that both operands are the same size
  static constexpr bool LinearAccess = true, MaybePacketAccess = true;

  static constexpr int LhsSizeAtCompileTime = Lhs::SizeAtCompileTime, RhsSizeAtCompileTime = Rhs::SizeAtCompileTime,
                       OuterSizeAtCompileTime = 1,
                       InnerSizeAtCompileTime = min_size_prefer_fixed(LhsSizeAtCompileTime, RhsSizeAtCompileTime),
                       SizeAtCompileTime = InnerSizeAtCompileTime;
};

template <typename Func, typename Lhs, typename Rhs>
struct binary_redux_evaluator {
  using Scalar = typename Func::result_type;
  using PacketType = typename packet_traits<Scalar>::type;
  using Traits = binary_redux_traits<Lhs, Rhs>;

  static constexpr bool IsVectorAtCompileTime = Lhs::IsVectorAtCompileTime && Rhs::IsVectorAtCompileTime,
                        IsRowMajor = Lhs::IsRowMajor, LinearAccess = Traits::LinearAccess,
                        PacketAccess = Traits::MaybePacketAccess && Func::PacketAccess,
                        UseAlignedMode =
                            LinearAccess ||
                            (Traits::InnerSizeAtCompileTime != Dynamic) &&
                                ((Traits::InnerSizeAtCompileTime % unpacket_traits<PacketType>::size) == 0);

  static constexpr int LhsAlignment = UseAlignedMode ? evaluator<Lhs>::Alignment : Unaligned,
                       RhsAlignment = UseAlignedMode ? evaluator<Rhs>::Alignment : Unaligned;

  static constexpr TraversalType PreferredTraversal =
      PacketAccess ? (LinearAccess ? LinearVectorizedTraversal : SliceVectorizedTraversal)
                   : (LinearAccess ? LinearTraversal : DefaultTraversal);

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE explicit binary_redux_evaluator(const Lhs& lhs, const Rhs& rhs,
                                                                        Func func = Func())
      : m_func(func),
        m_lhs(lhs),
        m_rhs(rhs),
        m_outerSize(lhs.outerSize()),
        m_innerSize(lhs.innerSize()),
        m_size(lhs.size()) {
    bool sizesMatch = lhs.size() == rhs.size();
    bool dimensionsMatch = (lhs.rows() == rhs.rows()) && (lhs.cols() == rhs.cols());
    EIGEN_UNUSED_VARIABLE(sizesMatch)
    EIGEN_UNUSED_VARIABLE(dimensionsMatch)
    eigen_assert((IsVectorAtCompileTime ? sizesMatch : dimensionsMatch) && "Incompatible dimensions");
  }

  Index innerSize() const { return m_innerSize.value(); }
  Index outerSize() const { return m_outerSize.value(); }
  Index size() const { return m_size.value(); }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar initialize() const { return m_func.initialize(); }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar coeff(const Scalar& value, Index row, Index col) const {
    return m_func(value, m_lhs.coeff(row, col), m_rhs.coeff(row, col));
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar coeffByOuterInner(const Scalar& value, Index outer, Index inner) const {
    Index row = IsRowMajor ? outer : inner;
    Index col = IsRowMajor ? inner : outer;
    return coeff(value, row, col);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar coeff(const Scalar& value, Index index) const {
    return m_func(value, m_lhs.coeff(index), m_rhs.coeff(index));
  }

  template <typename PacketType, int LhsMode = LhsAlignment, int RhsMode = RhsAlignment>
  EIGEN_STRONG_INLINE PacketType packet(PacketType value, Index row, Index col) const {
    return m_func.packetOp(value, m_lhs.template packet<LhsMode, PacketType>(row, col),
                           m_rhs.template packet<RhsMode, PacketType>(row, col));
  }

  template <typename PacketType, int LhsMode = LhsAlignment, int RhsMode = RhsAlignment>
  EIGEN_STRONG_INLINE PacketType packetByOuterInner(PacketType value, Index outer, Index inner) const {
    Index row = IsRowMajor ? outer : inner;
    Index col = IsRowMajor ? inner : outer;
    return packet<PacketType, LhsMode, RhsMode>(value, row, col);
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

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Func& func() { return m_func; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Func& func() const { return m_func; }

  const Func m_func;
  const evaluator<Lhs> m_lhs;
  const evaluator<Rhs> m_rhs;
  const variable_if_dynamic<Index, Traits::OuterSizeAtCompileTime> m_outerSize;
  const variable_if_dynamic<Index, Traits::InnerSizeAtCompileTime> m_innerSize;
  const variable_if_dynamic<Index, Traits::SizeAtCompileTime> m_size;
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
    Scalar scalarAccum = eval.initialize();
    Packet packetAccum = pset1<Packet>(scalarAccum);
    for (Index k = 0; k < packetEnd; k += kPacketSize) {
      packetAccum = eval.packet(packetAccum, k);
    }
    scalarAccum = eval.predux(packetAccum);
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
    Packet packetAccum = pset1<Packet>(eval.initialize());
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
