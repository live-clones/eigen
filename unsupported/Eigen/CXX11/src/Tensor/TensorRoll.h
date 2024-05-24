// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2024 Tobias Wood tobias@spinicist.org.uk
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_ROLL_H
#define EIGEN_CXX11_TENSOR_TENSOR_ROLL_H
// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

/** \class TensorRoll
 * \ingroup CXX11_Tensor_Module
 *
 * \brief Tensor roll (circular shift) elements class.
 *
 */
namespace internal {
template <typename RollDimensions, typename XprType>
struct traits<TensorRollOp<RollDimensions, XprType> > : public traits<XprType> {
  typedef typename XprType::Scalar Scalar;
  typedef traits<XprType> XprTraits;
  typedef typename XprTraits::StorageKind StorageKind;
  typedef typename XprTraits::Index Index;
  typedef typename XprType::Nested Nested;
  typedef std::remove_reference_t<Nested> Nested_;
  static constexpr int NumDimensions = XprTraits::NumDimensions;
  static constexpr int Layout = XprTraits::Layout;
  typedef typename XprTraits::PointerType PointerType;
};

template <typename RollDimensions, typename XprType>
struct eval<TensorRollOp<RollDimensions, XprType>, Eigen::Dense> {
  typedef const TensorRollOp<RollDimensions, XprType>& type;
};

template <typename RollDimensions, typename XprType>
struct nested<TensorRollOp<RollDimensions, XprType>, 1, typename eval<TensorRollOp<RollDimensions, XprType> >::type> {
  typedef TensorRollOp<RollDimensions, XprType> type;
};

}  // end namespace internal

template <typename RollDimensions, typename XprType>
class TensorRollOp : public TensorBase<TensorRollOp<RollDimensions, XprType>, WriteAccessors> {
 public:
  typedef TensorBase<TensorRollOp<RollDimensions, XprType>, WriteAccessors> Base;
  typedef typename Eigen::internal::traits<TensorRollOp>::Scalar Scalar;
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename Eigen::internal::nested<TensorRollOp>::type Nested;
  typedef typename Eigen::internal::traits<TensorRollOp>::StorageKind StorageKind;
  typedef typename Eigen::internal::traits<TensorRollOp>::Index Index;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorRollOp(const XprType& expr, const RollDimensions& roll_dims)
      : m_xpr(expr), m_roll_dims(roll_dims) {}

  EIGEN_DEVICE_FUNC const RollDimensions& roll() const { return m_roll_dims; }

  EIGEN_DEVICE_FUNC const internal::remove_all_t<typename XprType::Nested>& expression() const { return m_xpr; }

  EIGEN_TENSOR_INHERIT_ASSIGNMENT_OPERATORS(TensorRollOp)

 protected:
  typename XprType::Nested m_xpr;
  const RollDimensions m_roll_dims;
};

// Eval as rvalue
template <typename RollDimensions, typename ArgType, typename Device>
struct TensorEvaluator<const TensorRollOp<RollDimensions, ArgType>, Device> {
  typedef TensorRollOp<RollDimensions, ArgType> XprType;
  typedef typename XprType::Index Index;
  static constexpr int NumDims = internal::array_size<RollDimensions>::value;
  typedef DSizes<Index, NumDims> Dimensions;
  typedef typename XprType::Scalar Scalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename PacketType<CoeffReturnType, Device>::type PacketReturnType;
  static constexpr int PacketSize = PacketType<CoeffReturnType, Device>::size;
  typedef StorageMemory<CoeffReturnType, Device> Storage;
  typedef typename Storage::Type EvaluatorPointerType;

  static constexpr int Layout = TensorEvaluator<ArgType, Device>::Layout;
  enum {
    IsAligned = false,
    PacketAccess = TensorEvaluator<ArgType, Device>::PacketAccess,
    BlockAccess = false,
    PreferBlockAccess = false,
    CoordAccess = false,  // to be implemented
    RawAccess = false
  };
  typedef internal::TensorIntDivisor<Index> IndexDivisor;
  //===- Tensor block evaluation strategy (see TensorBlock.h) -------------===//
  typedef internal::TensorBlockNotImplemented TensorBlock;
  //===--------------------------------------------------------------------===//

  EIGEN_STRONG_INLINE TensorEvaluator(const XprType& op, const Device& device)
      : m_impl(op.expression(), device), m_rolls(op.roll()), m_device(device) {
    EIGEN_STATIC_ASSERT((NumDims > 0), Must_Have_At_Least_One_Dimension_To_Roll);

    // Compute strides
    m_dimensions = m_impl.dimensions();
    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      m_strides[0] = 1;
      for (int i = 1; i < NumDims; ++i) {
        m_strides[i] = m_strides[i - 1] * m_dimensions[i - 1];
        if (m_strides[i] > 0) m_fastStrides[i] = IndexDivisor(m_strides[i]);
      }
    } else {
      m_strides[NumDims - 1] = 1;
      for (int i = NumDims - 2; i >= 0; --i) {
        m_strides[i] = m_strides[i + 1] * m_dimensions[i + 1];
        if (m_strides[i] > 0) m_fastStrides[i] = IndexDivisor(m_strides[i]);
      }
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions& dimensions() const { return m_dimensions; }

  EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(EvaluatorPointerType) {
    m_impl.evalSubExprsIfNeeded(NULL);
    return true;
  }

#ifdef EIGEN_USE_THREADS
  template <typename EvalSubExprsCallback>
  EIGEN_STRONG_INLINE void evalSubExprsIfNeededAsync(EvaluatorPointerType, EvalSubExprsCallback done) {
    m_impl.evalSubExprsIfNeededAsync(nullptr, [done](bool) { done(true); });
  }
#endif  // EIGEN_USE_THREADS

  EIGEN_STRONG_INLINE void cleanup() { m_impl.cleanup(); }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index rollIndex(Index index) const {
    eigen_assert(index < dimensions().TotalSize());
    Index inputIndex = 0;
    auto Wrap = [](Index const i, Index const d) {
      auto const tmp = i % d;
      if (tmp < 0) {
        return tmp + d;
      } else {
        return tmp;
      }
    };
    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      EIGEN_UNROLL_LOOP
      for (int i = NumDims - 1; i > 0; --i) {
        Index idx = index / m_fastStrides[i];
        index -= idx * m_strides[i];
        inputIndex += Wrap(idx + m_rolls[i], m_dimensions[i]) * m_strides[i];
      }
      inputIndex += Wrap(index + m_rolls[0], m_dimensions[0]);
    } else {
      EIGEN_UNROLL_LOOP
      for (int i = 0; i < NumDims - 1; ++i) {
        Index idx = index / m_fastStrides[i];
        index -= idx * m_strides[i];
        inputIndex += Wrap(idx + m_rolls[i], m_dimensions[i]) * m_strides[i];
      }
      inputIndex += Wrap(index + m_rolls[NumDims - 1], m_dimensions[NumDims - 1]);
    }
    return inputIndex;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const {
    return m_impl.coeff(rollIndex(index));
  }

  template <int LoadMode>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE PacketReturnType packet(Index index) const {
    eigen_assert(index + PacketSize - 1 < dimensions().TotalSize());

    // TODO(ndjaitly): write a better packing routine that uses
    // local structure.
    EIGEN_ALIGN_MAX std::remove_const_t<CoeffReturnType> values[PacketSize];
    EIGEN_UNROLL_LOOP
    for (int i = 0; i < PacketSize; ++i) {
      values[i] = coeff(index + i);
    }
    PacketReturnType rslt = internal::pload<PacketReturnType>(values);
    return rslt;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorOpCost costPerCoeff(bool vectorized) const {
    double compute_cost = NumDims * (2 * TensorOpCost::AddCost<Index>() + 2 * TensorOpCost::MulCost<Index>() +
                                     TensorOpCost::DivCost<Index>());
    for (int i = 0; i < NumDims; ++i) {
      compute_cost += 2 * TensorOpCost::AddCost<Index>();
    }
    return m_impl.costPerCoeff(vectorized) + TensorOpCost(0, 0, compute_cost, false /* vectorized */, PacketSize);
  }

  EIGEN_DEVICE_FUNC typename Storage::Type data() const { return NULL; }

 protected:
  Dimensions m_dimensions;
  array<Index, NumDims> m_strides;
  array<IndexDivisor, NumDims> m_fastStrides;
  TensorEvaluator<ArgType, Device> m_impl;
  RollDimensions m_rolls;
  const Device EIGEN_DEVICE_REF m_device;
};
// Eval as lvalue

template <typename RollDimensions, typename ArgType, typename Device>
struct TensorEvaluator<TensorRollOp<RollDimensions, ArgType>, Device>
    : public TensorEvaluator<const TensorRollOp<RollDimensions, ArgType>, Device> {
  typedef TensorEvaluator<const TensorRollOp<RollDimensions, ArgType>, Device> Base;
  typedef TensorRollOp<RollDimensions, ArgType> XprType;
  typedef typename XprType::Index Index;
  static constexpr int NumDims = internal::array_size<RollDimensions>::value;
  typedef DSizes<Index, NumDims> Dimensions;

  static constexpr int Layout = TensorEvaluator<ArgType, Device>::Layout;
  enum {
    IsAligned = false,
    PacketAccess = TensorEvaluator<ArgType, Device>::PacketAccess,
    BlockAccess = false,
    PreferBlockAccess = false,
    CoordAccess = false,  // to be implemented
    RawAccess = false
  };
  EIGEN_STRONG_INLINE TensorEvaluator(const XprType& op, const Device& device) : Base(op, device) {}

  typedef typename XprType::Scalar Scalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename PacketType<CoeffReturnType, Device>::type PacketReturnType;
  static constexpr int PacketSize = PacketType<CoeffReturnType, Device>::size;

  //===- Tensor block evaluation strategy (see TensorBlock.h) -------------===//
  typedef internal::TensorBlockNotImplemented TensorBlock;
  //===--------------------------------------------------------------------===//

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions& dimensions() const { return this->m_dimensions; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar& coeffRef(Index index) const {
    return this->m_impl.coeffRef(this->rollIndex(index));
  }

  template <int StoreMode>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void writePacket(Index index, const PacketReturnType& x) const {
    eigen_assert(index + PacketSize - 1 < dimensions().TotalSize());

    // This code is pilfered from TensorMorphing.h
    EIGEN_ALIGN_MAX CoeffReturnType values[PacketSize];
    internal::pstore<CoeffReturnType, PacketReturnType>(values, x);
    EIGEN_UNROLL_LOOP
    for (int i = 0; i < PacketSize; ++i) {
      this->coeffRef(index + i) = values[i];
    }
  }
};

}  // end namespace Eigen

#endif  // EIGEN_CXX11_TENSOR_TENSOR_ROLL_H
