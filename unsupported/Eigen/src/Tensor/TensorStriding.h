// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_TENSOR_TENSOR_STRIDING_H
#define EIGEN_TENSOR_TENSOR_STRIDING_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {
template <typename Strides, typename XprType>
struct traits<TensorStridingOp<Strides, XprType> > : public traits<XprType> {
  typedef typename XprType::Scalar Scalar;
  typedef traits<XprType> XprTraits;
  typedef typename XprTraits::StorageKind StorageKind;
  typedef typename XprTraits::Index Index;
  static constexpr int NumDimensions = XprTraits::NumDimensions;
  static constexpr int Layout = XprTraits::Layout;
  typedef typename XprTraits::PointerType PointerType;
};

template <typename Strides, typename XprType>
struct eval<TensorStridingOp<Strides, XprType>, Eigen::Dense> {
  typedef const TensorStridingOp<Strides, XprType> EIGEN_DEVICE_REF type;
};

}  // end namespace internal

/**
 * \ingroup Tensor_Module
 *
 * \brief Tensor striding class.
 */
template <typename Strides, typename XprType>
class TensorStridingOp : public TensorBase<TensorStridingOp<Strides, XprType> > {
 public:
  typedef TensorBase<TensorStridingOp<Strides, XprType> > Base;
  typedef typename Eigen::internal::traits<TensorStridingOp>::Scalar Scalar;
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename Eigen::internal::ref_selector<TensorStridingOp>::type Nested;
  typedef typename Eigen::internal::traits<TensorStridingOp>::StorageKind StorageKind;
  typedef typename Eigen::internal::traits<TensorStridingOp>::Index Index;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorStridingOp(const XprType& expr, const Strides& dims)
      : m_xpr(expr), m_dims(dims) {}

  EIGEN_DEVICE_FUNC const Strides& strides() const { return m_dims; }

  EIGEN_DEVICE_FUNC const internal::remove_all_t<typename XprType::Nested>& expression() const { return m_xpr; }

  EIGEN_INHERIT_ASSIGNMENT_OPERATORS(TensorStridingOp)

 protected:
  typename XprType::Nested m_xpr;
  const Strides m_dims;
};

// Eval as rvalue
template <typename Strides, typename ArgType, typename Device>
struct TensorEvaluator<const TensorStridingOp<Strides, ArgType>, Device> {
  typedef TensorStridingOp<Strides, ArgType> XprType;
  typedef typename XprType::Index Index;
  static constexpr int NumDims = internal::array_size<typename TensorEvaluator<ArgType, Device>::Dimensions>::value;
  typedef DSizes<Index, NumDims> Dimensions;
  typedef typename XprType::Scalar Scalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename PacketType<CoeffReturnType, Device>::type PacketReturnType;
  static constexpr int PacketSize = PacketType<CoeffReturnType, Device>::size;
  typedef StorageMemory<CoeffReturnType, Device> Storage;
  typedef typename Storage::Type EvaluatorPointerType;

  static constexpr int Layout = TensorEvaluator<ArgType, Device>::Layout;
  typedef TensorEvaluator<const XprType, Device> Self;
  enum {
    IsAligned = false,
    // Packets are assembled from inner runs even when the nested evaluator
    // only exposes coefficient access.
    PacketAccess = (PacketType<CoeffReturnType, Device>::size > 1),
    BlockAccess = false,
    PreferBlockAccess = TensorEvaluator<ArgType, Device>::PreferBlockAccess,
    CoordAccess = false,  // to be implemented
    RawAccess = false
  };

  //===- Tensor block evaluation strategy (see TensorBlock.h) -------------===//
  typedef internal::TensorBlockNotImplemented TensorBlock;
  //===--------------------------------------------------------------------===//

  EIGEN_STRONG_INLINE TensorEvaluator(const XprType& op, const Device& device) : m_impl(op.expression(), device) {
    m_dimensions = m_impl.dimensions();
    m_is_identity = true;
    for (int i = 0; i < NumDims; ++i) {
      m_dimensions[i] = Eigen::numext::ceil(static_cast<float>(m_dimensions[i]) / op.strides()[i]);
      if (op.strides()[i] != 1) m_is_identity = false;
    }

    const typename TensorEvaluator<ArgType, Device>::Dimensions& input_dims = m_impl.dimensions();
    EIGEN_IF_CONSTEXPR (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      m_outputStrides[0] = 1;
      m_inputStrides[0] = 1;
      for (int i = 1; i < NumDims; ++i) {
        m_outputStrides[i] = m_outputStrides[i - 1] * m_dimensions[i - 1];
        m_inputStrides[i] = m_inputStrides[i - 1] * input_dims[i - 1];
        m_inputStrides[i - 1] *= op.strides()[i - 1];
      }
      m_inputStrides[NumDims - 1] *= op.strides()[NumDims - 1];
    } else {  // RowMajor
      m_outputStrides[NumDims - 1] = 1;
      m_inputStrides[NumDims - 1] = 1;
      for (int i = NumDims - 2; i >= 0; --i) {
        m_outputStrides[i] = m_outputStrides[i + 1] * m_dimensions[i + 1];
        m_inputStrides[i] = m_inputStrides[i + 1] * input_dims[i + 1];
        m_inputStrides[i + 1] *= op.strides()[i + 1];
      }
      m_inputStrides[0] *= op.strides()[0];
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions& dimensions() const { return m_dimensions; }

  EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(EvaluatorPointerType /*data*/) {
    m_impl.evalSubExprsIfNeeded(NULL);
    return true;
  }
  EIGEN_STRONG_INLINE void cleanup() { m_impl.cleanup(); }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const {
    if (m_is_identity) {
      return m_impl.coeff(index);
    }
    return m_impl.coeff(srcCoeff(index));
  }

  // Assembles a packet whose elements all lie in one inner-most run of the
  // output: the input indices form an arithmetic progression starting at
  // `base` with step `inner_stride`, so the index mapping is computed once
  // per packet instead of once per coefficient.
  template <int LoadMode, typename Self, bool ImplPacketAccess>
  struct InnerRunLoader {
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE static PacketReturnType Run(const Self& self, Index base,
                                                                      Index inner_stride) {
      EIGEN_ALIGN_TO_BOUNDARY(sizeof(PacketReturnType)) std::remove_const_t<CoeffReturnType> values[PacketSize];
      EIGEN_UNROLL_LOOP
      for (int i = 0; i < PacketSize; ++i) {
        values[i] = self.m_impl.coeff(base + i * inner_stride);
      }
      return internal::pload<PacketReturnType>(values);
    }
  };

  template <int LoadMode, typename Self>
  struct InnerRunLoader<LoadMode, Self, true> {
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE static PacketReturnType Run(const Self& self, Index base,
                                                                      Index inner_stride) {
      if (inner_stride == 1) {
        // Inner dimension not strided: one contiguous load.
        return self.m_impl.template packet<Unaligned>(base);
      }
      EIGEN_ALIGN_TO_BOUNDARY(sizeof(PacketReturnType)) std::remove_const_t<CoeffReturnType> values[PacketSize];
      EIGEN_UNROLL_LOOP
      for (int i = 0; i < PacketSize; ++i) {
        values[i] = self.m_impl.coeff(base + i * inner_stride);
      }
      return internal::pload<PacketReturnType>(values);
    }
  };

  template <int LoadMode, typename Self, bool ImplPacketAccess>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE static PacketReturnType LoadPacketViaInnerRun(const Self& self, Index index) {
    constexpr int inner_dim = (static_cast<int>(Layout) == static_cast<int>(ColMajor)) ? 0 : NumDims - 1;
    Index inner_pos;
    const Index base = self.srcCoeffInner(index, inner_pos);
    if (inner_pos + PacketSize <= self.m_dimensions[inner_dim]) {
      return InnerRunLoader<LoadMode, Self, ImplPacketAccess>::Run(self, base, self.m_inputStrides[inner_dim]);
    }

    // The packet crosses an inner-run boundary: assemble it scalar by scalar.
    EIGEN_ALIGN_TO_BOUNDARY(sizeof(PacketReturnType)) std::remove_const_t<CoeffReturnType> values[PacketSize];
    EIGEN_UNROLL_LOOP
    for (int i = 0; i < PacketSize; ++i) {
      values[i] = self.coeff(index + i);
    }
    return internal::pload<PacketReturnType>(values);
  }

  template <int LoadMode, typename Self, bool ImplPacketAccess>
  struct PacketLoader {
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE static PacketReturnType Run(const Self& self, Index index) {
      if (self.m_is_identity) {
        EIGEN_ALIGN_TO_BOUNDARY(sizeof(PacketReturnType)) std::remove_const_t<CoeffReturnType> values[PacketSize];
        EIGEN_UNROLL_LOOP
        for (int i = 0; i < PacketSize; ++i) {
          values[i] = self.m_impl.coeff(index + i);
        }
        return internal::pload<PacketReturnType>(values);
      }
      return LoadPacketViaInnerRun<LoadMode, Self, ImplPacketAccess>(self, index);
    }
  };

  template <int LoadMode, typename Self>
  struct PacketLoader<LoadMode, Self, true> {
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE static PacketReturnType Run(const Self& self, Index index) {
      if (self.m_is_identity) {
        return self.m_impl.template packet<LoadMode>(index);
      }
      return LoadPacketViaInnerRun<LoadMode, Self, true>(self, index);
    }
  };

  template <int LoadMode>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE PacketReturnType packet(Index index) const {
    eigen_assert(index + PacketSize - 1 < dimensions().TotalSize());
    return PacketLoader<LoadMode, Self, TensorEvaluator<ArgType, Device>::PacketAccess>::Run(*this, index);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorOpCost costPerCoeff(bool vectorized) const {
    const double compute_cost = m_is_identity
                                    ? TensorOpCost::AddCost<Index>()
                                    : (NumDims - 1) * (TensorOpCost::AddCost<Index>() + TensorOpCost::MulCost<Index>() +
                                                       TensorOpCost::DivCost<Index>()) +
                                          TensorOpCost::MulCost<Index>();
    constexpr int innerDim = (static_cast<int>(Layout) == static_cast<int>(ColMajor)) ? 0 : (NumDims - 1);
    return m_impl.costPerCoeff(vectorized && (m_is_identity || m_inputStrides[innerDim] == 1)) +
           // The index mapping is not vectorized per se, but it is computed once per packet.
           TensorOpCost(0, 0, compute_cost, vectorized, PacketSize);
  }

  EIGEN_DEVICE_FUNC typename Storage::Type data() const { return NULL; }

 protected:
  // Computes the input index of output index `index` and, as a by-product of
  // the same walk, the output's inner-dimension coordinate. The packet paths
  // use the latter to test whether a whole packet stays inside one inner-most
  // run without spending an extra division on it.
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index srcCoeffInner(Index index, Index& inner_pos) const {
    Index inputIndex = 0;
    EIGEN_IF_CONSTEXPR (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      EIGEN_UNROLL_LOOP
      for (int i = NumDims - 1; i > 0; --i) {
        const Index idx = index / m_outputStrides[i];
        inputIndex += idx * m_inputStrides[i];
        index -= idx * m_outputStrides[i];
      }
      inner_pos = index;
      inputIndex += index * m_inputStrides[0];
    } else {  // RowMajor
      EIGEN_UNROLL_LOOP
      for (int i = 0; i < NumDims - 1; ++i) {
        const Index idx = index / m_outputStrides[i];
        inputIndex += idx * m_inputStrides[i];
        index -= idx * m_outputStrides[i];
      }
      inner_pos = index;
      inputIndex += index * m_inputStrides[NumDims - 1];
    }
    return inputIndex;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index srcCoeff(Index index) const {
    Index inner_pos;
    return srcCoeffInner(index, inner_pos);
  }

  Dimensions m_dimensions;
  bool m_is_identity;
  array<Index, NumDims> m_outputStrides;
  array<Index, NumDims> m_inputStrides;
  TensorEvaluator<ArgType, Device> m_impl;
};

// Eval as lvalue
template <typename Strides, typename ArgType, typename Device>
struct TensorEvaluator<TensorStridingOp<Strides, ArgType>, Device>
    : public TensorEvaluator<const TensorStridingOp<Strides, ArgType>, Device> {
  typedef TensorStridingOp<Strides, ArgType> XprType;
  typedef TensorEvaluator<const XprType, Device> Base;
  static constexpr int NumDims = internal::array_size<typename TensorEvaluator<ArgType, Device>::Dimensions>::value;

  static constexpr int Layout = TensorEvaluator<ArgType, Device>::Layout;
  enum {
    IsAligned = false,
    // Packets are scattered into inner runs even when the nested evaluator
    // only exposes coefficient access.
    PacketAccess = (PacketType<typename XprType::CoeffReturnType, Device>::size > 1),
    PreferBlockAccess = false,
    CoordAccess = false,  // to be implemented
    RawAccess = false
  };

  EIGEN_STRONG_INLINE TensorEvaluator(const XprType& op, const Device& device) : Base(op, device) {}

  typedef typename XprType::Index Index;
  typedef typename XprType::Scalar Scalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename PacketType<CoeffReturnType, Device>::type PacketReturnType;
  static constexpr int PacketSize = PacketType<CoeffReturnType, Device>::size;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar& coeffRef(Index index) const {
    return this->m_impl.coeffRef(this->srcCoeff(index));
  }

  // Contiguous store into an unstrided inner run; only instantiated when the
  // nested evaluator has packet access.
  template <int StoreMode, typename Self, bool ImplPacketAccess>
  struct InnerRunWriter {
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE static bool Run(const Self&, Index, const PacketReturnType&) { return false; }
  };

  template <int StoreMode, typename Self>
  struct InnerRunWriter<StoreMode, Self, true> {
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE static bool Run(const Self& self, Index base, const PacketReturnType& x) {
      self.m_impl.template writePacket<Unaligned>(base, x);
      return true;
    }
  };

  template <int StoreMode>
  EIGEN_STRONG_INLINE void writePacket(Index index, const PacketReturnType& x) const {
    typedef TensorEvaluator<TensorStridingOp<Strides, ArgType>, Device> Self;
    constexpr bool ImplPacketAccess = bool(TensorEvaluator<ArgType, Device>::PacketAccess);
    eigen_assert(index + PacketSize - 1 < this->dimensions().TotalSize());

    // Mirrors the rvalue PacketLoader: within one inner-most run the target
    // input indices form an arithmetic progression, so the index mapping is
    // computed once per packet; an unstrided inner dimension becomes a
    // single contiguous store.
    constexpr int inner_dim = (static_cast<int>(Layout) == static_cast<int>(ColMajor)) ? 0 : NumDims - 1;
    Index inner_pos;
    const Index base = this->srcCoeffInner(index, inner_pos);
    if (inner_pos + PacketSize <= this->m_dimensions[inner_dim]) {
      const Index inner_stride = this->m_inputStrides[inner_dim];
      if (inner_stride == 1 && InnerRunWriter<StoreMode, Self, ImplPacketAccess>::Run(*this, base, x)) {
        return;
      }
      EIGEN_ALIGN_TO_BOUNDARY(sizeof(PacketReturnType)) Scalar values[PacketSize];
      internal::pstore<Scalar, PacketReturnType>(values, x);
      EIGEN_UNROLL_LOOP
      for (int i = 0; i < PacketSize; ++i) {
        this->m_impl.coeffRef(base + i * inner_stride) = values[i];
      }
      return;
    }

    // The packet crosses an inner-run boundary: scatter scalar by scalar.
    EIGEN_ALIGN_TO_BOUNDARY(sizeof(PacketReturnType)) Scalar values[PacketSize];
    internal::pstore<Scalar, PacketReturnType>(values, x);
    EIGEN_UNROLL_LOOP
    for (int i = 0; i < PacketSize; ++i) {
      this->coeffRef(index + i) = values[i];
    }
  }
};

}  // end namespace Eigen

#endif  // EIGEN_TENSOR_TENSOR_STRIDING_H
