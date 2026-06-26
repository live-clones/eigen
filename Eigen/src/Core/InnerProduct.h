// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2024 Charlie Schlosser <cs.schlosser@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_INNER_PRODUCT_EVAL_H
#define EIGEN_INNER_PRODUCT_EVAL_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

// recursively searches for the largest simd type that does not exceed Size, or the smallest if no such type exists
template <typename Scalar, int Size, typename Packet = typename packet_traits<Scalar>::type,
          bool Stop = (unpacket_traits<Packet>::size <= Size) ||
                      std::is_same<Packet, typename unpacket_traits<Packet>::half>::value>
struct find_inner_product_packet_helper;

template <typename Scalar, int Size, typename Packet>
struct find_inner_product_packet_helper<Scalar, Size, Packet, false> {
  using type = typename find_inner_product_packet_helper<Scalar, Size, typename unpacket_traits<Packet>::half>::type;
};

template <typename Scalar, int Size, typename Packet>
struct find_inner_product_packet_helper<Scalar, Size, Packet, true> {
  using type = Packet;
};

template <typename Scalar, int Size>
struct find_inner_product_packet : find_inner_product_packet_helper<Scalar, Size> {};

template <typename Scalar>
struct find_inner_product_packet<Scalar, Dynamic> {
  using type = typename packet_traits<Scalar>::type;
};

template <typename Lhs, typename Rhs>
struct inner_product_assert {
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Lhs)
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Rhs)
  EIGEN_STATIC_ASSERT_SAME_VECTOR_SIZE(Lhs, Rhs)
#ifndef EIGEN_NO_DEBUG
  static EIGEN_DEVICE_FUNC void run(const Lhs& lhs, const Rhs& rhs) {
    eigen_assert((lhs.size() == rhs.size()) && "Inner product: lhs and rhs vectors must have same size");
  }
#else
  static EIGEN_DEVICE_FUNC void run(const Lhs&, const Rhs&) {}
#endif
};

template <typename Func, typename Lhs, typename Rhs>
struct inner_product_evaluator {
  static constexpr int LhsFlags = evaluator<Lhs>::Flags;
  static constexpr int RhsFlags = evaluator<Rhs>::Flags;
  static constexpr int SizeAtCompileTime = size_prefer_fixed(Lhs::SizeAtCompileTime, Rhs::SizeAtCompileTime);
  static constexpr int MaxSizeAtCompileTime =
      min_size_prefer_fixed(Lhs::MaxSizeAtCompileTime, Rhs::MaxSizeAtCompileTime);
  static constexpr int LhsAlignment = evaluator<Lhs>::Alignment;
  static constexpr int RhsAlignment = evaluator<Rhs>::Alignment;

  using Scalar = typename Func::result_type;
  using Packet = typename find_inner_product_packet<Scalar, SizeAtCompileTime>::type;

  static constexpr bool Vectorize =
      bool(LhsFlags & RhsFlags & PacketAccessBit) && Func::PacketAccess &&
      ((MaxSizeAtCompileTime == Dynamic) || (unpacket_traits<Packet>::size <= MaxSizeAtCompileTime));

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE explicit inner_product_evaluator(const Lhs& lhs, const Rhs& rhs,
                                                                         Func func = Func())
      : m_func(func), m_lhs(lhs), m_rhs(rhs), m_size(lhs.size()) {
    inner_product_assert<Lhs, Rhs>::run(lhs, rhs);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index size() const { return m_size.value(); }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar coeff(Index index) const {
    return m_func.coeff(m_lhs.coeff(index), m_rhs.coeff(index));
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar coeff(const Scalar& value, Index index) const {
    return m_func.coeff(value, m_lhs.coeff(index), m_rhs.coeff(index));
  }

  template <typename PacketType, int LhsMode = LhsAlignment, int RhsMode = RhsAlignment>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE PacketType packet(Index index) const {
    return m_func.packet(m_lhs.template packet<LhsMode, PacketType>(index),
                         m_rhs.template packet<RhsMode, PacketType>(index));
  }

  template <typename PacketType, int LhsMode = LhsAlignment, int RhsMode = RhsAlignment>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE PacketType packet(const PacketType& value, Index index) const {
    return m_func.packet(value, m_lhs.template packet<LhsMode, PacketType>(index),
                         m_rhs.template packet<RhsMode, PacketType>(index));
  }

  const Func m_func;
  const evaluator<Lhs> m_lhs;
  const evaluator<Rhs> m_rhs;
  const variable_if_dynamic<Index, SizeAtCompileTime> m_size;
};

template <typename Evaluator, bool Vectorize = Evaluator::Vectorize>
struct inner_product_impl;

// scalar loop
template <typename Evaluator>
struct inner_product_impl<Evaluator, false> {
  using Scalar = typename Evaluator::Scalar;
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar run(const Evaluator& eval) {
    const Index size = eval.size();
    if (size == 0) return Scalar(0);

    Scalar result = eval.coeff(0);
    for (Index k = 1; k < size; k++) {
      result = eval.coeff(result, k);
    }

    return result;
  }
};

// vector loop
template <typename Evaluator>
struct inner_product_impl<Evaluator, true> {
  using UnsignedIndex = std::make_unsigned_t<Index>;
  using Scalar = typename Evaluator::Scalar;
  using Packet = typename Evaluator::Packet;
  static constexpr int PacketSize = unpacket_traits<Packet>::size;
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar run(const Evaluator& eval) {
    const UnsignedIndex size = static_cast<UnsignedIndex>(eval.size());
    if (size < PacketSize) return inner_product_impl<Evaluator, false>::run(eval);

    const UnsignedIndex packetEnd = numext::round_down(size, PacketSize);
    const UnsignedIndex quadEnd = numext::round_down(size, 4 * PacketSize);
    const UnsignedIndex numPackets = size / PacketSize;
    const UnsignedIndex numRemPackets = (packetEnd - quadEnd) / PacketSize;

    Packet presult0 = eval.template packet<Packet>(0 * PacketSize);
    if (numPackets >= 2) {
      Packet presult1 = eval.template packet<Packet>(1 * PacketSize);
      if (numPackets >= 3) {
        Packet presult2 = eval.template packet<Packet>(2 * PacketSize);
        if (numPackets >= 4) {
          Packet presult3 = eval.template packet<Packet>(3 * PacketSize);

          for (UnsignedIndex k = 4 * PacketSize; k < quadEnd; k += 4 * PacketSize) {
            presult0 = eval.packet(presult0, k + 0 * PacketSize);
            presult1 = eval.packet(presult1, k + 1 * PacketSize);
            presult2 = eval.packet(presult2, k + 2 * PacketSize);
            presult3 = eval.packet(presult3, k + 3 * PacketSize);
          }

          if (numRemPackets >= 1) {
            presult0 = eval.packet(presult0, quadEnd + 0 * PacketSize);
            if (numRemPackets >= 2) {
              presult1 = eval.packet(presult1, quadEnd + 1 * PacketSize);
              if (numRemPackets == 3) presult2 = eval.packet(presult2, quadEnd + 2 * PacketSize);
            }
          }

          presult2 = padd(presult2, presult3);
        }
        presult1 = padd(presult1, presult2);
      }
      presult0 = padd(presult0, presult1);
    }

    Scalar result = predux(presult0);
    for (UnsignedIndex k = packetEnd; k < size; k++) {
      result = eval.coeff(result, k);
    }

    return result;
  }
};

template <typename Scalar, bool Conj>
struct conditional_conj;

template <typename Scalar>
struct conditional_conj<Scalar, true> {
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar coeff(const Scalar& a) { return numext::conj(a); }
  template <typename Packet>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packet(const Packet& a) {
    return pconj(a);
  }
};

template <typename Scalar>
struct conditional_conj<Scalar, false> {
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar coeff(const Scalar& a) { return a; }
  template <typename Packet>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packet(const Packet& a) {
    return a;
  }
};

template <typename LhsScalar, typename RhsScalar, bool Conj>
struct scalar_inner_product_op {
  using result_type = typename ScalarBinaryOpTraits<LhsScalar, RhsScalar>::ReturnType;
  using conj_helper = conditional_conj<LhsScalar, Conj>;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE result_type coeff(const LhsScalar& a, const RhsScalar& b) const {
    return (conj_helper::coeff(a) * b);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE result_type coeff(const result_type& accum, const LhsScalar& a,
                                                          const RhsScalar& b) const {
    return (conj_helper::coeff(a) * b) + accum;
  }
  static constexpr bool PacketAccess = false;
};

// Partial specialization for packet access if and only if
// LhsScalar == RhsScalar == ScalarBinaryOpTraits<LhsScalar, RhsScalar>::ReturnType.
template <typename Scalar, bool Conj>
struct scalar_inner_product_op<
    Scalar,
    std::enable_if_t<std::is_same<typename ScalarBinaryOpTraits<Scalar, Scalar>::ReturnType, Scalar>::value, Scalar>,
    Conj> {
  using result_type = Scalar;
  using conj_helper = conditional_conj<Scalar, Conj>;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar coeff(const Scalar& a, const Scalar& b) const {
    return pmul(conj_helper::coeff(a), b);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar coeff(const Scalar& accum, const Scalar& a, const Scalar& b) const {
    return pmadd(conj_helper::coeff(a), b, accum);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packet(const Packet& a, const Packet& b) const {
    return pmul(conj_helper::packet(a), b);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packet(const Packet& accum, const Packet& a, const Packet& b) const {
    return pmadd(conj_helper::packet(a), b, accum);
  }
  static constexpr bool PacketAccess = packet_traits<Scalar>::HasMul && packet_traits<Scalar>::HasAdd;
};

template <typename Lhs, typename Rhs, bool Conj>
struct default_inner_product_impl {
  using LhsScalar = typename traits<Lhs>::Scalar;
  using RhsScalar = typename traits<Rhs>::Scalar;
  using Op = scalar_inner_product_op<LhsScalar, RhsScalar, Conj>;
  using Evaluator = inner_product_evaluator<Op, Lhs, Rhs>;
  using result_type = typename Evaluator::Scalar;
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE result_type run(const MatrixBase<Lhs>& a, const MatrixBase<Rhs>& b) {
    Evaluator eval(a.derived(), b.derived(), Op());
    return inner_product_impl<Evaluator>::run(eval);
  }
};

template <typename T>
struct has_direct_data_access {
  static constexpr bool value = bool(traits<T>::Flags & DirectAccessBit);
};

template <typename T, typename Enable = void>
struct dot_xpr_unwrapper {
  using type = T;
  static constexpr bool direct_access = has_direct_data_access<T>::value;
  static constexpr bool conj_lhs = true;
  static constexpr bool conj_rhs = false;
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE constexpr const T& get(
      const T& xpr) {
    return xpr;
  }
};

template <typename Scalar, typename Xpr>
struct dot_xpr_unwrapper<
    CwiseUnaryOp<internal::scalar_conjugate_op<Scalar>, Xpr>> {
  using type = Xpr;
  static constexpr bool direct_access = has_direct_data_access<Xpr>::value;
  static constexpr bool conj_lhs = false;
  static constexpr bool conj_rhs = true;
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE constexpr const Xpr& get(
      const CwiseUnaryOp<internal::scalar_conjugate_op<Scalar>, Xpr>& xpr) {
    return xpr.nestedExpression();
  }
};

template <typename Scalar, typename Xpr>
struct dot_xpr_unwrapper<
    const CwiseUnaryOp<internal::scalar_conjugate_op<Scalar>, Xpr>> {
  using type = Xpr;
  static constexpr bool direct_access = has_direct_data_access<Xpr>::value;
  static constexpr bool conj_lhs = false;
  static constexpr bool conj_rhs = true;
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE constexpr const Xpr& get(
      const CwiseUnaryOp<internal::scalar_conjugate_op<Scalar>, Xpr>& xpr) {
    return xpr.nestedExpression();
  }
};

template <bool ConjLhs, bool ConjRhs, typename LhsScalar, typename RhsScalar>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE typename ScalarBinaryOpTraits<LhsScalar, RhsScalar>::ReturnType conj_mul(
    const LhsScalar& x, const RhsScalar& y) {
  return conditional_conj<LhsScalar, ConjLhs>::coeff(x) * conditional_conj<RhsScalar, ConjRhs>::coeff(y);
}

// dot_traits abstracts the differences between same-type and mixed-type dot
// products
template <typename LhsScalar, typename RhsScalar, bool ConjLhs, bool ConjRhs,
          int LhsAlignment = Unaligned, int RhsAlignment = Unaligned,
          typename Enable = void>
struct dot_traits {
  using ResultType = typename ScalarBinaryOpTraits<LhsScalar, RhsScalar>::ReturnType;
  static constexpr bool Vectorizable = false;
};

// Specialization for Same-Type (Real/Real or Complex/Complex)
template <typename LhsScalar, typename RhsScalar, bool ConjLhs, bool ConjRhs,
          int LhsAlignment, int RhsAlignment>
struct dot_traits<
    LhsScalar, RhsScalar, ConjLhs, ConjRhs, LhsAlignment, RhsAlignment,
    std::enable_if_t<internal::is_same<LhsScalar, RhsScalar>::value>> {
  using ResultType = LhsScalar;
  using LhsPacket = typename packet_traits<LhsScalar>::type;
  using RhsPacket = LhsPacket;
  using ResPacket = LhsPacket;
  static constexpr bool Vectorizable = packet_traits<LhsScalar>::Vectorizable &&
                                       packet_traits<LhsScalar>::HasAdd &&
                                       packet_traits<LhsScalar>::HasMul;
  static constexpr int PacketSize = unpacket_traits<LhsPacket>::size;
  using ConjHelper = conj_helper<LhsPacket, RhsPacket, ConjLhs, ConjRhs>;

  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE LhsPacket
  load_lhs(const LhsScalar* ptr) {
    return ploadt<LhsPacket, LhsAlignment>(ptr);
  }
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE RhsPacket
  load_rhs(const RhsScalar* ptr) {
    return ploadt<RhsPacket, RhsAlignment>(ptr);
  }
};

// Specialization for Real LHS and Complex RHS (Vectorizable case)
template <typename LhsScalar, typename RhsScalar, bool ConjLhs, bool ConjRhs,
          int LhsAlignment, int RhsAlignment>
struct dot_traits<
    LhsScalar, RhsScalar, ConjLhs, ConjRhs, LhsAlignment, RhsAlignment,
    std::enable_if_t<
        !NumTraits<LhsScalar>::IsComplex && NumTraits<RhsScalar>::IsComplex &&
        packet_traits<RhsScalar>::Vectorizable &&
        packet_traits<RhsScalar>::HasAdd && packet_traits<RhsScalar>::HasMul>> {
  using ResultType = RhsScalar;
  using PacketCplx = typename packet_traits<RhsScalar>::type;
  static constexpr bool Vectorizable = true;
  using PacketReal = typename unpacket_traits<PacketCplx>::as_real;

  using LhsPacket = PacketReal;
  using RhsPacket = PacketCplx;
  using ResPacket = PacketCplx;
  static constexpr int PacketSize = unpacket_traits<PacketCplx>::size;
  using ConjHelper = conj_helper<PacketReal, PacketCplx, false, false>;

  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE LhsPacket
  load_lhs(const LhsScalar* ptr) {
    return ploaddup<PacketReal>(ptr);
  }
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE RhsPacket
  load_rhs(const RhsScalar* ptr) {
    return conditional_conj<RhsScalar, ConjRhs>::template packet<RhsPacket>(
        ploadt<PacketCplx, RhsAlignment>(ptr));
  }
};

// Specialization for Real LHS and Complex RHS (Non-Vectorizable case)
template <typename LhsScalar, typename RhsScalar, bool ConjLhs, bool ConjRhs,
          int LhsAlignment, int RhsAlignment>
struct dot_traits<LhsScalar, RhsScalar, ConjLhs, ConjRhs, LhsAlignment, RhsAlignment,
                  std::enable_if_t<!NumTraits<LhsScalar>::IsComplex &&
                                   NumTraits<RhsScalar>::IsComplex &&
                                   !(packet_traits<RhsScalar>::Vectorizable &&
                                     packet_traits<RhsScalar>::HasAdd &&
                                     packet_traits<RhsScalar>::HasMul)>> {
  using ResultType = RhsScalar;
  static constexpr bool Vectorizable = false;
};

// Specialization for Complex LHS and Real RHS (Vectorizable case)
template <typename LhsScalar, typename RhsScalar, bool ConjLhs, bool ConjRhs,
          int LhsAlignment, int RhsAlignment>
struct dot_traits<
    LhsScalar, RhsScalar, ConjLhs, ConjRhs, LhsAlignment, RhsAlignment,
    std::enable_if_t<
        NumTraits<LhsScalar>::IsComplex && !NumTraits<RhsScalar>::IsComplex &&
        packet_traits<LhsScalar>::Vectorizable &&
        packet_traits<LhsScalar>::HasAdd && packet_traits<LhsScalar>::HasMul>> {
  using ResultType = LhsScalar;
  using PacketCplx = typename packet_traits<LhsScalar>::type;
  static constexpr bool Vectorizable = true;
  using PacketReal = typename unpacket_traits<PacketCplx>::as_real;

  using LhsPacket = PacketCplx;
  using RhsPacket = PacketReal;
  using ResPacket = PacketCplx;
  static constexpr int PacketSize = unpacket_traits<PacketCplx>::size;
  using ConjHelper = conj_helper<PacketCplx, PacketReal, false, false>;

  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE LhsPacket
  load_lhs(const LhsScalar* ptr) {
    return conditional_conj<LhsScalar, ConjLhs>::template packet<LhsPacket>(
        ploadt<PacketCplx, LhsAlignment>(ptr));
  }
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE RhsPacket
  load_rhs(const RhsScalar* ptr) {
    return ploaddup<PacketReal>(ptr);
  }
};

// Specialization for Complex LHS and Real RHS (Non-Vectorizable case)
template <typename LhsScalar, typename RhsScalar, bool ConjLhs, bool ConjRhs,
          int LhsAlignment, int RhsAlignment>
struct dot_traits<LhsScalar, RhsScalar, ConjLhs, ConjRhs, LhsAlignment, RhsAlignment,
                  std::enable_if_t<NumTraits<LhsScalar>::IsComplex &&
                                   !NumTraits<RhsScalar>::IsComplex &&
                                   !(packet_traits<LhsScalar>::Vectorizable &&
                                     packet_traits<LhsScalar>::HasAdd &&
                                     packet_traits<LhsScalar>::HasMul)>> {
  using ResultType = LhsScalar;
  static constexpr bool Vectorizable = false;
};

template <typename LhsScalar, typename RhsScalar, bool ConjLhs, bool ConjRhs,
          int LhsAlignment, int RhsAlignment, bool Vectorizable>
struct generic_dot_impl_helper_impl;

template <typename LhsScalar, typename RhsScalar, bool ConjLhs, bool ConjRhs,
          int LhsAlignment, int RhsAlignment>
struct generic_dot_impl_helper_impl<LhsScalar, RhsScalar, ConjLhs, ConjRhs, LhsAlignment, RhsAlignment, true> {
  using Traits = dot_traits<LhsScalar, RhsScalar, ConjLhs, ConjRhs, LhsAlignment, RhsAlignment>;
  using ResultType = typename Traits::ResultType;

  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE ResultType
  run(const LhsScalar* lhs, const RhsScalar* rhs, Index size) {
    using ResPacket = typename Traits::ResPacket;
    constexpr int PacketSize = Traits::PacketSize;

    if (size < PacketSize) {
      ResultType res(0);
      for (Index i = 0; i < size; ++i) {
        res += conj_mul<ConjLhs, ConjRhs>(lhs[i], rhs[i]);
      }
      return res;
    }

    Index numPackets = size / PacketSize;
    Index quadPackets = numPackets / 4;
    Index remPackets = numPackets % 4;

    typename Traits::ConjHelper cj;
    ResPacket acc0, acc1, acc2, acc3;
    acc0 = cj.pmul(Traits::load_lhs(lhs + 0 * PacketSize),
                   Traits::load_rhs(rhs + 0 * PacketSize));

    if (numPackets >= 2) {
      acc1 = cj.pmul(Traits::load_lhs(lhs + 1 * PacketSize),
                     Traits::load_rhs(rhs + 1 * PacketSize));
    }
    if (numPackets >= 3) {
      acc2 = cj.pmul(Traits::load_lhs(lhs + 2 * PacketSize),
                     Traits::load_rhs(rhs + 2 * PacketSize));
    }
    if (numPackets >= 4) {
      acc3 = cj.pmul(Traits::load_lhs(lhs + 3 * PacketSize),
                     Traits::load_rhs(rhs + 3 * PacketSize));

      Index i = 4 * PacketSize;
      const Index limit = quadPackets * 4 * PacketSize;
      for (; i < limit; i += 4 * PacketSize) {
        acc0 = cj.pmadd(Traits::load_lhs(lhs + i + 0 * PacketSize),
                        Traits::load_rhs(rhs + i + 0 * PacketSize), acc0);
        acc1 = cj.pmadd(Traits::load_lhs(lhs + i + 1 * PacketSize),
                        Traits::load_rhs(rhs + i + 1 * PacketSize), acc1);
        acc2 = cj.pmadd(Traits::load_lhs(lhs + i + 2 * PacketSize),
                        Traits::load_rhs(rhs + i + 2 * PacketSize), acc2);
        acc3 = cj.pmadd(Traits::load_lhs(lhs + i + 3 * PacketSize),
                        Traits::load_rhs(rhs + i + 3 * PacketSize), acc3);
      }

      if (remPackets >= 1) {
        acc0 = cj.pmadd(Traits::load_lhs(lhs + i + 0 * PacketSize),
                        Traits::load_rhs(rhs + i + 0 * PacketSize), acc0);
      }
      if (remPackets >= 2) {
        acc1 = cj.pmadd(Traits::load_lhs(lhs + i + 1 * PacketSize),
                        Traits::load_rhs(rhs + i + 1 * PacketSize), acc1);
      }
      if (remPackets >= 3) {
        acc2 = cj.pmadd(Traits::load_lhs(lhs + i + 2 * PacketSize),
                        Traits::load_rhs(rhs + i + 2 * PacketSize), acc2);
      }

      acc2 = padd(acc2, acc3);
    }

    if (numPackets >= 3) acc1 = padd(acc1, acc2);
    if (numPackets >= 2) acc0 = padd(acc0, acc1);

    ResultType res = predux(acc0);
    Index packet_end = numPackets * PacketSize;
    for (Index i = packet_end; i < size; ++i) {
      res += conj_mul<ConjLhs, ConjRhs>(lhs[i], rhs[i]);
    }
    return res;
  }
};

template <typename LhsScalar, typename RhsScalar, bool ConjLhs, bool ConjRhs,
          int LhsAlignment, int RhsAlignment>
struct generic_dot_impl_helper_impl<LhsScalar, RhsScalar, ConjLhs, ConjRhs, LhsAlignment, RhsAlignment, false> {
  using Traits = dot_traits<LhsScalar, RhsScalar, ConjLhs, ConjRhs, LhsAlignment, RhsAlignment>;
  using ResultType = typename Traits::ResultType;

  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE ResultType
  run(const LhsScalar* lhs, const RhsScalar* rhs, Index size) {
    ResultType res(0);
    for (Index i = 0; i < size; ++i) {
      res += conj_mul<ConjLhs, ConjRhs>(lhs[i], rhs[i]);
    }
    return res;
  }
};

// Unified generic_dot_impl_helper
template <typename LhsScalar, typename RhsScalar, bool ConjLhs, bool ConjRhs,
          int LhsAlignment = Unaligned, int RhsAlignment = Unaligned,
          typename Enable = void>
struct generic_dot_impl_helper {
  using Traits = dot_traits<LhsScalar, RhsScalar, ConjLhs, ConjRhs, LhsAlignment, RhsAlignment>;
  using ResultType = typename Traits::ResultType;

  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE ResultType
  run(const LhsScalar* lhs, const RhsScalar* rhs, Index size) {
    return generic_dot_impl_helper_impl<LhsScalar, RhsScalar, ConjLhs, ConjRhs, LhsAlignment, RhsAlignment, Traits::Vectorizable>::run(lhs, rhs, size);
  }
};

template <typename Lhs, typename Rhs, bool Vectorize>
struct dot_impl_helper {
  using LhsScalar = typename traits<Lhs>::Scalar;
  using RhsScalar = typename traits<Rhs>::Scalar;
  using ResultType =
      typename ScalarBinaryOpTraits<LhsScalar, RhsScalar>::ReturnType;

  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE ResultType
  run(const MatrixBase<Lhs>& a, const MatrixBase<Rhs>& b) {
    return default_inner_product_impl<Lhs, Rhs, true>::run(a, b);
  }
};

template <typename Lhs, typename Rhs>
struct dot_impl_helper<Lhs, Rhs, true> {
  using LhsScalar = typename traits<Lhs>::Scalar;
  using RhsScalar = typename traits<Rhs>::Scalar;
  using ResultType =
      typename ScalarBinaryOpTraits<LhsScalar, RhsScalar>::ReturnType;

  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE ResultType
  run(const MatrixBase<Lhs>& a, const MatrixBase<Rhs>& b) {
    const auto& lhs_clean = dot_xpr_unwrapper<Lhs>::get(a.derived());
    const auto& rhs_clean = dot_xpr_unwrapper<Rhs>::get(b.derived());
    if (lhs_clean.innerStride() == 1 && rhs_clean.innerStride() == 1 &&
        (bool(Lhs::IsVectorAtCompileTime) || lhs_clean.rows() == 1 ||
         lhs_clean.cols() == 1) &&
        (bool(Rhs::IsVectorAtCompileTime) || rhs_clean.rows() == 1 ||
         rhs_clean.cols() == 1)) {
      using LhsClean = typename dot_xpr_unwrapper<Lhs>::type;
      using RhsClean = typename dot_xpr_unwrapper<Rhs>::type;
      constexpr int LhsAlignment = evaluator<LhsClean>::Alignment;
      constexpr int RhsAlignment = evaluator<RhsClean>::Alignment;
      return generic_dot_impl_helper<
          LhsScalar, RhsScalar, dot_xpr_unwrapper<Lhs>::conj_lhs,
          dot_xpr_unwrapper<Rhs>::conj_rhs, LhsAlignment, RhsAlignment>::run(
          lhs_clean.data(), rhs_clean.data(), lhs_clean.size());
    }
    return default_inner_product_impl<Lhs, Rhs, true>::run(a, b);
  }
};

template <typename Lhs, typename Rhs>
struct dot_impl {
  using LhsScalar = typename traits<Lhs>::Scalar;
  using RhsScalar = typename traits<Rhs>::Scalar;
  using ResultType =
      typename ScalarBinaryOpTraits<LhsScalar, RhsScalar>::ReturnType;

  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE ResultType
  run(const MatrixBase<Lhs>& a, const MatrixBase<Rhs>& b) {
    constexpr bool is_mixed = !internal::is_same<LhsScalar, RhsScalar>::value;
    constexpr bool is_vectorizable =
        dot_traits<LhsScalar, RhsScalar, dot_xpr_unwrapper<Lhs>::conj_lhs,
                   dot_xpr_unwrapper<Rhs>::conj_rhs>::Vectorizable;
    constexpr bool cond = dot_xpr_unwrapper<Lhs>::direct_access &&
                          dot_xpr_unwrapper<Rhs>::direct_access &&
                          is_vectorizable &&
                          (inner_stride_at_compile_time<
                               typename dot_xpr_unwrapper<Lhs>::type>::value != 1 ||
                           is_mixed);
    return dot_impl_helper<Lhs, Rhs, cond>::run(a, b);
  }
};

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_INNER_PRODUCT_EVAL_H
