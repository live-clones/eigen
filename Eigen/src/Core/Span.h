// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_SPAN_H
#define EIGEN_SPAN_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

#if EIGEN_COMP_CXXVER >= 20 && defined(__cpp_lib_span) && __cpp_lib_span >= 202002L
#define EIGEN_HAS_STD_SPAN 1
#include <span>
#endif

namespace Eigen {

namespace internal {

template <typename T, int N, int Align>
struct traits<Span<T, N, Align>> : traits<Vector<std::remove_const_t<T>, N>> {
  using TraitsBase = traits<Vector<std::remove_const_t<T>, N>>;

  static constexpr int InnerStrideAtCompileTime = 1;
  static constexpr int OuterStrideAtCompileTime = N;
  static constexpr int Alignment = Align & int(AlignedMask);
  static constexpr unsigned int Flags0 = TraitsBase::Flags & ~NestByRefBit;
  static constexpr unsigned int Flags = std::is_const<T>::value ? (Flags0 & ~LvalueBit) : Flags0;
};

template <typename T, int N, int Align>
struct evaluator<Span<T, N, Align>> : public mapbase_evaluator<Span<T, N, Align>, Vector<std::remove_const_t<T>, N>> {
  using XprType = Span<T, N, Align>;
  using PlainObjectType = Vector<std::remove_const_t<T>, N>;

  static constexpr unsigned int Flags = evaluator<PlainObjectType>::Flags;
  static constexpr int Alignment = traits<XprType>::Alignment;

  EIGEN_DEVICE_FUNC constexpr explicit evaluator(const XprType& span)
      : mapbase_evaluator<XprType, PlainObjectType>(span) {}
};

}  // namespace internal

/** \class Span
 * \ingroup Core_Module
 *
 * \brief A non-owning view over a contiguous 1-D array of data.
 *
 * \tparam T  Element type.
 * \tparam N  Number of elements at compile time.
 *
 * This class wraps a raw pointer and a length without taking ownership of the
 * underlying storage.
 *
 * Unlike `Map` and `Ref`, `Span` does not materialize buffers passed as an
 * index list to `IndexedView`'s `operator()`:
 *
 * \code
 * int idx[4] = {3, 1, 6, 5};
 * Eigen::Span<int, 4> ispan(idx, 4);
 * auto rows = A(ispan, Eigen::all).eval(); // gather rows 3, 1, 6, 5
 * A(ispan, Eigen::all).noalias() = B;      // scatter-write
 * \endcode
 *
 * When compiled as C++20 and the standard library provides `<span>`, `Span`
 * can additionally be constructed from `std::span`.
 *
 * \sa class `Map`, class `Ref`
 */
template <typename T, int N, int Align>
class Span : public MapBase<Span<T, N, Align>> {
 public:
  using Base = MapBase<Span>;

  EIGEN_DENSE_PUBLIC_INTERFACE(Span)

  using PointerType = typename Base::PointerType;

  // NOTE: `IndexedViewHelper` does not materialize `T` when `T == T::PlainObject`.
  using PlainObject = Span;

  /** Constructor in the fixed-size case.
   *
   * \param data pointer to the array to map
   */
  // Function template so it ranks below any array-lvalue constructor in overload resolution.
  template <int M = N, typename = std::enable_if_t<M != Dynamic>>
  EIGEN_DEVICE_FUNC constexpr explicit Span(PointerType data) : Base(data) {}

  /** Constructor in the dynamic-size case.
   *
   * \param data pointer to the array to map
   * \param size number of elements
   */
  EIGEN_DEVICE_FUNC constexpr Span(PointerType data, Index size) : Base(data, size) {}

#ifdef EIGEN_HAS_STD_SPAN
  /** Constructor from a mutable `std::span` */
  template <std::size_t Extent>
  EIGEN_DEVICE_FUNC constexpr Span(std::span<std::remove_const_t<T>, Extent> s) : Base(s.data(), Index(s.size())) {
    EIGEN_STATIC_ASSERT(N == Dynamic || Extent == std::dynamic_extent || N == Index(Extent),
                        SPAN_STATIC_EXTENT_DOES_NOT_MATCH_STD_SPAN_EXTENT)
  }

  /** Constructor from a constant `std::span` */
  template <std::size_t Extent, typename U = T, typename = std::enable_if_t<std::is_const_v<U>>>
  EIGEN_DEVICE_FUNC constexpr Span(std::span<const std::remove_const_t<T>, Extent> s)
      : Base(s.data(), Index(s.size())) {
    EIGEN_STATIC_ASSERT(N == Dynamic || Extent == std::dynamic_extent || N == Index(Extent),
                        SPAN_STATIC_EXTENT_DOES_NOT_MATCH_STD_SPAN_EXTENT)
  }
#endif  // EIGEN_HAS_STD_SPAN

  EIGEN_INHERIT_ASSIGNMENT_OPERATORS(Span)

  EIGEN_DEVICE_FUNC constexpr Index innerStride() const noexcept { return 1; }

  EIGEN_DEVICE_FUNC constexpr Index outerStride() const noexcept { return this->size(); }
};

}  // namespace Eigen

#endif  // EIGEN_SPAN_H
