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

// traits<Span<T,N>>
//
// Inheriting from traits<Matrix<Scalar,N,1>> gives XprKind = MatrixXpr so
// Span participates in the same expression hierarchy as VectorXf/VectorXd.
//
// The PlainObject entry here affects parts of Eigen that look it up via
// traits (e.g. nested_eval).  The critical override that prevents
// is_eigen_index_expression from firing is the PlainObject typedef on the
// Span class body itself — see DenseBase.h:203 and IndexedViewHelper.h:131.
template <typename T, int N>
struct traits<Span<T, N>> : traits<Matrix<std::remove_const_t<T>, N, 1>> {
  using TraitsBase = traits<Matrix<std::remove_const_t<T>, N, 1>>;
  using PlainObject = Span<T, N>;

  static constexpr int InnerStrideAtCompileTime = 1;
  static constexpr int OuterStrideAtCompileTime = N;
  static constexpr int Alignment = 0;
  static constexpr unsigned int Flags0 = TraitsBase::Flags & ~NestByRefBit;
  static constexpr unsigned int Flags = std::is_const<T>::value ? (Flags0 & ~LvalueBit) : Flags0;
};

// evaluator<Span<T,N>> — delegates to mapbase_evaluator just like Map and Ref.
template <typename T, int N>
struct evaluator<Span<T, N>> : public mapbase_evaluator<Span<T, N>, Matrix<std::remove_const_t<T>, N, 1>> {
  using XprType = Span<T, N>;
  using PlainObjectType = Matrix<std::remove_const_t<T>, N, 1>;

  static constexpr unsigned int Flags = evaluator<PlainObjectType>::Flags;
  static constexpr int Alignment = 0;

  EIGEN_DEVICE_FUNC constexpr explicit evaluator(const XprType& span)
      : mapbase_evaluator<XprType, PlainObjectType>(span) {}
};

}  // namespace internal

/** \class Span
 * \ingroup Core_Module
 *
 * \brief A non-owning view over a contiguous 1-D array of data.
 *
 * \tparam T  Element type.  Use \c const T for a read-only span.
 * \tparam N  Number of elements at compile time, or \c Dynamic (the default)
 *            for a runtime-sized span.
 *
 * This class wraps a raw pointer and a length without taking ownership of the
 * underlying storage.  It is a fully-fledged Eigen expression: assignments,
 * arithmetic, and SIMD-vectorised operations all work out of the box:
 * \code
 * float buf[8];
 * Eigen::Span<float, 8> s(buf);
 * s = Eigen::VectorXf::Ones(8);  // write into buf
 * VectorXf v = s + s;            // read from buf
 * \endcode
 *
 * Unlike Map and Ref, Span does not materialise (copy) a buffer passed as an
 * index list to IndexedView's \c operator():
 * \code
 * int idx[4] = {3, 1, 6, 5};
 * Eigen::Span<int, 4> ispan(idx, 4);
 * auto rows = A(ispan, Eigen::all);  // gather rows 3, 1, 6, 5
 * A(ispan, Eigen::all) = B;         // scatter-write
 * \endcode
 *
 * When compiled as C++20 and the standard library provides \c <span>, Span
 * can additionally be constructed from \c std::span.
 *
 * \sa class Map, class Ref
 */
template <typename T, int N>
class Span : public MapBase<Span<T, N>> {
 public:
  using Base = MapBase<Span<T, N>>;
  EIGEN_DENSE_PUBLIC_INTERFACE(Span)

  using PointerType = typename Base::PointerType;

  // Re-declare PlainObject to point back at Span<T,N>.
  // DenseBase<Span> defines PlainObject as Matrix<Scalar,N,1> (see DenseBase.h:203).
  // is_eigen_index_expression (IndexedViewHelper.h:131) checks T::PlainObject
  // on the class, not traits<T>::PlainObject.  By shadowing DenseBase::PlainObject
  // here we make is_eigen_index_expression<Span>::value == false, so Span passed
  // as an IndexedView index is never copied or eval()-ed.
  using PlainObject = Span<T, N>;

  EIGEN_DEVICE_FUNC constexpr Index innerStride() const noexcept { return 1; }

  EIGEN_DEVICE_FUNC constexpr Index outerStride() const noexcept { return this->size(); }

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
  /** Constructor from a mutable std::span (C++20). */
  template <std::size_t Extent>
  EIGEN_DEVICE_FUNC constexpr Span(std::span<std::remove_const_t<T>, Extent> s) : Base(s.data(), Index(s.size())) {
    EIGEN_STATIC_ASSERT(N == Dynamic || Extent == std::dynamic_extent || N == Index(Extent),
                        SPAN_STATIC_EXTENT_DOES_NOT_MATCH_STD_SPAN_EXTENT)
  }

  /** Constructor from a const std::span (C++20). */
  template <std::size_t Extent, typename U = T, typename = std::enable_if_t<std::is_const_v<U>>>
  EIGEN_DEVICE_FUNC constexpr Span(std::span<const std::remove_const_t<T>, Extent> s)
      : Base(s.data(), Index(s.size())) {
    EIGEN_STATIC_ASSERT(N == Dynamic || Extent == std::dynamic_extent || N == Index(Extent),
                        SPAN_STATIC_EXTENT_DOES_NOT_MATCH_STD_SPAN_EXTENT)
  }
#endif  // EIGEN_HAS_STD_SPAN

  EIGEN_INHERIT_ASSIGNMENT_OPERATORS(Span)
};

}  // namespace Eigen

#endif  // EIGEN_SPAN_H
