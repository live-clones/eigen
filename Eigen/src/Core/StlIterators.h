// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2018 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_STLITERATORS_H
#define EIGEN_STLITERATORS_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

template <typename IteratorType>
struct indexed_based_stl_iterator_traits;

template <typename Derived>
class indexed_based_stl_iterator_base {
 protected:
  using traits = indexed_based_stl_iterator_traits<Derived>;
  using XprType = typename traits::XprType;
  using non_const_iterator = indexed_based_stl_iterator_base<typename traits::non_const_iterator>;
  using const_iterator = indexed_based_stl_iterator_base<typename traits::const_iterator>;
  using other_iterator = std::conditional_t<std::is_const<XprType>::value, non_const_iterator, const_iterator>;

  friend class indexed_based_stl_iterator_base<typename traits::const_iterator>;
  friend class indexed_based_stl_iterator_base<typename traits::non_const_iterator>;

 public:
  using difference_type = Index;
  using iterator_category = std::random_access_iterator_tag;

  constexpr indexed_based_stl_iterator_base() noexcept = default;
  constexpr indexed_based_stl_iterator_base(XprType& xpr, Index index) noexcept : mp_xpr(&xpr), m_index(index) {}

  constexpr indexed_based_stl_iterator_base(const non_const_iterator& other) noexcept
      : mp_xpr(other.mp_xpr), m_index(other.m_index) {}

  constexpr indexed_based_stl_iterator_base& operator=(const non_const_iterator& other) {
    mp_xpr = other.mp_xpr;
    m_index = other.m_index;
    return *this;
  }

  constexpr Derived& operator++() {
    ++m_index;
    return derived();
  }
  constexpr Derived& operator--() {
    --m_index;
    return derived();
  }

  constexpr Derived operator++(int) {
    Derived prev(derived());
    operator++();
    return prev;
  }
  constexpr Derived operator--(int) {
    Derived prev(derived());
    operator--();
    return prev;
  }

  friend constexpr Derived operator+(const indexed_based_stl_iterator_base& a, Index b) {
    Derived ret(a.derived());
    ret += b;
    return ret;
  }
  friend constexpr Derived operator-(const indexed_based_stl_iterator_base& a, Index b) {
    Derived ret(a.derived());
    ret -= b;
    return ret;
  }
  friend constexpr Derived operator+(Index a, const indexed_based_stl_iterator_base& b) {
    Derived ret(b.derived());
    ret += a;
    return ret;
  }
  friend constexpr Derived operator-(Index a, const indexed_based_stl_iterator_base& b) {
    Derived ret(b.derived());
    ret -= a;
    return ret;
  }

  constexpr Derived& operator+=(Index b) {
    m_index += b;
    return derived();
  }
  constexpr Derived& operator-=(Index b) {
    m_index -= b;
    return derived();
  }

  constexpr difference_type operator-(const indexed_based_stl_iterator_base& other) const {
    eigen_assert(mp_xpr == other.mp_xpr);
    return m_index - other.m_index;
  }

  constexpr difference_type operator-(const other_iterator& other) const {
    eigen_assert(mp_xpr == other.mp_xpr);
    return m_index - other.m_index;
  }

  constexpr bool operator==(const indexed_based_stl_iterator_base& other) const {
    eigen_assert(mp_xpr == other.mp_xpr);
    return m_index == other.m_index;
  }
  constexpr bool operator!=(const indexed_based_stl_iterator_base& other) const {
    eigen_assert(mp_xpr == other.mp_xpr);
    return m_index != other.m_index;
  }
  constexpr bool operator<(const indexed_based_stl_iterator_base& other) const {
    eigen_assert(mp_xpr == other.mp_xpr);
    return m_index < other.m_index;
  }
  constexpr bool operator<=(const indexed_based_stl_iterator_base& other) const {
    eigen_assert(mp_xpr == other.mp_xpr);
    return m_index <= other.m_index;
  }
  constexpr bool operator>(const indexed_based_stl_iterator_base& other) const {
    eigen_assert(mp_xpr == other.mp_xpr);
    return m_index > other.m_index;
  }
  constexpr bool operator>=(const indexed_based_stl_iterator_base& other) const {
    eigen_assert(mp_xpr == other.mp_xpr);
    return m_index >= other.m_index;
  }

  constexpr bool operator==(const other_iterator& other) const {
    eigen_assert(mp_xpr == other.mp_xpr);
    return m_index == other.m_index;
  }
  constexpr bool operator!=(const other_iterator& other) const {
    eigen_assert(mp_xpr == other.mp_xpr);
    return m_index != other.m_index;
  }
  constexpr bool operator<(const other_iterator& other) const {
    eigen_assert(mp_xpr == other.mp_xpr);
    return m_index < other.m_index;
  }
  constexpr bool operator<=(const other_iterator& other) const {
    eigen_assert(mp_xpr == other.mp_xpr);
    return m_index <= other.m_index;
  }
  constexpr bool operator>(const other_iterator& other) const {
    eigen_assert(mp_xpr == other.mp_xpr);
    return m_index > other.m_index;
  }
  constexpr bool operator>=(const other_iterator& other) const {
    eigen_assert(mp_xpr == other.mp_xpr);
    return m_index >= other.m_index;
  }

 protected:
  constexpr Derived& derived() { return static_cast<Derived&>(*this); }
  constexpr const Derived& derived() const { return static_cast<const Derived&>(*this); }

  XprType* mp_xpr = nullptr;
  Index m_index = 0;
};

template <typename Derived>
class indexed_based_stl_reverse_iterator_base {
 protected:
  using traits = indexed_based_stl_iterator_traits<Derived>;
  using XprType = typename traits::XprType;
  using non_const_iterator = indexed_based_stl_reverse_iterator_base<typename traits::non_const_iterator>;
  using const_iterator = indexed_based_stl_reverse_iterator_base<typename traits::const_iterator>;
  using other_iterator = std::conditional_t<std::is_const<XprType>::value, non_const_iterator, const_iterator>;

  friend class indexed_based_stl_reverse_iterator_base<typename traits::const_iterator>;
  friend class indexed_based_stl_reverse_iterator_base<typename traits::non_const_iterator>;

 public:
  using difference_type = Index;
  using iterator_category = std::random_access_iterator_tag;

  constexpr indexed_based_stl_reverse_iterator_base() = default;
  constexpr indexed_based_stl_reverse_iterator_base(XprType& xpr, Index index) : mp_xpr(&xpr), m_index(index) {}

  constexpr indexed_based_stl_reverse_iterator_base(const non_const_iterator& other)
      : mp_xpr(other.mp_xpr), m_index(other.m_index) {}

  constexpr indexed_based_stl_reverse_iterator_base& operator=(const non_const_iterator& other) {
    mp_xpr = other.mp_xpr;
    m_index = other.m_index;
    return *this;
  }

  constexpr Derived& operator++() {
    --m_index;
    return derived();
  }
  constexpr Derived& operator--() {
    ++m_index;
    return derived();
  }

  constexpr Derived operator++(int) {
    Derived prev(derived());
    operator++();
    return prev;
  }
  constexpr Derived operator--(int) {
    Derived prev(derived());
    operator--();
    return prev;
  }

  friend constexpr Derived operator+(const indexed_based_stl_reverse_iterator_base& a, Index b) {
    Derived ret(a.derived());
    ret += b;
    return ret;
  }
  friend constexpr Derived operator-(const indexed_based_stl_reverse_iterator_base& a, Index b) {
    Derived ret(a.derived());
    ret -= b;
    return ret;
  }
  friend constexpr Derived operator+(Index a, const indexed_based_stl_reverse_iterator_base& b) {
    Derived ret(b.derived());
    ret += a;
    return ret;
  }
  friend constexpr Derived operator-(Index a, const indexed_based_stl_reverse_iterator_base& b) {
    Derived ret(b.derived());
    ret -= a;
    return ret;
  }

  constexpr Derived& operator+=(Index b) {
    m_index -= b;
    return derived();
  }
  constexpr Derived& operator-=(Index b) {
    m_index += b;
    return derived();
  }

  constexpr difference_type operator-(const indexed_based_stl_reverse_iterator_base& other) const {
    eigen_assert(mp_xpr == other.mp_xpr);
    return other.m_index - m_index;
  }

  constexpr difference_type operator-(const other_iterator& other) const {
    eigen_assert(mp_xpr == other.mp_xpr);
    return other.m_index - m_index;
  }

  constexpr bool operator==(const indexed_based_stl_reverse_iterator_base& other) const {
    eigen_assert(mp_xpr == other.mp_xpr);
    return m_index == other.m_index;
  }
  constexpr bool operator!=(const indexed_based_stl_reverse_iterator_base& other) const {
    eigen_assert(mp_xpr == other.mp_xpr);
    return m_index != other.m_index;
  }
  constexpr bool operator<(const indexed_based_stl_reverse_iterator_base& other) const {
    eigen_assert(mp_xpr == other.mp_xpr);
    return m_index > other.m_index;
  }
  constexpr bool operator<=(const indexed_based_stl_reverse_iterator_base& other) const {
    eigen_assert(mp_xpr == other.mp_xpr);
    return m_index >= other.m_index;
  }
  constexpr bool operator>(const indexed_based_stl_reverse_iterator_base& other) const {
    eigen_assert(mp_xpr == other.mp_xpr);
    return m_index < other.m_index;
  }
  constexpr bool operator>=(const indexed_based_stl_reverse_iterator_base& other) const {
    eigen_assert(mp_xpr == other.mp_xpr);
    return m_index <= other.m_index;
  }

  constexpr bool operator==(const other_iterator& other) const {
    eigen_assert(mp_xpr == other.mp_xpr);
    return m_index == other.m_index;
  }
  constexpr bool operator!=(const other_iterator& other) const {
    eigen_assert(mp_xpr == other.mp_xpr);
    return m_index != other.m_index;
  }
  constexpr bool operator<(const other_iterator& other) const {
    eigen_assert(mp_xpr == other.mp_xpr);
    return m_index > other.m_index;
  }
  constexpr bool operator<=(const other_iterator& other) const {
    eigen_assert(mp_xpr == other.mp_xpr);
    return m_index >= other.m_index;
  }
  constexpr bool operator>(const other_iterator& other) const {
    eigen_assert(mp_xpr == other.mp_xpr);
    return m_index < other.m_index;
  }
  constexpr bool operator>=(const other_iterator& other) const {
    eigen_assert(mp_xpr == other.mp_xpr);
    return m_index <= other.m_index;
  }

 protected:
  constexpr Derived& derived() { return static_cast<Derived&>(*this); }
  constexpr const Derived& derived() const { return static_cast<const Derived&>(*this); }

  XprType* mp_xpr = nullptr;
  Index m_index = 0;
};

template <typename XprType>
class pointer_based_stl_iterator {
  enum { is_lvalue = internal::is_lvalue<XprType>::value };
  using non_const_iterator = pointer_based_stl_iterator<std::remove_const_t<XprType>>;
  using const_iterator = pointer_based_stl_iterator<std::add_const_t<XprType>>;
  using other_iterator = std::conditional_t<std::is_const<XprType>::value, non_const_iterator, const_iterator>;

  friend class pointer_based_stl_iterator<std::add_const_t<XprType>>;
  friend class pointer_based_stl_iterator<std::remove_const_t<XprType>>;

 public:
  using difference_type = Index;
  using value_type = typename XprType::Scalar;
#if EIGEN_COMP_CXXVER >= 20 && defined(__cpp_lib_concepts) && __cpp_lib_concepts >= 202002L
  using iterator_category = std::conditional_t<XprType::InnerStrideAtCompileTime == 1, std::contiguous_iterator_tag,
                                               std::random_access_iterator_tag>;
#else
  using iterator_category = std::random_access_iterator_tag;
#endif
  using pointer = std::conditional_t<bool(is_lvalue), value_type*, const value_type*>;
  using reference = std::conditional_t<bool(is_lvalue), value_type&, const value_type&>;

  constexpr pointer_based_stl_iterator() noexcept = default;
  constexpr pointer_based_stl_iterator(XprType& xpr, Index index) noexcept
      : m_ptr(xpr.data() + index * xpr.innerStride()), m_incr(xpr.innerStride()) {}

  constexpr pointer_based_stl_iterator(const non_const_iterator& other) noexcept
      : m_ptr(other.m_ptr), m_incr(other.m_incr) {}

  constexpr pointer_based_stl_iterator& operator=(const non_const_iterator& other) noexcept {
    m_ptr = other.m_ptr;
    m_incr.setValue(other.m_incr);
    return *this;
  }

  constexpr reference operator*() const { return *m_ptr; }
  constexpr reference operator[](Index i) const { return *(m_ptr + i * m_incr.value()); }
  constexpr pointer operator->() const { return m_ptr; }

  constexpr pointer_based_stl_iterator& operator++() {
    m_ptr += m_incr.value();
    return *this;
  }
  constexpr pointer_based_stl_iterator& operator--() {
    m_ptr -= m_incr.value();
    return *this;
  }

  constexpr pointer_based_stl_iterator operator++(int) {
    pointer_based_stl_iterator prev(*this);
    operator++();
    return prev;
  }
  constexpr pointer_based_stl_iterator operator--(int) {
    pointer_based_stl_iterator prev(*this);
    operator--();
    return prev;
  }

  friend constexpr pointer_based_stl_iterator operator+(const pointer_based_stl_iterator& a, Index b) {
    pointer_based_stl_iterator ret(a);
    ret += b;
    return ret;
  }
  friend constexpr pointer_based_stl_iterator operator-(const pointer_based_stl_iterator& a, Index b) {
    pointer_based_stl_iterator ret(a);
    ret -= b;
    return ret;
  }
  friend constexpr pointer_based_stl_iterator operator+(Index a, const pointer_based_stl_iterator& b) {
    pointer_based_stl_iterator ret(b);
    ret += a;
    return ret;
  }
  friend constexpr pointer_based_stl_iterator operator-(Index a, const pointer_based_stl_iterator& b) {
    pointer_based_stl_iterator ret(b);
    ret -= a;
    return ret;
  }

  constexpr pointer_based_stl_iterator& operator+=(Index b) {
    m_ptr += b * m_incr.value();
    return *this;
  }
  constexpr pointer_based_stl_iterator& operator-=(Index b) {
    m_ptr -= b * m_incr.value();
    return *this;
  }

  constexpr difference_type operator-(const pointer_based_stl_iterator& other) const {
    return (m_ptr - other.m_ptr) / m_incr.value();
  }

  constexpr difference_type operator-(const other_iterator& other) const {
    return (m_ptr - other.m_ptr) / m_incr.value();
  }

  constexpr bool operator==(const pointer_based_stl_iterator& other) const { return m_ptr == other.m_ptr; }
  constexpr bool operator!=(const pointer_based_stl_iterator& other) const { return m_ptr != other.m_ptr; }
  constexpr bool operator<(const pointer_based_stl_iterator& other) const { return m_ptr < other.m_ptr; }
  constexpr bool operator<=(const pointer_based_stl_iterator& other) const { return m_ptr <= other.m_ptr; }
  constexpr bool operator>(const pointer_based_stl_iterator& other) const { return m_ptr > other.m_ptr; }
  constexpr bool operator>=(const pointer_based_stl_iterator& other) const { return m_ptr >= other.m_ptr; }

  constexpr bool operator==(const other_iterator& other) const { return m_ptr == other.m_ptr; }
  constexpr bool operator!=(const other_iterator& other) const { return m_ptr != other.m_ptr; }
  constexpr bool operator<(const other_iterator& other) const { return m_ptr < other.m_ptr; }
  constexpr bool operator<=(const other_iterator& other) const { return m_ptr <= other.m_ptr; }
  constexpr bool operator>(const other_iterator& other) const { return m_ptr > other.m_ptr; }
  constexpr bool operator>=(const other_iterator& other) const { return m_ptr >= other.m_ptr; }

 protected:
  pointer m_ptr = nullptr;
  internal::variable_if_dynamic<Index, XprType::InnerStrideAtCompileTime> m_incr{XprType::InnerStrideAtCompileTime};
};

template <typename XprType_>
struct indexed_based_stl_iterator_traits<generic_randaccess_stl_iterator<XprType_>> {
  using XprType = XprType_;
  using non_const_iterator = generic_randaccess_stl_iterator<std::remove_const_t<XprType>>;
  using const_iterator = generic_randaccess_stl_iterator<std::add_const_t<XprType>>;
};

template <typename XprType>
class generic_randaccess_stl_iterator
    : public indexed_based_stl_iterator_base<generic_randaccess_stl_iterator<XprType>> {
 public:
  using value_type = typename XprType::Scalar;

 protected:
  enum {
    has_direct_access = (internal::traits<XprType>::Flags & DirectAccessBit) ? 1 : 0,
    is_lvalue = internal::is_lvalue<XprType>::value
  };

  using Base = indexed_based_stl_iterator_base<generic_randaccess_stl_iterator>;
  using Base::m_index;
  using Base::mp_xpr;

  // TODO: currently const Transpose/Reshape expressions never returns const references,
  // so lets return by value too.
  // typedef std::conditional_t<bool(has_direct_access), const value_type&, const value_type> read_only_ref_t;
  using read_only_ref_t = const value_type;

 public:
  using pointer = std::conditional_t<bool(is_lvalue), value_type*, const value_type*>;
  using reference = std::conditional_t<bool(is_lvalue), value_type&, read_only_ref_t>;

  constexpr generic_randaccess_stl_iterator() = default;
  constexpr generic_randaccess_stl_iterator(XprType& xpr, Index index) : Base(xpr, index) {}
  constexpr generic_randaccess_stl_iterator(const typename Base::non_const_iterator& other) : Base(other) {}
  using Base::operator=;

  constexpr reference operator*() const { return (*mp_xpr)(m_index); }
  constexpr reference operator[](Index i) const { return (*mp_xpr)(m_index + i); }
  constexpr pointer operator->() const { return &((*mp_xpr)(m_index)); }
};

template <typename XprType_, DirectionType Direction>
struct indexed_based_stl_iterator_traits<subvector_stl_iterator<XprType_, Direction>> {
  using XprType = XprType_;
  using non_const_iterator = subvector_stl_iterator<std::remove_const_t<XprType>, Direction>;
  using const_iterator = subvector_stl_iterator<std::add_const_t<XprType>, Direction>;
};

template <typename XprType, DirectionType Direction>
class subvector_stl_iterator : public indexed_based_stl_iterator_base<subvector_stl_iterator<XprType, Direction>> {
 protected:
  enum { is_lvalue = internal::is_lvalue<XprType>::value };

  using Base = indexed_based_stl_iterator_base<subvector_stl_iterator>;
  using Base::m_index;
  using Base::mp_xpr;

  using SubVectorType = std::conditional_t<Direction == Vertical, typename XprType::ColXpr, typename XprType::RowXpr>;
  using ConstSubVectorType =
      std::conditional_t<Direction == Vertical, typename XprType::ConstColXpr, typename XprType::ConstRowXpr>;

 public:
  using reference = std::conditional_t<bool(is_lvalue), SubVectorType, ConstSubVectorType>;
  using value_type = typename reference::PlainObject;

 private:
  class subvector_stl_iterator_ptr {
   public:
    constexpr subvector_stl_iterator_ptr(const reference& subvector) : m_subvector(subvector) {}
    constexpr reference* operator->() { return &m_subvector; }

   private:
    reference m_subvector;
  };

 public:
  using pointer = subvector_stl_iterator_ptr;

  constexpr subvector_stl_iterator() = default;
  constexpr subvector_stl_iterator(XprType& xpr, Index index) : Base(xpr, index) {}

  constexpr reference operator*() const { return (*mp_xpr).template subVector<Direction>(m_index); }
  constexpr reference operator[](Index i) const { return (*mp_xpr).template subVector<Direction>(m_index + i); }
  constexpr pointer operator->() const { return (*mp_xpr).template subVector<Direction>(m_index); }
};

template <typename XprType_, DirectionType Direction>
struct indexed_based_stl_iterator_traits<subvector_stl_reverse_iterator<XprType_, Direction>> {
  using XprType = XprType_;
  using non_const_iterator = subvector_stl_reverse_iterator<std::remove_const_t<XprType>, Direction>;
  using const_iterator = subvector_stl_reverse_iterator<std::add_const_t<XprType>, Direction>;
};

template <typename XprType, DirectionType Direction>
class subvector_stl_reverse_iterator
    : public indexed_based_stl_reverse_iterator_base<subvector_stl_reverse_iterator<XprType, Direction>> {
 protected:
  enum { is_lvalue = internal::is_lvalue<XprType>::value };

  using Base = indexed_based_stl_reverse_iterator_base<subvector_stl_reverse_iterator>;
  using Base::m_index;
  using Base::mp_xpr;

  using SubVectorType = std::conditional_t<Direction == Vertical, typename XprType::ColXpr, typename XprType::RowXpr>;
  using ConstSubVectorType =
      std::conditional_t<Direction == Vertical, typename XprType::ConstColXpr, typename XprType::ConstRowXpr>;

 public:
  using reference = std::conditional_t<bool(is_lvalue), SubVectorType, ConstSubVectorType>;
  using value_type = typename reference::PlainObject;

 private:
  class subvector_stl_reverse_iterator_ptr {
   public:
    constexpr subvector_stl_reverse_iterator_ptr(const reference& subvector) : m_subvector(subvector) {}
    constexpr reference* operator->() { return &m_subvector; }

   private:
    reference m_subvector;
  };

 public:
  using pointer = subvector_stl_reverse_iterator_ptr;

  constexpr subvector_stl_reverse_iterator() = default;
  constexpr subvector_stl_reverse_iterator(XprType& xpr, Index index) : Base(xpr, index) {}

  constexpr reference operator*() const { return (*mp_xpr).template subVector<Direction>(m_index); }
  constexpr reference operator[](Index i) const { return (*mp_xpr).template subVector<Direction>(m_index + i); }
  constexpr pointer operator->() const { return (*mp_xpr).template subVector<Direction>(m_index); }
};

}  // namespace internal

/** returns an iterator to the first element of the 1D vector or array
 * \only_for_vectors
 * \sa end(), cbegin()
 */
template <typename Derived>
constexpr typename DenseBase<Derived>::iterator DenseBase<Derived>::begin() {
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
  return iterator(derived(), 0);
}

/** const version of begin() */
template <typename Derived>
constexpr typename DenseBase<Derived>::const_iterator DenseBase<Derived>::begin() const {
  return cbegin();
}

/** returns a read-only const_iterator to the first element of the 1D vector or array
 * \only_for_vectors
 * \sa cend(), begin()
 */
template <typename Derived>
constexpr typename DenseBase<Derived>::const_iterator DenseBase<Derived>::cbegin() const {
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
  return const_iterator(derived(), 0);
}

/** returns an iterator to the element following the last element of the 1D vector or array
 * \only_for_vectors
 * \sa begin(), cend()
 */
template <typename Derived>
constexpr typename DenseBase<Derived>::iterator DenseBase<Derived>::end() {
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
  return iterator(derived(), size());
}

/** const version of end() */
template <typename Derived>
constexpr typename DenseBase<Derived>::const_iterator DenseBase<Derived>::end() const {
  return cend();
}

/** returns a read-only const_iterator to the element following the last element of the 1D vector or array
 * \only_for_vectors
 * \sa cbegin(), end()
 */
template <typename Derived>
constexpr typename DenseBase<Derived>::const_iterator DenseBase<Derived>::cend() const {
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
  return const_iterator(derived(), size());
}

}  // namespace Eigen

#endif  // EIGEN_STLITERATORS_H
