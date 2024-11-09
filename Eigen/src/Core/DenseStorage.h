// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2006-2009 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2010-2013 Hauke Heibel <hauke.heibel@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_MATRIXSTORAGE_H
#define EIGEN_MATRIXSTORAGE_H

#ifdef EIGEN_DENSE_STORAGE_CTOR_PLUGIN
#define EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN(X) \
  X;                                                \
  EIGEN_DENSE_STORAGE_CTOR_PLUGIN;
#else
#define EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN(X)
#endif

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

#if defined(EIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT)
#define EIGEN_MAKE_UNALIGNED_ARRAY_ASSERT(Alignment)
#else
#define EIGEN_MAKE_UNALIGNED_ARRAY_ASSERT(Alignment)                                                \
  eigen_assert((internal::is_constant_evaluated() || (std::uintptr_t(m_array) % Alignment == 0)) && \
               "this assertion is explained here: "                                                 \
               "http://eigen.tuxfamily.org/dox-devel/group__TopicUnalignedArrayAssert.html"         \
               " **** READ THIS WEB PAGE !!! ****");
#endif

#if EIGEN_STACK_ALLOCATION_LIMIT
#define EIGEN_MAKE_STACK_ALLOCATION_ASSERT(X) \
  EIGEN_STATIC_ASSERT(X <= EIGEN_STACK_ALLOCATION_LIMIT, OBJECT_ALLOCATED_ON_STACK_IS_TOO_BIG)
#else
#define EIGEN_MAKE_STACK_ALLOCATION_ASSERT(X)
#endif

/** \internal
 * Static array. If the MatrixOrArrayOptions require auto-alignment, the array will be automatically aligned:
 * to 16 bytes boundary if the total size is a multiple of 16 bytes.
 */
template <typename T, int Size, int MatrixOrArrayOptions,
          int Alignment = (MatrixOrArrayOptions & DontAlign) ? 0 : compute_default_alignment<T, Size>::value>
struct plain_array {
  EIGEN_ALIGN_TO_BOUNDARY(Alignment) T m_array[Size];
#if defined(EIGEN_NO_DEBUG) || defined(EIGEN_TESTING_PLAINOBJECT_CTOR) || !defined(EIGEN_DENSE_STORAGE_CTOR_PLUGIN)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr plain_array() = default;
#else
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr plain_array() {
    EIGEN_MAKE_UNALIGNED_ARRAY_ASSERT(Alignment)
    EIGEN_MAKE_STACK_ALLOCATION_ASSERT(Size * sizeof(T))
    EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN(Index size = Size)
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr plain_array(const plain_array& other) {
    EIGEN_MAKE_UNALIGNED_ARRAY_ASSERT(Alignment)
    EIGEN_MAKE_STACK_ALLOCATION_ASSERT(Size * sizeof(T))
    EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN(Index size = Size)
    smart_copy(other.m_array, other.m_array + Size, m_array);
  }
#endif
};

template <typename T, int Size, int MatrixOrArrayOptions>
struct plain_array<T, Size, MatrixOrArrayOptions, 0> {
  T m_array[Size];
#if defined(EIGEN_NO_DEBUG) || defined(EIGEN_TESTING_PLAINOBJECT_CTOR) || !defined(EIGEN_DENSE_STORAGE_CTOR_PLUGIN)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr plain_array() = default;
#else
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr plain_array() {
    EIGEN_MAKE_STACK_ALLOCATION_ASSERT(Size * sizeof(T))
    EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN(Index size = Size)
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr plain_array(const plain_array& other) {
    EIGEN_MAKE_STACK_ALLOCATION_ASSERT(Size * sizeof(T))
    EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN(Index size = Size)
    smart_copy(other.m_array, other.m_array + Size, m_array);
  }
#endif
};

template <typename T, int MatrixOrArrayOptions, int Alignment>
struct plain_array<T, 0, MatrixOrArrayOptions, Alignment> {
  T m_array[1];
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr plain_array() = default;
};

template <typename T, int Size, int Options, int Alignment>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void swap_plain_array(plain_array<T, Size, Options, Alignment>& a,
                                                                      plain_array<T, Size, Options, Alignment>& b,
                                                                      Index a_size, Index b_size) {
  Index commonSize = numext::mini(a_size, b_size);
  std::swap_ranges(a.m_array, a.m_array + commonSize, b.m_array);
  if (a_size > b_size)
    std::move(a.m_array + commonSize, a.m_array + a_size, b.m_array + commonSize);
  else if (b_size > a_size)
    std::move(b.m_array + commonSize, b.m_array + b_size, a.m_array + commonSize);
}

}  // end namespace internal

/** \internal
 *
 * \class DenseStorage
 * \ingroup Core_Module
 *
 * \brief Stores the data of a matrix
 *
 * This class stores the data of fixed-size, dynamic-size or mixed matrices
 * in a way as compact as possible.
 *
 * \sa Matrix
 */
template <typename T, int Size, int Rows, int Cols, int Options>
class DenseStorage {
  internal::plain_array<T, Size, Options> m_data;

 public:
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage() = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(const DenseStorage&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(Index /*size*/, Index /*rows*/, Index /*cols*/) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(DenseStorage&&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage& operator=(const DenseStorage&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage& operator=(DenseStorage&&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void swap(DenseStorage& other) { numext::swap(m_data, other.m_data); }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void conservativeResize(Index /*size*/, Index /*rows*/,
                                                                          Index /*cols*/) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void resize(Index /*size*/, Index /*rows*/, Index /*cols*/) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index rows() const { return Rows; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index cols() const { return Cols; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index size() const { return Rows * Cols; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr T* data() { return m_data.m_array; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr const T* data() const { return m_data.m_array; }
};
template <typename T, int Size, int Cols, int Options>
class DenseStorage<T, Size, Dynamic, Cols, Options> {
  internal::plain_array<T, Size, Options> m_data;
  Index m_rows = 0;

 public:
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE DenseStorage() = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(const DenseStorage& other) : m_rows(other.m_rows) {
    internal::smart_copy(other.m_data.m_array, other.m_data.m_array + size(), m_data.m_array);
    EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN(Index size = Size)
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(Index /*size*/, Index rows, Index /*cols*/)
      : m_rows(rows) {
    EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN(Index size = Size)
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(DenseStorage&& other) : m_rows(other.m_rows) {
    std::move(other.m_data.m_array, other.m_data.m_array + size(), m_data.m_array);
    other.m_rows = 0;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage& operator=(const DenseStorage& other) {
    m_rows = other.m_rows;
    internal::smart_copy(other.m_data.m_array, other.m_data.m_array + size(), m_data.m_array);
    return *this;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage& operator=(DenseStorage&& other) {
    m_rows = other.m_rows;
    std::move(other.m_data.m_array, other.m_data.m_array + size(), m_data.m_array);
    other.m_rows = 0;
    return *this;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void swap(DenseStorage& other) {
    internal::swap_plain_array(m_data, other.m_data, size(), other.size());
    numext::swap(m_rows, other.m_rows);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void conservativeResize(Index /*size*/, Index rows, Index /*cols*/) {
    m_rows = rows;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void resize(Index /*size*/, Index rows, Index /*cols*/) {
    m_rows = rows;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index rows() const { return m_rows; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index cols() const { return Cols; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index size() const { return m_rows * Cols; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr T* data() { return m_data.m_array; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr const T* data() const { return m_data.m_array; }
};
template <typename T, int Size, int Rows, int Options>
class DenseStorage<T, Size, Rows, Dynamic, Options> {
  internal::plain_array<T, Size, Options> m_data;
  Index m_cols = 0;

 public:
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE DenseStorage() = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(const DenseStorage& other) : m_cols(other.m_cols) {
    internal::smart_copy(other.m_data.m_array, other.m_data.m_array + size(), m_data.m_array);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(Index /*size*/, Index /*rows*/, Index cols)
      : m_cols(cols) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(DenseStorage&& other) : m_cols(other.m_cols) {
    std::move(other.m_data.m_array, other.m_data.m_array + size(), m_data.m_array);
    other.m_cols = 0;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage& operator=(const DenseStorage& other) {
    m_cols = other.m_cols;
    internal::smart_copy(other.m_data.m_array, other.m_data.m_array + size(), m_data.m_array);
    return *this;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage& operator=(DenseStorage&& other) {
    m_cols = other.m_cols;
    std::move(other.m_data.m_array, other.m_data.m_array + size(), m_data.m_array);
    other.m_cols = 0;
    return *this;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void swap(DenseStorage& other) {
    internal::swap_plain_array(m_data, other.m_data, size(), other.size());
    numext::swap(m_cols, other.m_cols);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void conservativeResize(Index /*size*/, Index /*rows*/, Index cols) {
    m_cols = cols;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void resize(Index /*size*/, Index /*rows*/, Index cols) {
    m_cols = cols;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index rows() const { return Rows; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index cols() const { return m_cols; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index size() const { return Rows * m_cols; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr T* data() { return m_data.m_array; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr const T* data() const { return m_data.m_array; }
};
template <typename T, int Size, int Options>
class DenseStorage<T, Size, Dynamic, Dynamic, Options> {
  internal::plain_array<T, Size, Options> m_data;
  Index m_rows = 0;
  Index m_cols = 0;

 public:
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE DenseStorage() = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(const DenseStorage& other)
      : m_rows(other.m_rows), m_cols(other.m_cols) {
    internal::smart_copy(other.m_data.m_array, other.m_data.m_array + size(), m_data.m_array);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(Index /*size*/, Index rows, Index cols)
      : m_rows(rows), m_cols(cols) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(DenseStorage&& other)
      : m_rows(other.m_rows), m_cols(other.m_cols) {
    std::move(other.m_data.m_array, other.m_data.m_array + size(), m_data.m_array);
    other.m_rows = 0;
    other.m_cols = 0;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage& operator=(const DenseStorage& other) {
    m_rows = other.m_rows;
    m_cols = other.m_cols;
    internal::smart_copy(other.m_data.m_array, other.m_data.m_array + size(), m_data.m_array);
    return *this;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage& operator=(DenseStorage&& other) {
    m_rows = other.m_rows;
    m_cols = other.m_cols;
    std::move(other.m_data.m_array, other.m_data.m_array + size(), m_data.m_array);
    other.m_rows = 0;
    other.m_cols = 0;
    return *this;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void swap(DenseStorage& other) {
    internal::swap_plain_array(m_data, other.m_data, size(), other.size());
    numext::swap(m_rows, other.m_rows);
    numext::swap(m_cols, other.m_cols);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void conservativeResize(Index /*size*/, Index rows, Index cols) {
    m_rows = rows;
    m_cols = cols;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void resize(Index /*size*/, Index rows, Index cols) {
    m_rows = rows;
    m_cols = cols;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index rows() const { return m_rows; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index cols() const { return m_cols; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index size() const { return m_rows * m_cols; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr T* data() { return m_data.m_array; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr const T* data() const { return m_data.m_array; }
};
// zero-sized variants
template <typename T, int Rows, int Cols, int Options>
class DenseStorage<T, 0, Rows, Cols, Options> {
 public:
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage() = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(const DenseStorage&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(Index /*size*/, Index /*rows*/, Index /*cols*/) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(DenseStorage&&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage& operator=(const DenseStorage&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage& operator=(DenseStorage&&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void swap(DenseStorage&) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void conservativeResize(Index /*size*/, Index /*rows*/,
                                                                          Index /*cols*/) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void resize(Index /*size*/, Index /*rows*/, Index /*cols*/) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index rows() const { return Rows; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index cols() const { return Cols; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index size() const { return Rows * Cols; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr T* data() { return nullptr; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr const T* data() const { return nullptr; }
};
template <typename T, int Cols, int Options>
class DenseStorage<T, 0, Dynamic, Cols, Options> {
  Index m_rows = 0;

 public:
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage() = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(const DenseStorage&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(Index /*size*/, Index rows, Index /*cols*/)
      : m_rows(rows) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(DenseStorage&&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage& operator=(const DenseStorage&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage& operator=(DenseStorage&&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void swap(DenseStorage& other) { numext::swap(m_rows, other.m_rows); }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void conservativeResize(Index /*size*/, Index rows, Index /*cols*/) {
    m_rows = rows;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void resize(Index /*size*/, Index rows, Index /*cols*/) {
    m_rows = rows;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index rows() const { return m_rows; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index cols() const { return Cols; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index size() const { return m_rows * Cols; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr T* data() { return nullptr; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr const T* data() const { return nullptr; }
};
template <typename T, int Rows, int Options>
class DenseStorage<T, 0, Rows, Dynamic, Options> {
  Index m_cols = 0;

 public:
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage() = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(const DenseStorage&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(Index /*size*/, Index /*rows*/, Index cols)
      : m_cols(cols) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(DenseStorage&&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage& operator=(const DenseStorage&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage& operator=(DenseStorage&&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void swap(DenseStorage& other) { numext::swap(m_cols, other.m_cols); }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void conservativeResize(Index /*size*/, Index /*rows*/, Index cols) {
    m_cols = cols;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void resize(Index /*size*/, Index /*rows*/, Index cols) {
    m_cols = cols;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index rows() const { return Rows; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index cols() const { return m_cols; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index size() const { return Rows * m_cols; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr T* data() { return nullptr; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr const T* data() const { return nullptr; }
};
template <typename T, int Options>
class DenseStorage<T, 0, Dynamic, Dynamic, Options> {
  Index m_rows = 0;
  Index m_cols = 0;

 public:
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage() = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(const DenseStorage&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(Index /*size*/, Index rows, Index cols)
      : m_rows(rows), m_cols(cols) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(DenseStorage&&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage& operator=(const DenseStorage&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage& operator=(DenseStorage&&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void swap(DenseStorage& other) {
    numext::swap(m_rows, other.m_rows);
    numext::swap(m_cols, other.m_cols);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void conservativeResize(Index /*size*/, Index rows, Index cols) {
    m_rows = rows;
    m_cols = cols;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void resize(Index /*size*/, Index rows, Index cols) {
    m_rows = rows;
    m_cols = cols;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index rows() const { return m_rows; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index cols() const { return m_cols; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index size() const { return m_rows * m_cols; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr T* data() { return nullptr; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr const T* data() const { return nullptr; }
};
// dynamic-sized variants
template <typename T, int Rows, int Cols, int Options>
class DenseStorage<T, Dynamic, Rows, Cols, Options> {
  // special case of a fixed-size object with dynamic memory allocation
  static constexpr int Size = Rows * Cols;
  static constexpr bool Align = (Options & DontAlign) == 0;
  T* m_data = nullptr;

 public:
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage() = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(const DenseStorage& other)
      : m_data(internal::conditional_aligned_new_auto<T, Align>(Size)) {
    EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN(Index size = Size)
    internal::smart_copy(other.m_data, other.m_data + Size, m_data);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(Index /*size*/, Index /*rows*/, Index /*cols*/)
      : m_data(internal::conditional_aligned_new_auto<T, Align>(Size)) {
    EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN(Index size = Size)
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(DenseStorage&& other) noexcept : m_data(other.m_data) {
    other.m_data = nullptr;
  }
  EIGEN_DEVICE_FUNC ~DenseStorage() { internal::conditional_aligned_delete_auto<T, Align>(m_data, Size); }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage& operator=(const DenseStorage& other) {
    resize(Size, Rows, Cols);
    internal::smart_copy(other.m_data, other.m_data + Size, m_data);
    return *this;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage& operator=(DenseStorage&& other) {
    this->swap(other);
    return *this;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void swap(DenseStorage& other) { numext::swap(m_data, other.m_data); }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void conservativeResize(Index /*size*/, Index /*rows*/,
                                                                          Index /*cols*/) {
    if (m_data == nullptr) m_data = internal::conditional_aligned_new_auto<T, Align>(Size);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void resize(Index /*size*/, Index /*rows*/, Index /*cols*/) {
    if (m_data == nullptr) {
      m_data = internal::conditional_aligned_new_auto<T, Align>(Size);
      EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN(Index size = Size)
    }
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index rows() const { return Rows; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index cols() const { return Cols; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index size() const { return Rows * Cols; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr T* data() { return m_data; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr const T* data() const { return m_data; }
};
template <typename T, int Cols, int Options>
class DenseStorage<T, Dynamic, Dynamic, Cols, Options> {
  static constexpr int Size = Dynamic;
  static constexpr bool Align = (Options & DontAlign) == 0;
  T* m_data = nullptr;
  Index m_rows = 0;

 public:
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage() = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(const DenseStorage& other)
      : m_data(internal::conditional_aligned_new_auto<T, Align>(other.size())), m_rows(other.m_rows) {
    Index size = other.size();
    EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN({})
    internal::smart_copy(other.m_data, other.m_data + size, m_data);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(Index size, Index rows, Index /*cols*/)
      : m_data(internal::conditional_aligned_new_auto<T, Align>(size)), m_rows(rows) {
    EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN({})
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(DenseStorage&& other) noexcept
      : m_data(other.m_data), m_rows(other.m_rows) {
    other.m_data = nullptr;
    other.m_rows = 0;
  }
  EIGEN_DEVICE_FUNC ~DenseStorage() { internal::conditional_aligned_delete_auto<T, Align>(m_data, size()); }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage& operator=(const DenseStorage& other) {
    resize(other.size(), other.rows(), other.cols());
    internal::smart_copy(other.m_data, other.m_data + other.size(), m_data);
    return *this;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage& operator=(DenseStorage&& other) {
    this->swap(other);
    return *this;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void swap(DenseStorage& other) {
    numext::swap(m_data, other.m_data);
    numext::swap(m_rows, other.m_rows);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void conservativeResize(Index size, Index rows, Index /*cols*/) {
    Index oldSize = this->size();
    m_data = internal::conditional_aligned_realloc_new_auto<T, Align>(m_data, size, oldSize);
    m_rows = rows;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void resize(Index size, Index rows, Index /*cols*/) {
    Index oldSize = this->size();
    if (oldSize != size) {
      internal::conditional_aligned_delete_auto<T, Align>(m_data, oldSize);
      m_data = internal::conditional_aligned_new_auto<T, Align>(size);
      EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN({})
    }
    m_rows = rows;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index rows() const { return m_rows; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index cols() const { return Cols; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index size() const { return m_rows * Cols; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr T* data() { return m_data; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr const T* data() const { return m_data; }
};
template <typename T, int Rows, int Options>
class DenseStorage<T, Dynamic, Rows, Dynamic, Options> {
  static constexpr int Size = Dynamic;
  static constexpr bool Align = (Options & DontAlign) == 0;
  T* m_data = nullptr;
  Index m_cols = 0;

 public:
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage() = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(const DenseStorage& other)
      : m_data(internal::conditional_aligned_new_auto<T, Align>(other.size())), m_cols(other.m_cols) {
    Index size = other.size();
    EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN({})
    internal::smart_copy(other.m_data, other.m_data + size, m_data);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(Index size, Index /*rows*/, Index cols)
      : m_data(internal::conditional_aligned_new_auto<T, Align>(size)), m_cols(cols) {
    EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN({})
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(DenseStorage&& other) noexcept
      : m_data(other.m_data), m_cols(other.m_cols) {
    other.m_data = nullptr;
    other.m_cols = 0;
  }
  EIGEN_DEVICE_FUNC ~DenseStorage() { internal::conditional_aligned_delete_auto<T, Align>(m_data, size()); }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage& operator=(const DenseStorage& other) {
    resize(other.size(), other.rows(), other.cols());
    internal::smart_copy(other.m_data, other.m_data + other.size(), m_data);
    return *this;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage& operator=(DenseStorage&& other) {
    this->swap(other);
    return *this;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void swap(DenseStorage& other) {
    numext::swap(m_data, other.m_data);
    numext::swap(m_cols, other.m_cols);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void conservativeResize(Index size, Index /*rows*/, Index cols) {
    Index oldSize = this->size();
    m_data = internal::conditional_aligned_realloc_new_auto<T, Align>(m_data, size, oldSize);
    m_cols = cols;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void resize(Index size, Index /*rows*/, Index cols) {
    Index oldSize = this->size();
    if (oldSize != size) {
      internal::conditional_aligned_delete_auto<T, Align>(m_data, oldSize);
      m_data = internal::conditional_aligned_new_auto<T, Align>(size);
      EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN({})
    }
    m_cols = cols;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index rows() const { return Rows; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index cols() const { return m_cols; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index size() const { return Rows * m_cols; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr T* data() { return m_data; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr const T* data() const { return m_data; }
};
template <typename T, int Options>
class DenseStorage<T, Dynamic, Dynamic, Dynamic, Options> {
  static constexpr int Size = Dynamic;
  static constexpr bool Align = (Options & DontAlign) == 0;
  T* m_data = nullptr;
  Index m_rows = 0;
  Index m_cols = 0;

 public:
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage() = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(const DenseStorage& other)
      : m_data(internal::conditional_aligned_new_auto<T, Align>(other.size())),
        m_rows(other.m_rows),
        m_cols(other.m_cols) {
    Index size = other.size();
    EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN({})
    internal::smart_copy(other.m_data, other.m_data + size, m_data);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(Index size, Index rows, Index cols)
      : m_data(internal::conditional_aligned_new_auto<T, Align>(size)), m_rows(rows), m_cols(cols) {
    EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN({})
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(DenseStorage&& other) noexcept
      : m_data(other.m_data), m_rows(other.m_rows), m_cols(other.m_cols) {
    other.m_data = nullptr;
    other.m_rows = 0;
    other.m_cols = 0;
  }
  EIGEN_DEVICE_FUNC ~DenseStorage() { internal::conditional_aligned_delete_auto<T, Align>(m_data, size()); }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage& operator=(const DenseStorage& other) {
    resize(other.size(), other.rows(), other.cols());
    internal::smart_copy(other.m_data, other.m_data + other.size(), m_data);
    return *this;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage& operator=(DenseStorage&& other) {
    this->swap(other);
    return *this;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void swap(DenseStorage& other) {
    numext::swap(m_data, other.m_data);
    numext::swap(m_rows, other.m_rows);
    numext::swap(m_cols, other.m_cols);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void conservativeResize(Index size, Index rows, Index cols) {
    Index oldSize = this->size();
    m_data = internal::conditional_aligned_realloc_new_auto<T, Align>(m_data, size, oldSize);
    m_rows = rows;
    m_cols = cols;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void resize(Index size, Index rows, Index cols) {
    Index oldSize = this->size();
    if (oldSize != size) {
      internal::conditional_aligned_delete_auto<T, Align>(m_data, oldSize);
      m_data = internal::conditional_aligned_new_auto<T, Align>(size);
      EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN({})
    }
    m_rows = rows;
    m_cols = cols;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index rows() const { return m_rows; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index cols() const { return m_cols; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index size() const { return m_rows * m_cols; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr T* data() { return m_data; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr const T* data() const { return m_data; }
};
}  // end namespace Eigen

#endif  // EIGEN_MATRIX_H
