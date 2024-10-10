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

template <typename T, int Size>
struct check_static_allocation_size {
#if EIGEN_STACK_ALLOCATION_LIMIT
  EIGEN_STATIC_ASSERT(Size * sizeof(T) <= EIGEN_STACK_ALLOCATION_LIMIT, OBJECT_ALLOCATED_ON_STACK_IS_TOO_BIG)
#endif
};

#if defined(EIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT)
#define EIGEN_MAKE_UNALIGNED_ARRAY_ASSERT(sizemask)
#else
#define EIGEN_MAKE_UNALIGNED_ARRAY_ASSERT(sizemask)                                                \
  eigen_assert((internal::is_constant_evaluated() || (std::uintptr_t(array) & (sizemask)) == 0) && \
               "this assertion is explained here: "                                                \
               "http://eigen.tuxfamily.org/dox-devel/group__TopicUnalignedArrayAssert.html"        \
               " **** READ THIS WEB PAGE !!! ****");
#endif

/** \internal
 * Static array. If the MatrixOrArrayOptions require auto-alignment, the array will be automatically aligned:
 * to 16 bytes boundary if the total size is a multiple of 16 bytes.
 */
template <typename T, int Size, int MatrixOrArrayOptions,
          int Alignment = (MatrixOrArrayOptions & DontAlign) ? 0 : compute_default_alignment<T, Size>::value>
struct plain_array : check_static_allocation_size<T, Size> {
  EIGEN_ALIGN_TO_BOUNDARY(Alignment) T array[Size];
#if defined(EIGEN_NO_DEBUG) || defined(EIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr plain_array() = default;
#else
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr plain_array() { EIGEN_MAKE_UNALIGNED_ARRAY_ASSERT(Alignment - 1); }
#endif
};

template <typename T, int Size, int MatrixOrArrayOptions>
struct plain_array<T, Size, MatrixOrArrayOptions, 0> : check_static_allocation_size<T, Size> {
  T array[Size];
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr plain_array() = default;
};

template <typename T, int MatrixOrArrayOptions, int Alignment>
struct plain_array<T, 0, MatrixOrArrayOptions, Alignment> {
  T array[1];
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr plain_array() = default;
};

struct plain_array_helper {
  template <typename T, int Size, int MatrixOrArrayOptions, int Alignment>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE static void copy(
      const plain_array<T, Size, MatrixOrArrayOptions, Alignment>& src, const Eigen::Index size,
      plain_array<T, Size, MatrixOrArrayOptions, Alignment>& dst) {
    smart_copy(src.array, src.array + size, dst.array);
  }

  template <typename T, int Size, int MatrixOrArrayOptions, int Alignment>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE static void swap(plain_array<T, Size, MatrixOrArrayOptions, Alignment>& a,
                                                         const Eigen::Index a_size,
                                                         plain_array<T, Size, MatrixOrArrayOptions, Alignment>& b,
                                                         const Eigen::Index b_size) {
    if (a_size < b_size) {
      std::swap_ranges(b.array, b.array + a_size, a.array);
      smart_move(b.array + a_size, b.array + b_size, a.array + a_size);
    } else if (a_size > b_size) {
      std::swap_ranges(a.array, a.array + b_size, b.array);
      smart_move(a.array + b_size, a.array + a_size, b.array + b_size);
    } else {
      std::swap_ranges(a.array, a.array + a_size, b.array);
    }
  }
};

// this class is intended to be inherited by DenseStorage to take advantage of empty base optimization
template <int Rows, int Cols>
struct DenseStorageIndices {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices() = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices(const DenseStorageIndices&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices(DenseStorageIndices&&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices& operator=(const DenseStorageIndices&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices& operator=(DenseStorageIndices&&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices(Index /*rows*/, Index /*cols*/) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index rows() const { return Rows; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index cols() const { return Cols; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void set(Index /*rows*/, Index /*cols*/) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void swap(DenseStorageIndices& /*other*/) noexcept {}
};
template <int Rows>
struct DenseStorageIndices<Rows, Dynamic> {
  Index m_cols;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices() : m_cols(0) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices(const DenseStorageIndices&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices(DenseStorageIndices&&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices& operator=(const DenseStorageIndices&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices& operator=(DenseStorageIndices&&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices(Index /*rows*/, Index cols) : m_cols(cols) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index rows() const { return Rows; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index cols() const { return m_cols; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void set(Index /*rows*/, Index cols) { m_cols = cols; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void swap(DenseStorageIndices& other) noexcept {
    numext::swap(m_cols, other.m_cols);
  }
};
template <int Cols>
struct DenseStorageIndices<Dynamic, Cols> {
  Index m_rows;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices() : m_rows(0) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices(const DenseStorageIndices&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices(DenseStorageIndices&&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices& operator=(const DenseStorageIndices&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices& operator=(DenseStorageIndices&&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices(Index rows, Index /*cols*/) : m_rows(rows) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index rows() const { return m_rows; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index cols() const { return Cols; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void set(Index rows, Index /*cols*/) { m_rows = rows; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void swap(DenseStorageIndices& other) noexcept {
    numext::swap(m_rows, other.m_rows);
  }
};
template <>
struct DenseStorageIndices<Dynamic, Dynamic> {
  Index m_rows;
  Index m_cols;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices() : m_rows(0), m_cols(0) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices(const DenseStorageIndices&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices(DenseStorageIndices&&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices& operator=(const DenseStorageIndices&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices& operator=(DenseStorageIndices&&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices(Index rows, Index cols)
      : m_rows(rows), m_cols(cols) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index rows() const { return m_rows; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index cols() const { return m_cols; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void set(Index rows, Index cols) {
    m_rows = rows;
    m_cols = cols;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void swap(DenseStorageIndices& other) noexcept {
    numext::swap(m_rows, other.m_rows);
    numext::swap(m_cols, other.m_cols);
  }
};

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
class DenseStorage : internal::DenseStorageIndices<Rows, Cols> {
  using Base = internal::DenseStorageIndices<Rows, Cols>;

  internal::plain_array<T, Size, Options> m_data;

 public:
  using Base::cols;
  using Base::rows;
#ifndef EIGEN_DENSE_STORAGE_CTOR_PLUGIN
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE DenseStorage() = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(const DenseStorage&) = default;
#else
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE DenseStorage() { EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN(Index size = Size) }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(const DenseStorage& other)
      : Base(other), m_data(other.m_data) {
    EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN(Index size = Size)
  }
#endif
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(DenseStorage&&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage& operator=(const DenseStorage&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage& operator=(DenseStorage&&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(Index size, Index rows, Index cols) : Base(rows, cols) {
    EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN({})
    EIGEN_UNUSED_VARIABLE(size);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void swap(DenseStorage& other) noexcept {
    numext::swap(m_data, other.m_data);
    Base::swap(other);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void conservativeResize(Index /*size*/, Index rows, Index cols) {
    Base::set(rows, cols);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void resize(Index /*size*/, Index rows, Index cols) {
    Base::set(rows, cols);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr T* data() { return m_data.array; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr const T* data() const { return m_data.array; }
};
// null matrix specialization
template <typename T, int Rows, int Cols, int Options>
class DenseStorage<T, 0, Rows, Cols, Options> : internal::DenseStorageIndices<Rows, Cols> {
  using Base = internal::DenseStorageIndices<Rows, Cols>;

 public:
  using Base::cols;
  using Base::rows;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage() = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(const DenseStorage&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(DenseStorage&&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage& operator=(const DenseStorage&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage& operator=(DenseStorage&&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(Index /*size*/, Index rows, Index cols)
      : Base(rows, cols) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void swap(DenseStorage& other) noexcept { Base::swap(other); }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void conservativeResize(Index /*size*/, Index rows, Index cols) {
    Base::set(rows, cols);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void resize(Index /*size*/, Index rows, Index cols) {
    Base::set(rows, cols);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr T* data() { return nullptr; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr const T* data() const { return nullptr; }
};
// dynamic matrix specialization
template <typename T, int Rows, int Cols, int Options>
class DenseStorage<T, Dynamic, Rows, Cols, Options> : internal::DenseStorageIndices<Rows, Cols> {
  using Base = internal::DenseStorageIndices<Rows, Cols>;
  static constexpr int Size = Dynamic;
  static constexpr bool Align = (Options & DontAlign) == 0;

  T* m_data;

 public:
  using Base::cols;
  using Base::rows;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage() : m_data(nullptr) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(const DenseStorage& other)
      : Base(other), m_data(internal::conditional_aligned_new_auto<T, Align>(other.rows() * other.cols())) {
    Index size = other.rows() * other.cols();
    EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN({})
    internal::smart_copy(other.m_data, other.m_data + size, m_data);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(DenseStorage&& other) noexcept
      : Base(other), m_data(other.m_data) {
    other.set(0, 0);
    other.m_data = nullptr;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage& operator=(const DenseStorage& other) {
    Base::set(other.rows(), other.cols());
    Index size = other.rows() * other.cols();
    m_data = internal::conditional_aligned_new_auto<T, Align>(size);
    EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN({})
    internal::smart_copy(other.m_data, other.m_data + size, m_data);
    return *this;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage& operator=(DenseStorage&& other) noexcept {
    this->swap(other);
    return *this;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(Index size, Index rows, Index cols)
      : Base(rows, cols), m_data(internal::conditional_aligned_new_auto<T, Align>(size)) {
    EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN({})
  }
  EIGEN_DEVICE_FUNC ~DenseStorage() {
    Index size = this->rows() * this->cols();
    internal::conditional_aligned_delete_auto<T, Align>(m_data, size);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void swap(DenseStorage& other) noexcept {
    numext::swap(m_data, other.m_data);
    Base::swap(other);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void conservativeResize(Index size, Index rows, Index cols) {
    Index oldSize = this->rows() * this->cols();
    m_data = internal::conditional_aligned_realloc_new_auto<T, Align>(m_data, size, oldSize);
    Base::set(rows, cols);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void resize(Index size, Index rows, Index cols) {
    Index oldSize = this->rows() * this->cols();
    if (size != oldSize) {
      internal::conditional_aligned_delete_auto<T, Align>(m_data, oldSize);
      if (size > 0)  // >0 and not simply !=0 to let the compiler knows that size cannot be negative
      {
        m_data = internal::conditional_aligned_new_auto<T, Align>(size);
        EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN({})
      } else
        m_data = nullptr;
    }
    Base::set(rows, cols);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr T* data() { return m_data; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr const T* data() const { return m_data; }
};
}  // end namespace Eigen

#endif  // EIGEN_MATRIX_H
