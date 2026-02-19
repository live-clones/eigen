// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_COMPRESSED_STORAGE_H
#define EIGEN_COMPRESSED_STORAGE_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

template <typename Scalar, typename StorageIndex, int MaxSize_>
struct InnerStorage {
  Scalar values[MaxSize_];
  StorageIndex indices[MaxSize_];
  Index size = MaxSize_;
  Index allocatedSize = MaxSize_;

  InnerStorage() = default;
  ~InnerStorage() = default;
};

template <typename Scalar, typename StorageIndex>
struct InnerStorage<Scalar, StorageIndex, Dynamic> {
  Scalar* values;
  StorageIndex* indices;
  Index size;
  Index allocatedSize;

  InnerStorage() : values(0), indices(0), size(0), allocatedSize(0) {}

  ~InnerStorage() {
    conditional_aligned_delete_auto<Scalar, true>(values, allocatedSize);
    conditional_aligned_delete_auto<StorageIndex, true>(indices, allocatedSize);
  }
};

template <typename Scalar, typename StorageIndex, int MaxSize>
void inner_storage_swap(InnerStorage<Scalar, StorageIndex, MaxSize>& lhs,
                        InnerStorage<Scalar, StorageIndex, MaxSize>& rhs) {
  std::swap(lhs.values, rhs.values);
  std::swap(lhs.indices, rhs.indices);
  std::swap(lhs.size, rhs.size);
  std::swap(lhs.allocatedSize, rhs.allocatedSize);
}

template <typename InnerStorage_>
void inner_storage_assign_impl(InnerStorage_& lhs, const InnerStorage_& rhs) {
  internal::smart_copy(rhs.values, rhs.values + lhs.size, lhs.values);
  internal::smart_copy(rhs.indices, rhs.indices + lhs.size, lhs.indices);
}

/** \internal
 * Stores a sparse set of values as a list of values and a list of indices.
 *
 */
template <typename Scalar_, typename StorageIndex_, int MaxSize = Dynamic>
class CompressedStorage {
 public:
  typedef Scalar_ Scalar;
  typedef StorageIndex_ StorageIndex;

 protected:
  typedef typename NumTraits<Scalar>::Real RealScalar;

  template <int MS = MaxSize, typename std::enable_if<MS == Dynamic, int>::type = 0>
  void before_equal(const CompressedStorage& other) {
    resize(other.size());
  }

  template <int MS = MaxSize, typename std::enable_if<MS != Dynamic, int>::type = 0>
  void before_equal(const CompressedStorage& other) {
    eigen_assert(other.size() == MaxSize);
  }

 public:
  CompressedStorage() : m_storage() {}

  explicit CompressedStorage(Index size) : m_storage() { resize(size); }

  CompressedStorage(const CompressedStorage& other) : m_storage() { *this = other; }

  CompressedStorage& operator=(const CompressedStorage& other) {
    before_equal(other);
    inner_storage_assign_impl(m_storage, other.m_storage);
    return *this;
  }

  void swap(CompressedStorage& other) { inner_storage_swap(m_storage, other.m_storage); }

  ~CompressedStorage() = default;

  void reserve(Index size) {
    Index newAllocatedSize = m_storage.size + size;
    if (newAllocatedSize > m_storage.allocatedSize) reallocate(newAllocatedSize);
  }

  void squeeze() {
    if (m_storage.allocatedSize > m_storage.size) reallocate(m_storage.size);
  }

  void resize(Index size, double reserveSizeFactor = 0) {
    eigen_assert(MaxSize == Dynamic && "CompressedStorage can be resized only for Dynamic sizes");
    if (m_storage.allocatedSize < size) {
      // Avoid underflow on the std::min<Index> call by choosing the smaller index type.
      using SmallerIndexType =
          typename std::conditional<static_cast<size_t>((std::numeric_limits<Index>::max)()) <
                                        static_cast<size_t>((std::numeric_limits<StorageIndex>::max)()),
                                    Index, StorageIndex>::type;
      Index realloc_size =
          (std::min<Index>)(NumTraits<SmallerIndexType>::highest(), size + Index(reserveSizeFactor * double(size)));
      if (realloc_size < size) internal::throw_std_bad_alloc();
      reallocate(realloc_size);
    }
    m_storage.size = size;
  }

  void append(const Scalar& v, Index i) {
    Index id = m_storage.size;
    resize(m_storage.size + 1, 1);
    m_storage.values[id] = v;
    m_storage.indices[id] = internal::convert_index<StorageIndex>(i);
  }

  inline Index size() const { return m_storage.size; }
  inline Index allocatedSize() const { return m_storage.allocatedSize; }
  inline void clear() { m_storage.size = 0; }

  const Scalar* valuePtr() const { return m_storage.values; }
  Scalar* valuePtr() { return m_storage.values; }
  const StorageIndex* indexPtr() const { return m_storage.indices; }
  StorageIndex* indexPtr() { return m_storage.indices; }

  inline Scalar& value(Index i) {
    eigen_internal_assert(m_storage.values != 0);
    return m_storage.values[i];
  }
  inline const Scalar& value(Index i) const {
    eigen_internal_assert(m_storage.values != 0);
    return m_storage.values[i];
  }

  inline StorageIndex& index(Index i) {
    eigen_internal_assert(m_storage.indices != 0);
    return m_storage.indices[i];
  }
  inline const StorageIndex& index(Index i) const {
    eigen_internal_assert(m_storage.indices != 0);
    return m_storage.indices[i];
  }

  /** \returns the largest \c k such that for all \c j in [0,k) index[\c j]\<\a key */
  inline Index searchLowerIndex(Index key) const { return searchLowerIndex(0, m_storage.size, key); }

  /** \returns the largest \c k in [start,end) such that for all \c j in [start,k) index[\c j]\<\a key */
  inline Index searchLowerIndex(Index start, Index end, Index key) const {
    return static_cast<Index>(
        std::distance(m_storage.indices, std::lower_bound(m_storage.indices + start, m_storage.indices + end, key)));
  }

  /** \returns the stored value at index \a key
   * If the value does not exist, then the value \a defaultValue is returned without any insertion. */
  inline Scalar at(Index key, const Scalar& defaultValue = Scalar(0)) const {
    if (m_storage.size == 0)
      return defaultValue;
    else if (key == m_storage.indices[m_storage.size - 1])
      return m_storage.values[m_storage.size - 1];
    // ^^  optimization: let's first check if it is the last coefficient
    // (very common in high level algorithms)
    const Index id = searchLowerIndex(0, m_storage.size - 1, key);
    return ((id < m_storage.size) && (m_storage.indices[id] == key)) ? m_storage.values[id] : defaultValue;
  }

  /** Like at(), but the search is performed in the range [start,end) */
  inline Scalar atInRange(Index start, Index end, Index key, const Scalar& defaultValue = Scalar(0)) const {
    if (start >= end)
      return defaultValue;
    else if (end > start && key == m_storage.indices[end - 1])
      return m_storage.values[end - 1];
    // ^^  optimization: let's first check if it is the last coefficient
    // (very common in high level algorithms)
    const Index id = searchLowerIndex(start, end - 1, key);
    return ((id < end) && (m_storage.indices[id] == key)) ? m_storage.values[id] : defaultValue;
  }

  /** \returns a reference to the value at index \a key
   * If the value does not exist, then the value \a defaultValue is inserted
   * such that the keys are sorted. */
  inline Scalar& atWithInsertion(Index key, const Scalar& defaultValue = Scalar(0)) {
    Index id = searchLowerIndex(0, m_storage.size, key);
    if (id >= m_storage.size || m_storage.indices[id] != key) {
      if (m_storage.allocatedSize < m_storage.size + 1) {
        Index newAllocatedSize = 2 * (m_storage.size + 1);
        m_storage.values = conditional_aligned_realloc_new_auto<Scalar, true>(m_storage.values, newAllocatedSize,
                                                                              m_storage.allocatedSize);
        m_storage.indices = conditional_aligned_realloc_new_auto<StorageIndex, true>(
            m_storage.indices, newAllocatedSize, m_storage.allocatedSize);
        m_storage.allocatedSize = newAllocatedSize;
      }
      if (m_storage.size > id) {
        internal::smart_memmove(m_storage.values + id, m_storage.values + m_storage.size, m_storage.values + id + 1);
        internal::smart_memmove(m_storage.indices + id, m_storage.indices + m_storage.size, m_storage.indices + id + 1);
      }
      m_storage.size++;
      m_storage.indices[id] = internal::convert_index<StorageIndex>(key);
      m_storage.values[id] = defaultValue;
    }
    return m_storage.values[id];
  }

  inline void moveChunk(Index from, Index to, Index chunkSize) {
    eigen_internal_assert(chunkSize >= 0 && to + chunkSize <= m_storage.size);
    internal::smart_memmove(m_storage.values + from, m_storage.values + from + chunkSize, m_storage.values + to);
    internal::smart_memmove(m_storage.indices + from, m_storage.indices + from + chunkSize, m_storage.indices + to);
  }

 protected:
  inline void reallocate(Index size) {
#ifdef EIGEN_SPARSE_COMPRESSED_STORAGE_REALLOCATE_PLUGIN
    EIGEN_SPARSE_COMPRESSED_STORAGE_REALLOCATE_PLUGIN
#endif
    eigen_internal_assert(size != m_storage.allocatedSize);
    m_storage.values =
        conditional_aligned_realloc_new_auto<Scalar, true>(m_storage.values, size, m_storage.allocatedSize);
    m_storage.indices =
        conditional_aligned_realloc_new_auto<StorageIndex, true>(m_storage.indices, size, m_storage.allocatedSize);
    m_storage.allocatedSize = size;
  }

 protected:
  using Storage = InnerStorage<Scalar, StorageIndex, MaxSize>;
  Storage m_storage;
};

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_COMPRESSED_STORAGE_H