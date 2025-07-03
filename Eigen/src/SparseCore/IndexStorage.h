// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_INDEXSTORAGE_STORAGE_H
#define EIGEN_INDEXSTORAGE_STORAGE_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

template <typename StorageIndex, int MaxSize_>
struct IndexStorage {
  StorageIndex outerIndex[MaxSize_ + 1] = {0};
  StorageIndex* innerNonZeros;  // not used in static size
  StorageIndex* outerPtr() { return outerIndex; }
  const StorageIndex* outerPtr() const { return outerIndex; }
  StorageIndex* innerNZPtr() { return innerNonZeros; }
  const StorageIndex* innerNZPtr() const { return innerNonZeros; }
  IndexStorage() : innerNonZeros(0) {}
  ~IndexStorage() = default;
};

template <typename StorageIndex>
struct IndexStorage<StorageIndex, Dynamic> {
  StorageIndex* outerIndex;
  StorageIndex* innerNonZeros;  // optional, if null then the data is compressed
  StorageIndex* outerPtr() { return outerIndex; }
  const StorageIndex* outerPtr() const { return outerIndex; }
  StorageIndex* innerNZPtr() { return innerNonZeros; }
  const StorageIndex* innerNZPtr() const { return innerNonZeros; }
  IndexStorage() : outerIndex(0), innerNonZeros(0) {}
  ~IndexStorage() = default;
};

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_INDEXSTORAGE_STORAGE_H