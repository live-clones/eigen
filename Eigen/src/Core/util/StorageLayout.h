// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2022, Erik Schultheis <erik.schultheis@aalto.fi>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_STORAGE_LAYOUT_H
#define EIGEN_STORAGE_LAYOUT_H

#include "../InternalHeaderCheck.h"

namespace Eigen {
namespace internal {

/** \defgroup storage_order Storage Order Utilities
  * \ingroup Core_Module
  *
  * Helper functions to deal with storage layouts. There are two storage layouts,
  * column major (\ref `StorageOrder::ColMajor`) and row major (\ref `StorageOrder::RowMajor`).
  * More information about these storage orders can be found on the page \ref TopicStorageOrders.
  * There are two way how the storage order is represented within Eigen. The first is
  * as part of a general flags bitfield, such as the `Options` parameter of the `Matrix`
  * template. The second is as a strong enum of type \ref StorageOrder, which is used wherever
  * the storage order is not combined in a bit flag. Whenever possible, the strong enum should be
  * preferred, because it reduces the likelihood of programming errors caused by misuse of the
  * bit-field flags.
  *
  * This file provides some helper functions which allow calling code to interact with these
  * two representations in a uniform manner, and to convert between them.
  *
  * To extract the storage order from a flags bit-field, the \ref `get_storage_order()` function
  * should be used. To set the storage order of an existing bit field, use \ref `with_storage_order()`.
  * In general, all bit manipulation corresponding to storage order should be handled through
  * these utility functions.
  *
  * \sa TopicStorageOrders
  */

/// \ingroup enums
/// Strong enum to distinguish column major and row major storage order.
/// \note The values of the enumerators are deliberately set to irregular values so that any casting to the
/// corresponding bit-field flags will cause visible problems.
/// \sa TopicStorageOrders, storage_order
enum class StorageOrder {
    ColMajor = 0xC01,   //!< Column-major storage order.
    RowMajor = 0x120    //!< Row-major storage order.
};

/// \ingroup storage_order
/// Returns whether the bit-field represented by flags corresponds to row-major layout
/// \todo Should flags be signed or unsigned?
constexpr bool is_row_major(int flags) {
  return flags & RowMajorBit;
}

/// \ingroup storage_order
/// Returns whether the storage order is row-major.
constexpr bool is_row_major(StorageOrder order) {
  return order == StorageOrder::RowMajor;
}

/// \ingroup storage_order
/// Returns whether the bit-field represented by flags corresponds to col-major layout
constexpr bool is_col_major(int flags) {
  return !is_row_major(flags);
}

/// \ingroup storage_order
/// Returns whether the storage order is col-major.
constexpr bool is_col_major(StorageOrder order) {
  return order == StorageOrder::ColMajor;
}

/// \ingroup storage_order
/// Returns the transposed storage order
constexpr StorageOrder transposed(StorageOrder order) {
  switch (order) {
    case StorageOrder::RowMajor:
      return StorageOrder::ColMajor;
    case StorageOrder::ColMajor:
      return StorageOrder::RowMajor;
  }
}

/// \ingroup storage_order
/// Based on the bit-field `flags`, returns the storage order as a strong enum.
constexpr StorageOrder get_storage_order(int flags) {
  return is_row_major(flags) ? StorageOrder::RowMajor : StorageOrder::ColMajor;
}

/// \ingroup storage_order
/// Returns whether the two bit-fields encode the same storage order
constexpr bool has_same_storage_order(int flags_a, int flags_b) {
  return get_storage_order(flags_a) == get_storage_order(flags_b);
}

/// \ingroup storage_order
/// Extracts the storage-order part from the `flags` bit field
constexpr unsigned int storage_order_flag(unsigned int flags) {
  return flags & RowMajorBit;
}

/// \ingroup storage_order
/// Returns the bit pattern corresponding to the storage order.
constexpr unsigned int storage_order_flag(StorageOrder order) {
  switch (order) {
    case StorageOrder::RowMajor:
      return RowMajorBit;
    case StorageOrder::ColMajor:
      return 0;
  }
}

/// \ingroup storage_order
/// Adjusts the bit field so that it corresponds to the given storage order
constexpr unsigned int with_storage_order(unsigned int flags, StorageOrder order) {
  return (flags & ~RowMajorBit) | storage_order_flag(order);
}
} // end namespace internal

using internal::StorageOrder;

} // end namespace Eigen

#endif // EIGEN_STORAGE_LAYOUT_H
