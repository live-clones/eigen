// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2022 Erik Schultheis <erik.schultheis@aalto.fi>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

namespace Eigen {
namespace internal {
  inline std::ostream& operator<<(std::ostream& stream, StorageOrder order) {
    switch (order) {
      case StorageOrder::RowMajor:
        return stream << "RowMajor";
      case StorageOrder::ColMajor:
        return stream << "ColMajor";
    }
  }
}
}

EIGEN_DECLARE_TEST(storage_order)
{
  using namespace Eigen::internal;
  VERIFY( is_row_major(StorageOptions::RowMajor) );
  VERIFY( is_row_major(StorageOrder::RowMajor) );
  VERIFY( !is_row_major(StorageOptions::ColMajor) );
  VERIFY( !is_row_major(StorageOrder::ColMajor) );

  VERIFY( !is_col_major(StorageOptions::RowMajor) );
  VERIFY( !is_col_major(StorageOrder::RowMajor) );
  VERIFY( is_col_major(StorageOptions::ColMajor) );
  VERIFY( is_col_major(StorageOrder::ColMajor) );

  VERIFY_IS_EQUAL(get_storage_order(StorageOptions::RowMajor), StorageOrder::RowMajor);
  VERIFY_IS_EQUAL(get_storage_order(StorageOptions::ColMajor), StorageOrder::ColMajor);

  VERIFY_IS_EQUAL(storage_order_flag(StorageOrder::RowMajor), StorageOptions::RowMajor);
  VERIFY_IS_EQUAL(storage_order_flag(StorageOrder::ColMajor), StorageOptions::ColMajor);
  VERIFY_IS_EQUAL(storage_order_flag(StorageOptions::RowMajor), StorageOptions::RowMajor);
  VERIFY_IS_EQUAL(storage_order_flag(StorageOptions::ColMajor), StorageOptions::ColMajor);
  // check layout bit extraction with other options present.
  VERIFY_IS_EQUAL(storage_order_flag(StorageOptions::RowMajor | StorageOptions::DontAlign), StorageOptions::RowMajor);
  VERIFY_IS_EQUAL(storage_order_flag(StorageOptions::ColMajor | StorageOptions::DontAlign), StorageOptions::ColMajor);

  // check layout bit flip without disturbing other options
  VERIFY_IS_EQUAL(with_storage_order(StorageOptions::ColMajor | StorageOptions::DontAlign, StorageOrder::RowMajor),
                  StorageOptions::RowMajor | StorageOptions::DontAlign);
  VERIFY_IS_EQUAL(with_storage_order(StorageOptions::RowMajor | StorageOptions::DontAlign, StorageOrder::ColMajor),
                  StorageOptions::ColMajor | StorageOptions::DontAlign);
  
  // transpose operation
  VERIFY_IS_EQUAL(transposed(StorageOrder::RowMajor), StorageOrder::ColMajor);
  VERIFY_IS_EQUAL(transposed(StorageOrder::ColMajor), StorageOrder::RowMajor);

  // same storage order
  VERIFY(has_same_storage_order(storage_order_flag(StorageOrder::ColMajor) | StorageOptions::DontAlign,
                                storage_order_flag(StorageOrder::ColMajor)));
  VERIFY(!has_same_storage_order(storage_order_flag(StorageOrder::ColMajor) | StorageOptions::DontAlign,
                                 storage_order_flag(StorageOrder::RowMajor)));
  VERIFY(has_same_storage_order(storage_order_flag(StorageOrder::RowMajor),
                                storage_order_flag(StorageOrder::RowMajor) | StorageOptions::DontAlign));
  VERIFY(!has_same_storage_order(storage_order_flag(StorageOrder::RowMajor),
                                 storage_order_flag(StorageOrder::ColMajor) | StorageOptions::DontAlign));
}