// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Jack Feds <jackf10oh@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_LEFT_KRONECKER_H
#define EIGEN_LEFT_KRONECKER_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

// forward declaration ---------------------------------------------
template <typename ArgTpe>
class RightKroneckerImpl;

namespace internal {

// type traits =======================================================================
template <class ArgType>
struct traits<RightKroneckerImpl<ArgType>> : public traits<ArgType> {
  enum {
    RowsAtCompileTime = Dynamic,
    ColsAtCompileTime = Dynamic,
    MaxRowsAtCompileTime = Dynamic,
    MaxColsAtCompileTime = Dynamic,
    Options = traits<ArgType>::Options,
    Flags = Options & (~NestByRefBit) & (~LvalueBit),  // remove nest by ref, lvalue
    SupportedAccessPatterns = InnerRandomAccessPattern
  };
};

}  // end namespace internal

// expression class =======================================================================
template <class ArgType>
class RightKroneckerImpl : public Eigen::SparseMatrixBase<RightKroneckerImpl<ArgType>> {
 public:
  using ArgTypeNested = typename internal::ref_selector<ArgType>::type;
  using Base = Eigen::SparseMatrixBase<RightKroneckerImpl<ArgType>>;
  EIGEN_SPARSE_PUBLIC_INTERFACE(RightKroneckerImpl<ArgType>)

  RightKroneckerImpl(const ArgType& arg_init, StorageIndex n) : m_arg(arg_init), m_prod_before(n) {
    eigen_assert(n > 0);
  }

  EIGEN_STRONG_INLINE StorageIndex rows() const { return m_prod_before * m_arg.rows(); }
  EIGEN_STRONG_INLINE StorageIndex cols() const { return m_prod_before * m_arg.cols(); }

  ArgTypeNested m_arg;
  StorageIndex m_prod_before;
};

namespace internal {

// the evaluator =======================================================================
template <typename ArgType>
struct evaluator<RightKroneckerImpl<ArgType>> : evaluator_base<RightKroneckerImpl<ArgType>> {
  using XprType = RightKroneckerImpl<ArgType>;
  using ArgTypeNested = typename nested_eval<ArgType, XprType::ColsAtCompileTime>::type;
  using ArgTypeNestedCleaned = typename remove_all<ArgTypeNested>::type;
  using CoeffReturnType = typename XprType::CoeffReturnType;
  using StorageIndex = typename XprType::StorageIndex;
  using Scalar = typename XprType::Scalar;

  enum {
    CoeffReadCost = evaluator<ArgTypeNestedCleaned>::CoeffReadCost,
    Flags = traits<RightKroneckerImpl<ArgType>>::Flags
  };

  // custom InnerIterator ----------------------------------
  struct InnerIterator {
    InnerIterator(const evaluator& eval, Index outer_idx)
        : m_eval(eval),
          m_outer_idx(outer_idx),
          m_offset(outer_idx % eval.m_prod_before),
          m_wrapped_it(eval.m_argImpl, outer_idx / eval.m_prod_before){};

    EIGEN_STRONG_INLINE operator bool() const { return m_wrapped_it; }
    EIGEN_STRONG_INLINE void operator++() { ++m_wrapped_it; }
    EIGEN_STRONG_INLINE StorageIndex row() const {
      return (traits<RightKroneckerImpl<ArgType>>::Flags & RowMajorBit) ? m_outer_idx : index();
    }
    EIGEN_STRONG_INLINE StorageIndex col() const {
      return (traits<RightKroneckerImpl<ArgType>>::Flags & RowMajorBit) ? index() : m_outer_idx;
    }
    EIGEN_STRONG_INLINE StorageIndex index() const { return m_offset + m_wrapped_it.index() * m_eval.m_prod_before; }
    EIGEN_STRONG_INLINE Scalar value() const { return m_wrapped_it.value(); }

    const evaluator& m_eval;
    StorageIndex m_outer_idx;
    StorageIndex m_offset;
    typename evaluator<ArgTypeNestedCleaned>::InnerIterator m_wrapped_it;
  };

  evaluator(const XprType& xpr) : m_argImpl(xpr.m_arg), m_xpr(xpr), m_prod_before(xpr.m_prod_before){};

  EIGEN_STRONG_INLINE StorageIndex rows() const { return m_xpr.rows(); };
  EIGEN_STRONG_INLINE StorageIndex cols() const { return m_xpr.cols(); };
  EIGEN_STRONG_INLINE StorageIndex innerSize() const {
    return (traits<RightKroneckerImpl<ArgType>>::Flags & RowMajorBit) ? cols() : rows();
  }
  EIGEN_STRONG_INLINE StorageIndex outerSize() const {
    return (traits<RightKroneckerImpl<ArgType>>::Flags & RowMajorBit) ? rows() : cols();
  }
  EIGEN_STRONG_INLINE StorageIndex nonZerosEstimate() const { return m_xpr.nonZerosEstimate(); }

  evaluator<ArgTypeNestedCleaned> m_argImpl;
  const XprType& m_xpr;
  StorageIndex m_prod_before;
};

}  // end namespace internal

// the entry point =======================================================================
template <class ArgType>
RightKroneckerImpl<ArgType> RightKronecker(const SparseMatrixBase<ArgType>& arg,
                                           typename internal::traits<ArgType>::StorageIndex n) {
  return RightKroneckerImpl<ArgType>(arg.derived(), n);
}

}  // end namespace Eigen

#endif  // RightKronecker.h