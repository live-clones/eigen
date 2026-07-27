// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Jack Feds <jackf10oh@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_DOUBLE_KRONECKER_H
#define EIGEN_DOUBLE_KRONECKER_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

// Forward declarations ---------------------------------------------
template<typename ArgTpe>
class DoubleKronecker;

namespace internal {

// type traits =======================================================================
template<class ArgType>
struct traits<DoubleKronecker<ArgType> > 
  : public traits<ArgType>
{
  enum {
    RowsAtCompileTime = Dynamic,
    ColsAtCompileTime = Dynamic,
    MaxRowsAtCompileTime = Dynamic,
    MaxColsAtCompileTime = Dynamic,
    Options = traits<ArgType>::Options,
    Flags = Options & (~NestByRefBit) & (~LvalueBit), // remove nest by ref, lvalue 
    SupportedAccessPatterns = InnerRandomAccessPattern
  };
};

} // end namespace internal

// expression class ======================================================================= 
template<class ArgType>
class DoubleKronecker : public Eigen::SparseMatrixBase< DoubleKronecker<ArgType> > 
{
  public:
    // typedefs
    typedef typename internal::ref_selector<ArgType>::type ArgTypeNested;
    using Base = Eigen::SparseMatrixBase< DoubleKronecker<ArgType> >;
    EIGEN_SPARSE_PUBLIC_INTERFACE(DoubleKronecker<ArgType>)

    // constructors 
    DoubleKronecker(const ArgType& arg_init, StorageIndex n, StorageIndex m)
      : m_arg(arg_init), m_prod_before(n), m_prod_after(m) 
    { eigen_assert((m>0) && (n>0)); }
    
    // member functions 
    EIGEN_STRONG_INLINE StorageIndex rows() const { return m_prod_before * m_arg.rows() * m_prod_after; }
    EIGEN_STRONG_INLINE StorageIndex cols() const { return m_prod_before * m_arg.cols() * m_prod_after; }

    // member data 
    ArgTypeNested m_arg;
    StorageIndex m_prod_before; 
    StorageIndex m_prod_after; 
};

namespace internal{

// the evaluator =======================================================================
template<typename ArgType>
struct evaluator< DoubleKronecker<ArgType> > : evaluator_base< DoubleKronecker<ArgType> > {

  // typedefs -------------------------------------------------- 
  typedef DoubleKronecker<ArgType> XprType;
  typedef typename nested_eval<ArgType, XprType::ColsAtCompileTime>::type ArgTypeNested;
  typedef typename remove_all<ArgTypeNested>::type ArgTypeNestedCleaned;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename XprType::StorageIndex StorageIndex; 
  typedef typename XprType::Scalar Scalar; 

  // Flags ------------------------------------------------------
  enum { CoeffReadCost = evaluator<ArgTypeNestedCleaned>::CoeffReadCost, Flags = traits<DoubleKronecker<ArgType>>::Flags };

  // custom InnerIterator ----------------------------------
  struct InnerIterator{
    // Constructor ================================================================
    InnerIterator(const evaluator& eval, Index outer_idx)
      : m_eval(eval), 
      m_outer_idx(outer_idx),
      m_offset(eval.m_prod_before * eval.m_xpr.m_arg.innerSize() * (outer_idx / (eval.m_prod_before * eval.m_xpr.m_arg.outerSize())) + outer_idx % eval.m_prod_before),
      m_wrapped_it(eval.m_argImpl, (outer_idx / eval.m_prod_before) % eval.m_xpr.m_arg.outerSize())
    {};

    // Member Funcs ===================================================
    EIGEN_STRONG_INLINE operator bool() const { return m_wrapped_it; }
    EIGEN_STRONG_INLINE void operator++(){ ++m_wrapped_it; }
    EIGEN_STRONG_INLINE StorageIndex row() const { return (traits<DoubleKronecker<ArgType>>::Flags & RowMajorBit) ? m_outer_idx : index(); }
    EIGEN_STRONG_INLINE StorageIndex col() const { return (traits<DoubleKronecker<ArgType>>::Flags & RowMajorBit) ? index() : m_outer_idx; }
    EIGEN_STRONG_INLINE StorageIndex index() const { return m_offset + m_wrapped_it.index() * m_eval.m_prod_before; }
    EIGEN_STRONG_INLINE Scalar value() const { return m_wrapped_it.value(); }

    // member data ------------------------------------------
    const evaluator& m_eval; 
    typename evaluator<ArgTypeNestedCleaned>::InnerIterator m_wrapped_it;
    StorageIndex m_outer_idx; 
    StorageIndex m_offset; 

  }; // end InnerIterator 

  // Constructors ======================================================== 
  evaluator(const XprType& xpr) 
    : m_argImpl(xpr.m_arg), 
    m_xpr(xpr), 
    m_prod_before(xpr.m_prod_before), 
    m_prod_after(xpr.m_prod_after)
  {};
 
  // Member Functions ========================================================
  EIGEN_STRONG_INLINE StorageIndex rows() const {return m_xpr.rows(); };
  EIGEN_STRONG_INLINE StorageIndex cols() const {return m_xpr.cols(); }; 
  EIGEN_STRONG_INLINE StorageIndex innerSize() const { return (traits<DoubleKronecker<ArgType>>::Flags & RowMajorBit) ? cols() : rows(); }
  EIGEN_STRONG_INLINE StorageIndex outerSize() const { return (traits<DoubleKronecker<ArgType>>::Flags & RowMajorBit) ? rows() : cols(); }
  EIGEN_STRONG_INLINE StorageIndex nonZerosEstimate() const { return m_xpr.nonZerosEstimate(); }
 
  // Member Data ------------------------------------------------------
  evaluator<ArgTypeNestedCleaned> m_argImpl;
  const XprType& m_xpr;  
  StorageIndex m_prod_before; 
  StorageIndex m_prod_after; 
};

} // end namespace internal

// the entry point ======================================================================= 
template<class ArgType>
DoubleKronecker<ArgType> make_DoubleKronecker(const SparseMatrixBase<ArgType>& arg, typename internal::traits<ArgType>::StorageIndex n, typename internal::traits<ArgType>::StorageIndex m) {
  return DoubleKronecker<ArgType>(arg.derived(), n, m);
}

} // end namespace Eigen

#endif // DoubleKronecker.hpp