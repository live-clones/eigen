// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SPARSE_PERMUTATION_H
#define EIGEN_SPARSE_PERMUTATION_H

// This file implements sparse * permutation products

#include "./InternalHeaderCheck.h"

namespace Eigen { 

namespace internal {

template<typename ExpressionType, typename PlainObjectType, bool NeedEval = !is_same<ExpressionType, PlainObjectType>::value>
struct XprHelper
{
    XprHelper(const ExpressionType& xpr) : m_xpr(xpr) {}
    inline const PlainObjectType& xpr() const { return m_xpr; }
    const PlainObjectType m_xpr;
};
template<typename ExpressionType, typename PlainObjectType>
struct XprHelper<ExpressionType, PlainObjectType, false>
{
    XprHelper(const ExpressionType& xpr) : m_xpr(xpr) {}
    inline const PlainObjectType& xpr() const { return m_xpr; }
    const PlainObjectType& m_xpr;
};

template<typename PermDerived, bool NeedInverseEval>
struct PermHelper
{
    typedef typename PermDerived::IndicesType IndicesType;
    typedef typename IndicesType::Scalar StorageIndex;
    typedef typename PermutationMatrix<IndicesType::SizeAtCompileTime, IndicesType::MaxSizeAtCompileTime, StorageIndex> type;
    PermHelper(const PermDerived& perm) : m_perm(perm.derived().inverse()) {}
    inline const type& perm() const { return m_perm; }
    const type m_perm;
};
template<typename PermDerived>
struct PermHelper<PermDerived, false>
{
    typedef PermDerived type;
    PermHelper(const PermDerived& perm) : m_perm(perm) {}
    inline const type& perm() const { return m_perm; }
    const type& m_perm;
};

template<typename ExpressionType, int Side, bool Transposed>
struct permutation_matrix_product<ExpressionType, Side, Transposed, SparseShape>
{
    typedef typename nested_eval<ExpressionType, 1>::type MatrixType;
    typedef remove_all_t<MatrixType> MatrixTypeCleaned;

    typedef typename MatrixTypeCleaned::Scalar Scalar;
    typedef typename MatrixTypeCleaned::StorageIndex StorageIndex;

    // the actual "return type" is `Dest`. this is a temporary type
    typedef SparseMatrix<Scalar, MatrixTypeCleaned::IsRowMajor ? RowMajor : ColMajor, StorageIndex> ReturnType;
    typedef XprHelper<ExpressionType, ReturnType> XprHelper;

    static constexpr bool OuterPermutation = ExpressionType::IsRowMajor ? Side == OnTheLeft : Side == OnTheRight;
    static constexpr bool InversePermutation = Transposed ? Side == OnTheLeft : Side == OnTheRight;

    template <typename Dest, typename PermutationType>
    static inline void run(Dest& dst, const PermutationType& perm, const ExpressionType& xpr) {
      typedef PermHelper<PermutationType, InversePermutation && !OuterPermutation> InnerPermHelper;
      typedef typename InnerPermHelper::type InnerPermType;

      // if ExpressionType is not ReturnType, evaluate `xpr` (allocation)
      // otherwise, just reference `xpr`
      // TODO: handle trivial expressions such as CwiseBinaryOp without temporary
      const XprHelper xprHelper(xpr);
      const ReturnType& tmp = xprHelper.xpr();

      // if inverse permutation of inner indices is requested, calculate perm.inverse() (allocation)
      // otherwise, just reference `perm`
      const InnerPermHelper permHelper(perm);
      const InnerPermType& innerPerm = permHelper.perm();

      ReturnType result(tmp.rows(), tmp.cols());

      for (Index j = 0; j < tmp.outerSize(); j++) {
        Index jp = OuterPermutation ? perm.indices().coeff(j) : j;
        Index jsrc = InversePermutation ? jp : j;
        Index jdst = InversePermutation ? j : jp;
        Index begin = tmp.outerIndexPtr()[jsrc];
        Index end = tmp.isCompressed() ? tmp.outerIndexPtr()[jsrc + 1] : begin + tmp.innerNonZeroPtr()[jsrc];
        result.outerIndexPtr()[jdst + 1] += end - begin;
      }

      std::partial_sum(result.outerIndexPtr(), result.outerIndexPtr() + result.outerSize() + 1, result.outerIndexPtr());
      result.resizeNonZeros(result.nonZeros());

      for (Index j = 0; j < tmp.outerSize(); j++) {
        Index jp = OuterPermutation ? perm.indices().coeff(j) : j;
        Index jsrc = InversePermutation ? jp : j;
        Index jdst = InversePermutation ? j : jp;
        Index begin = tmp.outerIndexPtr()[jsrc];
        Index end = tmp.isCompressed() ? tmp.outerIndexPtr()[jsrc + 1] : begin + tmp.innerNonZeroPtr()[jsrc];
        Index target = result.outerIndexPtr()[jdst];
        if (OuterPermutation)
          smart_copy(tmp.innerIndexPtr() + begin, tmp.innerIndexPtr() + end, result.innerIndexPtr() + target);
        else
          std::transform(tmp.innerIndexPtr() + begin, tmp.innerIndexPtr() + end, result.innerIndexPtr() + target,
                         [&innerPerm](Index i) { return innerPerm.indices().coeff(i); });
        smart_copy(tmp.valuePtr() + begin, tmp.valuePtr() + end, result.valuePtr() + target);
      }
      // if the inner indices were permuted, they must be sorted
      if (!OuterPermutation) result.sortInnerIndices();
      dst = std::move(result);
    }
};

}

namespace internal {

template <int ProductTag> struct product_promote_storage_type<Sparse,             PermutationStorage, ProductTag> { typedef Sparse ret; };
template <int ProductTag> struct product_promote_storage_type<PermutationStorage, Sparse,             ProductTag> { typedef Sparse ret; };

// TODO, the following two overloads are only needed to define the right temporary type through 
// typename traits<permutation_sparse_matrix_product<Rhs,Lhs,OnTheRight,false> >::ReturnType
// whereas it should be correctly handled by traits<Product<> >::PlainObject

template<typename Lhs, typename Rhs, int ProductTag>
struct product_evaluator<Product<Lhs, Rhs, AliasFreeProduct>, ProductTag, PermutationShape, SparseShape>
  : public evaluator<typename permutation_matrix_product<Rhs,OnTheLeft,false,SparseShape>::ReturnType>
{
  typedef Product<Lhs, Rhs, AliasFreeProduct> XprType;
  typedef typename permutation_matrix_product<Rhs,OnTheLeft,false,SparseShape>::ReturnType PlainObject;
  typedef evaluator<PlainObject> Base;

  enum {
    Flags = Base::Flags | EvalBeforeNestingBit
  };

  explicit product_evaluator(const XprType& xpr)
    : m_result(xpr.rows(), xpr.cols())
  {
    internal::construct_at<Base>(this, m_result);
    generic_product_impl<Lhs, Rhs, PermutationShape, SparseShape, ProductTag>::evalTo(m_result, xpr.lhs(), xpr.rhs());
  }

protected:
  PlainObject m_result;
};

template<typename Lhs, typename Rhs, int ProductTag>
struct product_evaluator<Product<Lhs, Rhs, AliasFreeProduct>, ProductTag, SparseShape, PermutationShape >
  : public evaluator<typename permutation_matrix_product<Lhs,OnTheRight,false,SparseShape>::ReturnType>
{
  typedef Product<Lhs, Rhs, AliasFreeProduct> XprType;
  typedef typename permutation_matrix_product<Lhs,OnTheRight,false,SparseShape>::ReturnType PlainObject;
  typedef evaluator<PlainObject> Base;

  enum {
    Flags = Base::Flags | EvalBeforeNestingBit
  };

  explicit product_evaluator(const XprType& xpr)
    : m_result(xpr.rows(), xpr.cols())
  {
    ::new (static_cast<Base*>(this)) Base(m_result);
    generic_product_impl<Lhs, Rhs, SparseShape, PermutationShape, ProductTag>::evalTo(m_result, xpr.lhs(), xpr.rhs());
  }

protected:
  PlainObject m_result;
};

} // end namespace internal

/** \returns the matrix with the permutation applied to the columns
  */
template<typename SparseDerived, typename PermDerived>
inline const Product<SparseDerived, PermDerived, AliasFreeProduct>
operator*(const SparseMatrixBase<SparseDerived>& matrix, const PermutationBase<PermDerived>& perm)
{ return Product<SparseDerived, PermDerived, AliasFreeProduct>(matrix.derived(), perm.derived()); }

/** \returns the matrix with the permutation applied to the rows
  */
template<typename SparseDerived, typename PermDerived>
inline const Product<PermDerived, SparseDerived, AliasFreeProduct>
operator*( const PermutationBase<PermDerived>& perm, const SparseMatrixBase<SparseDerived>& matrix)
{ return  Product<PermDerived, SparseDerived, AliasFreeProduct>(perm.derived(), matrix.derived()); }


/** \returns the matrix with the inverse permutation applied to the columns.
  */
template<typename SparseDerived, typename PermutationType>
inline const Product<SparseDerived, Inverse<PermutationType>, AliasFreeProduct>
operator*(const SparseMatrixBase<SparseDerived>& matrix, const InverseImpl<PermutationType, PermutationStorage>& tperm)
{ return Product<SparseDerived, Inverse<PermutationType>, AliasFreeProduct>(matrix.derived(), tperm.derived()); }

/** \returns the matrix with the inverse permutation applied to the rows.
  */
template<typename SparseDerived, typename PermutationType>
inline const Product<Inverse<PermutationType>, SparseDerived, AliasFreeProduct>
operator*(const InverseImpl<PermutationType,PermutationStorage>& tperm, const SparseMatrixBase<SparseDerived>& matrix)
{ return Product<Inverse<PermutationType>, SparseDerived, AliasFreeProduct>(tperm.derived(), matrix.derived()); }

} // end namespace Eigen

#endif // EIGEN_SPARSE_SELFADJOINTVIEW_H
