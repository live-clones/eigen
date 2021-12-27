// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015-2019 Godeffroy Valet <godeffroy.valet@m4x.org>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_INDEXED_H
#define EIGEN_CXX11_TENSOR_TENSOR_INDEXED_H 

#include "TensorDimensions.h"

#include <array>
#include <type_traits>
#include <utility>

namespace Eigen {

/** \file 
  * \ingroup CXX11_Tensor_Module
  * 
  * The following example shows how and why to use the indexed notation :
  * \include TensorIndexed.cpp
  * Output: 
  * \verbinclude TensorIndexed.out
  *
  * This feature requires C++14.
  *
  */

//! Base class of all TensorIndex
struct TensorIndexBase{};
//! In a tensor expression such as "A(i,j)*B(j,k)", i,j,k are each a TensorIndex with unique id_
template<int id_>
struct TensorIndex:TensorIndexBase{};

// We define some common indices which can be imported with "using namespace Eigen::tensor_indices;"
namespace tensor_indices {
  EIGEN_UNUSED static const TensorIndex<'i'> i;
  EIGEN_UNUSED static const TensorIndex<'j'> j;
  EIGEN_UNUSED static const TensorIndex<'k'> k;
  EIGEN_UNUSED static const TensorIndex<'l'> l;
  EIGEN_UNUSED static const TensorIndex<'m'> m;
  EIGEN_UNUSED static const TensorIndex<'n'> n;
  EIGEN_UNUSED static const TensorIndex<'w'> w;
  EIGEN_UNUSED static const TensorIndex<'x'> x;
  EIGEN_UNUSED static const TensorIndex<'y'> y;
  EIGEN_UNUSED static const TensorIndex<'z'> z;
}

using DimensionIndex = unsigned;

//! BoundTensorIndex<id,dim> links TensorIndex<id> to the dim-th dimension of a Tensor
template<int id, DimensionIndex dim>
struct BoundTensorIndex{};

namespace internal {
  //! counts the number of arguments which are true, at compile time if possible
  constexpr unsigned static_count()
  {
	  return 0;
  }
  template<typename BoolT, typename... BoolTs>
  constexpr unsigned static_count(BoolT b, BoolTs... bools)
  {
	  return (b ? 1 : 0) + static_count(bools...);
  }

  //! A tool to keep a list of indices sorted at compile time
  template<typename... Indices>
  struct sorted_indices_t 
  {
    template<typename Index>
    auto insert(Index index) 
    {
      return sorted_indices_insert(sorted_indices_t<>{}, *this, index);
    }
  };

  //linear insert, todo: dichotomic insert, there probably is an implementation of this somewhere
  template<int... SmallerIndexIds, int IndexId, int... IndexIds, 
           DimensionIndex... PrevBoundDims, DimensionIndex BoundDim, DimensionIndex... BoundDims, 
           int NewIndexId, DimensionIndex NewBoundDim, 
           typename EnableIf=typename internal::enable_if<(NewIndexId <= IndexId)>::type>
  auto sorted_indices_insert(sorted_indices_t<BoundTensorIndex<SmallerIndexIds, PrevBoundDims>...>,
              sorted_indices_t<BoundTensorIndex<IndexId, BoundDim>, 
                               BoundTensorIndex<IndexIds, BoundDims>...>,
                               BoundTensorIndex<NewIndexId, NewBoundDim>) 
  {
    EIGEN_STATIC_ASSERT(NewIndexId != IndexId, "sorted_indices_t cannot have duplicates");
    return sorted_indices_t<BoundTensorIndex<SmallerIndexIds, PrevBoundDims>..., 
                            BoundTensorIndex<NewIndexId, NewBoundDim>,
                            BoundTensorIndex<IndexId, BoundDim>,
                            BoundTensorIndex<IndexIds, BoundDims>...>{};
  }
  template<int... SmallerIndexIds, int IndexId, int... IndexIds, 
           DimensionIndex... PrevBoundDims, DimensionIndex BoundDim, DimensionIndex... BoundDims, 
           int NewIndexId, DimensionIndex NewBoundDim,
           typename internal::enable_if<(NewIndexId > IndexId), bool>::type=true>
  auto sorted_indices_insert(sorted_indices_t<BoundTensorIndex<SmallerIndexIds, PrevBoundDims>...>,
                             sorted_indices_t<BoundTensorIndex<IndexId, BoundDim>, 
                             BoundTensorIndex<IndexIds, BoundDims>...>,
                             BoundTensorIndex<NewIndexId, NewBoundDim>) 
  {
    return sorted_indices_insert(sorted_indices_t<BoundTensorIndex<SmallerIndexIds, PrevBoundDims>..., 
                                 BoundTensorIndex<IndexId, BoundDim>>{},
                                 sorted_indices_t<BoundTensorIndex<IndexIds, BoundDims>...>{},
                                 BoundTensorIndex<NewIndexId, NewBoundDim>{});
  }
  template<int... SmallerIndexIds, 
           DimensionIndex... PrevBoundDims, 
           int NewIndexId, DimensionIndex NewBoundDim>
  auto sorted_indices_insert(sorted_indices_t<BoundTensorIndex<SmallerIndexIds, PrevBoundDims>...>,
                             sorted_indices_t<>,
                             BoundTensorIndex<NewIndexId, NewBoundDim>) 
  {
    return sorted_indices_t<BoundTensorIndex<SmallerIndexIds, PrevBoundDims>..., 
                            BoundTensorIndex<NewIndexId, NewBoundDim>>{};
  }

  template<int... IndexIds2, DimensionIndex... BoundDims2>
  auto sorted_indices_merge(sorted_indices_t<>, sorted_indices_t<BoundTensorIndex<IndexIds2, BoundDims2>...> b)
  {
    return b;
  }
  template<int IndexId, DimensionIndex BoundDim, 
           int... IndexIds1, DimensionIndex... BoundDims1, 
           int... IndexIds2, DimensionIndex... BoundDims2>
  auto sorted_indices_merge(sorted_indices_t<BoundTensorIndex<IndexId, BoundDim>, BoundTensorIndex<IndexIds1, BoundDims1>...>, 
                            sorted_indices_t<BoundTensorIndex<IndexIds2, BoundDims2>...> b) 
  {
    return sorted_indices_merge(sorted_indices_t<BoundTensorIndex<IndexIds1, BoundDims1>...>{}, 
                                b.insert(BoundTensorIndex<IndexId, BoundDim>{}));
  }

  //! Transforms any T const& to T& and leaves anything else unchanged
  template<typename T>
  struct const_ref_to_ref
  {
    using type = T;
  };
  template<typename T>
  struct const_ref_to_ref<T const&>
  {
    using type = T&;
  };
  template<typename T>
  using const_ref_to_ref_t = typename const_ref_to_ref<T>::type;
}

//! \brief This class stores any indexed tensor expression such as my_tensor_expression(i,k,j) or A(i,j)*B(j,k)+C(i,k).
//!
//! If TensorExpr is an actual tensor, a reference to it is stored to allow assignment.
//! If TensorExpr is an expression, a copy of it is stored because the expression might be a temporary object.
//! In any case, BoundIndices keep track of which TensorIndex is linked to which dimension of the TensorExpr.
//! BoundIndices are always kept sorted by the ids of their TensorIndex.
//! For example, the expression A(i,k,j) will have {i:0,j:2,k:1} as BoundIndices
//! The fact that they are sorted makes it easy to ensure that operands of = or + operations share the same set of indices
//! It also makes it easy to shuffle the dimensions of those operands before calling the underlying operations
template<typename TensorExpr, typename... BoundIndices>
class IndexedTensor
{
 public:

  typedef internal::const_ref_to_ref_t<typename internal::remove_const<typename TensorExpr::Nested>::type> _TensorExpr;

  const _TensorExpr& expression() const { return m_tensor_expr; }
        _TensorExpr& expression()       { return m_tensor_expr; }

  IndexedTensor(TensorExpr&& t) : m_tensor_expr(std::move(t)){}
  IndexedTensor(TensorExpr&  t) : m_tensor_expr(t){}
  // IndexedTensor(TensorExpr&& t) = default;
  // IndexedTensor(TensorExpr&  t) = default;

  template<typename OtherTensorExpr, typename... OtherBoundIndices>
  void operator=(IndexedTensor<OtherTensorExpr, OtherBoundIndices...> const& other);
  template<typename OtherTensorExpr, typename... OtherBoundIndices>
  void operator=(IndexedTensor<OtherTensorExpr, OtherBoundIndices...> const&& other);
  template<typename OtherTensorExpr, typename... OtherBoundIndices>
  void operator+=(IndexedTensor<OtherTensorExpr, OtherBoundIndices...> const& other);
  template<typename OtherTensorExpr, typename... OtherBoundIndices>
  void operator-=(IndexedTensor<OtherTensorExpr, OtherBoundIndices...> const& other);

 private:
  _TensorExpr m_tensor_expr;
};

//! An IndexedTensor with no indices is a tensor of order zero, which is a scalar. 
//! So, it can be assigned to a scalar instead of a tensor.
template<typename TensorExpr>
class IndexedTensor<TensorExpr>
{
 public:
  typedef internal::const_ref_to_ref_t<typename TensorExpr::Nested> _TensorExpr;

  IndexedTensor(TensorExpr&& t) : m_tensor_expr(std::forward<TensorExpr>(t)){}

  using Scalar = typename TensorExpr::Scalar;
  static const int DataLayout = internal::traits<TensorExpr>::Layout;

  operator Scalar() const
  {
      TensorFixedSize<Scalar, Sizes<>, DataLayout> result = m_tensor_expr;
      return *result.data();
  }

 private:
  _TensorExpr m_tensor_expr;
};

namespace internal {

  // chipping - done with the operator()

  //! \brief A helper function to TensorBase::operator()->TensorIndexed.
  //!
  //! This is the exit point of make_indexed_tensor(), which is called by TensorBase::operator().
  //! All dimensions are either chipped or linked to a TensorIndex, so we can constructs the IndexedTensor
  template<typename TensorExpr, int... IndexIds, DimensionIndex... BoundDims> 
  auto make_indexed_tensor(TensorExpr&& tensor_expr, 
                           internal::sorted_indices_t<BoundTensorIndex<IndexIds, BoundDims>...>)
  {
    return IndexedTensor<typename std::remove_reference<TensorExpr>::type, 
                         BoundTensorIndex<IndexIds, BoundDims>...>(std::forward<TensorExpr>(tensor_expr));
  }
  //! Some dimensions remain to be chipped or linked to a TensorIndex.
  //! If the current_index is a number, we chip the tensor_expr immediately, and continue to iterate over the remaining dimensions
  template<typename TensorExpr, typename... PrevIndices, typename... Indices> inline 
  auto make_indexed_tensor(TensorExpr&& tensor_expr,
                           internal::sorted_indices_t<PrevIndices...> prev_indices, 
                           DimensionIndex current_index, Indices... indices)
  {
    return make_indexed_tensor(tensor_expr.chip(current_index, sizeof...(PrevIndices)), 
                               prev_indices, indices...);
  }
  //! Some dimensions remain to be chipped or linked to a TensorIndex.
  //! If the current_index is a TensorIndex, we link it to the current dimension and continue to iterate over the remaining dimensions
  template<typename TensorExpr, typename... PrevIndices, typename... Indices, int id> inline 
  auto make_indexed_tensor(TensorExpr&& tensor_expr, 
                           internal::sorted_indices_t<PrevIndices...> prev_indices, 
                           TensorIndex<id>, Indices... indices)
  {
    return make_indexed_tensor(std::forward<TensorExpr>(tensor_expr), 
                               prev_indices.insert(BoundTensorIndex<id, sizeof...(PrevIndices)>{}), 
                               indices...);
  }

  // transposing, assignment - done with the operator=

  //! \brief A helper function to IndexedTensor::operator=.
  //!
  //! BoundTensorIndex's are used to shuffle operands in a specific order (sorted by the ids of the TensorIndex's),
  //! then a simple tensor assignment is made.
  template<typename TensorExpr, int... IndexIds, DimensionIndex... BoundDims, 
           typename OtherTensorExpr, /*same IndexIds*/ DimensionIndex... OtherBoundDims> 
  void indexed_tensor_assign(IndexedTensor<TensorExpr, BoundTensorIndex<IndexIds, BoundDims>...>* this_, 
                             IndexedTensor<OtherTensorExpr, BoundTensorIndex<IndexIds, OtherBoundDims>...> const& other)
  {
    // this_->tensor_expr.shuffle(internal::make_array(static_cast<int>(BoundDims)...))
    //   = other.tensor_expr.shuffle(internal::make_array(static_cast<int>(OtherBoundDims)...));
    this_->expression().shuffle(internal::make_array(static_cast<int>(BoundDims)...))
      = other.expression().shuffle(internal::make_array(static_cast<int>(OtherBoundDims)...));
  }

} // namespace internal

template<typename TensorExpr, typename... BoundIndices> 
template<typename OtherTensorExpr, typename... OtherBoundIndices>
void IndexedTensor<TensorExpr, BoundIndices...>::operator=(IndexedTensor<OtherTensorExpr, OtherBoundIndices...> const&& other)
{
  internal::indexed_tensor_assign(this, other);
}
template<typename TensorExpr, typename... BoundIndices> 
template<typename OtherTensorExpr, typename... OtherBoundIndices>
void IndexedTensor<TensorExpr, BoundIndices...>::operator=(IndexedTensor<OtherTensorExpr, OtherBoundIndices...> const& other)
{
  internal::indexed_tensor_assign(this, other);
}

namespace internal {

  // contraction, tensor product - done with the operator*

  //! Calculates new dimension indices after a contraction
  //! During a contraction, some dimensions disappear.
  //! For example A(i,j,k)*B(j) contracts the dimension j.
  //! BoundIndices are {i:0,j:1,k:2} for A and {j:0} for B, but after contration,
  //! BoundIndices are {i:0,k:1} for A and {} for B, and those need to be computed
  //! This function takes the RemovedDims {1} for A or {0} for B, and one 
  //! remaining_dim at a time, and it returns the updated dimension index 
  template<DimensionIndex... RemovedDims>
  constexpr DimensionIndex update_dim_index(DimensionIndex remaining_dim)
  {
    return remaining_dim - internal::static_count((remaining_dim > RemovedDims)...);
  }

  template<DimensionIndex... Ints>
  using DimensionIds = std::integer_sequence<DimensionIndex, Ints...>;
  template<int... Ints>
  using TensorIndexIds = std::integer_sequence<int, Ints...>;

  //! \brief A helper function to IndexedTensor::operator*.
  //!
  //! This is the exit point of make_contraction_indices().
  //! It returns the list of pairs of dimensions to be contracted,
  //! and the BoundIndices of the resulting expression.
  //! Indices remaining from the first operand are put first.
  template<int... UncontractedIndexIds1, DimensionIndex... UncontractedBoundDims1, 
           int... UncontractedIndexIds2, DimensionIndex... UncontractedBoundDims2,
           DimensionIndex... ContractedBoundDims1,
           DimensionIndex... ContractedBoundDims2>
  auto make_contraction_indices(TensorIndexIds<>, DimensionIds<>, 
                                TensorIndexIds<>, DimensionIds<>, 
                                TensorIndexIds<UncontractedIndexIds1...>, DimensionIds<UncontractedBoundDims1...>, 
                                TensorIndexIds<UncontractedIndexIds2...>, DimensionIds<UncontractedBoundDims2...>,
                                DimensionIds<ContractedBoundDims1...>, DimensionIds<ContractedBoundDims2...>)
  {
    return std::make_pair(
      std::array<Eigen::IndexPair<int>, sizeof...(ContractedBoundDims1)>{
        {Eigen::IndexPair<int>{static_cast<int>(ContractedBoundDims1), static_cast<int>(ContractedBoundDims2)}...}},
        sorted_indices_merge(
          internal::sorted_indices_t<BoundTensorIndex<UncontractedIndexIds1, update_dim_index<ContractedBoundDims1...>(UncontractedBoundDims1)>...>{},
           internal::sorted_indices_t<BoundTensorIndex<UncontractedIndexIds2, update_dim_index<ContractedBoundDims2...>(UncontractedBoundDims2) + sizeof...(UncontractedBoundDims1)>...>{})
      );
  }
  //! If all dimensions from one operand have been treated, there is nothing left to be contracted,
  //! so we put all remaing indices from the other operand as uncontracted indices
  template<int... IndexIds1, DimensionIndex... BoundDims1,
           int... UncontractedIndexIds1, DimensionIndex... UncontractedBoundDims1, 
           int... UncontractedIndexIds2, DimensionIndex... UncontractedBoundDims2,
           DimensionIndex... ContractedBoundDims1,
           DimensionIndex... ContractedBoundDims2>
  auto make_contraction_indices(TensorIndexIds<IndexIds1...>, DimensionIds<BoundDims1...>, 
                                TensorIndexIds<            >, DimensionIds<             >, 
                                TensorIndexIds<UncontractedIndexIds1...>, DimensionIds<UncontractedBoundDims1...>, 
                                TensorIndexIds<UncontractedIndexIds2...>, DimensionIds<UncontractedBoundDims2...>,
                                DimensionIds<ContractedBoundDims1...>, DimensionIds<ContractedBoundDims2...>)
  {
    return make_contraction_indices(
        TensorIndexIds<>{}, DimensionIds<>{}, 
        TensorIndexIds<>{}, DimensionIds<>{}, 
        TensorIndexIds<UncontractedIndexIds1..., IndexIds1...>{}, DimensionIds<UncontractedBoundDims1..., BoundDims1...>{},
        TensorIndexIds<UncontractedIndexIds2...              >{}, DimensionIds<UncontractedBoundDims2...               >{},
        DimensionIds<ContractedBoundDims1...>{}, 
        DimensionIds<ContractedBoundDims2...>{});
  }
  //! If all dimensions from one operand have been treated, there is nothing left to be contracted,
  //! so we put all remaing indices from the other operand as uncontracted indices
  template<int... IndexIds2, DimensionIndex... BoundDims2,
           int... UncontractedIndexIds1, DimensionIndex... UncontractedBoundDims1, 
           int... UncontractedIndexIds2, DimensionIndex... UncontractedBoundDims2,
           DimensionIndex... ContractedBoundDims1,
           DimensionIndex... ContractedBoundDims2>
  auto make_contraction_indices(TensorIndexIds<            >, DimensionIds<             >, 
                                TensorIndexIds<IndexIds2...>, DimensionIds<BoundDims2...>, 
                                TensorIndexIds<UncontractedIndexIds1...>, DimensionIds<UncontractedBoundDims1...>, 
                                TensorIndexIds<UncontractedIndexIds2...>, DimensionIds<UncontractedBoundDims2...>,
                                DimensionIds<ContractedBoundDims1...>, DimensionIds<ContractedBoundDims2...>)
  {
    return make_contraction_indices(
        TensorIndexIds<>{}, DimensionIds<>{}, 
        TensorIndexIds<>{}, DimensionIds<>{}, 
        TensorIndexIds<UncontractedIndexIds1...              >{}, DimensionIds<UncontractedBoundDims1...               >{}, 
        TensorIndexIds<UncontractedIndexIds2..., IndexIds2...>{}, DimensionIds<UncontractedBoundDims2..., BoundDims2...>{}, 
        DimensionIds<ContractedBoundDims1...>{}, 
        DimensionIds<ContractedBoundDims2...>{});
  }
  // forward declaration
  template<int IndexId1, int... IndexIds1, DimensionIndex BoundDim1, DimensionIndex... BoundDims1, 
           int IndexId2, int... IndexIds2, DimensionIndex BoundDim2, DimensionIndex... BoundDims2,
           int... UncontractedIndexIds1, DimensionIndex... UncontractedBoundDims1, 
           int... UncontractedIndexIds2, DimensionIndex... UncontractedBoundDims2,
           DimensionIndex... ContractedBoundDims1,
           DimensionIndex... ContractedBoundDims2,
           typename internal::enable_if<(IndexId1 > IndexId2), bool>::type=true>
  auto make_contraction_indices(TensorIndexIds<IndexId1, IndexIds1...>, DimensionIds<BoundDim1, BoundDims1...>, 
                                TensorIndexIds<IndexId2, IndexIds2...>, DimensionIds<BoundDim2, BoundDims2...>, 
                                TensorIndexIds<UncontractedIndexIds1...>, DimensionIds<UncontractedBoundDims1...>, 
                                TensorIndexIds<UncontractedIndexIds2...>, DimensionIds<UncontractedBoundDims2...>,
                                DimensionIds<ContractedBoundDims1...>, DimensionIds<ContractedBoundDims2...>);
  // forward declaration
  template<int IndexId1, int... IndexIds1, DimensionIndex BoundDim1, DimensionIndex... BoundDims1, 
           int IndexId2, int... IndexIds2, DimensionIndex BoundDim2, DimensionIndex... BoundDims2,
           int... UncontractedIndexIds1, DimensionIndex... UncontractedBoundDims1, 
           int... UncontractedIndexIds2, DimensionIndex... UncontractedBoundDims2,
           DimensionIndex... ContractedBoundDims1,
           DimensionIndex... ContractedBoundDims2,
           typename internal::enable_if<(IndexId1 < IndexId2), bool>::type=true>
  auto make_contraction_indices(TensorIndexIds<IndexId1, IndexIds1...>, DimensionIds<BoundDim1, BoundDims1...>, 
                                TensorIndexIds<IndexId2, IndexIds2...>, DimensionIds<BoundDim2, BoundDims2...>, 
                                TensorIndexIds<UncontractedIndexIds1...>, DimensionIds<UncontractedBoundDims1...>, 
                                TensorIndexIds<UncontractedIndexIds2...>, DimensionIds<UncontractedBoundDims2...>,
                                DimensionIds<ContractedBoundDims1...>, DimensionIds<ContractedBoundDims2...>);
  //! We iterate over all operand indices from the smallest to the biggest.
  //! Here, IndexId1 is the smallest and is also present in the other operand,
  //! so we put it as a contracted TensorIndex and continue
  template<int IndexId1, int... IndexIds1, DimensionIndex BoundDim1, DimensionIndex... BoundDims1, 
           int IndexId2, int... IndexIds2, DimensionIndex BoundDim2, DimensionIndex... BoundDims2,
           int... UncontractedIndexIds1, DimensionIndex... UncontractedBoundDims1, 
           int... UncontractedIndexIds2, DimensionIndex... UncontractedBoundDims2,
           DimensionIndex... ContractedBoundDims1,
           DimensionIndex... ContractedBoundDims2,
           typename internal::enable_if<(IndexId1 == IndexId2), bool>::type=true>
  auto make_contraction_indices(TensorIndexIds<IndexId1, IndexIds1...>, DimensionIds<BoundDim1, BoundDims1...>, 
                                TensorIndexIds<IndexId2, IndexIds2...>, DimensionIds<BoundDim2, BoundDims2...>, 
                                TensorIndexIds<UncontractedIndexIds1...>, DimensionIds<UncontractedBoundDims1...>, 
                                TensorIndexIds<UncontractedIndexIds2...>, DimensionIds<UncontractedBoundDims2...>,
                                DimensionIds<ContractedBoundDims1...>, DimensionIds<ContractedBoundDims2...>)
  {
    return make_contraction_indices(
        TensorIndexIds<IndexIds1...>{}, DimensionIds<BoundDims1...>{}, 
        TensorIndexIds<IndexIds2...>{}, DimensionIds<BoundDims2...>{}, 
        TensorIndexIds<UncontractedIndexIds1...>{}, DimensionIds<UncontractedBoundDims1...>{},
        TensorIndexIds<UncontractedIndexIds2...>{}, DimensionIds<UncontractedBoundDims2...>{},
        DimensionIds<BoundDim1, ContractedBoundDims1...>{}, 
        DimensionIds<BoundDim2, ContractedBoundDims2...>{});
  }
  //! We iterate over all operand indices from the smallest to the biggest.
  //! Here, IndexId1 is the smallest and is not present in the other operand,
  //! so we put it as an uncontracted TensorIndex and continue
  template<int IndexId1, int... IndexIds1, DimensionIndex BoundDim1, DimensionIndex... BoundDims1, 
           int IndexId2, int... IndexIds2, DimensionIndex BoundDim2, DimensionIndex... BoundDims2,
           int... UncontractedIndexIds1, DimensionIndex... UncontractedBoundDims1, 
           int... UncontractedIndexIds2, DimensionIndex... UncontractedBoundDims2,
           DimensionIndex... ContractedBoundDims1,
           DimensionIndex... ContractedBoundDims2,
           typename internal::enable_if<(IndexId1 < IndexId2), bool>::type>
  auto make_contraction_indices(TensorIndexIds<IndexId1, IndexIds1...>, DimensionIds<BoundDim1, BoundDims1...>, 
                                TensorIndexIds<IndexId2, IndexIds2...>, DimensionIds<BoundDim2, BoundDims2...>, 
                                TensorIndexIds<UncontractedIndexIds1...>, DimensionIds<UncontractedBoundDims1...>, 
                                TensorIndexIds<UncontractedIndexIds2...>, DimensionIds<UncontractedBoundDims2...>,
                                DimensionIds<ContractedBoundDims1...>, DimensionIds<ContractedBoundDims2...>)
  {
    return make_contraction_indices(
        TensorIndexIds<         IndexIds1...>{}, DimensionIds<          BoundDims1...>{}, 
        TensorIndexIds<IndexId2,IndexIds2...>{}, DimensionIds<BoundDim2,BoundDims2...>{}, 
        TensorIndexIds<UncontractedIndexIds1..., IndexId1>{}, DimensionIds<UncontractedBoundDims1..., BoundDim1>{},
        TensorIndexIds<UncontractedIndexIds2...          >{}, DimensionIds<UncontractedBoundDims2...           >{},
        DimensionIds<ContractedBoundDims1...>{}, 
        DimensionIds<ContractedBoundDims2...>{});
  }
  //! We iterate over all operand indices from the smallest to the biggest.
  //! Here, IndexId2 is the smallest and is not present in the other operand,
  //! so we put it as an uncontracted TensorIndex and continue
  template<int IndexId1, int... IndexIds1, DimensionIndex BoundDim1, DimensionIndex... BoundDims1, 
           int IndexId2, int... IndexIds2, DimensionIndex BoundDim2, DimensionIndex... BoundDims2,
           int... UncontractedIndexIds1, DimensionIndex... UncontractedBoundDims1, 
           int... UncontractedIndexIds2, DimensionIndex... UncontractedBoundDims2,
           DimensionIndex... ContractedBoundDims1,
           DimensionIndex... ContractedBoundDims2,
           typename internal::enable_if<(IndexId1 > IndexId2), bool>::type>
  auto make_contraction_indices(TensorIndexIds<IndexId1, IndexIds1...>, DimensionIds<BoundDim1, BoundDims1...>, 
                                TensorIndexIds<IndexId2, IndexIds2...>, DimensionIds<BoundDim2, BoundDims2...>, 
                                TensorIndexIds<UncontractedIndexIds1...>, DimensionIds<UncontractedBoundDims1...>, 
                                TensorIndexIds<UncontractedIndexIds2...>, DimensionIds<UncontractedBoundDims2...>,
                                DimensionIds<ContractedBoundDims1...>, DimensionIds<ContractedBoundDims2...>)
  {
    return make_contraction_indices(
      TensorIndexIds<IndexId1,IndexIds1...>{}, DimensionIds<BoundDim1,BoundDims1...>{}, 
      TensorIndexIds<         IndexIds2...>{}, DimensionIds<          BoundDims2...>{}, 
      TensorIndexIds<UncontractedIndexIds1...          >{}, DimensionIds<UncontractedBoundDims1...           >{}, 
      TensorIndexIds<UncontractedIndexIds2..., IndexId2>{}, DimensionIds<UncontractedBoundDims2..., BoundDim2>{}, 
      DimensionIds<ContractedBoundDims1...>{}, 
      DimensionIds<ContractedBoundDims2...>{});
  }

} // namespace internal

//! \brief This is the operator* between two indexed tensors.
//!
//! It can be any combination of tensor contractions and tensor products.
template<typename Expr1, typename Expr2, 
         int... IndexIds1, DimensionIndex... BoundDims1, 
         int... IndexIds2, DimensionIndex... BoundDims2>
auto operator*(IndexedTensor<Expr1, BoundTensorIndex<IndexIds1, BoundDims1>...> const& a, 
               IndexedTensor<Expr2, BoundTensorIndex<IndexIds2, BoundDims2>...> const& b)
{
  // make_contraction_indices() computes which dimension goes where,
  // so that we can call TensorBase::contract()
  auto contraction_indices = internal::make_contraction_indices(
    internal::TensorIndexIds<IndexIds1...>{}, internal::DimensionIds<BoundDims1...>{},
    internal::TensorIndexIds<IndexIds2...>{}, internal::DimensionIds<BoundDims2...>{},
    internal::TensorIndexIds<>{}, internal::DimensionIds<>{},
    internal::TensorIndexIds<>{}, internal::DimensionIds<>{},
    internal::DimensionIds<>{}, internal::DimensionIds<>{});
  auto contracted_indices = contraction_indices.first;
  auto remaining_indices = contraction_indices.second;
  // return internal::make_indexed_tensor(a.tensor_expr.contract(b.tensor_expr, contracted_indices), remaining_indices);
  return internal::make_indexed_tensor(a.expression().contract(b.expression(), contracted_indices), remaining_indices);
}

namespace internal {

  // operator+

  //! \brief A helper function to IndexedTensor::operator+.
  //!
  //! BoundTensorIndex's are used to shuffle operands in a specific order (sorted by the ids of the TensorIndex's),
  //! then a simple tensor addition is made.
  template<typename Expr1, typename Expr2, int... IndexIds, DimensionIndex... BoundDimsA, DimensionIndex... BoundDimsB, size_t... Is>
  auto indexed_tensor_add(IndexedTensor<Expr1, BoundTensorIndex<IndexIds, BoundDimsA>...> const& a, 
                          IndexedTensor<Expr2, BoundTensorIndex<IndexIds, BoundDimsB>...> const& b,
                          std::index_sequence<Is...>)
  {
    // return internal::make_indexed_tensor(a.tensor_expr.shuffle(internal::make_array(static_cast<int>(BoundDimsA)...))
    //   + b.tensor_expr.shuffle(internal::make_array(static_cast<int>(BoundDimsB)...)), 
    //                           internal::sorted_indices_t<BoundTensorIndex<IndexIds, static_cast<DimensionIndex>(Is)>...>{});
    return internal::make_indexed_tensor(a.expression().shuffle(internal::make_array(static_cast<int>(BoundDimsA)...))
      + b.expression().shuffle(internal::make_array(static_cast<int>(BoundDimsB)...)), 
                         internal::sorted_indices_t<BoundTensorIndex<IndexIds, static_cast<DimensionIndex>(Is)>...>{});
  }

} // namespace internal

//! This is the operator+ for indexed tensor notation.
template<typename Expr1, typename Expr2, typename... BoundTensorIndices1, typename... BoundTensorIndices2>
auto operator+(IndexedTensor<Expr1, BoundTensorIndices1...> const& a, 
               IndexedTensor<Expr2, BoundTensorIndices2...> const& b)
{
  return internal::indexed_tensor_add(a, b, std::make_index_sequence<sizeof...(BoundTensorIndices1)>{});
}

// operator-

//! This is the unary operator- for indexed tensor notation.
template<typename Expr1, int... IndexIds, DimensionIndex... BoundDims>
auto operator-(IndexedTensor<Expr1, BoundTensorIndex<IndexIds, BoundDims>...> const& a)
{
  // return internal::make_indexed_tensor(-a.tensor_expr, internal::sorted_indices_t<BoundTensorIndex<IndexIds, BoundDims>...>{});
  return internal::make_indexed_tensor(-a.expression(), internal::sorted_indices_t<BoundTensorIndex<IndexIds, BoundDims>...>{});
}

//! This is the binary operator- for indexed tensor notation.
template<typename Expr1, typename Expr2, typename... BoundTensorIndices1, typename... BoundTensorIndices2>
auto operator-(IndexedTensor<Expr1, BoundTensorIndices1...> const& a, 
               IndexedTensor<Expr2, BoundTensorIndices2...> const& b)
{
  return a + (-b);
}

//operator += and -=
template<typename TensorExpr, typename... BoundIndices> template<typename OtherTensorExpr, typename... OtherBoundIndices>
void IndexedTensor<TensorExpr, BoundIndices...>::operator+=(IndexedTensor<OtherTensorExpr, OtherBoundIndices...> const& other)
{
  *this = *this + other;
}
template<typename TensorExpr, typename... BoundIndices> template<typename OtherTensorExpr, typename... OtherBoundIndices>
void IndexedTensor<TensorExpr, BoundIndices...>::operator-=(IndexedTensor<OtherTensorExpr, OtherBoundIndices...> const& other)
{
  *this = *this - other;
}

} // namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_INDEXED_H
