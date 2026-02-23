// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015-2019 Godeffroy Valet <godeffroy.valet@m4x.org>
// Copyright (C) 2022-2024 Jonas Breuling <breuling@inm.uni-stuttgart.de>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_INDEXED_H
#define EIGEN_CXX11_TENSOR_TENSOR_INDEXED_H

#include "./InternalHeaderCheck.h"

#include "TensorDimensions.h"
#include "TensorIndexList.h"

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

/** Base class of all TensorIndex. */
struct TensorIndexBase {};
/** In a tensor expression such as "A(i, j) * B(j, k)", i, j, k are each a
 *  TensorIndex with unique id_.
 */
template <int id_>
struct TensorIndex : TensorIndexBase {};

// We define some common indices which can be imported with "using namespace
// Eigen::tensor_indices;"
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
}  // namespace tensor_indices

using DimensionIndex = unsigned;

/** BoundTensorIndex<id,dim> links TensorIndex<id> to the dim-th dimension of
 * a Tensor
 */
template <int id, DimensionIndex dim>
struct BoundTensorIndex {};

namespace internal {
// A tool to keep a list of indices sorted at compile time
template <typename... Indices>
struct sorted_indices_t {
  template <typename Index>
  constexpr auto insert(Index index) {
    return sorted_indices_insert(sorted_indices_t<>{}, *this, index);
  }
};

// Linear insert a new BoundTensorIndex.
template <int... SmallerIndexIds, int IndexId, int... IndexIds, DimensionIndex... PrevBoundDims,
          DimensionIndex BoundDim, DimensionIndex... BoundDims, int NewIndexId, DimensionIndex NewBoundDim,
          typename EnableIf = typename std::enable_if_t<(NewIndexId <= IndexId), bool>>
constexpr auto sorted_indices_insert(
    sorted_indices_t<BoundTensorIndex<SmallerIndexIds, PrevBoundDims>...>,
    sorted_indices_t<BoundTensorIndex<IndexId, BoundDim>, BoundTensorIndex<IndexIds, BoundDims>...>,
    BoundTensorIndex<NewIndexId, NewBoundDim>) {
  EIGEN_STATIC_ASSERT(NewIndexId != IndexId, "sorted_indices_t cannot have duplicates");
  return sorted_indices_t<BoundTensorIndex<SmallerIndexIds, PrevBoundDims>...,
                          BoundTensorIndex<NewIndexId, NewBoundDim>, BoundTensorIndex<IndexId, BoundDim>,
                          BoundTensorIndex<IndexIds, BoundDims>...>{};
}
template <int... SmallerIndexIds, int IndexId, int... IndexIds, DimensionIndex... PrevBoundDims,
          DimensionIndex BoundDim, DimensionIndex... BoundDims, int NewIndexId, DimensionIndex NewBoundDim,
          typename std::enable_if_t<(NewIndexId > IndexId), bool> = true>
constexpr auto sorted_indices_insert(
    sorted_indices_t<BoundTensorIndex<SmallerIndexIds, PrevBoundDims>...>,
    sorted_indices_t<BoundTensorIndex<IndexId, BoundDim>, BoundTensorIndex<IndexIds, BoundDims>...>,
    BoundTensorIndex<NewIndexId, NewBoundDim>) {
  return sorted_indices_insert(
      sorted_indices_t<BoundTensorIndex<SmallerIndexIds, PrevBoundDims>..., BoundTensorIndex<IndexId, BoundDim>>{},
      sorted_indices_t<BoundTensorIndex<IndexIds, BoundDims>...>{}, BoundTensorIndex<NewIndexId, NewBoundDim>{});
}
template <int... SmallerIndexIds, DimensionIndex... PrevBoundDims, int NewIndexId, DimensionIndex NewBoundDim>
constexpr auto sorted_indices_insert(sorted_indices_t<BoundTensorIndex<SmallerIndexIds, PrevBoundDims>...>,
                                     sorted_indices_t<>, BoundTensorIndex<NewIndexId, NewBoundDim>) {
  return sorted_indices_t<BoundTensorIndex<SmallerIndexIds, PrevBoundDims>...,
                          BoundTensorIndex<NewIndexId, NewBoundDim>>{};
}

// Merge two sorted_indices_t.
template <int... IndexIds2, DimensionIndex... BoundDims2>
constexpr auto sorted_indices_merge(sorted_indices_t<>,
                                    sorted_indices_t<BoundTensorIndex<IndexIds2, BoundDims2>...> b) {
  return b;
}
template <int IndexId, DimensionIndex BoundDim, int... IndexIds1, DimensionIndex... BoundDims1, int... IndexIds2,
          DimensionIndex... BoundDims2>
constexpr auto sorted_indices_merge(
    sorted_indices_t<BoundTensorIndex<IndexId, BoundDim>, BoundTensorIndex<IndexIds1, BoundDims1>...>,
    sorted_indices_t<BoundTensorIndex<IndexIds2, BoundDims2>...> b) {
  return sorted_indices_merge(sorted_indices_t<BoundTensorIndex<IndexIds1, BoundDims1>...>{},
                              b.insert(BoundTensorIndex<IndexId, BoundDim>{}));
}

// Transforms any T const& to T& and leaves anything else unchanged.
template <typename T>
struct const_ref_to_ref {
  using type = T;
};
template <typename T>
struct const_ref_to_ref<T const&> {
  using type = T&;
};
template <typename T>
using const_ref_to_ref_t = typename const_ref_to_ref<T>::type;
}  // namespace internal

/** \brief This class stores any indexed tensor expression such as
 * my_tensor_expression(i,k,j) or A(i,j)*B(j,k)+C(i,k).
 *
 * If TensorExpr is an actual tensor, a reference to it is stored to allow
 * assignment. If TensorExpr is an expression, a copy of it is stored because
 * the expression might be a temporary object. In any case, BoundIndices keep
 * track of which TensorIndex is linked to which dimension of the TensorExpr.
 * BoundIndices are always kept sorted by the ids of their TensorIndex. For
 * example, the expression A(i,k,j) will have {i:0,j:2,k:1} as BoundIndices
 * The fact that they are sorted makes it easy to ensure that operands of
 * = or + operations share the same set of indices. It also makes it easy
 * to shuffle the dimensions of those operands before calling the underlying
 * operations.
 */
template <typename TensorExpr, typename... BoundIndices>
class IndexedTensor {
 public:
  typedef internal::const_ref_to_ref_t<typename internal::remove_const<typename TensorExpr::Nested>::type> _TensorExpr;

  const _TensorExpr& expression() const { return m_tensor_expr; }
  _TensorExpr& expression() { return m_tensor_expr; }

  IndexedTensor(TensorExpr&& t) : m_tensor_expr(std::move(t)) {}
  IndexedTensor(TensorExpr& t) : m_tensor_expr(t) {}

  void operator=(const typename TensorExpr::Scalar& other) { this->expression().setConstant(other); }
  template <typename OtherTensorExpr, typename... OtherBoundIndices>
  void operator=(IndexedTensor<OtherTensorExpr, OtherBoundIndices...> const& other);
  template <typename OtherTensorExpr, typename... OtherBoundIndices>
  void operator=(IndexedTensor<OtherTensorExpr, OtherBoundIndices...> const&& other);
  template <typename OtherTensorExpr, typename... OtherBoundIndices>
  void operator+=(IndexedTensor<OtherTensorExpr, OtherBoundIndices...> const& other);
  template <typename OtherTensorExpr, typename... OtherBoundIndices>
  void operator-=(IndexedTensor<OtherTensorExpr, OtherBoundIndices...> const& other);

 private:
  _TensorExpr m_tensor_expr;
};

/** An IndexedTensor with no indices is a tensor of order zero, which is a
 * scalar. So, it can be assigned to a scalar instead of a tensor.
 */
template <typename TensorExpr>
class IndexedTensor<TensorExpr> {
 public:
  typedef internal::const_ref_to_ref_t<typename TensorExpr::Nested> _TensorExpr;

  IndexedTensor(TensorExpr&& t) : m_tensor_expr(std::forward<TensorExpr>(t)) {}

  using Scalar = typename TensorExpr::Scalar;
  static const int DataLayout = internal::traits<TensorExpr>::Layout;

  operator Scalar() const {
    TensorFixedSize<Scalar, Sizes<>, DataLayout> result = m_tensor_expr;
    return *result.data();
  }

 private:
  _TensorExpr m_tensor_expr;
};

namespace internal {

// chipping - done with the operator()

/** \brief A helper function to TensorBase::operator()->TensorIndexed.
 *
 * This is the exit point of make_indexed_tensor(), which is called by
 * TensorBase::operator(). All dimensions are either chipped or linked to
 * a TensorIndex, so we can constructs the IndexedTensor.
 */
template <typename TensorExpr, int... IndexIds, DimensionIndex... BoundDims>
auto make_indexed_tensor(TensorExpr&& tensor_expr,
                         internal::sorted_indices_t<BoundTensorIndex<IndexIds, BoundDims>...>) {
  return IndexedTensor<typename std::remove_reference<TensorExpr>::type, BoundTensorIndex<IndexIds, BoundDims>...>(
      std::forward<TensorExpr>(tensor_expr));
}
/** Some dimensions remain to be chipped or linked to a TensorIndex. If the
 * current_index is a number, we chip the tensor_expr immediately, and
 * continue to iterate over the remaining dimensions.
 */
template <typename TensorExpr, typename... PrevIndices, typename... Indices>
inline auto make_indexed_tensor(TensorExpr&& tensor_expr, internal::sorted_indices_t<PrevIndices...> prev_indices,
                                DimensionIndex current_index, Indices... indices) {
  return make_indexed_tensor(tensor_expr.template chip<sizeof...(PrevIndices)>(current_index), prev_indices,
                             indices...);
}
/** Some dimensions remain to be chipped or linked to a TensorIndex. If the
 * current_index is a TensorIndex, we link it to the current dimension and
 * continue to iterate over the remaining dimensions.
 */
template <typename TensorExpr, typename... PrevIndices, typename... Indices, int id>
inline auto make_indexed_tensor(TensorExpr&& tensor_expr, internal::sorted_indices_t<PrevIndices...> prev_indices,
                                TensorIndex<id>, Indices... indices) {
  return make_indexed_tensor(std::forward<TensorExpr>(tensor_expr),
                             prev_indices.insert(BoundTensorIndex<id, sizeof...(PrevIndices)>{}), indices...);
}

// transposing, assignment - done with the operator=

/** \brief A helper function to IndexedTensor::operator=.
 *
 * BoundTensorIndex's are used to shuffle operands in a specific order
 * (sorted by the ids of the TensorIndex's), then a simple tensor
 * assignment is made.
 */
template <typename TensorExpr, int... IndexIds, DimensionIndex... BoundDims, typename OtherTensorExpr,
          /*same IndexIds*/ DimensionIndex... OtherBoundDims>
void indexed_tensor_assign(IndexedTensor<TensorExpr, BoundTensorIndex<IndexIds, BoundDims>...>* this_,
                           IndexedTensor<OtherTensorExpr, BoundTensorIndex<IndexIds, OtherBoundDims>...> const& other) {
  this_->expression().shuffle(Eigen::IndexList<Eigen::type2index<BoundDims>...>{}) =
      other.expression().shuffle(Eigen::IndexList<Eigen::type2index<OtherBoundDims>...>{});
}

}  // namespace internal

template <typename TensorExpr, typename... BoundIndices>
template <typename OtherTensorExpr, typename... OtherBoundIndices>
void IndexedTensor<TensorExpr, BoundIndices...>::operator=(
    IndexedTensor<OtherTensorExpr, OtherBoundIndices...> const&& other) {
  internal::indexed_tensor_assign(this, other);
}
template <typename TensorExpr, typename... BoundIndices>
template <typename OtherTensorExpr, typename... OtherBoundIndices>
void IndexedTensor<TensorExpr, BoundIndices...>::operator=(
    IndexedTensor<OtherTensorExpr, OtherBoundIndices...> const& other) {
  internal::indexed_tensor_assign(this, other);
}

namespace internal {
// contraction, tensor product - done with the operator*

/** \brief Calculates new dimension indices after a contraction.
 *
 * During a contraction, some dimensions disappear. For example
 * A(i,j,k)*B(j) contracts the dimension j. BoundIndices are {i:0,j:1,k:2}
 * for A and {j:0} for B, but after contration, BoundIndices are {i:0,k:1}
 * for A and {} for B, and those need to be computed. This function takes
 * the RemovedDims {1} for A or {0} for B, and one remaining_dim at a time,
 * and it returns the updated dimension index.
 */
template <DimensionIndex RemainingDim, DimensionIndex... RemovedDims>
constexpr DimensionIndex update_dim_index() {
  return RemainingDim - internal::reduce_count<(RemainingDim > RemovedDims)...>::value;
}

template <DimensionIndex... Ints>
using DimensionIds = std::integer_sequence<DimensionIndex, Ints...>;
template <int... Ints>
using TensorIndexIds = std::integer_sequence<int, Ints...>;

/** \brief A helper function to IndexedTensor::operator*.
 *
 * This is the exit point of make_contraction_indices() that performs a
 * true contraction. It returns the list of pairs of dimensions to be
 * contracted, and the BoundIndices of the resulting expression. Indices
 * remaining from the first operand are put first.
 */
template <int... UncontractedIndexIds1, DimensionIndex... UncontractedBoundDims1, int... UncontractedIndexIds2,
          DimensionIndex... UncontractedBoundDims2, DimensionIndex... ContractedBoundDims1,
          DimensionIndex... ContractedBoundDims2>
auto make_contraction_indices(TensorIndexIds<>, DimensionIds<>, TensorIndexIds<>, DimensionIds<>,
                              TensorIndexIds<UncontractedIndexIds1...>, DimensionIds<UncontractedBoundDims1...>,
                              TensorIndexIds<UncontractedIndexIds2...>, DimensionIds<UncontractedBoundDims2...>,
                              DimensionIds<ContractedBoundDims1...>, DimensionIds<ContractedBoundDims2...>) {
  return std::make_pair(
      Eigen::IndexPairList<Eigen::type2indexpair<ContractedBoundDims1, ContractedBoundDims2>...>{},
      sorted_indices_merge(
          internal::sorted_indices_t<BoundTensorIndex<
              UncontractedIndexIds1, update_dim_index<UncontractedBoundDims1, ContractedBoundDims1...>()>...>{},
          internal::sorted_indices_t<BoundTensorIndex<
              UncontractedIndexIds2, update_dim_index<UncontractedBoundDims2, ContractedBoundDims2...>() +
                                         sizeof...(UncontractedBoundDims1)>...>{}));
}
/** This is the exit point of make_contraction_indices() that performs no
 * contraction but a tensor product (handled by a tensor contraction with
 * an empty array of IndexPairs).
 */
template <int... UncontractedIndexIds1, DimensionIndex... UncontractedBoundDims1, int... UncontractedIndexIds2,
          DimensionIndex... UncontractedBoundDims2>
auto make_contraction_indices(TensorIndexIds<>, DimensionIds<>, TensorIndexIds<>, DimensionIds<>,
                              TensorIndexIds<UncontractedIndexIds1...>, DimensionIds<UncontractedBoundDims1...>,
                              TensorIndexIds<UncontractedIndexIds2...>, DimensionIds<UncontractedBoundDims2...>,
                              DimensionIds<>, DimensionIds<>) {
  return std::make_pair(
      std::array<Eigen::IndexPair<int>, 0>{},
      sorted_indices_merge(
          internal::sorted_indices_t<BoundTensorIndex<UncontractedIndexIds1, UncontractedBoundDims1>...>{},
          internal::sorted_indices_t<BoundTensorIndex<
              UncontractedIndexIds2, UncontractedBoundDims2 + sizeof...(UncontractedBoundDims1)>...>{}));
}
/** If all dimensions from one operand have been treated, there is nothing
 * left to be contracted, so we put all remaing indices from the other
 * operand as uncontracted indices.
 */
template <int... IndexIds1, DimensionIndex... BoundDims1, int... UncontractedIndexIds1,
          DimensionIndex... UncontractedBoundDims1, int... UncontractedIndexIds2,
          DimensionIndex... UncontractedBoundDims2, DimensionIndex... ContractedBoundDims1,
          DimensionIndex... ContractedBoundDims2>
auto make_contraction_indices(TensorIndexIds<IndexIds1...>, DimensionIds<BoundDims1...>, TensorIndexIds<>,
                              DimensionIds<>, TensorIndexIds<UncontractedIndexIds1...>,
                              DimensionIds<UncontractedBoundDims1...>, TensorIndexIds<UncontractedIndexIds2...>,
                              DimensionIds<UncontractedBoundDims2...>, DimensionIds<ContractedBoundDims1...>,
                              DimensionIds<ContractedBoundDims2...>) {
  return make_contraction_indices(TensorIndexIds<>{}, DimensionIds<>{}, TensorIndexIds<>{}, DimensionIds<>{},
                                  TensorIndexIds<UncontractedIndexIds1..., IndexIds1...>{},
                                  DimensionIds<UncontractedBoundDims1..., BoundDims1...>{},
                                  TensorIndexIds<UncontractedIndexIds2...>{}, DimensionIds<UncontractedBoundDims2...>{},
                                  DimensionIds<ContractedBoundDims1...>{}, DimensionIds<ContractedBoundDims2...>{});
}
/** If all dimensions from one operand have been treated, there is nothing
 * left to be contracted, so we put all remaing indices from the other
 * operand as uncontracted indices.
 */
template <int... IndexIds2, DimensionIndex... BoundDims2, int... UncontractedIndexIds1,
          DimensionIndex... UncontractedBoundDims1, int... UncontractedIndexIds2,
          DimensionIndex... UncontractedBoundDims2, DimensionIndex... ContractedBoundDims1,
          DimensionIndex... ContractedBoundDims2>
auto make_contraction_indices(TensorIndexIds<>, DimensionIds<>, TensorIndexIds<IndexIds2...>,
                              DimensionIds<BoundDims2...>, TensorIndexIds<UncontractedIndexIds1...>,
                              DimensionIds<UncontractedBoundDims1...>, TensorIndexIds<UncontractedIndexIds2...>,
                              DimensionIds<UncontractedBoundDims2...>, DimensionIds<ContractedBoundDims1...>,
                              DimensionIds<ContractedBoundDims2...>) {
  return make_contraction_indices(TensorIndexIds<>{}, DimensionIds<>{}, TensorIndexIds<>{}, DimensionIds<>{},
                                  TensorIndexIds<UncontractedIndexIds1...>{}, DimensionIds<UncontractedBoundDims1...>{},
                                  TensorIndexIds<UncontractedIndexIds2..., IndexIds2...>{},
                                  DimensionIds<UncontractedBoundDims2..., BoundDims2...>{},
                                  DimensionIds<ContractedBoundDims1...>{}, DimensionIds<ContractedBoundDims2...>{});
}
// forward declaration
template <int IndexId1, int... IndexIds1, DimensionIndex BoundDim1, DimensionIndex... BoundDims1, int IndexId2,
          int... IndexIds2, DimensionIndex BoundDim2, DimensionIndex... BoundDims2, int... UncontractedIndexIds1,
          DimensionIndex... UncontractedBoundDims1, int... UncontractedIndexIds2,
          DimensionIndex... UncontractedBoundDims2, DimensionIndex... ContractedBoundDims1,
          DimensionIndex... ContractedBoundDims2, typename std::enable_if_t<(IndexId1 > IndexId2), bool> = true>
auto make_contraction_indices(TensorIndexIds<IndexId1, IndexIds1...>, DimensionIds<BoundDim1, BoundDims1...>,
                              TensorIndexIds<IndexId2, IndexIds2...>, DimensionIds<BoundDim2, BoundDims2...>,
                              TensorIndexIds<UncontractedIndexIds1...>, DimensionIds<UncontractedBoundDims1...>,
                              TensorIndexIds<UncontractedIndexIds2...>, DimensionIds<UncontractedBoundDims2...>,
                              DimensionIds<ContractedBoundDims1...>, DimensionIds<ContractedBoundDims2...>);
// forward declaration
template <int IndexId1, int... IndexIds1, DimensionIndex BoundDim1, DimensionIndex... BoundDims1, int IndexId2,
          int... IndexIds2, DimensionIndex BoundDim2, DimensionIndex... BoundDims2, int... UncontractedIndexIds1,
          DimensionIndex... UncontractedBoundDims1, int... UncontractedIndexIds2,
          DimensionIndex... UncontractedBoundDims2, DimensionIndex... ContractedBoundDims1,
          DimensionIndex... ContractedBoundDims2, typename std::enable_if_t<(IndexId1 < IndexId2), bool> = true>
auto make_contraction_indices(TensorIndexIds<IndexId1, IndexIds1...>, DimensionIds<BoundDim1, BoundDims1...>,
                              TensorIndexIds<IndexId2, IndexIds2...>, DimensionIds<BoundDim2, BoundDims2...>,
                              TensorIndexIds<UncontractedIndexIds1...>, DimensionIds<UncontractedBoundDims1...>,
                              TensorIndexIds<UncontractedIndexIds2...>, DimensionIds<UncontractedBoundDims2...>,
                              DimensionIds<ContractedBoundDims1...>, DimensionIds<ContractedBoundDims2...>);
/** We iterate over all operand indices from the smallest to the biggest.
 * Here, IndexId1 is the smallest and is also present in the other operand,
 * so we put it as a contracted TensorIndex and continue.
 */
template <int IndexId1, int... IndexIds1, DimensionIndex BoundDim1, DimensionIndex... BoundDims1, int IndexId2,
          int... IndexIds2, DimensionIndex BoundDim2, DimensionIndex... BoundDims2, int... UncontractedIndexIds1,
          DimensionIndex... UncontractedBoundDims1, int... UncontractedIndexIds2,
          DimensionIndex... UncontractedBoundDims2, DimensionIndex... ContractedBoundDims1,
          DimensionIndex... ContractedBoundDims2, typename std::enable_if_t<(IndexId1 == IndexId2), bool> = true>
auto make_contraction_indices(TensorIndexIds<IndexId1, IndexIds1...>, DimensionIds<BoundDim1, BoundDims1...>,
                              TensorIndexIds<IndexId2, IndexIds2...>, DimensionIds<BoundDim2, BoundDims2...>,
                              TensorIndexIds<UncontractedIndexIds1...>, DimensionIds<UncontractedBoundDims1...>,
                              TensorIndexIds<UncontractedIndexIds2...>, DimensionIds<UncontractedBoundDims2...>,
                              DimensionIds<ContractedBoundDims1...>, DimensionIds<ContractedBoundDims2...>) {
  return make_contraction_indices(TensorIndexIds<IndexIds1...>{}, DimensionIds<BoundDims1...>{},
                                  TensorIndexIds<IndexIds2...>{}, DimensionIds<BoundDims2...>{},
                                  TensorIndexIds<UncontractedIndexIds1...>{}, DimensionIds<UncontractedBoundDims1...>{},
                                  TensorIndexIds<UncontractedIndexIds2...>{}, DimensionIds<UncontractedBoundDims2...>{},
                                  DimensionIds<BoundDim1, ContractedBoundDims1...>{},
                                  DimensionIds<BoundDim2, ContractedBoundDims2...>{});
}
/** We iterate over all operand indices from the smallest to the biggest.
 * Here, IndexId1 is the smallest and is not present in the other operand,
 * so we put it as an uncontracted TensorIndex and continue.
 */
template <int IndexId1, int... IndexIds1, DimensionIndex BoundDim1, DimensionIndex... BoundDims1, int IndexId2,
          int... IndexIds2, DimensionIndex BoundDim2, DimensionIndex... BoundDims2, int... UncontractedIndexIds1,
          DimensionIndex... UncontractedBoundDims1, int... UncontractedIndexIds2,
          DimensionIndex... UncontractedBoundDims2, DimensionIndex... ContractedBoundDims1,
          DimensionIndex... ContractedBoundDims2, typename std::enable_if_t<(IndexId1 < IndexId2), bool>>
auto make_contraction_indices(TensorIndexIds<IndexId1, IndexIds1...>, DimensionIds<BoundDim1, BoundDims1...>,
                              TensorIndexIds<IndexId2, IndexIds2...>, DimensionIds<BoundDim2, BoundDims2...>,
                              TensorIndexIds<UncontractedIndexIds1...>, DimensionIds<UncontractedBoundDims1...>,
                              TensorIndexIds<UncontractedIndexIds2...>, DimensionIds<UncontractedBoundDims2...>,
                              DimensionIds<ContractedBoundDims1...>, DimensionIds<ContractedBoundDims2...>) {
  return make_contraction_indices(TensorIndexIds<IndexIds1...>{}, DimensionIds<BoundDims1...>{},
                                  TensorIndexIds<IndexId2, IndexIds2...>{}, DimensionIds<BoundDim2, BoundDims2...>{},
                                  TensorIndexIds<UncontractedIndexIds1..., IndexId1>{},
                                  DimensionIds<UncontractedBoundDims1..., BoundDim1>{},
                                  TensorIndexIds<UncontractedIndexIds2...>{}, DimensionIds<UncontractedBoundDims2...>{},
                                  DimensionIds<ContractedBoundDims1...>{}, DimensionIds<ContractedBoundDims2...>{});
}
/** We iterate over all operand indices from the smallest to the biggest.
 * Here, IndexId2 is the smallest and is not present in the other operand,
 * so we put it as an uncontracted TensorIndex and continue.
 */
template <int IndexId1, int... IndexIds1, DimensionIndex BoundDim1, DimensionIndex... BoundDims1, int IndexId2,
          int... IndexIds2, DimensionIndex BoundDim2, DimensionIndex... BoundDims2, int... UncontractedIndexIds1,
          DimensionIndex... UncontractedBoundDims1, int... UncontractedIndexIds2,
          DimensionIndex... UncontractedBoundDims2, DimensionIndex... ContractedBoundDims1,
          DimensionIndex... ContractedBoundDims2, typename std::enable_if_t<(IndexId1 > IndexId2), bool>>
auto make_contraction_indices(TensorIndexIds<IndexId1, IndexIds1...>, DimensionIds<BoundDim1, BoundDims1...>,
                              TensorIndexIds<IndexId2, IndexIds2...>, DimensionIds<BoundDim2, BoundDims2...>,
                              TensorIndexIds<UncontractedIndexIds1...>, DimensionIds<UncontractedBoundDims1...>,
                              TensorIndexIds<UncontractedIndexIds2...>, DimensionIds<UncontractedBoundDims2...>,
                              DimensionIds<ContractedBoundDims1...>, DimensionIds<ContractedBoundDims2...>) {
  return make_contraction_indices(TensorIndexIds<IndexId1, IndexIds1...>{}, DimensionIds<BoundDim1, BoundDims1...>{},
                                  TensorIndexIds<IndexIds2...>{}, DimensionIds<BoundDims2...>{},
                                  TensorIndexIds<UncontractedIndexIds1...>{}, DimensionIds<UncontractedBoundDims1...>{},
                                  TensorIndexIds<UncontractedIndexIds2..., IndexId2>{},
                                  DimensionIds<UncontractedBoundDims2..., BoundDim2>{},
                                  DimensionIds<ContractedBoundDims1...>{}, DimensionIds<ContractedBoundDims2...>{});
}

}  // namespace internal

/** \brief This is the operator* between two indexed tensors.
 *
 * It can be any combination of tensor contractions and tensor products.
 */
template <typename Expr1, typename Expr2, int... IndexIds1, DimensionIndex... BoundDims1, int... IndexIds2,
          DimensionIndex... BoundDims2>
auto operator*(IndexedTensor<Expr1, BoundTensorIndex<IndexIds1, BoundDims1>...> const& a,
               IndexedTensor<Expr2, BoundTensorIndex<IndexIds2, BoundDims2>...> const& b) {
  // make_contraction_indices() computes which dimension goes where,
  // so that we can call TensorBase::contract()
  auto contraction_indices = internal::make_contraction_indices(
      internal::TensorIndexIds<IndexIds1...>{}, internal::DimensionIds<BoundDims1...>{},
      internal::TensorIndexIds<IndexIds2...>{}, internal::DimensionIds<BoundDims2...>{}, internal::TensorIndexIds<>{},
      internal::DimensionIds<>{}, internal::TensorIndexIds<>{}, internal::DimensionIds<>{}, internal::DimensionIds<>{},
      internal::DimensionIds<>{});
  auto contracted_indices = contraction_indices.first;
  auto remaining_indices = contraction_indices.second;
  return internal::make_indexed_tensor(a.expression().contract(b.expression(), contracted_indices), remaining_indices);
}

/** \brief This is the operator* between an indexed tensors and a real number an vice versa.
 */
template <typename Expr, int... IndexIds, DimensionIndex... BoundDims, typename Scalar = typename Expr::Scalar>
auto operator*(const typename Expr::Scalar& a, IndexedTensor<Expr, BoundTensorIndex<IndexIds, BoundDims>...> const& b) {
  return internal::make_indexed_tensor(a * b.expression(),
                                       internal::sorted_indices_t<BoundTensorIndex<IndexIds, BoundDims>...>{});
}
template <typename Expr, int... IndexIds, DimensionIndex... BoundDims>
auto operator*(IndexedTensor<Expr, BoundTensorIndex<IndexIds, BoundDims>...> const& a, const typename Expr::Scalar& b) {
  return internal::make_indexed_tensor(b * a.expression(),
                                       internal::sorted_indices_t<BoundTensorIndex<IndexIds, BoundDims>...>{});
}

namespace internal {

// operator+

/** \brief A helper function to IndexedTensor::operator+.
 *
 * BoundTensorIndex's are used to shuffle operands in a specific order
 * (sorted by the ids of the TensorIndex's), then a simple tensor addition
 * is made.
 */
template <typename Expr1, typename Expr2, int... IndexIds, DimensionIndex... BoundDimsA, DimensionIndex... BoundDimsB,
          size_t... Is>
auto indexed_tensor_add(IndexedTensor<Expr1, BoundTensorIndex<IndexIds, BoundDimsA>...> const& a,
                        IndexedTensor<Expr2, BoundTensorIndex<IndexIds, BoundDimsB>...> const& b,
                        std::index_sequence<Is...>) {
  return internal::make_indexed_tensor(
      a.expression().shuffle(Eigen::IndexList<Eigen::type2index<BoundDimsA>...>{}) +
          b.expression().shuffle(Eigen::IndexList<Eigen::type2index<BoundDimsB>...>{}),
      internal::sorted_indices_t<BoundTensorIndex<IndexIds, static_cast<DimensionIndex>(Is)>...>{});
}

}  // namespace internal

/** This is the operator+ for indexed tensor notation. */
template <typename Expr1, typename Expr2, typename... BoundTensorIndices1, typename... BoundTensorIndices2>
auto operator+(IndexedTensor<Expr1, BoundTensorIndices1...> const& a,
               IndexedTensor<Expr2, BoundTensorIndices2...> const& b) {
  return internal::indexed_tensor_add(a, b, std::make_index_sequence<sizeof...(BoundTensorIndices1)>{});
}

// operator-

/** This is the unary operator- for indexed tensor notation. */
template <typename Expr1, int... IndexIds, DimensionIndex... BoundDims>
auto operator-(IndexedTensor<Expr1, BoundTensorIndex<IndexIds, BoundDims>...> const& a) {
  return internal::make_indexed_tensor(-a.expression(),
                                       internal::sorted_indices_t<BoundTensorIndex<IndexIds, BoundDims>...>{});
}

/** This is the binary operator- for indexed tensor notation. */
template <typename Expr1, typename Expr2, typename... BoundTensorIndices1, typename... BoundTensorIndices2>
auto operator-(IndexedTensor<Expr1, BoundTensorIndices1...> const& a,
               IndexedTensor<Expr2, BoundTensorIndices2...> const& b) {
  return a + (-b);
}

/** operator += and -= */
template <typename TensorExpr, typename... BoundIndices>
template <typename OtherTensorExpr, typename... OtherBoundIndices>
void IndexedTensor<TensorExpr, BoundIndices...>::operator+=(
    IndexedTensor<OtherTensorExpr, OtherBoundIndices...> const& other) {
  *this = *this + other;
}
template <typename TensorExpr, typename... BoundIndices>
template <typename OtherTensorExpr, typename... OtherBoundIndices>
void IndexedTensor<TensorExpr, BoundIndices...>::operator-=(
    IndexedTensor<OtherTensorExpr, OtherBoundIndices...> const& other) {
  *this = *this - other;
}

}  // namespace Eigen

#endif  // EIGEN_CXX11_TENSOR_TENSOR_INDEXED_H
