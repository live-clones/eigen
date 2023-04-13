// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#if !defined(EIGEN_PARSED_BY_DOXYGEN)

protected:
// define some aliases to ease readability

template <typename Indices>
using IvcRowType = typename internal::IndexedViewCompatibleType<Indices, RowsAtCompileTime>::type;

template <typename Indices>
using IvcColType = typename internal::IndexedViewCompatibleType<Indices, ColsAtCompileTime>::type;

template <typename Indices>
using IvcType = typename internal::IndexedViewCompatibleType<Indices, SizeAtCompileTime>::type;

typedef typename internal::IndexedViewCompatibleType<Index, 1>::type IvcIndex;

template <typename Indices>
inline IvcRowType<Indices> ivcRow(const Indices& indices) const {
  return internal::makeIndexedViewCompatible(
      indices, internal::variable_if_dynamic<Index, RowsAtCompileTime>(derived().rows()), Specialized);
}

template <typename Indices>
inline IvcColType<Indices> ivcCol(const Indices& indices) const {
  return internal::makeIndexedViewCompatible(
      indices, internal::variable_if_dynamic<Index, ColsAtCompileTime>(derived().cols()), Specialized);
}

template <typename Indices>
inline IvcColType<Indices> ivcSize(const Indices& indices) const {
  return internal::makeIndexedViewCompatible(
      indices, internal::variable_if_dynamic<Index, SizeAtCompileTime>(derived().size()), Specialized);
}

// this helper class assumes internal::valid_indexed_view_overload<RowIndices, ColIndices>::value == true
template <typename DenseDerived, typename RowIndices, typename ColIndices,
          bool UseSymbolic = internal::traits<IndexedView<DenseDerived, IvcRowType<RowIndices>, IvcColType<ColIndices>>>::ReturnAsScalar,
          bool UseBlock = internal::traits<IndexedView<DenseDerived, IvcRowType<RowIndices>, IvcColType<ColIndices>>>::ReturnAsBlock,
          bool UseGeneric = internal::traits<IndexedView<DenseDerived, IvcRowType<RowIndices>, IvcColType<ColIndices>>>::ReturnAsIndexedView>
struct IndexedView_selector;

// Generic
template <typename DenseDerived, typename RowIndices, typename ColIndices>
struct IndexedView_selector<DenseDerived, RowIndices, ColIndices, false, false, true> {
  using ReturnType = IndexedView<DenseDerived, IvcRowType<RowIndices>, IvcColType<ColIndices>>;
  using ConstReturnType = IndexedView<const DenseDerived, IvcRowType<RowIndices>, IvcColType<ColIndices>>;

  static inline ReturnType run(DenseDerived& derived, const RowIndices& rowIndices, const ColIndices& colIndices) {
    return ReturnType(derived, derived.ivcRow(rowIndices), derived.ivcCol(colIndices));
  }
  static inline ConstReturnType run(const DenseDerived& derived, const RowIndices& rowIndices,
                                    const ColIndices& colIndices) {
    return ConstReturnType(derived, derived.ivcRow(rowIndices), derived.ivcCol(colIndices));
  }
};

// Block
template <typename DenseDerived, typename RowIndices, typename ColIndices>
struct IndexedView_selector<DenseDerived, RowIndices, ColIndices, false, true, false> {
  using IndexedViewType = IndexedView<DenseDerived, IvcRowType<RowIndices>, IvcColType<ColIndices>>;
  using ConstIndexedViewType = IndexedView<const DenseDerived, IvcRowType<RowIndices>, IvcColType<ColIndices>>;
  using ReturnType = typename internal::traits<IndexedViewType>::BlockType;
  using ConstReturnType = typename internal::traits<ConstIndexedViewType>::BlockType;

  static inline ReturnType run(DenseDerived& derived, const RowIndices& rowIndices, const ColIndices& colIndices) {
    IvcRowType<RowIndices> actualRowIndices = derived.ivcRow(rowIndices);
    IvcColType<ColIndices> actualColIndices = derived.ivcCol(colIndices);
    return ReturnType(derived, internal::first(actualRowIndices), internal::first(actualColIndices),
                      internal::index_list_size(actualRowIndices), internal::index_list_size(actualColIndices));
  }
  static inline ConstReturnType run(const DenseDerived& derived, const RowIndices& rowIndices,
                                    const ColIndices& colIndices) {
    IvcRowType<RowIndices> actualRowIndices = derived.ivcRow(rowIndices);
    IvcColType<ColIndices> actualColIndices = derived.ivcCol(colIndices);
    return ConstReturnType(derived, internal::first(actualRowIndices), internal::first(actualColIndices),
                           internal::index_list_size(actualRowIndices), internal::index_list_size(actualColIndices));
  }
};

// Symbolic
template <typename DenseDerived, typename RowIndices, typename ColIndices>
struct IndexedView_selector<DenseDerived, RowIndices, ColIndices, true, false, false> {
  using ReturnType = typename DenseBase<DenseDerived>::Scalar&;
  using ConstReturnType = typename DenseBase<DenseDerived>::CoeffReturnType;

  template <bool IsLValue = internal::is_lvalue<DenseDerived>::value, std::enable_if_t<IsLValue, bool> = true>
  static inline ReturnType run(DenseDerived& derived, const RowIndices& rowIndices, const ColIndices& colIndices) {
    return derived(internal::eval_expr_given_size(rowIndices, derived.rows()),
                   internal::eval_expr_given_size(colIndices, derived.cols()));
  }
  static inline ConstReturnType run(const DenseDerived& derived, const RowIndices& rowIndices,
                                    const ColIndices& colIndices) {
    return derived(internal::eval_expr_given_size(rowIndices, derived.rows()),
                   internal::eval_expr_given_size(colIndices, derived.cols()));
  }
};

// this helper class assumes internal::is_valid_index_type<Indices>::value == false
template <typename DenseDerived, typename Indices, bool UseSymbolic = symbolic::is_symbolic<Indices>::value,
          bool UseBlock = !UseSymbolic && internal::get_compile_time_incr<IvcType<Indices>>::value == 1,
          bool UseGeneric = !UseSymbolic && !UseBlock>
struct VectorIndexedView_selector;

// Generic
template <typename DenseDerived, typename Indices>
struct VectorIndexedView_selector<DenseDerived, Indices, false, false, true> {

  static constexpr bool IsRowMajor = DenseBase<DenseDerived>::IsRowMajor;

  using RowMajorReturnType = IndexedView<DenseDerived, IvcIndex, IvcType<Indices>>;
  using ConstRowMajorReturnType = IndexedView<const DenseDerived, IvcIndex, IvcType<Indices>>;

  using ColMajorReturnType = IndexedView<DenseDerived, IvcType<Indices>, IvcIndex>;
  using ConstColMajorReturnType = IndexedView<const DenseDerived, IvcType<Indices>, IvcIndex>;

  using ReturnType = typename internal::conditional<IsRowMajor, RowMajorReturnType, ColMajorReturnType>::type;
  using ConstReturnType =
      typename internal::conditional<IsRowMajor, ConstRowMajorReturnType, ConstColMajorReturnType>::type;

  template <bool UseRowMajor = IsRowMajor, std::enable_if_t<UseRowMajor, bool> = true>
  static inline RowMajorReturnType run(DenseDerived& derived, const Indices& indices) {
    return RowMajorReturnType(derived, IvcIndex(0), derived.ivcCol(indices));
  }
  template <bool UseRowMajor = IsRowMajor, std::enable_if_t<UseRowMajor, bool> = true>
  static inline ConstRowMajorReturnType run(const DenseDerived& derived, const Indices& indices) {
    return ConstRowMajorReturnType(derived, IvcIndex(0), derived.ivcCol(indices));
  }

  template <bool UseRowMajor = IsRowMajor, std::enable_if_t<!UseRowMajor, bool> = true>
  static inline ColMajorReturnType run(DenseDerived& derived, const Indices& indices) {
    return ColMajorReturnType(derived, derived.ivcRow(indices), IvcIndex(0));
  }
  template <bool UseRowMajor = IsRowMajor, std::enable_if_t<!UseRowMajor, bool> = true>
  static inline ConstColMajorReturnType run(const DenseDerived& derived, const Indices& indices) {
    return ConstColMajorReturnType(derived, derived.ivcRow(indices), IvcIndex(0));
  }
};

// Block
template <typename DenseDerived, typename Indices>
struct VectorIndexedView_selector<DenseDerived, Indices, false, true, false> {

  using ReturnType = VectorBlock<DenseDerived, internal::array_size<Indices>::value>;
  using ConstReturnType = VectorBlock<const DenseDerived, internal::array_size<Indices>::value>;

  static inline ReturnType run(DenseDerived& derived, const Indices& indices) {
    IvcType<Indices> actualIndices = derived.ivcSize(indices);
    return ReturnType(derived, internal::first(actualIndices), internal::index_list_size(actualIndices));
  }
  static inline ConstReturnType run(const DenseDerived& derived, const Indices& indices) {
    IvcType<Indices> actualIndices = derived.ivcSize(indices);
    return ConstReturnType(derived, internal::first(actualIndices), internal::index_list_size(actualIndices));
  }
};

// Symbolic
template <typename DenseDerived, typename Indices>
struct VectorIndexedView_selector<DenseDerived, Indices, true, false, false> {

  using ReturnType = typename DenseBase<DenseDerived>::Scalar&;
  using ConstReturnType = typename DenseBase<DenseDerived>::CoeffReturnType;

  template <bool IsLValue = internal::is_lvalue<DenseDerived>::value, std::enable_if_t<IsLValue, bool> = true>
  static inline ReturnType run(DenseDerived& derived, const Indices& id) {
    return derived(internal::eval_expr_given_size(id, derived.size()));
  }

  static inline ConstReturnType run(const DenseDerived& derived, const Indices& id) {
    return derived(internal::eval_expr_given_size(id, derived.size()));
  }
};

public:
// Overloads for 2D matrices/arrays

// non-const versions

template <typename RowIndices, typename ColIndices>
using IndexedViewType = std::enable_if_t<internal::valid_indexed_view_overload<RowIndices, ColIndices>::value,
                                         typename IndexedView_selector<Derived, RowIndices, ColIndices>::ReturnType>;

template <typename RowIndices, typename ColIndices>
IndexedViewType<RowIndices, ColIndices> operator()(const RowIndices& rowIndices, const ColIndices& colIndices) {
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
  using Impl = IndexedView_selector<Derived, RowIndices, ColIndices>;
  return Impl::run(derived(), rowIndices, colIndices);
}

template <typename RowT, size_t RowSize, typename ColIndices, typename RowIndices = Array<RowT, RowSize, 1>>
IndexedViewType<RowIndices, ColIndices> operator()(const RowT (&rowIndices)[RowSize], const ColIndices& colIndices) {
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
  using Impl = IndexedView_selector<Derived, RowIndices, ColIndices>;
  return Impl::run(derived(), RowIndices{rowIndices}, colIndices);
}

template <typename RowIndices, typename ColT, size_t ColSize, typename ColIndices = Array<ColT, ColSize, 1>>
IndexedViewType<RowIndices, ColIndices> operator()(const RowIndices& rowIndices, const ColT (&colIndices)[ColSize]) {
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
  using Impl = IndexedView_selector<Derived, RowIndices, ColIndices>;
  return Impl::run(derived(), rowIndices, ColIndices{colIndices});
}

template <typename RowT, size_t RowSize, typename ColT, size_t ColSize, typename RowIndices = Array<RowT, RowSize, 1>,
          typename ColIndices = Array<ColT, ColSize, 1>>
IndexedViewType<RowIndices, ColIndices> operator()(const RowT (&rowIndices)[RowSize],
                                                   const ColT (&colIndices)[ColSize]) {
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
  using Impl = IndexedView_selector<Derived, RowIndices, ColIndices>;
  return Impl::run(derived(), RowIndices{rowIndices}, ColIndices{colIndices});
}

// const versions

template <typename RowIndices, typename ColIndices>
using ConstIndexedViewType =
    std::enable_if_t<internal::valid_indexed_view_overload<RowIndices, ColIndices>::value,
                     typename IndexedView_selector<Derived, RowIndices, ColIndices>::ConstReturnType>;

template <typename RowIndices, typename ColIndices>
ConstIndexedViewType<RowIndices, ColIndices> operator()(const RowIndices& rowIndices,
                                                        const ColIndices& colIndices) const {
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
  using Impl = IndexedView_selector<Derived, RowIndices, ColIndices>;
  return Impl::run(derived(), rowIndices, colIndices);
}

template <typename RowT, size_t RowSize, typename ColIndices, typename RowIndices = Array<RowT, RowSize, 1>>
ConstIndexedViewType<RowIndices, ColIndices> operator()(const RowT (&rowIndices)[RowSize],
                                                        const ColIndices& colIndices) const {
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
  using Impl = IndexedView_selector<Derived, RowIndices, ColIndices>;
  return Impl::run(derived(), RowIndices{rowIndices}, colIndices);
}

template <typename RowIndices, typename ColT, size_t ColSize, typename ColIndices = Array<ColT, ColSize, 1>>
ConstIndexedViewType<RowIndices, ColIndices> operator()(const RowIndices& rowIndices,
                                                        const ColT (&colIndices)[ColSize]) const {
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
  using Impl = IndexedView_selector<Derived, RowIndices, ColIndices>;
  return Impl::run(derived(), rowIndices, ColIndices{colIndices});
}

template <typename RowT, size_t RowSize, typename ColT, size_t ColSize, typename RowIndices = Array<RowT, RowSize, 1>,
          typename ColIndices = Array<ColT, ColSize, 1>>
ConstIndexedViewType<RowIndices, ColIndices> operator()(const RowT (&rowIndices)[RowSize],
                                                        const ColT (&colIndices)[ColSize]) const {
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
  using Impl = IndexedView_selector<Derived, RowIndices, ColIndices>;
  return Impl::run(derived(), RowIndices{rowIndices}, ColIndices{colIndices});
}

// Overloads for 1D vectors/arrays

// non-const versions

template <typename Indices>
using VectorIndexedViewType = std::enable_if_t<!internal::is_valid_index_type<Indices>::value,
                                               typename VectorIndexedView_selector<Derived, Indices>::ReturnType>;

template <typename Indices>
VectorIndexedViewType<Indices> operator()(const Indices& indices) {
  using Impl = VectorIndexedView_selector<Derived, Indices>;
  return Impl::run(derived(), indices);
}

template <typename IdxT, size_t Size, typename Indices = Array<IdxT, Size, 1>>
VectorIndexedViewType<Indices> operator()(const IdxT (&indices)[Size]) {
  using Impl = VectorIndexedView_selector<Derived, Indices>;
  return Impl::run(derived(), Indices{indices});
}

// const versions

template <typename Indices>
using ConstVectorIndexedViewType =
    std::enable_if_t<!internal::is_valid_index_type<Indices>::value,
                     typename VectorIndexedView_selector<Derived, Indices>::ConstReturnType>;

template <typename Indices>
ConstVectorIndexedViewType<Indices> operator()(const Indices& indices) const {
  using Impl = VectorIndexedView_selector<Derived, Indices>;
  return Impl::run(derived(), indices);
}

template <typename IdxT, size_t Size, typename Indices = Array<IdxT, Size, 1>>
ConstVectorIndexedViewType<Indices> operator()(const IdxT (&indices)[Size]) const {
  using Impl = VectorIndexedView_selector<Derived, Indices>;
  return Impl::run(derived(), Indices{indices});
}

#else // EIGEN_PARSED_BY_DOXYGEN

/**
  * \returns a generic submatrix view defined by the rows and columns indexed \a rowIndices and \a colIndices respectively.
  *
  * Each parameter must either be:
  *  - An integer indexing a single row or column
  *  - Eigen::placeholders::all indexing the full set of respective rows or columns in increasing order
  *  - An ArithmeticSequence as returned by the Eigen::seq and Eigen::seqN functions
  *  - Any %Eigen's vector/array of integers or expressions
  *  - Plain C arrays: \c int[N]
  *  - And more generally any type exposing the following two member functions:
  * \code
  * <integral type> operator[](<integral type>) const;
  * <integral type> size() const;
  * \endcode
  * where \c <integral \c type>  stands for any integer type compatible with Eigen::Index (i.e. \c std::ptrdiff_t).
  *
  * The last statement implies compatibility with \c std::vector, \c std::valarray, \c std::array, many of the Range-v3's ranges, etc.
  *
  * If the submatrix can be represented using a starting position \c (i,j) and positive sizes \c (rows,columns), then this
  * method will returns a Block object after extraction of the relevant information from the passed arguments. This is the case
  * when all arguments are either:
  *  - An integer
  *  - Eigen::placeholders::all
  *  - An ArithmeticSequence with compile-time increment strictly equal to 1, as returned by Eigen::seq(a,b), and Eigen::seqN(a,N).
  *
  * Otherwise a more general IndexedView<Derived,RowIndices',ColIndices'> object will be returned, after conversion of the inputs
  * to more suitable types \c RowIndices' and \c ColIndices'.
  *
  * For 1D vectors and arrays, you better use the operator()(const Indices&) overload, which behave the same way but taking a single parameter.
  *
  * See also this <a href="https://stackoverflow.com/questions/46110917/eigen-replicate-items-along-one-dimension-without-useless-allocations">question</a> and its answer for an example of how to duplicate coefficients.
  *
  * \sa operator()(const Indices&), class Block, class IndexedView, DenseBase::block(Index,Index,Index,Index)
  */
template<typename RowIndices, typename ColIndices>
IndexedView_or_Block
operator()(const RowIndices& rowIndices, const ColIndices& colIndices);

/** This is an overload of operator()(const RowIndices&, const ColIndices&) for 1D vectors or arrays
  *
  * \only_for_vectors
  */
template<typename Indices>
IndexedView_or_VectorBlock
operator()(const Indices& indices);

#endif  // EIGEN_PARSED_BY_DOXYGEN
