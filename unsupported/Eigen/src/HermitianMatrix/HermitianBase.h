// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2018 David Tellenbach <david.tellenbach@tellnotes.org>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_HERMITIANBASE_H
#define EIGEN_HERMITIANBASE_H
namespace Eigen {

template<typename Derived>
class HermitianBase : public EigenBase<Derived>
{
  public:
#ifndef EIGEN_PARSED_BY_DOXYGEN
    typedef typename internal::traits<Derived>::NestedExpression NestedExpression;
    typedef typename internal::traits<Derived>::NestedExpression Nested;
    typedef typename internal::traits<Derived>::DenseType DenseType;
    typedef typename NestedExpression::Scalar Scalar;
    typedef typename internal::conditional<bool(internal::traits<Derived>::Flags&LvalueBit), const Scalar&,
      typename internal::conditional<internal::is_arithmetic<Scalar>::value, Scalar, const Scalar>::type>::type
      CoeffReturnType;
#endif

    enum {
      UpLo = internal::traits<Derived>::UpLo,
      RowsAtCompileTime = internal::traits<Derived>::RowsAtCompileTime,
      ColsAtCompileTime = internal::traits<Derived>::ColsAtCompileTime,
      MaxRowsAtCompileTime = internal::traits<Derived>::MaxRowsAtCompileTime,
      MaxColsAtCompileTime = internal::traits<Derived>::MaxColsAtCompileTime,
      SizeAtCompileTime = RowsAtCompileTime * ColsAtCompileTime,
      MaxSizeAtCompileTime = MaxRowsAtCompileTime * MaxColsAtCompileTime,
      IsRowMajor = NestedExpression::IsRowMajor,
      IsVectorAtCompileTime = 0
    };

    template <typename OtherDerived>
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    Derived& operator=(const HermitianBase<OtherDerived>& other)
    {
      internal::call_assignment(derived(), other.derived());
      return derived();
    }

    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    Derived& _set(const HermitianBase<OtherDerived>& other)
    {
      internal::call_assignment(derived(), other.derived());
      return derived();
    }

    template <typename OtherDerived>
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    Derived& operator=(const DenseBase<OtherDerived>& other)
    {
      internal::call_assignment(derived(), other.derived());
      return derived();
    }
   

    /** \returns constant reference to Derived */
    EIGEN_DEVICE_FUNC
    inline const Derived& derived() const
    {
      return *static_cast<const Derived*>(this);
    }

    /** \returns reference to Derived */
    EIGEN_DEVICE_FUNC
    inline Derived& derived()
    {
      return *static_cast<Derived*>(this);
    }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    Scalar coeff(Index row, Index col) const
    {
      eigen_internal_assert(row >= 0 && row < rows()
                         && col >= 0 && col < cols());
      return internal::evaluator<HermitianBase<Derived> >(derived()).coeff(row, col);
    }

    /** \returns a reference to the coefficient at given the given row and
     *  column */
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    HermitianMatrixCoeffReturnHelper<Scalar, NestedExpression> operator()(Index row, Index col)
    {
      eigen_assert(row >= 0 && row < rows() && col >= 0 && col < cols());
      return coeffRef(row, col);
    }

    /** Short version: don't use this function, use
     *  \link operator()(Index,Index) \endlink instead.
     *
     *  Long version: this function is similar to
     *  \link operator()(Index,Index) \endlink, but without the assertion.
     *  Use this for limiting the performance cost of debugging code when doing
     *  repeated coefficient access. Only use this when it is guaranteed that the
     *  parameters \a row and \a col are in range.
     *
     *  If EIGEN_INTERNAL_DEBUGGING is defined, an assertion will be made, making this
     *  function equivalent to \link operator()(Index,Index) \endlink.
     *
     *  \sa operator()(Index,Index), coeff(Index, Index) const
     */
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    HermitianMatrixCoeffReturnHelper<Scalar, NestedExpression> coeffRef(Index row, Index col)
    {
      eigen_internal_assert(row >= 0 && row < rows() && col >= 0 && col < cols());
      return internal::evaluator<HermitianBase<Derived> >(derived()).coeffRef(row, col);
    }

    /** \returns a const reference to the nested expression */
    EIGEN_DEVICE_FUNC
    const NestedExpression& nestedExpression() const
    {
      return derived().nestedExpression();
    }

    /** \returns a reference to the nested expression */
    EIGEN_DEVICE_FUNC
    NestedExpression& nestedExpression()
    { 
      return derived().nestedExpression();
    }
    
    /** \returns the number of columns. \sa rows(), ColsAtCompileTime*/
    EIGEN_DEVICE_FUNC
    Index cols()
    {
      return derived().cols();
    }

    /** \returns the number of columns. \sa rows(), ColsAtCompileTime*/
    EIGEN_DEVICE_FUNC
    Index cols() const
    {
      return derived().cols();
    }

    /** \returns the number of rows. \sa cols(), RowsAtCompileTime */
    EIGEN_DEVICE_FUNC
    Index rows()
    {
      return derived().rows();
    }

    /** \returns the number of rows. \sa cols(), RowsAtCompileTime */
    EIGEN_DEVICE_FUNC
    Index rows() const
    {
      return derived().rows();
    }

    EIGEN_DEVICE_FUNC
    DenseType toDenseMatrix() const { return derived(); }

    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    Derived& operator+=(const HermitianBase<OtherDerived>& other)
    {
      call_assignment(nestedExpression(), other.nestedExpression(),
                      internal::add_assign_op<Scalar, typename OtherDerived::Scalar>());
      return derived();
    }

    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    Derived& operator-=(const HermitianBase<OtherDerived>& other)
    {
      call_assignment(nestedExpression(), other.nestedExpression(),
                      internal::sub_assign_op<Scalar, typename OtherDerived::Scalar>());
      return derived();
    }

    /** \returns CwiseBinaryOp that represents an abstract sum expression */
    template<typename OtherDerived> 
    EIGEN_STRONG_INLINE const
    CwiseBinaryOp<internal::scalar_sum_op<typename internal::traits<Derived>::Scalar,
                  typename internal::traits<OtherDerived>::Scalar >, const Derived, const OtherDerived>
    operator+(const HermitianBase<OtherDerived> &other) const
    { 
      return CwiseBinaryOp<internal::scalar_sum_op<typename internal::traits<Derived>::Scalar,
                  typename internal::traits<OtherDerived>::Scalar >, const Derived, const OtherDerived>
              (derived(), other.derived());
    }

    /** \returns CwiseBinaryOp that represents an abstract difference expression */
    template<typename OtherDerived> 
    EIGEN_STRONG_INLINE const
    CwiseBinaryOp<internal::scalar_difference_op<typename internal::traits<Derived>::Scalar,
                  typename internal::traits<OtherDerived>::Scalar >, const Derived, const OtherDerived>
    operator-(const HermitianBase<OtherDerived> &other) const
    { 
      return CwiseBinaryOp<internal::scalar_difference_op<typename internal::traits<Derived>::Scalar,
                  typename internal::traits<OtherDerived>::Scalar >, const Derived, const OtherDerived>
              (derived(), other.derived());
    }


    // hermitian * hermitian
    template<typename OtherDerived> 
    EIGEN_STRONG_INLINE const
    Product<Derived, OtherDerived, LazyProduct>
    operator*(const HermitianBase<OtherDerived>& rhs) const
    { 
      return Product<Derived, OtherDerived, LazyProduct>(derived(), rhs.derived());
    }

    // hermitian * dense
    template<typename OtherDerived> 
    EIGEN_STRONG_INLINE const
    Product<Derived, OtherDerived, LazyProduct>
    operator*(const MatrixBase<OtherDerived> &other) const
    { 
      return Product<Derived, OtherDerived, LazyProduct>(derived(), other.derived());
    } 

    typedef typename internal::add_const_on_value_type<typename internal::eval<Derived>::type>::type EvalReturnType;

    /** \returns the matrix or vector obtained by evaluating this expression.
      *
      * Notice that in the case of a plain matrix or vector (not an expression) this function just returns
      * a const reference, in order to avoid a useless copy.
      * 
      * \warning Be careful with eval() and the auto C++ keyword, as detailed in this \link TopicPitfalls_auto_keyword page \endlink.
      */
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE EvalReturnType eval() const
    {
      // Even though MSVC does not honor strong inlining when the return type
      // is a dynamic matrix, we desperately need strong inlining for fixed
      // size types on MSVC.
      return typename internal::eval<Derived>::type(derived());
    }
};

} // namespace Eigen

#endif // EIGEN_HERMITIANBASE_H
