// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Pavel Guzenfeld
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CONCAT_OP_H
#define EIGEN_CONCAT_OP_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

template <int Direction, typename LhsType, typename RhsType>
struct traits<ConcatOp<Direction, LhsType, RhsType>> : traits<LhsType> {
  typedef typename ScalarBinaryOpTraits<typename LhsType::Scalar, typename RhsType::Scalar>::ReturnType Scalar;
  typedef typename traits<LhsType>::StorageKind StorageKind;
  typedef typename traits<LhsType>::XprKind XprKind;
  typedef typename ref_selector<LhsType>::type LhsTypeNested;
  typedef typename ref_selector<RhsType>::type RhsTypeNested;
  typedef std::remove_reference_t<LhsTypeNested> LhsTypeNested_;
  typedef std::remove_reference_t<RhsTypeNested> RhsTypeNested_;
  enum {
    // For vertical concat (stacking rows): rows add up, cols must match
    // For horizontal concat (stacking cols): cols add up, rows must match
    LhsRows = int(LhsType::RowsAtCompileTime),
    RhsRows = int(RhsType::RowsAtCompileTime),
    LhsCols = int(LhsType::ColsAtCompileTime),
    RhsCols = int(RhsType::ColsAtCompileTime),

    RowsAtCompileTime = Direction == Vertical
                            ? (LhsRows == Dynamic || RhsRows == Dynamic ? int(Dynamic) : LhsRows + RhsRows)
                            : size_prefer_fixed(LhsRows, RhsRows),
    ColsAtCompileTime = Direction == Horizontal
                            ? (LhsCols == Dynamic || RhsCols == Dynamic ? int(Dynamic) : LhsCols + RhsCols)
                            : size_prefer_fixed(LhsCols, RhsCols),

    LhsMaxRows = int(LhsType::MaxRowsAtCompileTime),
    RhsMaxRows = int(RhsType::MaxRowsAtCompileTime),
    LhsMaxCols = int(LhsType::MaxColsAtCompileTime),
    RhsMaxCols = int(RhsType::MaxColsAtCompileTime),

    MaxRowsAtCompileTime =
        Direction == Vertical
            ? (LhsMaxRows == Dynamic || RhsMaxRows == Dynamic ? int(Dynamic) : LhsMaxRows + RhsMaxRows)
            : max_size_prefer_dynamic(LhsMaxRows, RhsMaxRows),
    MaxColsAtCompileTime =
        Direction == Horizontal
            ? (LhsMaxCols == Dynamic || RhsMaxCols == Dynamic ? int(Dynamic) : LhsMaxCols + RhsMaxCols)
            : max_size_prefer_dynamic(LhsMaxCols, RhsMaxCols),

    IsRowMajor = MaxRowsAtCompileTime == 1 && MaxColsAtCompileTime != 1   ? 1
                 : MaxColsAtCompileTime == 1 && MaxRowsAtCompileTime != 1 ? 0
                 : (int(LhsType::Flags) & RowMajorBit)                    ? 1
                                                                          : 0,
    Flags = IsRowMajor ? RowMajorBit : 0
  };
};

}  // namespace internal

/**
 * \class ConcatOp
 * \ingroup Core_Module
 *
 * \brief Expression of the concatenation of two dense expressions
 *
 * \tparam Direction either \c Vertical or \c Horizontal
 * \tparam LhsType the type of the left-hand side expression
 * \tparam RhsType the type of the right-hand side expression
 *
 * This class represents an expression of the concatenation of two dense expressions,
 * either vertically (stacking rows) or horizontally (stacking columns).
 *
 * It is the return type of hcat() and vcat() and typically this is the only way it is used.
 *
 * \sa hcat(), vcat()
 */
template <int Direction, typename LhsType, typename RhsType>
class ConcatOp : public internal::dense_xpr_base<ConcatOp<Direction, LhsType, RhsType>>::type {
  typedef typename internal::traits<ConcatOp>::LhsTypeNested LhsTypeNested;
  typedef typename internal::traits<ConcatOp>::RhsTypeNested RhsTypeNested;
  typedef typename internal::traits<ConcatOp>::LhsTypeNested_ LhsTypeNested_;
  typedef typename internal::traits<ConcatOp>::RhsTypeNested_ RhsTypeNested_;

 public:
  typedef typename internal::dense_xpr_base<ConcatOp>::type Base;
  EIGEN_DENSE_PUBLIC_INTERFACE(ConcatOp)
  typedef internal::remove_all_t<LhsType> LhsNestedExpression;
  typedef internal::remove_all_t<RhsType> RhsNestedExpression;

  template <typename OriginalLhsType, typename OriginalRhsType>
  EIGEN_DEVICE_FUNC constexpr inline ConcatOp(const OriginalLhsType& lhs, const OriginalRhsType& rhs)
      : m_lhs(lhs), m_rhs(rhs) {
    EIGEN_STATIC_ASSERT((internal::is_same<std::remove_const_t<LhsType>, OriginalLhsType>::value),
                        THE_MATRIX_OR_EXPRESSION_THAT_YOU_PASSED_DOES_NOT_HAVE_THE_EXPECTED_TYPE)
    EIGEN_STATIC_ASSERT((internal::is_same<std::remove_const_t<RhsType>, OriginalRhsType>::value),
                        THE_MATRIX_OR_EXPRESSION_THAT_YOU_PASSED_DOES_NOT_HAVE_THE_EXPECTED_TYPE)
    if (Direction == Vertical) {
      eigen_assert(lhs.cols() == rhs.cols() && "vcat: number of columns must match");
    } else {
      eigen_assert(lhs.rows() == rhs.rows() && "hcat: number of rows must match");
    }
  }

  EIGEN_DEVICE_FUNC constexpr Index rows() const {
    return Direction == Vertical ? m_lhs.rows() + m_rhs.rows() : m_lhs.rows();
  }
  EIGEN_DEVICE_FUNC constexpr Index cols() const {
    return Direction == Horizontal ? m_lhs.cols() + m_rhs.cols() : m_lhs.cols();
  }

  EIGEN_DEVICE_FUNC constexpr const LhsTypeNested_& lhs() const { return m_lhs; }
  EIGEN_DEVICE_FUNC constexpr const RhsTypeNested_& rhs() const { return m_rhs; }

 protected:
  LhsTypeNested m_lhs;
  RhsTypeNested m_rhs;
};

// Evaluator for ConcatOp
namespace internal {

template <int Direction, typename LhsType, typename RhsType>
struct evaluator<ConcatOp<Direction, LhsType, RhsType>> : evaluator_base<ConcatOp<Direction, LhsType, RhsType>> {
  typedef ConcatOp<Direction, LhsType, RhsType> XprType;
  typedef typename XprType::CoeffReturnType CoeffReturnType;

  typedef typename nested_eval<LhsType, 1>::type LhsNested;
  typedef typename nested_eval<RhsType, 1>::type RhsNested;
  typedef remove_all_t<LhsNested> LhsNestedCleaned;
  typedef remove_all_t<RhsNested> RhsNestedCleaned;

  enum {
    CoeffReadCost = plain_enum_max(evaluator<LhsNestedCleaned>::CoeffReadCost,
                                   evaluator<RhsNestedCleaned>::CoeffReadCost) +
                    NumTraits<typename XprType::Scalar>::AddCost,  // cost of the branch
    Flags = (traits<XprType>::Flags & RowMajorBit),
    Alignment = 0  // conservative: no alignment guarantees across boundary
  };

  EIGEN_DEVICE_FUNC constexpr EIGEN_STRONG_INLINE explicit evaluator(const XprType& xpr)
      : m_lhs(xpr.lhs()),
        m_rhs(xpr.rhs()),
        m_lhsImpl(m_lhs),
        m_rhsImpl(m_rhs),
        m_lhsRows(xpr.lhs().rows()),
        m_lhsCols(xpr.lhs().cols()) {}

  EIGEN_DEVICE_FUNC constexpr EIGEN_STRONG_INLINE CoeffReturnType coeff(Index row, Index col) const {
    if (Direction == Vertical) {
      if (row < m_lhsRows.value())
        return m_lhsImpl.coeff(row, col);
      else
        return m_rhsImpl.coeff(row - m_lhsRows.value(), col);
    } else {
      if (col < m_lhsCols.value())
        return m_lhsImpl.coeff(row, col);
      else
        return m_rhsImpl.coeff(row, col - m_lhsCols.value());
    }
  }

 protected:
  const LhsNested m_lhs;
  const RhsNested m_rhs;
  evaluator<LhsNestedCleaned> m_lhsImpl;
  evaluator<RhsNestedCleaned> m_rhsImpl;
  const variable_if_dynamic<Index, LhsType::RowsAtCompileTime> m_lhsRows;
  const variable_if_dynamic<Index, LhsType::ColsAtCompileTime> m_lhsCols;
};

}  // namespace internal

/**
 * \relates ConcatOp
 * \returns an expression of \a lhs and \a rhs concatenated horizontally (side by side).
 *
 * Both arguments must have the same number of rows.
 * To concatenate more than two expressions, chain calls: \c hcat(hcat(a, b), c).
 *
 * Example: \code
 * Matrix2d A, B;
 * auto C = hcat(A, B);  // C is 2x4
 * \endcode
 *
 * \sa vcat(), ConcatOp
 */
template <typename Lhs, typename Rhs>
EIGEN_DEVICE_FUNC inline const ConcatOp<Horizontal, Lhs, Rhs> hcat(const DenseBase<Lhs>& lhs,
                                                                   const DenseBase<Rhs>& rhs) {
  return ConcatOp<Horizontal, Lhs, Rhs>(lhs.derived(), rhs.derived());
}

/**
 * \relates ConcatOp
 * \returns an expression of \a lhs and \a rhs concatenated vertically (stacked on top of each other).
 *
 * Both arguments must have the same number of columns.
 * To concatenate more than two expressions, chain calls: \c vcat(vcat(a, b), c).
 *
 * Example: \code
 * Matrix2d A, B;
 * auto C = vcat(A, B);  // C is 4x2
 * \endcode
 *
 * \sa hcat(), ConcatOp
 */
template <typename Lhs, typename Rhs>
EIGEN_DEVICE_FUNC inline const ConcatOp<Vertical, Lhs, Rhs> vcat(const DenseBase<Lhs>& lhs, const DenseBase<Rhs>& rhs) {
  return ConcatOp<Vertical, Lhs, Rhs>(lhs.derived(), rhs.derived());
}

}  // end namespace Eigen

#endif  // EIGEN_CONCAT_OP_H
