// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_SPARSETRIANGULARSOLVER_H
#define EIGEN_SPARSETRIANGULARSOLVER_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

template <typename Lhs, typename Rhs, int Mode,
          int UpLo = (Mode & Lower)   ? Lower
                     : (Mode & Upper) ? Upper
                                      : -1,
          int StorageOrder = int(traits<Lhs>::Flags) & RowMajorBit>
struct sparse_solve_triangular_selector;

// forward substitution, row-major
template <typename Lhs, typename Rhs, int Mode>
struct sparse_solve_triangular_selector<Lhs, Rhs, Mode, Lower, RowMajor> {
  typedef typename Rhs::Scalar Scalar;
  typedef evaluator<Lhs> LhsEval;
  typedef typename evaluator<Lhs>::InnerIterator LhsIterator;
  static void run(const Lhs& lhs, Rhs& other) {
    LhsEval lhsEval(lhs);
    for (Index col = 0; col < other.cols(); ++col) {
      for (Index i = 0; i < lhs.rows(); ++i) {
        Scalar tmp = other.coeff(i, col);
        Scalar lastVal(0);
        Index lastIndex = 0;
        for (LhsIterator it(lhsEval, i); it; ++it) {
          lastVal = it.value();
          lastIndex = it.index();
          if (lastIndex == i) break;
          tmp = numext::madd<Scalar>(-lastVal, other.coeff(lastIndex, col), tmp);
        }
        EIGEN_IF_CONSTEXPR (Mode & UnitDiag)
          other.coeffRef(i, col) = tmp;
        else {
          eigen_assert(lastIndex == i);
          other.coeffRef(i, col) = tmp / lastVal;
        }
      }
    }
  }
};

// backward substitution, row-major
template <typename Lhs, typename Rhs, int Mode>
struct sparse_solve_triangular_selector<Lhs, Rhs, Mode, Upper, RowMajor> {
  typedef typename Rhs::Scalar Scalar;
  typedef evaluator<Lhs> LhsEval;
  typedef typename evaluator<Lhs>::InnerIterator LhsIterator;
  static void run(const Lhs& lhs, Rhs& other) {
    LhsEval lhsEval(lhs);
    for (Index col = 0; col < other.cols(); ++col) {
      for (Index i = lhs.rows() - 1; i >= 0; --i) {
        Scalar tmp = other.coeff(i, col);
        Scalar l_ii(0);
        LhsIterator it(lhsEval, i);
        while (it && it.index() < i) ++it;
        EIGEN_IF_CONSTEXPR (!(Mode & UnitDiag)) {
          eigen_assert(it && it.index() == i);
          l_ii = it.value();
          ++it;
        } else if (it && it.index() == i)
          ++it;
        for (; it; ++it) {
          tmp = numext::madd<Scalar>(-it.value(), other.coeff(it.index(), col), tmp);
        }

        EIGEN_IF_CONSTEXPR (Mode & UnitDiag)
          other.coeffRef(i, col) = tmp;
        else
          other.coeffRef(i, col) = tmp / l_ii;
      }
    }
  }
};

// forward substitution, col-major
template <typename Lhs, typename Rhs, int Mode>
struct sparse_solve_triangular_selector<Lhs, Rhs, Mode, Lower, ColMajor> {
  typedef typename Rhs::Scalar Scalar;
  typedef evaluator<Lhs> LhsEval;
  typedef typename evaluator<Lhs>::InnerIterator LhsIterator;
  static void run(const Lhs& lhs, Rhs& other) {
    LhsEval lhsEval(lhs);
    for (Index col = 0; col < other.cols(); ++col) {
      for (Index i = 0; i < lhs.cols(); ++i) {
        Scalar& tmp = other.coeffRef(i, col);
        if (!numext::is_exactly_zero(tmp))  // optimization when other is actually sparse
        {
          LhsIterator it(lhsEval, i);
          while (it && it.index() < i) ++it;
          EIGEN_IF_CONSTEXPR (!(Mode & UnitDiag)) {
            eigen_assert(it && it.index() == i);
            tmp /= it.value();
          }
          if (it && it.index() == i) ++it;
          for (; it; ++it) {
            other.coeffRef(it.index(), col) = numext::madd<Scalar>(-tmp, it.value(), other.coeffRef(it.index(), col));
          }
        }
      }
    }
  }
};

// backward substitution, col-major
template <typename Lhs, typename Rhs, int Mode>
struct sparse_solve_triangular_selector<Lhs, Rhs, Mode, Upper, ColMajor> {
  typedef typename Rhs::Scalar Scalar;
  typedef evaluator<Lhs> LhsEval;
  typedef typename evaluator<Lhs>::InnerIterator LhsIterator;
  static void run(const Lhs& lhs, Rhs& other) {
    LhsEval lhsEval(lhs);
    for (Index col = 0; col < other.cols(); ++col) {
      for (Index i = lhs.cols() - 1; i >= 0; --i) {
        Scalar& tmp = other.coeffRef(i, col);
        if (!numext::is_exactly_zero(tmp))  // optimization when other is actually sparse
        {
          EIGEN_IF_CONSTEXPR (!(Mode & UnitDiag)) {
            // TODO: replace this with a binary search. make sure the binary search is safe for partially sorted
            // elements
            LhsIterator it(lhsEval, i);
            while (it && it.index() != i) ++it;
            eigen_assert(it && it.index() == i);
            other.coeffRef(i, col) /= it.value();
          }
          LhsIterator it(lhsEval, i);
          for (; it && it.index() < i; ++it) {
            other.coeffRef(it.index(), col) = numext::madd<Scalar>(-tmp, it.value(), other.coeffRef(it.index(), col));
          }
        }
      }
    }
  }
};

}  // end namespace internal

#ifndef EIGEN_PARSED_BY_DOXYGEN

template <typename ExpressionType, unsigned int Mode>
template <typename OtherDerived>
void TriangularViewImpl<ExpressionType, Mode, Sparse>::solveInPlace(MatrixBase<OtherDerived>& other) const {
  eigen_assert(derived().cols() == derived().rows() && derived().cols() == other.rows());
  eigen_assert((!(Mode & ZeroDiag)) && bool(Mode & (Upper | Lower)));

  enum { copy = internal::traits<OtherDerived>::Flags & RowMajorBit };

  typedef std::conditional_t<copy, typename internal::plain_matrix_type_column_major<OtherDerived>::type, OtherDerived&>
      OtherCopy;
  OtherCopy otherCopy(other.derived());

  internal::sparse_solve_triangular_selector<ExpressionType, std::remove_reference_t<OtherCopy>, Mode>::run(
      derived().nestedExpression(), otherCopy);

  if (copy) other = otherCopy;
}
#endif

// pure sparse path

namespace internal {

template <typename Lhs, typename Rhs, int Mode,
          int UpLo = (Mode & Lower)   ? Lower
                     : (Mode & Upper) ? Upper
                                      : -1,
          int StorageOrder = int(Lhs::Flags) & RowMajorBit>
struct sparse_solve_triangular_sparse_selector;

// forward and backward substitution, col-major
template <typename Lhs, typename Rhs, int Mode, int UpLo>
struct sparse_solve_triangular_sparse_selector<Lhs, Rhs, Mode, UpLo, ColMajor> {
  typedef typename Rhs::Scalar Scalar;
  typedef typename promote_index_type<typename traits<Lhs>::StorageIndex, typename traits<Rhs>::StorageIndex>::type
      StorageIndex;
  static void run(const Lhs& lhs, Rhs& other) {
    const bool IsLower = (UpLo == Lower);
    AmbiVector<Scalar, StorageIndex> tempVector(other.rows() * 2);
    tempVector.setBounds(0, other.rows());

    Rhs res(other.rows(), other.cols());
    res.reserve(other.nonZeros());

    for (Index col = 0; col < other.cols(); ++col) {
      // FIXME: estimate the number of non-zeros per column for better allocation.
      tempVector.init(.99 /*float(other.col(col).nonZeros())/float(other.rows())*/);
      tempVector.setZero();
      tempVector.restart();
      for (typename Rhs::InnerIterator rhsIt(other, col); rhsIt; ++rhsIt) {
        tempVector.coeffRef(rhsIt.index()) = rhsIt.value();
      }

      for (Index i = IsLower ? 0 : lhs.cols() - 1; IsLower ? i < lhs.cols() : i >= 0; i += IsLower ? 1 : -1) {
        tempVector.restart();
        Scalar& ci = tempVector.coeffRef(i);
        if (!numext::is_exactly_zero(ci)) {
          // find
          typename Lhs::InnerIterator it(lhs, i);
          EIGEN_IF_CONSTEXPR (!(Mode & UnitDiag)) {
            EIGEN_IF_CONSTEXPR (IsLower) {
              eigen_assert(it.index() == i);
              ci /= it.value();
            } else
              ci /= lhs.coeff(i, i);
          }
          tempVector.restart();
          EIGEN_IF_CONSTEXPR (IsLower) {
            if (it.index() == i) ++it;
            for (; it; ++it) {
              tempVector.coeffRef(it.index()) = numext::madd<Scalar>(-ci, it.value(), tempVector.coeffRef(it.index()));
            }
          } else {
            for (; it && it.index() < i; ++it) {
              tempVector.coeffRef(it.index()) = numext::madd<Scalar>(-ci, it.value(), tempVector.coeffRef(it.index()));
            }
          }
        }
      }

      // FIXME: compute a reference value to filter zeros.
      for (typename AmbiVector<Scalar, StorageIndex>::Iterator it(tempVector /*,1e-12*/); it; ++it) {
        // FIXME: use insertBack for better performance.
        res.insert(it.index(), col) = it.value();
      }
    }
    res.finalize();
    other = res.markAsRValue();
  }
};

// Reach-based (Gilbert-Peierls) sparse triangular solve, col-major, for lower OR
// upper. Only the columns reachable from each rhs column's pattern are touched, so
// the cost is O(|reach| + flops) per column instead of the O(n)-per-column
// AmbiVector sweep (which also pays a coeff(i,i) binary search per row in the upper
// case). More specialized than the generic AmbiVector selector above, so it is
// selected for both Lower/ColMajor and Upper/ColMajor. The body is shared; only the
// solve's Upper flag differs between the two specializations below.
template <bool Upper, typename Lhs, typename Rhs, int Mode>
void run_sparse_reach_triangular_solve(const Lhs& lhs, Rhs& other) {
  typedef typename Rhs::Scalar Scalar;
  typedef typename traits<Lhs>::StorageIndex StorageIndex;
  Index size = lhs.rows();

  // Reused across all rhs columns, so repeated columns stay allocation-free. We use
  // reach_solve_dense, which leaves the solution values in xwork and the reached
  // indices in iwork[top..n): we read values straight out of xwork at insert time and
  // clear as we go, so no outIdx/outVal snapshot is needed. It also expects the rhs
  // pre-scattered into xwork, which we do while reading the column -- so only the rhs
  // pattern bIdx is materialized, not its values.
  //
  // One StorageIndex allocation of 3n, carved xi | pstack | bIdx: the reach reads bIdx
  // while writing the disjoint xi/pstack, so they coexist safely. None of it needs
  // zero-init -- every slot is written before it is read -- so only mark and xwork are
  // zeroed. All three buffers are restored by every column.
  Matrix<StorageIndex, Dynamic, 1> iwork(3 * size);
  Matrix<uint8_t, Dynamic, 1> mark = Matrix<uint8_t, Dynamic, 1>::Zero(size);
  Matrix<Scalar, Dynamic, 1> xwork = Matrix<Scalar, Dynamic, 1>::Zero(size);
  StorageIndex* xi = iwork.data();
  StorageIndex* bIdx = iwork.data() + 2 * size;  // reach roots; disjoint from xi | pstack

  Rhs res(other.rows(), other.cols());
  res.reserve(other.nonZeros());

  for (Index col = 0; col < other.cols(); ++col) {
    Index bCount = 0;
    for (typename Rhs::InnerIterator it(other, col); it; ++it) {
      bIdx[bCount] = StorageIndex(it.index());
      xwork[it.index()] = it.value();  // scatter as we read; no separate bVal buffer
      ++bCount;
    }
    if (bCount == 0) continue;

    Index top = reach_solve_dense<Upper, bool(Mode & UnitDiag)>(lhs, bIdx, bCount, xi, mark.data(), xwork.data());

    // The reach is in topological order; sort the reached indices ascending so the
    // column is written with increasing inner index. Then read values from xwork and
    // clear it and mark for the next column.
    std::sort(xi + top, xi + size);
    for (Index k = top; k < size; ++k) {
      StorageIndex j = xi[k];
      res.insert(j, col) = xwork[j];
      xwork[j] = Scalar(0);
      mark[j] = 0;
    }
  }
  res.finalize();
  other = res.markAsRValue();
}

template <typename Lhs, typename Rhs, int Mode>
struct sparse_solve_triangular_sparse_selector<Lhs, Rhs, Mode, Lower, ColMajor> {
  static void run(const Lhs& lhs, Rhs& other) { run_sparse_reach_triangular_solve<false, Lhs, Rhs, Mode>(lhs, other); }
};

template <typename Lhs, typename Rhs, int Mode>
struct sparse_solve_triangular_sparse_selector<Lhs, Rhs, Mode, Upper, ColMajor> {
  static void run(const Lhs& lhs, Rhs& other) { run_sparse_reach_triangular_solve<true, Lhs, Rhs, Mode>(lhs, other); }
};

}  // end namespace internal

#ifndef EIGEN_PARSED_BY_DOXYGEN
template <typename ExpressionType, unsigned int Mode>
template <typename OtherDerived>
void TriangularViewImpl<ExpressionType, Mode, Sparse>::solveInPlace(SparseMatrixBase<OtherDerived>& other) const {
  eigen_assert(derived().cols() == derived().rows() && derived().cols() == other.rows());
  eigen_assert((!(Mode & ZeroDiag)) && bool(Mode & (Upper | Lower)));

  internal::sparse_solve_triangular_sparse_selector<ExpressionType, OtherDerived, Mode>::run(
      derived().nestedExpression(), other.derived());
}
#endif

}  // end namespace Eigen

#endif  // EIGEN_SPARSETRIANGULARSOLVER_H
