// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_TRIANGULAR_REACH_SOLVER_H
#define EIGEN_TRIANGULAR_REACH_SOLVER_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

// PROTOTYPE: Gilbert-Peierls sparse-rhs triangular solve.
//
// Solves L x = b for a lower-triangular, column-major, compressed L whose
// columns are sorted with the diagonal stored first (Li[Lp[j]] == j), and a
// sparse right-hand side b. The cost is O(|reach| + flops): only the columns
// reachable from b's pattern are touched, independent of the dimension n.
//
// The reach strategy is a compile-time policy: the two variants share one DFS
// scaffold and differ only in how many successors each node exposes.
enum class TriangularReach {
  // Scan every sub-diagonal entry of a column. Valid for ANY triangular L.
  Dfs,
  // Follow only the first sub-diagonal entry (= elimination-tree parent). Valid
  // ONLY for a symbolic Cholesky factor, whose fill property guarantees the
  // remaining sub-diagonal entries are reached transitively through the parent.
  EliminationTree
};

// Computes reach_{G(L)}(pattern(b)) via a non-recursive depth-first search,
// emitting the reached columns into xi[top..n) in topological (solve) order and
// returning top. `mark` (length n) is a visited flag that must be all-zero on
// entry; every reached node is flagged, and since the reached set is exactly the
// output, the caller clears those flags again while gathering (no reset needed).
// `xi` and `pstack` are size-n scratch; the DFS stack occupies xi[0..head] while
// the output grows down from xi[n), and head < top holds so they never overlap.
template <TriangularReach Reach, typename StorageIndex>
Index lower_triangular_reach(const StorageIndex* Lp, const StorageIndex* Li, const StorageIndex* bIdx, Index bCount,
                             StorageIndex* xi, StorageIndex* pstack, StorageIndex* mark, Index n) {
  Index top = n;
  for (Index r = 0; r < bCount; ++r) {
    StorageIndex root = bIdx[r];
    if (mark[root]) continue;

    Index head = 0;
    xi[0] = root;
    while (head >= 0) {
      StorageIndex j = xi[head];
      Index colStart = Lp[j] + 1;  // skip the diagonal stored at Lp[j]
      Index colEnd = Lp[j + 1];
      if (!mark[j]) {
        mark[j] = 1;
        pstack[head] = StorageIndex(colStart);
      }
      // Dfs exposes the whole sub-diagonal; EliminationTree only the parent.
      Index scanEnd =
          (Reach == TriangularReach::EliminationTree) ? (colStart < colEnd ? colStart + 1 : colStart) : colEnd;
      bool done = true;
      for (Index p = pstack[head]; p < scanEnd; ++p) {
        StorageIndex i = Li[p];
        if (mark[i]) continue;  // already visited
        pstack[head] = StorageIndex(p + 1);
        xi[++head] = i;  // descend
        done = false;
        break;
      }
      if (done) {         // no unvisited successor: postorder j
        xi[--top] = j;
        --head;
      }
    }
  }
  return top;
}

// Alternative reach via a mark-on-push worklist: nodes are marked when pushed
// (so the stack stays within n and needs no pstack), the whole column is scanned
// at once (no pause/resume, hence no per-depth state), and the collected set is
// then sorted -- ascending index order is a valid topological order for a lower
// solve. Cost is O(|reach| log|reach|) vs the DFS's O(|reach|); it trades that
// log factor for holding only one column cursor at a time, which suits an
// InnerIterator-based column source. Scratch: xi (n, shared stack+output) and
// mark (n, all-zero in, reach flagged out). Returns top; xi[top..n) is sorted.
template <TriangularReach Reach, typename StorageIndex>
Index lower_triangular_reach_worklist(const StorageIndex* Lp, const StorageIndex* Li, const StorageIndex* bIdx,
                                      Index bCount, StorageIndex* xi, StorageIndex* mark, Index n) {
  Index top = n;
  Index sp = 0;  // worklist occupies xi[0..sp); output grows down from xi[n); sp <= top always
  for (Index r = 0; r < bCount; ++r) {
    StorageIndex root = bIdx[r];
    if (!mark[root]) {
      mark[root] = 1;
      xi[sp++] = root;
    }
  }
  while (sp > 0) {
    StorageIndex j = xi[--sp];
    xi[--top] = j;  // collect
    Index colStart = Lp[j] + 1, colEnd = Lp[j + 1];
    Index scanEnd =
        (Reach == TriangularReach::EliminationTree) ? (colStart < colEnd ? colStart + 1 : colStart) : colEnd;
    for (Index p = colStart; p < scanEnd; ++p) {
      StorageIndex i = Li[p];
      if (!mark[i]) {
        mark[i] = 1;
        xi[sp++] = i;
      }
    }
  }
  std::sort(xi + top, xi + n);
  return top;
}

// Numeric sweep over a precomputed reach: solves L x = b in place on the dense
// accumulator x (zero except where b was scattered), touching only the reached
// columns in the order produced by lower_triangular_reach.
template <bool UnitDiag, typename StorageIndex, typename Scalar>
void lower_triangular_solve_over_reach(const StorageIndex* Lp, const StorageIndex* Li, const Scalar* Lx,
                                       const StorageIndex* xi, Index top, Index n, Scalar* x) {
  for (Index k = top; k < n; ++k) {
    StorageIndex j = xi[k];
    Index p = Lp[j];
    EIGEN_IF_CONSTEXPR(!UnitDiag) {
      x[j] /= Lx[p];  // diagonal stored first
      ++p;
    }
    else {
      if (p < Lp[j + 1] && Li[p] == j) ++p;  // skip an explicit unit diagonal
    }
    Scalar xj = x[j];
    for (; p < Lp[j + 1]; ++p) x[Li[p]] -= Lx[p] * xj;
  }
}

// Solve L x = b for a lower-triangular, column-major, diagonal-first L and a
// sparse rhs (bIdx/bVal), writing the solution's nonzero indices/values into the
// caller's outIdx/outVal (each with capacity >= n) in topological order, and
// returning their count.
//
// Borrow-a-buffer, like the elimination_tree helpers: the caller owns the scratch
// and passes it in, so a reused buffer makes repeated solves allocation-free.
//   - iwork: >= 3n StorageIndex, carved into xi | pstack | mark (each length n).
//   - xwork: >= n Scalar, the dense accumulator.
// Preconditions (both restored on exit, so one setup suffices for many solves):
//   - iwork's mark third (iwork[2n, 3n)) is all-zero.
//   - xwork is all-zero.
template <TriangularReach Reach, bool UnitDiag, typename StorageIndex, typename Scalar>
Index sparse_reach_lower_solve(const StorageIndex* Lp, const StorageIndex* Li, const Scalar* Lx, Index n,
                               const StorageIndex* bIdx, const Scalar* bVal, Index bCount, StorageIndex* iwork,
                               Scalar* xwork, StorageIndex* outIdx, Scalar* outVal) {
  StorageIndex* xi = iwork;
  StorageIndex* pstack = iwork + n;
  StorageIndex* mark = iwork + 2 * n;
  Scalar* x = xwork;

  for (Index r = 0; r < bCount; ++r) x[bIdx[r]] = bVal[r];
  Index top = lower_triangular_reach<Reach>(Lp, Li, bIdx, bCount, xi, pstack, mark, n);
  lower_triangular_solve_over_reach<UnitDiag>(Lp, Li, Lx, xi, top, n, x);

  Index count = 0;
  for (Index k = top; k < n; ++k) {  // gather, restoring x and mark to all-zero
    StorageIndex j = xi[k];
    outIdx[count] = j;
    outVal[count] = x[j];
    x[j] = Scalar(0);
    mark[j] = 0;
    ++count;
  }
  return count;
}

// Convenience overload: no scratch supplied, so allocate (and zero) it for this
// single solve.
template <TriangularReach Reach, bool UnitDiag, typename StorageIndex, typename Scalar>
Index sparse_reach_lower_solve(const StorageIndex* Lp, const StorageIndex* Li, const Scalar* Lx, Index n,
                               const StorageIndex* bIdx, const Scalar* bVal, Index bCount, StorageIndex* outIdx,
                               Scalar* outVal) {
  Matrix<StorageIndex, Dynamic, 1> iwork = Matrix<StorageIndex, Dynamic, 1>::Zero(3 * n);
  Matrix<Scalar, Dynamic, 1> xwork = Matrix<Scalar, Dynamic, 1>::Zero(n);
  return sparse_reach_lower_solve<Reach, UnitDiag>(Lp, Li, Lx, n, bIdx, bVal, bCount, iwork.data(), xwork.data(),
                                                   outIdx, outVal);
}

// ---------------------------------------------------------------------------
// Iterator-driven path: for a lower-triangular, column-major sparse expression
// that does NOT expose raw CSC storage (has_compressed_access is false). Columns
// are read through the expression's evaluator InnerIterator, which yields the
// diagonal first and the sub-diagonal after (the same sorted precondition as the
// pointer path). Because an InnerIterator can't cheaply hold DFS resume state,
// the reach uses the worklist+sort form; the log factor is empirically ~free.
// ---------------------------------------------------------------------------

// Reach via a mark-on-push worklist that opens one column at a time through
// InnerIterator. Same contract as lower_triangular_reach_worklist otherwise.
template <TriangularReach Reach, typename Eval, typename StorageIndex>
Index lower_triangular_reach_iter(const Eval& mat, const StorageIndex* bIdx, Index bCount, StorageIndex* xi,
                                  StorageIndex* mark, Index n) {
  Index top = n;
  Index sp = 0;
  for (Index r = 0; r < bCount; ++r) {
    StorageIndex root = bIdx[r];
    if (!mark[root]) {
      mark[root] = 1;
      xi[sp++] = root;
    }
  }
  while (sp > 0) {
    StorageIndex j = xi[--sp];
    xi[--top] = j;  // collect
    typename Eval::InnerIterator it(mat, j);
    if (it) ++it;  // skip the diagonal stored first
    for (; it; ++it) {
      StorageIndex i = StorageIndex(it.index());
      if (!mark[i]) {
        mark[i] = 1;
        xi[sp++] = i;
      }
      if (Reach == TriangularReach::EliminationTree) break;  // only the parent
    }
  }
  std::sort(xi + top, xi + n);
  return top;
}

// Numeric sweep over the reach, reading columns through InnerIterator.
template <bool UnitDiag, typename Eval, typename StorageIndex, typename Scalar>
void lower_triangular_solve_over_reach_iter(const Eval& mat, const StorageIndex* xi, Index top, Index n, Scalar* x) {
  for (Index k = top; k < n; ++k) {
    StorageIndex j = xi[k];
    typename Eval::InnerIterator it(mat, j);
    EIGEN_IF_CONSTEXPR(!UnitDiag) {
      x[j] /= it.value();  // diagonal stored first
      ++it;
    }
    else {
      if (it && StorageIndex(it.index()) == j) ++it;  // skip an explicit unit diagonal
    }
    Scalar xj = x[j];
    for (; it; ++it) x[it.index()] -= it.value() * xj;
  }
}

// Iterator counterpart of the borrow-a-buffer sparse_reach_lower_solve. Uses the
// same 3n / n workspace layout (the pstack third is simply left unused), so the
// two paths share one workspace contract.
template <TriangularReach Reach, bool UnitDiag, typename LhsType, typename StorageIndex, typename Scalar>
Index sparse_reach_lower_solve_iter(const LhsType& lhs, Index n, const StorageIndex* bIdx, const Scalar* bVal,
                                    Index bCount, StorageIndex* iwork, Scalar* xwork, StorageIndex* outIdx,
                                    Scalar* outVal) {
  evaluator<LhsType> mat(lhs);
  StorageIndex* xi = iwork;
  StorageIndex* mark = iwork + 2 * n;
  Scalar* x = xwork;

  for (Index r = 0; r < bCount; ++r) x[bIdx[r]] = bVal[r];
  Index top = lower_triangular_reach_iter<Reach>(mat, bIdx, bCount, xi, mark, n);
  lower_triangular_solve_over_reach_iter<UnitDiag>(mat, xi, top, n, x);

  Index count = 0;
  for (Index k = top; k < n; ++k) {  // gather, restoring x and mark to all-zero
    StorageIndex j = xi[k];
    outIdx[count] = j;
    outVal[count] = x[j];
    x[j] = Scalar(0);
    mark[j] = 0;
    ++count;
  }
  return count;
}

// Policy dispatch: an expression that exposes raw storage takes the pointer + DFS
// fast path; anything else takes the evaluator + worklist path. Tag dispatch (not
// if-constexpr) keeps the untaken branch from being instantiated, so outerIndexPtr()
// is never named on a type that lacks it.
template <TriangularReach Reach, bool UnitDiag, typename LhsType, typename StorageIndex, typename Scalar>
Index sparse_reach_lower_solve_dispatch(std::true_type /*compressed*/, const LhsType& lhs, Index n,
                                        const StorageIndex* bIdx, const Scalar* bVal, Index bCount,
                                        StorageIndex* iwork, Scalar* xwork, StorageIndex* outIdx, Scalar* outVal) {
  return sparse_reach_lower_solve<Reach, UnitDiag>(lhs.outerIndexPtr(), lhs.innerIndexPtr(), lhs.valuePtr(), n, bIdx,
                                                   bVal, bCount, iwork, xwork, outIdx, outVal);
}
template <TriangularReach Reach, bool UnitDiag, typename LhsType, typename StorageIndex, typename Scalar>
Index sparse_reach_lower_solve_dispatch(std::false_type /*iterator*/, const LhsType& lhs, Index n,
                                        const StorageIndex* bIdx, const Scalar* bVal, Index bCount,
                                        StorageIndex* iwork, Scalar* xwork, StorageIndex* outIdx, Scalar* outVal) {
  return sparse_reach_lower_solve_iter<Reach, UnitDiag>(lhs, n, bIdx, bVal, bCount, iwork, xwork, outIdx, outVal);
}

// Expression overload: solve L x = b for a sparse-expression lower-triangular L
// and sparse rhs, selecting the pointer or iterator path at compile time. Borrow
// a 3n StorageIndex / n Scalar workspace as in the pointer overload.
template <TriangularReach Reach, bool UnitDiag, typename LhsDerived, typename StorageIndex, typename Scalar>
Index sparse_reach_lower_solve(const SparseMatrixBase<LhsDerived>& lhs, const StorageIndex* bIdx, const Scalar* bVal,
                               Index bCount, StorageIndex* iwork, Scalar* xwork, StorageIndex* outIdx,
                               Scalar* outVal) {
  return sparse_reach_lower_solve_dispatch<Reach, UnitDiag>(
      std::integral_constant<bool, has_compressed_access<LhsDerived>::value>{}, lhs.derived(), lhs.rows(), bIdx, bVal,
      bCount, iwork, xwork, outIdx, outVal);
}

// Convenience expression overload: allocate (and zero) the workspace for one solve.
template <TriangularReach Reach, bool UnitDiag, typename LhsDerived, typename StorageIndex, typename Scalar>
Index sparse_reach_lower_solve(const SparseMatrixBase<LhsDerived>& lhs, const StorageIndex* bIdx, const Scalar* bVal,
                               Index bCount, StorageIndex* outIdx, Scalar* outVal) {
  Index n = lhs.rows();
  Matrix<StorageIndex, Dynamic, 1> iwork = Matrix<StorageIndex, Dynamic, 1>::Zero(3 * n);
  Matrix<Scalar, Dynamic, 1> xwork = Matrix<Scalar, Dynamic, 1>::Zero(n);
  return sparse_reach_lower_solve<Reach, UnitDiag>(lhs, bIdx, bVal, bCount, iwork.data(), xwork.data(), outIdx,
                                                   outVal);
}

}  // namespace internal

}  // namespace Eigen

#endif  // EIGEN_TRIANGULAR_REACH_SOLVER_H
