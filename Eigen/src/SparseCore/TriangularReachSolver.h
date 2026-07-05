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
// Solves T x = b for a column-major triangular T (lower OR upper) with sorted
// columns, and a sparse right-hand side b. The cost is O(|reach| + flops): only
// the columns reachable from b's pattern are touched, independent of the dimension.
//
// Direction (lower vs upper) is a compile-time bool `Upper`, but it affects very
// little. The reach DFS itself is direction-AGNOSTIC: a lower-triangular column
// holds only entries with inner index >= j, an upper one only <= j, so scanning the
// whole stored column and skipping the self-diagonal (via `mark`) yields exactly the
// right reach successors either way -- the same trick `cs_spsolve` uses to serve
// both lo=1 and lo=0 from one cs_reach. Only two things depend on `Upper`:
//   - the numeric sweep, where the diagonal is the FIRST stored entry of a column
//     for lower and the LAST for upper;
//   - the iterator worklist reach, whose collected set is sorted into topological
//     order (ascending for lower, descending for upper).
//
// A separate elimination-tree reach (lower_triangular_ereach) specializes the lower
// forward solve for a Cholesky-type factor: O(reach), no resume stack, no sort.

// ===========================================================================
// Reach (direction-agnostic): any triangular T, raw CSC storage.
// ===========================================================================

// Computes reach_{G(T)}(pattern(b)) via a non-recursive depth-first search,
// emitting the reached columns into xi[top..n) in topological (solve) order and
// returning top. `mark` (length-n byte array; a 0/1 visited flag needs no more)
// must be all-zero on entry; every reached node is flagged, and since the reached
// set is exactly the output, the caller clears those flags again while gathering
// (no reset needed). `xi` and `pstack` are size-n scratch; the DFS stack occupies
// xi[0..head] while the output grows down from xi[n), and head < top holds so they
// never overlap. The whole stored column is scanned; the diagonal is index j itself,
// so mark[j] (set on entry to j) skips it -- no diagonal-position knowledge needed.
// `innerNonZeroPtr` is the per-column nonzero count: pass it for an
// uncompressed matrix so column j ends at outerIndexPtr[j]+innerNonZeroPtr[j]; pass nullptr (compressed)
// to end at outerIndexPtr[j+1].
template <typename StorageIndex>
Index triangular_reach(const StorageIndex* outerIndexPtr, const StorageIndex* innerIndexPtr,
                       const StorageIndex* innerNonZeroPtr, const StorageIndex* bIdx, Index bCount, StorageIndex* xi,
                       StorageIndex* pstack, uint8_t* mark, Index n) {
  Index top = n;
  for (Index r = 0; r < bCount; ++r) {
    StorageIndex root = bIdx[r];
    if (mark[root]) continue;

    Index head = 0;
    xi[0] = root;
    while (head >= 0) {
      StorageIndex j = xi[head];
      Index colBeg = outerIndexPtr[j];
      Index colEnd = innerNonZeroPtr ? outerIndexPtr[j] + innerNonZeroPtr[j] : outerIndexPtr[j + 1];
      if (!mark[j]) {
        mark[j] = 1;
        pstack[head] = StorageIndex(colBeg);
      }
      bool done = true;
      for (Index p = pstack[head]; p < colEnd; ++p) {
        StorageIndex i = innerIndexPtr[p];
        if (mark[i]) continue;  // self-diagonal (i == j) or already visited
        pstack[head] = StorageIndex(p + 1);
        xi[++head] = i;  // descend
        done = false;
        break;
      }
      if (done) {  // no unvisited successor: postorder j
        xi[--top] = j;
        --head;
      }
    }
  }
  return top;
}

// ===========================================================================
// Elimination-tree reach: lower Cholesky-type factor with a known etree.
// ===========================================================================

// Reach of a sparse rhs pattern through a Cholesky-type factor's elimination tree
// (Davis's cs_ereach). Because each node's only reach-successor is its etree
// parent, the DFS collapses to walking parent pointers: from each root we ascend
// the tree -- marking as we go -- until we meet an already-visited node, recording
// that path, then splice it onto the output xi[top..n). The splice reverses the
// path in place (safe: len <= top always, so read and write ranges never clobber),
// leaving the whole output in topological (solve) order. Hence, unlike the general
// worklist reach, this needs NO final sort and NO resume stack: O(|reach|).
//
// The etree is not passed in: for a Cholesky-type factor stored diagonal-first, the
// parent of column j is simply its first sub-diagonal entry innerIndexPtr[outerIndexPtr[j]+1] (the least
// i > j with L(i,j) != 0), so we read the parent straight from L. A column with no
// sub-diagonal entry is a tree root and ends the walk. `innerNonZeroPtr` is
// nullptr for a compressed factor, else the per-column count (col j ends at
// outerIndexPtr[j]+innerNonZeroPtr[j]). `xi` is length-n scratch holding both the current path (xi[0..len))
// and the growing output (xi[top..n)). `mark` is n bytes, all-zero on entry; the
// reached set is flagged out (the caller clears it during gather).
template <typename StorageIndex>
Index lower_triangular_ereach(const StorageIndex* outerIndexPtr, const StorageIndex* innerIndexPtr,
                              const StorageIndex* innerNonZeroPtr, const StorageIndex* bIdx, Index bCount,
                              StorageIndex* xi, uint8_t* mark, Index n) {
  Index top = n;
  for (Index r = 0; r < bCount; ++r) {
    Index len = 0;
    StorageIndex i = bIdx[r];
    while (i >= 0 && !mark[i]) {
      xi[len++] = i;  // ascend, recording the path child-first
      mark[i] = 1;
      Index colStart = outerIndexPtr[i] + 1;  // skip the diagonal stored first
      Index colEnd = innerNonZeroPtr ? outerIndexPtr[i] + innerNonZeroPtr[i] : outerIndexPtr[i + 1];
      i = colStart < colEnd ? innerIndexPtr[colStart] : StorageIndex(-1);  // parent, or root => stop
    }
    while (len > 0) xi[--top] = xi[--len];  // splice the path in, keeping topo order
  }
  return top;
}

// ===========================================================================
// Numeric sweep over a precomputed reach.
// ===========================================================================

// Solves T x = b in place on the dense accumulator x (zero except where b was
// scattered), touching only the reached columns in the order produced by the reach.
// For lower the diagonal is the first stored entry of a column and the off-diagonal
// (updated) entries follow; for upper the diagonal is the last entry and the
// off-diagonal entries precede it.
template <bool Upper, bool UnitDiag, typename StorageIndex, typename Scalar>
void triangular_solve_over_reach(const StorageIndex* outerIndexPtr, const StorageIndex* innerIndexPtr,
                                 const Scalar* valuePtr, const StorageIndex* innerNonZeroPtr, const StorageIndex* xi,
                                 Index top, Index n, Scalar* x) {
  for (Index k = top; k < n; ++k) {
    StorageIndex j = xi[k];
    Index colBeg = outerIndexPtr[j];
    Index colEnd = innerNonZeroPtr ? outerIndexPtr[j] + innerNonZeroPtr[j] : outerIndexPtr[j + 1];
    Index offBeg, offEnd;
    EIGEN_IF_CONSTEXPR (Upper) {
      // diagonal is the last stored entry; off-diagonal entries precede it
      offBeg = colBeg;
      EIGEN_IF_CONSTEXPR (!UnitDiag) {
        x[j] /= valuePtr[colEnd - 1];
        offEnd = colEnd - 1;
      } else {
        offEnd = (colEnd > colBeg && innerIndexPtr[colEnd - 1] == j) ? colEnd - 1 : colEnd;
      }
    } else {
      // diagonal is the first stored entry; off-diagonal entries follow it
      offEnd = colEnd;
      EIGEN_IF_CONSTEXPR (!UnitDiag) {
        x[j] /= valuePtr[colBeg];
        offBeg = colBeg + 1;
      } else {
        offBeg = (colEnd > colBeg && innerIndexPtr[colBeg] == j) ? colBeg + 1 : colBeg;
      }
    }
    Scalar xj = x[j];
    for (Index p = offBeg; p < offEnd; ++p) x[innerIndexPtr[p]] -= valuePtr[p] * xj;
  }
}

// ===========================================================================
// Borrow-a-buffer solve wrappers.
//
// Like the elimination_tree helpers, the caller owns the scratch and passes it in,
// so a reused buffer makes repeated solves allocation-free. All buffers are
// restored on exit, so one setup suffices for many solves.
// ===========================================================================

// Gather a computed reach into caller arrays, restoring x and mark to all-zero, and
// return the count. Direction-agnostic: xi[top..n) holds the reached indices (in
// topological order) and x holds their values. This is the "clean primitive" finish
// -- it snapshots the values into outVal because it then wipes x. A caller that
// consumes the solution immediately (the sparse selector below) skips this and reads
// values straight out of x, avoiding the outIdx/outVal buffers entirely.
template <typename StorageIndex, typename Scalar>
Index gather_reach_solution(const StorageIndex* xi, Index top, Index n, Scalar* x, uint8_t* mark, StorageIndex* outIdx,
                            Scalar* outVal) {
  Index count = 0;
  for (Index k = top; k < n; ++k) {
    StorageIndex j = xi[k];
    outIdx[count] = j;
    outVal[count] = x[j];
    x[j] = Scalar(0);
    mark[j] = 0;
    ++count;
  }
  return count;
}

// Core: compute reach(pattern(b)) and run the numeric sweep, returning top -- WITHOUT
// any cleanup. The rhs must ALREADY be scattered into xwork (xwork[bIdx[r]] = value)
// by the caller; bIdx is the rhs pattern (the reach roots). Pulling the scatter out
// lets a caller reading the rhs through an iterator scatter as it reads, dropping the
// separate value array. On return, xi = iwork[top..n) holds the reached columns in
// topological order, xwork holds their solution values, and mark is set on the reached
// set; the caller consumes xwork/xi and clears them (see gather_reach_solution).
// Solving T x = b for a column-major, sorted triangular T (lower or upper):
//   - iwork: >= 2n StorageIndex, carved into xi | pstack (each length n).
//   - mark:  >= n bytes, all-zero.
//   - xwork: >= n Scalar, the dense accumulator, zero except b scattered on bIdx.
// `innerNonZeroPtr` is nullptr for a compressed T, or the per-column nonzero
// count for an uncompressed T (columns then end at outerIndexPtr[j]+innerNonZeroPtr[j]).
template <bool Upper, bool UnitDiag, typename StorageIndex, typename Scalar>
Index reach_solve_dense(const StorageIndex* outerIndexPtr, const StorageIndex* innerIndexPtr, const Scalar* valuePtr,
                        const StorageIndex* innerNonZeroPtr, Index n, const StorageIndex* bIdx, Index bCount,
                        StorageIndex* iwork, uint8_t* mark, Scalar* xwork) {
  Index top = triangular_reach(outerIndexPtr, innerIndexPtr, innerNonZeroPtr, bIdx, bCount, iwork, iwork + n, mark, n);
  triangular_solve_over_reach<Upper, UnitDiag>(outerIndexPtr, innerIndexPtr, valuePtr, innerNonZeroPtr, iwork, top, n,
                                               xwork);
  return top;
}

// Clean, self-restoring primitive: scatter + core + gather. Takes the rhs as
// (bIdx/bVal), scatters it, and writes the solution's nonzero indices/values into
// outIdx/outVal (each capacity >= n) in topological order, returning the count; x and
// mark are restored to all-zero.
template <bool Upper, bool UnitDiag, typename StorageIndex, typename Scalar>
Index sparse_reach_triangular_solve(const StorageIndex* outerIndexPtr, const StorageIndex* innerIndexPtr,
                                    const Scalar* valuePtr, const StorageIndex* innerNonZeroPtr, Index n,
                                    const StorageIndex* bIdx, const Scalar* bVal, Index bCount, StorageIndex* iwork,
                                    uint8_t* mark, Scalar* xwork, StorageIndex* outIdx, Scalar* outVal) {
  for (Index r = 0; r < bCount; ++r) xwork[bIdx[r]] = bVal[r];
  Index top = reach_solve_dense<Upper, UnitDiag>(outerIndexPtr, innerIndexPtr, valuePtr, innerNonZeroPtr, n, bIdx,
                                                 bCount, iwork, mark, xwork);
  return gather_reach_solution(iwork, top, n, xwork, mark, outIdx, outVal);
}

// Convenience overload: no scratch supplied, so allocate (and zero) it for this
// single solve.
template <bool Upper, bool UnitDiag, typename StorageIndex, typename Scalar>
Index sparse_reach_triangular_solve(const StorageIndex* outerIndexPtr, const StorageIndex* innerIndexPtr,
                                    const Scalar* valuePtr, const StorageIndex* innerNonZeroPtr, Index n,
                                    const StorageIndex* bIdx, const Scalar* bVal, Index bCount, StorageIndex* outIdx,
                                    Scalar* outVal) {
  Matrix<StorageIndex, Dynamic, 1> iwork = Matrix<StorageIndex, Dynamic, 1>::Zero(2 * n);
  Matrix<uint8_t, Dynamic, 1> mark = Matrix<uint8_t, Dynamic, 1>::Zero(n);
  Matrix<Scalar, Dynamic, 1> xwork = Matrix<Scalar, Dynamic, 1>::Zero(n);
  return sparse_reach_triangular_solve<Upper, UnitDiag>(outerIndexPtr, innerIndexPtr, valuePtr, innerNonZeroPtr, n,
                                                        bIdx, bVal, bCount, iwork.data(), mark.data(), xwork.data(),
                                                        outIdx, outVal);
}

// Elimination-tree solve: same contract as the general lower solve, but the reach
// walks the factor's etree (read from L) instead of doing a full-column DFS, so it
// needs neither a resume stack nor a final sort. Only valid for a complete
// Cholesky-type lower factor L. Workspace:
//   - iwork: >= n StorageIndex (xi only; the etree reach needs no pstack).
//   - mark:  >= n bytes, all-zero.
//   - xwork: >= n Scalar, all-zero.
template <bool UnitDiag, typename StorageIndex, typename Scalar>
Index sparse_reach_lower_solve_etree(const StorageIndex* outerIndexPtr, const StorageIndex* innerIndexPtr,
                                     const Scalar* valuePtr, const StorageIndex* innerNonZeroPtr, Index n,
                                     const StorageIndex* bIdx, const Scalar* bVal, Index bCount, StorageIndex* iwork,
                                     uint8_t* mark, Scalar* xwork, StorageIndex* outIdx, Scalar* outVal) {
  StorageIndex* xi = iwork;
  Scalar* x = xwork;

  for (Index r = 0; r < bCount; ++r) x[bIdx[r]] = bVal[r];
  Index top = lower_triangular_ereach(outerIndexPtr, innerIndexPtr, innerNonZeroPtr, bIdx, bCount, xi, mark, n);
  triangular_solve_over_reach<false, UnitDiag>(outerIndexPtr, innerIndexPtr, valuePtr, innerNonZeroPtr, xi, top, n, x);

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

// Convenience overload of the etree solve: allocate (and zero) the workspace.
template <bool UnitDiag, typename StorageIndex, typename Scalar>
Index sparse_reach_lower_solve_etree(const StorageIndex* outerIndexPtr, const StorageIndex* innerIndexPtr,
                                     const Scalar* valuePtr, const StorageIndex* innerNonZeroPtr, Index n,
                                     const StorageIndex* bIdx, const Scalar* bVal, Index bCount, StorageIndex* outIdx,
                                     Scalar* outVal) {
  Matrix<StorageIndex, Dynamic, 1> iwork = Matrix<StorageIndex, Dynamic, 1>::Zero(n);
  Matrix<uint8_t, Dynamic, 1> mark = Matrix<uint8_t, Dynamic, 1>::Zero(n);
  Matrix<Scalar, Dynamic, 1> xwork = Matrix<Scalar, Dynamic, 1>::Zero(n);
  return sparse_reach_lower_solve_etree<UnitDiag>(outerIndexPtr, innerIndexPtr, valuePtr, innerNonZeroPtr, n, bIdx,
                                                  bVal, bCount, iwork.data(), mark.data(), xwork.data(), outIdx,
                                                  outVal);
}

// ---------------------------------------------------------------------------
// Iterator-driven path: for a triangular, column-major sparse expression that does
// NOT expose raw CSC storage (has_compressed_access is false). Columns are read
// through the expression's evaluator InnerIterator. Because an InnerIterator can't
// cheaply hold DFS resume state, the reach uses a mark-on-push worklist plus a final
// sort into topological order (ascending index for lower, descending for upper --
// both valid topological orders for the respective solve); the log factor is
// empirically ~free.
// ---------------------------------------------------------------------------

// Reach via a mark-on-push worklist that opens one column at a time through
// InnerIterator. The whole column is scanned; the self-diagonal is skipped by mark
// (as in the pointer DFS). Scratch: xi (n, shared stack+output) and mark (n bytes,
// all-zero in, reach flagged out). Returns top; xi[top..n) is sorted into the
// topological order for the solve direction.
template <bool Upper, typename Eval, typename StorageIndex>
Index triangular_reach_iter(const Eval& mat, const StorageIndex* bIdx, Index bCount, StorageIndex* xi, uint8_t* mark,
                            Index n) {
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
    for (typename Eval::InnerIterator it(mat, j); it; ++it) {
      StorageIndex i = StorageIndex(it.index());
      if (!mark[i]) {  // self-diagonal (i == j) already has mark[j] == 1
        mark[i] = 1;
        xi[sp++] = i;
      }
    }
  }
  EIGEN_IF_CONSTEXPR (Upper) {
    std::sort(xi + top, xi + n, [](StorageIndex a, StorageIndex b) { return a > b; });  // descending
  } else {
    std::sort(xi + top, xi + n);  // ascending
  }
  return top;
}

// Numeric sweep over the reach, reading columns through InnerIterator.
template <bool Upper, bool UnitDiag, typename Eval, typename StorageIndex, typename Scalar>
void triangular_solve_over_reach_iter(const Eval& mat, const StorageIndex* xi, Index top, Index n, Scalar* x) {
  for (Index k = top; k < n; ++k) {
    StorageIndex j = xi[k];
    EIGEN_IF_CONSTEXPR (Upper) {
      // diagonal is the last entry; above-diagonal entries (index < j) precede it.
      EIGEN_IF_CONSTEXPR (!UnitDiag) {
        Scalar d(1);
        for (typename Eval::InnerIterator dt(mat, j); dt; ++dt)
          if (StorageIndex(dt.index()) == j) d = dt.value();
        x[j] /= d;
      }
      Scalar xj = x[j];
      for (typename Eval::InnerIterator it(mat, j); it && StorageIndex(it.index()) < j; ++it)
        x[it.index()] -= it.value() * xj;
    } else {
      typename Eval::InnerIterator it(mat, j);
      EIGEN_IF_CONSTEXPR (!UnitDiag) {
        x[j] /= it.value();  // diagonal stored first
        ++it;
      } else {
        if (it && StorageIndex(it.index()) == j) ++it;  // skip an explicit unit diagonal
      }
      Scalar xj = x[j];
      for (; it; ++it) x[it.index()] -= it.value() * xj;
    }
  }
}

// Iterator core: reach + numeric, returning top (no cleanup), the iterator counterpart
// of reach_solve_dense -- xwork must already hold the scattered rhs. Uses the same
// 2n / n(bytes) / n workspace layout (the pstack half of iwork is left unused), so the
// two general paths share one workspace contract.
template <bool Upper, bool UnitDiag, typename LhsType, typename StorageIndex, typename Scalar>
Index reach_solve_dense_iter(const LhsType& lhs, Index n, const StorageIndex* bIdx, Index bCount, StorageIndex* iwork,
                             uint8_t* mark, Scalar* xwork) {
  evaluator<LhsType> mat(lhs);
  Index top = triangular_reach_iter<Upper>(mat, bIdx, bCount, iwork, mark, n);
  triangular_solve_over_reach_iter<Upper, UnitDiag>(mat, iwork, top, n, xwork);
  return top;
}

// Clean iterator primitive: scatter + core + gather.
template <bool Upper, bool UnitDiag, typename LhsType, typename StorageIndex, typename Scalar>
Index sparse_reach_triangular_solve_iter(const LhsType& lhs, Index n, const StorageIndex* bIdx, const Scalar* bVal,
                                         Index bCount, StorageIndex* iwork, uint8_t* mark, Scalar* xwork,
                                         StorageIndex* outIdx, Scalar* outVal) {
  for (Index r = 0; r < bCount; ++r) xwork[bIdx[r]] = bVal[r];
  Index top = reach_solve_dense_iter<Upper, UnitDiag>(lhs, n, bIdx, bCount, iwork, mark, xwork);
  return gather_reach_solution(iwork, top, n, xwork, mark, outIdx, outVal);
}

// Policy dispatch for the core (returns top): an expression that exposes raw storage
// takes the pointer + DFS fast path; anything else takes the evaluator + worklist
// path. Tag dispatch (not if-constexpr) keeps the untaken branch from being
// instantiated, so outerIndexPtr() is never named on a type that lacks it.
//
// CompressedAccessBit is a compile-time capability, not a guarantee the instance
// is compressed: an uncompressed SparseMatrix keeps per-column gaps addressed via
// innerNonZeroPtr(), so its columns do NOT run to outerIndexPtr()[j+1]. Passing
// innerNonZeroPtr() through keeps the raw-pointer path valid either way -- it is
// nullptr exactly when compressed (columns end at outerIndexPtr[j+1]) and the per-column
// count otherwise (columns end at outerIndexPtr[j]+innerNonZeroPtr[j]).
template <bool Upper, bool UnitDiag, typename LhsType, typename StorageIndex, typename Scalar>
Index reach_solve_dense_dispatch(std::true_type /*compressed*/, const LhsType& lhs, Index n, const StorageIndex* bIdx,
                                 Index bCount, StorageIndex* iwork, uint8_t* mark, Scalar* xwork) {
  return reach_solve_dense<Upper, UnitDiag>(lhs.outerIndexPtr(), lhs.innerIndexPtr(), lhs.valuePtr(),
                                            lhs.innerNonZeroPtr(), n, bIdx, bCount, iwork, mark, xwork);
}
template <bool Upper, bool UnitDiag, typename LhsType, typename StorageIndex, typename Scalar>
Index reach_solve_dense_dispatch(std::false_type /*iterator*/, const LhsType& lhs, Index n, const StorageIndex* bIdx,
                                 Index bCount, StorageIndex* iwork, uint8_t* mark, Scalar* xwork) {
  return reach_solve_dense_iter<Upper, UnitDiag>(lhs, n, bIdx, bCount, iwork, mark, xwork);
}

// Expression core: solve T x = b for a sparse-expression triangular T with the rhs
// PRE-SCATTERED into xwork (bIdx is its pattern), selecting the pointer or iterator
// path at compile time; RETURNS top with the solution left in xwork and the reach in
// iwork[top..n) (see reach_solve_dense). This is what the sparse selector uses -- it
// scatters the rhs as it reads it and consumes xwork directly, so no bVal/outIdx/outVal.
template <bool Upper, bool UnitDiag, typename LhsDerived, typename StorageIndex, typename Scalar>
Index reach_solve_dense(const SparseMatrixBase<LhsDerived>& lhs, const StorageIndex* bIdx, Index bCount,
                        StorageIndex* iwork, uint8_t* mark, Scalar* xwork) {
  return reach_solve_dense_dispatch<Upper, UnitDiag>(
      std::integral_constant<bool, has_compressed_access<LhsDerived>::value>{}, lhs.derived(), lhs.rows(), bIdx, bCount,
      iwork, mark, xwork);
}

// Clean expression primitive: scatter + expression core + gather. Borrow a 2n
// StorageIndex / n byte / n Scalar workspace as in the pointer overload.
template <bool Upper, bool UnitDiag, typename LhsDerived, typename StorageIndex, typename Scalar>
Index sparse_reach_triangular_solve(const SparseMatrixBase<LhsDerived>& lhs, const StorageIndex* bIdx,
                                    const Scalar* bVal, Index bCount, StorageIndex* iwork, uint8_t* mark, Scalar* xwork,
                                    StorageIndex* outIdx, Scalar* outVal) {
  for (Index r = 0; r < bCount; ++r) xwork[bIdx[r]] = bVal[r];
  Index top = reach_solve_dense<Upper, UnitDiag>(lhs, bIdx, bCount, iwork, mark, xwork);
  return gather_reach_solution(iwork, top, lhs.rows(), xwork, mark, outIdx, outVal);
}

// Convenience expression overload: allocate (and zero) the workspace for one solve.
template <bool Upper, bool UnitDiag, typename LhsDerived, typename StorageIndex, typename Scalar>
Index sparse_reach_triangular_solve(const SparseMatrixBase<LhsDerived>& lhs, const StorageIndex* bIdx,
                                    const Scalar* bVal, Index bCount, StorageIndex* outIdx, Scalar* outVal) {
  Index n = lhs.rows();
  Matrix<StorageIndex, Dynamic, 1> iwork = Matrix<StorageIndex, Dynamic, 1>::Zero(2 * n);
  Matrix<uint8_t, Dynamic, 1> mark = Matrix<uint8_t, Dynamic, 1>::Zero(n);
  Matrix<Scalar, Dynamic, 1> xwork = Matrix<Scalar, Dynamic, 1>::Zero(n);
  return sparse_reach_triangular_solve<Upper, UnitDiag>(lhs, bIdx, bVal, bCount, iwork.data(), mark.data(),
                                                        xwork.data(), outIdx, outVal);
}

}  // namespace internal

}  // namespace Eigen

#endif  // EIGEN_TRIANGULAR_REACH_SOLVER_H
