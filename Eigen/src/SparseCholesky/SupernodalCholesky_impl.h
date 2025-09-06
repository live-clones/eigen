#ifndef EIGEN_SUPERNODAL_CHOLESKY_IMPL_H
#define EIGEN_SUPERNODAL_CHOLESKY_IMPL_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {
namespace internal {

template <typename Scalar, typename StorageIndex>
struct supernodal_chol_helper {
  using CholMatrixType = SparseMatrix<Scalar, ColMajor, StorageIndex>;
  using InnerIterator = typename CholMatrixType::InnerIterator;
  using VectorI = Matrix<StorageIndex, Dynamic, 1>;

  /// means the values is empty.
  /// in the etrees this means the column is a root
  static constexpr StorageIndex kEmpty = -1;

  struct Supernodes {
    /// boundaries of each supernode (`size = #supernodes + 1`)
    VectorI supernodes;

    /// id of the final supernode each column belongs to
    /// (`size = #columns`)
    VectorI sn_id;

    /// #nonzero rows in each supernode.
    VectorI Snz;
  };

  // Supernode Algorithm Adapted from CHOLMOD
  // Y. Chen, T. A. Davis, W. W. Hager, and S. Rajamanickam,
  // Algorithm 887:
  // CHOLMOD, supernodal sparse Cholesky factorization and update/downdate,
  // ACM Trans. on Mathematical Software, 35(3), 2008, pp. 22:1--22:14.
  // https://dl.acm.org/doi/abs/10.1145/1391989.1391995

  // T. A. Davis and W. W. Hager,
  // Dynamic supernodes in sparse Cholesky update/downdate and triangular solves,
  // ACM Trans. on Mathematical Software, 35(4), 2009, pp. 27:1--27:23.
  // https://doi.org/10.1145/1462173.1462176

  static Supernodes compute_supernodes(const VectorI& parent, const VectorI& colcount) {
    const StorageIndex n = parent.size();
    eigen_assert(colcount.size() == n && "colcount must have size n");

    if (n == 0) {
      return Supernodes{VectorI::Zero(1), VectorI::Zero(0), VectorI::Zero(0)};
    }

    /// id of the final supernode each column belongs to
    VectorI sn_id = VectorI::Constant(n, kEmpty);

    VectorI supernodes;

    // Fundamental supernodes

    /// #children of each node
    VectorI childCount = VectorI::Zero(n);

    for (StorageIndex j = 0; j < n; ++j)
      // make sure its not a root node
      if (parent[j] != kEmpty) childCount[parent[j]]++;

    // fundamental supernode test:
    // (1) parent[j] == j + 1  (j+1 is parent of j in etree)
    // (2) nnz(L(:,j)) - nnz(L(:,j-1)) == 1
    //       (they have the same # of nonzeros below the diagonal
    //        minus 1 for the diagonal)
    // (3) j has at most one child (to ensure a chain of nodes)
    // once these three conditions are met then we know they have the same
    // sparsity structure below the diagonal. Since the principle of column
    // replication ensures that the sparsity pattern of column j is replicated
    // into column j + 1 if j + 1 is the parent of j

    /// fundamental supernodes boundaries
    VectorI fn_supernodes;

    fn_supernodes.resize(n + 1);
    StorageIndex m = 0;

    fn_supernodes[m++] = 0;
    for (StorageIndex j = 1; j < n; ++j) {
      const bool parentChain = (parent[j - 1] == j);
      const bool colNest = (colcount[j - 1] == colcount[j] + 1);
      const bool singleChild = (childCount[j] <= 1);
      if (!(parentChain && colNest && singleChild)) {
        // j is the leading node of a supernode
        fn_supernodes[m++] = j;
      }
    }
    fn_supernodes[m++] = n;
    fn_supernodes.conservativeResize(m);

    // note we no longer need the childCount. may want to do something with this info later

    /// #fundamental supernodes
    const StorageIndex nfsuper = static_cast<StorageIndex>(fn_supernodes.size()) - 1;

    /// id of the fundamental supernode each column belongs to
    VectorI fundamental_sn_id(n);

    for (StorageIndex s = 0; s < nfsuper; ++s)
      for (StorageIndex k = fn_supernodes[s]; k < fn_supernodes[s + 1]; ++k) fundamental_sn_id[k] = s;

    /// Fundamental supernodal etree (ie: assembly tree)
    VectorI f_sn_etree = VectorI::Constant(nfsuper, kEmpty);

    for (int s = 0; s < nfsuper; ++s) {
      const int jlast = fn_supernodes[s + 1] - 1;
      const int pcol = parent[jlast];
      f_sn_etree[s] = (pcol == kEmpty) ? kEmpty : fundamental_sn_id[pcol];
    }

    // Relaxed amalgamation
    // Here we want to merge neighboring fundamental supernodes
    // while ensuring that the resulting supernode is not too sparse.

    // Default parameters from CHOLMOD
    // https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/d558c83006d63d1dc62004f30042b3ca484f3f94/CHOLMOD/Utility/t_cholmod_defaults.c#L54-L59
    constexpr StorageIndex nrelax0 = 4;
    constexpr StorageIndex nrelax1 = 16;
    constexpr StorageIndex nrelax2 = 48;
    constexpr double zrelax0 = 0.8;
    constexpr double zrelax1 = 0.1;
    constexpr double zrelax2 = 0.05;

    /// #columns in each fundamental supernode
    VectorI Nscol(nfsuper);

    /// #nonzero rows in each fundamental supernode.
    /// at the start this is just the nnz of the leading column
    VectorI Snz(nfsuper);

    /// accumulated new zeros introduced by earlier merges (starts at 0).
    VectorI Zeros = VectorI::Zero(nfsuper);

    VectorI Merged = VectorI::Constant(nfsuper, kEmpty);

    for (StorageIndex s = 0; s < nfsuper; ++s) {
      Nscol[s] = fn_supernodes[s + 1] - fn_supernodes[s];  // width
      eigen_assert(s <= fn_supernodes[s] && "supernode index is invalid");
      Snz[s] = colcount[fn_supernodes[s]];  // nnz of leading col
    }

    for (StorageIndex s = nfsuper - 2; s >= 0; --s) {
      StorageIndex ss = f_sn_etree[s];

      if (ss == kEmpty) continue;

      while (Merged[ss] != kEmpty) ss = Merged[ss];

      const StorageIndex sparent = ss;

      for (ss = f_sn_etree[s]; ss != kEmpty && Merged[ss] != kEmpty;) {
        const StorageIndex snext = Merged[ss];
        Merged[ss] = sparent;
        ss = snext;
      }

      if (sparent != s + 1) continue;

      /// width of the left supernode
      const StorageIndex nscol0 = Nscol[s];

      // width of the right supernode
      const StorageIndex nscol1 = Nscol[s + 1];

      const StorageIndex ns = nscol0 + nscol1;

      /// zeros accumulated in the right block
      StorageIndex totzeros = Zeros[s + 1];

      /// leading column nonzeros in the right supernode
      const double l_nnz_1 = static_cast<double>(Snz[s + 1]);

      bool merge = false;

      // Always merge if the resulting supernode is small enough
      if (ns <= nrelax0) merge = true;

      // Estimate additional nonzeros if we append `s` on the left
      else {
        /// leading column nonzeros in the left supernode
        const double l_nnz_0 = static_cast<double>(Snz[s]);

        /// the estimated number of new zero entries introduced if you force the left block
        // to share the same structure as the right block in a merged supernode.
        const double xnewzeros = static_cast<double>(nscol0) * (l_nnz_1 + nscol0 - l_nnz_0);

        const StorageIndex newzeros_ll = 1LL * nscol0 * (Snz[s + 1] + nscol0 - Snz[s]);

        if (xnewzeros == 0.0) {
          // no new nonzero rows so we merge
          merge = true;
        } else {
          /// total artificial zeros after the merge.
          const double xtotzeros = static_cast<double>(totzeros) + xnewzeros;

          /// merged width
          const double xns = static_cast<double>(ns);

          const double xtotsize = (xns * (xns + 1.0) / 2.0) + xns * (l_nnz_1 - nscol1);

          /// fill ratio after the merge; how many artificial nonzeros pop up from the merge.
          /// larger z means more sparse
          const double z = xtotzeros / std::max(1.0, xtotsize);

          const double int_guard = static_cast<double>(std::numeric_limits<int>::max()) / sizeof(double);

          totzeros += static_cast<StorageIndex>(newzeros_ll);

          if (xtotsize < int_guard &&
              ((ns <= nrelax1 && z < zrelax0) || (ns <= nrelax2 && z < zrelax1) || (z < zrelax2))) {
            merge = true;
          }
        }
      }

      if (merge) {
        Zeros[s] = totzeros;
        Merged[s + 1] = s;
        Snz[s] = nscol0 + Snz[s + 1];
        Nscol[s] += Nscol[s + 1];
      }
    }

    // Emit relaxed supernodes

    /// Count how many supernodes
    StorageIndex nsuper = 1;

    for (StorageIndex s = 0; s < nfsuper; ++s)
      if (Merged[s] == kEmpty) nsuper++;

    VectorI relaxed(nsuper);

    StorageIndex r = 0;
    for (StorageIndex s = 0; s < nfsuper; ++s)
      if (Merged[s] == kEmpty) relaxed[r++] = fn_supernodes[s];

    relaxed[r++] = n;

    supernodes.resize(relaxed.size());
    for (StorageIndex i = 0; i < supernodes.size(); ++i) supernodes[i] = relaxed[static_cast<StorageIndex>(i)];

    for (StorageIndex s = 0; s + 1 < relaxed.size(); ++s)
      for (StorageIndex k = relaxed[s]; k < relaxed[s + 1]; ++k) sn_id[k] = s;

    Supernodes out;
    out.supernodes = std::move(supernodes);
    out.sn_id = std::move(sn_id);

    // Build compact Snz
    VectorI Snz_relaxed(relaxed.size() - 1);
    StorageIndex t = 0;
    for (StorageIndex s = 0; s < nfsuper; ++s) {
      if (Merged[s] == kEmpty) {
        Snz_relaxed[t++] = Snz[s];
      }
    }
    out.Snz = std::move(Snz_relaxed);
    return out;
  }

  static VectorI compute_supernodal_etree(const VectorI& parent, const Supernodes& supe) {
    /// #supernodes
    const StorageIndex nsuper = supe.supernodes.size() - 1;

    VectorI s_parent = VectorI::Constant(nsuper, kEmpty);

    for (StorageIndex s = 0; s < nsuper; ++s) {
      const StorageIndex jlast = supe.supernodes[s + 1] - 1;
      const StorageIndex pcol = parent[jlast];
      s_parent[s] = (pcol < 0) ? kEmpty : supe.sn_id[pcol];
    }
    return s_parent;
  }

  /// Lp (size `# supernodes + 1`)
  /// Li List of row indices used by a super node `s`
  /// Li[Lp[s] ... Lp[s+1]-1]

  /**
   * @brief finds which supernodes column `j` contributes to in the factor `L`
   *
   * @param j the column we are building the pattern for
   * @param Ap column pointers of A
   * @param Ai row indices of A
   * @param Anz
   * @param sn_id column to supernode mapping
   * @param s_parent supernodal elimination tree
   * @param mark
   * @param k1 supernode start column (k1 <= j)
   * @param flag
   * @param Li List of row indices used by supernodes
   * @param Lp supernode column pointers (size `# supernodes + 1`)
   */
  static void sn_contribution(StorageIndex j, const VectorI& Ap, const VectorI& Ai, const VectorI* Anz,
                              const VectorI& sn_id, const VectorI& s_parent, StorageIndex mark, StorageIndex k1,
                              VectorI& flag, VectorI& Li, VectorI& Lp) {
    StorageIndex p = Ap[j];
    const StorageIndex pend = (Anz == nullptr) ? Ap[j + 1] : (p + (*Anz)[j]);

    for (; p < pend; ++p) {
      const StorageIndex i = Ai[p];

      if (i < k1) {
        // climb up supernodal etree
        for (StorageIndex si = sn_id[i]; flag[si] < mark; si = s_parent[si]) {
          Li[Lp[si]++] = j;
          flag[si] = mark;
        }
      } else {
        break;
      }
    }
  }

  /**
   * @brief
   *
   * @param supe supernode structure
   * @param[out] ssize output size of Li
   * @param[out] xsize output size of Lx
   */
  static void compute_supernodal_sizes(const Supernodes& supe, StorageIndex& ssize, StorageIndex& xsize) {
    const StorageIndex nsuper = supe.supernodes.size() - 1;

    ssize = 0;
    xsize = 0;

    for (StorageIndex s = 0; s < nsuper; ++s) {
      /// number of columns in this supernode
      uint64_t nscol = static_cast<uint64_t>(supe.supernodes[s + 1] - supe.supernodes[s]);

      /// number of rows (including diagonal block)
      uint64_t nsrow = static_cast<uint64_t>(supe.Snz[s]);

      // add to ssize (row index pool length)
      ssize += static_cast<size_t>(nsrow);

      // each supernode block has nscol*nsrow entries
      uint64_t c = nscol * nsrow;
      xsize += static_cast<size_t>(c);
    }

    // ensure results fit w/i limits
    eigen_assert(ssize >= 0 && ssize <= std::numeric_limits<StorageIndex>::max() && "Li size too large");
    eigen_assert(xsize >= 0 && xsize <= std::numeric_limits<StorageIndex>::max() && "Lx size too large");
  }

  // Lx = data (dense blocks).
  // Lpx = per-supernode pointers into that data (size nsuper+1).
  // Li = pattern (row indices).
  // Lpi = per-supernode pointers into that pattern (size nsuper+1).

  /**
   * @brief allocate `L`
   *
   * @param supe supernode structure
   * @param[out] Lpx per-supernode pointers into that data (size nsuper+1)
   * @param[out] Li pattern (row indices) (size `nsrows[s]` for supernode `s`)
   * @param[out] Lpi per-supernode pointers into that pattern (size nsuper+1)
   * @param[out] ssize total size of `Li`
   * @param[out] xsize total size of `Lx`
   */
  static void allocate_L(const Supernodes& supe, VectorI& Lpx, VectorI& Li, VectorI& Lpi, StorageIndex& ssize,
                         StorageIndex& xsize) {
    const StorageIndex nsuper = supe.supernodes.size() - 1;

    // sizes
    compute_supernodal_sizes(supe, ssize, xsize);

    Lpi.resize(nsuper + 1);
    Lpx.resize(nsuper + 1);
    Li.resize(ssize);

    // fill column pointers (Lpi)
    {
      StorageIndex p = 0;
      for (StorageIndex s = 0; s < nsuper; ++s) {
        Lpi[s] = p;
        p += supe.Snz[s];
      }
      Lpi[nsuper] = p;
      eigen_assert(p == ssize);
    }

    // fill block value pointers (Lpx)
    {
      StorageIndex q = 0;
      for (StorageIndex s = 0; s < nsuper; ++s) {
        const StorageIndex nscol = supe.supernodes[s + 1] - supe.supernodes[s];
        const StorageIndex nsrow = supe.Snz[s];
        Lpx[s] = q;
        q += nscol * nsrow;
      }
      Lpx[nsuper] = q;
      eigen_assert(q == xsize);
    }
  }

  /**
   * @brief Fill the supernodal pattern `Li`.
   *
   * @param supe supernode structure
   * @param s_parent supernodal etree (size nsuper)
   * @param Ap, Ai CSC of A
   * @param Anz per-column counts
   * @param Li row indices pool (allocated)
   * @param Lpi per-supernode pointers into Li
   */
  static void build_sn_pattern(const Supernodes& supe, const VectorI& s_parent, const VectorI& Ap, const VectorI& Ai,
                               const VectorI* Anz, VectorI& Li, const VectorI& Lpi) {
    const StorageIndex nsuper = supe.supernodes.size() - 1;

    /// working copy of Lpi
    VectorI Lpi2 = Lpi;

    /// per-supernode flag array
    VectorI flag = VectorI::Zero(nsuper);
    StorageIndex mark = 1;

    auto clear_flag = [&]() {
      if (mark == std::numeric_limits<StorageIndex>::max()) {
        flag.setZero();
        mark = 1;
      } else {
        ++mark;
      }
    };

    for (StorageIndex s = 0; s < nsuper; ++s) {
      const StorageIndex k1 = supe.supernodes[s];
      const StorageIndex k2 = supe.supernodes[s + 1];

      // put rows `k1...k2-1` in leading column of supernode `s`
      for (StorageIndex k = k1; k < k2; ++k) {
        Li[Lpi2[s]++] = k;
      }

      // traverse for each column k in this supernode
      for (StorageIndex k = k1; k < k2; ++k) {
        clear_flag();
        flag[s] = mark;  // mark this supernode
        sn_contribution(k, Ap, Ai, Anz, supe.sn_id, s_parent, mark, k1, flag, Li, Lpi2);
      }
    }

#ifndef NDEBUG
    for (StorageIndex s = 0; s < nsuper; ++s) {
      eigen_assert(Lpi2[s] == Lpi[s + 1] && "Li fill mismatch");
    }
#endif
  }

  /**
   * @brief Compute largest update matrix
   *
   * @param supe
   * @param Lpi
   * @param Li
   * @param[out] maxcsize largest panel area among all updates
   * @param[out] maxesize largest esize = (# extra rows beyond diagonal block) for any supernode
   */
  static void compute_max_update(const Supernodes& supe, const VectorI& Lpi, const VectorI& Li, StorageIndex& maxcsize,
                                 StorageIndex& maxesize) {
    const StorageIndex nsuper = supe.supernodes.size() - 1;
    eigen_assert(supe.Snz.size() == nsuper);
    eigen_assert(Lpi.size() == nsuper + 1);

    maxcsize = StorageIndex(1);
    maxesize = StorageIndex(1);

    for (StorageIndex d = 0; d < nsuper; ++d) {
      /// number of supernode columns
      const StorageIndex nscol = supe.supernodes[d + 1] - supe.supernodes[d];

      /// Start of "extra" rows (after diagonal block), end of block:
      StorageIndex p = Lpi[d] + nscol;

      StorageIndex plast = p;

      const StorageIndex pend = Lpi[d + 1];

      /// number of extra rows used in solves for this supernode
      const StorageIndex esize = pend - p;

      // makeshift `std::max` function
      if (esize > maxesize) maxesize = esize;

      // If no extra rows, skip
      if (p == pend) continue;

      // Current group label = supernode id of the first extra-row index
      StorageIndex slast = supe.sn_id[Li[p]];

      // Walk until pend, including a sentinel step at p==pend
      for (; p <= pend; ++p) {
        const StorageIndex s = (p == pend) ? kEmpty : supe.sn_id[Li[p]];

        if (s != slast) {
          /// number of rows in this group
          const StorageIndex ndrow1 = p - plast;

          /// number of rows in the trailing panel
          const StorageIndex ndrow2 = pend - plast;

          const uint64_t c = uint64_t(ndrow1) * uint64_t(ndrow2);
          const uint64_t cap = uint64_t(std::numeric_limits<StorageIndex>::max());

          const StorageIndex csize = StorageIndex(c > cap ? cap : c);

          if (csize > maxcsize) maxcsize = csize;

          plast = p;
          slast = s;
        }
      }
    }
  }

  /**
   * @brief
   *
   * @param A
   * @param supe supernode struct
   * @param Lpi pointers into Li (size nsuper+1)
   * @param Lpx pointers into Lx (size nsuper+1)
   * @param Li row indices per supernode (size = Lpi[nsuper])
   * @param beta
   * @param[out] Lx dense storage for all supernodes (size = Lpx[nsuper])
   */
  static bool compute_numeric(const CholMatrixType& A, const Supernodes& supe, const VectorI& Lpi, const VectorI& Lpx,
                              const VectorI& Li, Scalar beta, Matrix<Scalar, Dynamic, 1>& Lx) {
    using std::max;

    const VectorI& supernodes = supe.supernodes;
    const VectorI& Snz = supe.Snz;
    const VectorI& sn_id = supe.sn_id;

    const StorageIndex n = static_cast<StorageIndex>(A.cols());
    const StorageIndex nsuper = static_cast<StorageIndex>(supernodes.size()) - 1;
    eigen_assert(static_cast<StorageIndex>(A.rows()) == n);

    // allocate numeric storage
    Lx.resize(Lpx[nsuper]);
    Lx.setZero();

    /// Map[i] -> local row within current supernode s (or -1)
    VectorI Map(n);
    Map.setConstant(kEmpty);

    /// head of child list for each supernode
    VectorI Head(nsuper);
    Head.setConstant(kEmpty);

    /// sibling link for child list
    VectorI Next(nsuper);
    Next.setConstant(kEmpty);

    /// where child d resumes for next ancestor
    VectorI Lpos(nsuper);
    Lpos.setZero();

    // figure out a safe scratch size for C and for RelativeMap
    StorageIndex maxcsize = 1, maxesize = 1;
    compute_max_update(supe, Lpi, Li, maxcsize, maxesize);

    std::vector<Scalar> Cbuf(static_cast<size_t>(maxcsize));

    // strided map type (outer stride = leading dimension)
    using DenseMat = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;
    using StrideType = Eigen::Stride<Dynamic, Dynamic>;
    using BlockMap = Eigen::Map<DenseMat, 0, StrideType>;

    // main supernode loop
    for (StorageIndex s = 0; s < nsuper; ++s) {
      const StorageIndex k1 = supernodes[s];
      const StorageIndex k2 = supernodes[s + 1];

      /// width of supernode
      const StorageIndex nscol = k2 - k1;

      /// pointer to first row of s in Li
      const StorageIndex psi = Lpi[s];

      /// pointer to first row of s in Lx
      const StorageIndex psx = Lpx[s];

      /// #nonzero rows in this s
      const StorageIndex nsrow = Snz[s];

      // build Map for the rows of this supernode
      for (StorageIndex k = 0; k < nsrow; ++k) Map[Li[psi + k]] = k;

      // copy A(:, k1:k2-1) into dense block (lower part only)
      for (StorageIndex k = k1; k < k2; ++k) {
        for (typename CholMatrixType::InnerIterator it(A, k); it; ++it) {
          const StorageIndex i = static_cast<StorageIndex>(it.row());

          // keep lower triangle including diag
          if (i < k) continue;

          const StorageIndex imap = Map[i];

          if (imap >= 0 && imap < nsrow) {
            Lx[psx + (k - k1) * nsrow + imap] += it.value();
          }
        }
      }

      // add beta to the diagonal of the supernode (top-left nscol-by-nscol block)
      if (beta != Scalar(0)) {
        StorageIndex pk = psx;
        for (StorageIndex j = 0; j < nscol; ++j) {
          Lx[pk] += beta;
          pk += nsrow + 1;
        }
      }

      // Apply updates from children already factorized and queued at Head[s]
      StorageIndex dnext = Head[s];
      while (dnext != kEmpty) {
        const StorageIndex d = dnext;

        // preserve next pointer
        dnext = Next[d];

        /// width of child
        const StorageIndex ndcol = supernodes[d + 1] - supernodes[d];

        const StorageIndex pdi = Lpi[d];
        const StorageIndex pdend = Lpi[d + 1];
        const StorageIndex pdx = Lpx[d];

        /// total rows in child
        const StorageIndex ndrow = pdend - pdi;

        const StorageIndex p = Lpos[d];

        /// first relevant row (global index Li[pdi1])
        StorageIndex pdi1 = pdi + p;
        StorageIndex pdi2 = pdi1;
        while (pdi2 < pdend && Li[pdi2] < k2) ++pdi2;

        /// rows within [k1..k2)
        const StorageIndex ndrow1 = pdi2 - pdi1;

        /// rows in [k1..n)
        const StorageIndex ndrow2 = pdend - pdi1;

        /// rows strictly below k2
        const StorageIndex ndrow3 = ndrow2 - ndrow1;

        if (ndrow1 <= 0) continue;

        /// Map the child's dense block Ld (size ndrow by ndcol, leading dim = ndrow)
        BlockMap Ld(Lx.data() + pdx, ndrow, ndcol, StrideType(ndrow, 1));

        // Subblocks relative to p
        /// rows in [k1..k2)
        auto L1 = Ld.block(p, 0, ndrow1, ndcol);

        /// rows in (k2..n)
        auto L2 = Ld.block(p + ndrow1, 0, ndrow3, ndcol);

        // Build C (ndrow2 by ndrow1) (col-major)
        Eigen::Map<DenseMat> C(Cbuf.data(), ndrow2, ndrow1);
        if (ndrow1 > 0) {
          // C1 = L1 * L1'
          C.topRows(ndrow1).noalias() = L1 * L1.adjoint();
          if (ndrow3 > 0) {
            // C2 = L2 * L1'
            C.bottomRows(ndrow3).noalias() = L2 * L1.adjoint();
          }
        }

        /// where each row of C lands within current parent s
        VectorI RelativeMap(maxesize);

        for (StorageIndex i = 0; i < ndrow2; ++i) {
          const StorageIndex gi = Li[pdi1 + i];  // global row index
          RelativeMap[i] = Map[gi];              // local row in s
          eigen_assert(RelativeMap[i] >= 0 && RelativeMap[i] < nsrow);
        }

        // Assemble: L_s( RelativeMap[i], RelativeMap[j] ) -= C(i,j)
        BlockMap Sblock(Lx.data() + psx, nsrow, nscol, StrideType(nsrow, 1));
        for (StorageIndex j = 0; j < ndrow1; ++j) {
          const StorageIndex cj = RelativeMap[j];  // parent column index (within s)
          eigen_assert(cj >= 0 && cj < nscol);
          for (StorageIndex i = j; i < ndrow2; ++i) {
            const StorageIndex ri = RelativeMap[i];  // parent row index (within s)
            eigen_assert(ri >= j && ri < nsrow);
            Sblock(ri, cj) -= C(i, j);
          }
        }

        // Prepare child d for its next ancestor
        Lpos[d] = pdi2 - pdi;  // skip rows already used
        if (Lpos[d] < ndrow) {
          const StorageIndex dancestor = supe.sn_id[Li[pdi2]];
          // enqueue at head of dancestor
          Next[d] = Head[dancestor];
          Head[dancestor] = d;
        }
      }  // while children

      // Factorize diagonal block S1 (nscol by nscol) and overwrite with its Cholesky L1
      // Map to a square view with outer stride = nsrow
      BlockMap Sfull(Lx.data() + psx, nsrow, nscol, StrideType(nsrow, 1));
      BlockMap S1(Lx.data() + psx, nscol, nscol, StrideType(nsrow, 1));  // top nscol rows

      // Compute LL^T = S1 (lower)
      Eigen::LLT<DenseMat> llt;
      DenseMat S1copy = S1;  // make a compact square copy
      llt.compute(S1copy);
      if (llt.info() != Eigen::Success) {
        // err: not SPD
        return false;
      }
      S1.setZero();
      S1.template triangularView<Eigen::Lower>() = llt.matrixL();

      // Triangular solve for subdiagonal block: S2 := S2 * inv(L1')  (on the right)
      const StorageIndex nsrow2 = nsrow - nscol;
      if (nsrow2 > 0) {
        auto S2 = Sfull.bottomRows(nsrow2);  // (nsrow2 bu nscol)

        // Solve on the right by transposing: (L1 * X^T = S2^T)
        DenseMat S2T = S2.transpose();
        auto L1view = S1.template triangularView<Eigen::Lower>();
        L1view.solveInPlace(S2T);
        S2 = S2T.transpose();
      }

      // queue this supernode as a child of its parent (if any)
      if (nsrow2 > 0) {
        Lpos[s] = nscol;  // next time, child rows start below the diag block
        const StorageIndex parent_s = supe.sn_id[Li[psi + nscol]];
        if (parent_s != kEmpty) {
          Next[s] = Head[parent_s];
          Head[parent_s] = s;
        }
      }

      // clear map entries we set for s
      for (StorageIndex k = 0; k < nsrow; ++k) Map[Li[psi + k]] = kEmpty;
    }

    return true;  // yay it worked
  }

  /**
   * @brief Export the supernodal factor `L` into a sparse matrix in column-major CSC format.
   *
   * @param supe supernode structure
   * @param Lpi per-supernode pointers into Li (size nsuper+1)
   * @param Lpx per-supernode pointers into Lx (size nsuper+1)
   * @param Li row indices per supernode (size = Lpi[nsuper])
   * @param Lx dense storage for all supernodes (size = Lpx[nsuper])
   * @param n order of the matrix
   * @param drop_tol drop tolerance; remove values with abs() <= drop_tol
   * @return CholMatrixType Eigen matrix
   */
  static CholMatrixType export_sparse(const Supernodes& supe, const VectorI& Lpi, const VectorI& Lpx, const VectorI& Li,
                                      const Matrix<Scalar, Dynamic, 1>& Lx, StorageIndex n,
                                      Scalar drop_tol = Scalar(0)) {
    const StorageIndex nsuper = static_cast<StorageIndex>(supe.supernodes.size()) - 1;

    // count nnz(L) quickly: for a supernode with nscol by nsrow (lower form),
    // column c (0..nscol-1) has (nsrow - c) entries.
    size_t nnz = 0;
    for (StorageIndex s = 0; s < nsuper; ++s) {
      const StorageIndex nscol = supe.supernodes[s + 1] - supe.supernodes[s];
      const StorageIndex nsrow = supe.Snz[s];
      for (StorageIndex c = 0; c < nscol; ++c) nnz += size_t(nsrow - c);
    }

    CholMatrixType L(n, n);
    L.reserve(nnz);

    // emit columns in order with startVec/insertBack
    for (StorageIndex s = 0; s < nsuper; ++s) {
      const StorageIndex k1 = supe.supernodes[s];
      const StorageIndex k2 = supe.supernodes[s + 1];
      const StorageIndex nscol = k2 - k1;
      const StorageIndex nsrow = supe.Snz[s];

      /// start of row index list for s
      const StorageIndex psi = Lpi[s];

      /// start of dense block for s
      const StorageIndex psx = Lpx[s];

      // dense block is column-major with leading dimension nsrow
      for (StorageIndex c = 0; c < nscol; ++c) {
        const StorageIndex j = k1 + c;  // global column in L
        L.startVec(j);

        // lower part for this column within the supernode:
        // local rows r = c..nsrow-1 map to global rows Li[psi + r]
        const StorageIndex col_offset = psx + c * nsrow;
        for (StorageIndex r = c; r < nsrow; ++r) {
          const StorageIndex i = Li[psi + r];   // global row
          const Scalar v = Lx[col_offset + r];  // value L(i,j)

          // drop exact & near zeros values
          if (drop_tol == Scalar(0) ? (v != Scalar(0)) : (std::abs(v) > drop_tol)) {
            // i >= j is guaranteed by construction
            L.insertBack(i, j) = v;
          }
        }
      }
    }

    L.finalize();
    return L;
  }
};
}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_SUPERNODAL_CHOLESKY_IMPL_H