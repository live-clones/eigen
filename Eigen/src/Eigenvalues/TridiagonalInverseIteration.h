// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_TRIDIAGONAL_INVERSE_ITERATION_H
#define EIGEN_TRIDIAGONAL_INVERSE_ITERATION_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {
namespace internal {

/** \internal
 *
 * Self-contained deterministic pseudo-random generator (SplitMix64) producing reproducible
 * values in the open interval (-1, 1). Used to seed the inverse-iteration start vectors.
 *
 * \c internal::random draws from the global \c rand() state, which is neither reproducible
 * across runs nor safe to call from several threads; a deterministic generator instead keeps
 * the eigenvector signs and the per-cluster reorthogonalization stable regardless of how the
 * columns are scheduled.
 */
struct inverse_iteration_rng {
  numext::uint64_t state;

  explicit inverse_iteration_rng(numext::uint64_t seed) : state(seed) {}

  template <typename RealScalar>
  RealScalar next() {
    // SplitMix64: a strong avalanche so that consecutive per-column seeds yield well-separated
    // (uncorrelated) start vectors -- consecutive LCG seeds would produce near-parallel starts that
    // collapse under the cluster reorthogonalization.
    state += 0x9E3779B97F4A7C15ULL;
    numext::uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    // Top 32 bits give a uniform integer in [0, 2^32); map to [0, 1) and then to (-1, 1).
    const numext::uint32_t hi = numext::uint32_t(z >> 32);
    const RealScalar u = RealScalar(hi) * (RealScalar(1) / RealScalar(4294967296.0));
    return RealScalar(2) * u - RealScalar(1);
  }
};

/** \internal
 *
 * In-place LU factorization of the symmetric tridiagonal matrix \f$ T - \lambda I \f$ by
 * Gaussian elimination with partial pivoting and implicit row interchanges (LAPACK's xLAGTF):
 *   \f$ T - \lambda I = P L U \f$,
 * with \c P a permutation, \c L unit lower bidiagonal, and \c U upper triangular with up to two
 * non-zero super-diagonals. This is the careful factorization that makes inverse iteration on the
 * deliberately near-singular \f$ (T - \lambda I) \f$ (\f$ \lambda \f$ \f$ \approx \f$ a true
 * eigenvalue) robust: the pivots are never replaced here -- tiny ones are handled, when they would
 * otherwise overflow the back-solve, by tridiagonal_lagts().
 *
 * On entry \a d holds the diagonal of T, and both \a du and \a dl hold the (shared) off-diagonal of
 * the symmetric T. On exit:
 *  - \a d   holds the \c n diagonal elements of U,
 *  - \a du  holds the \c n-1 first super-diagonal elements of U,
 *  - \a du2 holds the \c n-2 second super-diagonal elements of U,
 *  - \a dl  holds the \c n-1 multipliers of L,
 *  - \a piv[k] (k < n-1) is 1 if rows k and k+1 were interchanged at step k, 0 otherwise.
 *
 * \param[in,out] d   diagonal (length \c n).
 * \param[in,out] du  first super-diagonal (length \c n-1, indices 0..n-2).
 * \param[in,out] dl  sub-diagonal / L multipliers (length \c n-1, indices 0..n-2).
 * \param[out]    du2 second super-diagonal of U (length \c n-2, indices 0..n-3).
 * \param[out]    piv row-interchange flags (length \c n).
 * \param[in]     lambda the shift \f$ \lambda \f$.
 * \param[in]     n      dimension of T.
 */
template <typename RealScalar>
void tridiagonal_lagtf(RealScalar* d, RealScalar* du, RealScalar* dl, RealScalar* du2, Index* piv, RealScalar lambda,
                       Index n) {
  using std::abs;
  d[0] -= lambda;
  piv[n - 1] = 0;
  if (n == 1) return;

  RealScalar scale1 = abs(d[0]) + abs(du[0]);
  for (Index k = 0; k < n - 1; ++k) {
    d[k + 1] -= lambda;
    RealScalar scale2 = abs(dl[k]) + abs(d[k + 1]);
    if (k < n - 2) scale2 += abs(du[k + 1]);
    // Scaled magnitudes of the two candidate pivots; pick the larger (partial pivoting).
    const RealScalar piv1 = numext::is_exactly_zero(d[k]) ? RealScalar(0) : abs(d[k]) / scale1;
    if (numext::is_exactly_zero(dl[k])) {
      piv[k] = 0;
      scale1 = scale2;
      if (k < n - 2) du2[k] = RealScalar(0);
    } else {
      const RealScalar piv2 = abs(dl[k]) / scale2;
      if (piv2 <= piv1) {
        // No interchange: eliminate using the diagonal pivot d[k] (non-zero, since piv2 > 0).
        piv[k] = 0;
        scale1 = scale2;
        dl[k] = dl[k] / d[k];
        d[k + 1] -= dl[k] * du[k];
        if (k < n - 2) du2[k] = RealScalar(0);
      } else {
        // Interchange rows k and k+1; the sub-diagonal entry becomes the pivot.
        piv[k] = 1;
        const RealScalar mult = d[k] / dl[k];
        d[k] = dl[k];
        const RealScalar temp = d[k + 1];
        d[k + 1] = du[k] - mult * temp;
        if (k < n - 2) {
          du2[k] = du[k + 1];
          du[k + 1] = -mult * du2[k];
        }
        du[k] = temp;
        dl[k] = mult;
      }
    }
  }
}

/** \internal
 *
 * Overflow-safe solution of \f$ (T - \lambda I) x = b \f$ from the xLAGTF factorization
 * \f$ T - \lambda I = P L U \f$ (LAPACK's xLAGTS with JOB = -1). The right-hand side \a b is
 * overwritten with the solution. Diagonal elements of \c U that are so small the back-solve would
 * overflow are perturbed (by a quantity of order \f$ \mathrm{eps} \cdot \|U\| \f$, with the sign of
 * the pivot, doubling until safe) -- exactly the situation produced by inverse iteration, where
 * \f$ \lambda \f$ is intentionally a near-eigenvalue and \f$ U \f$ is nearly singular.
 *
 * \param[in]     d   diagonal of U (length \c n).
 * \param[in]     rcp elementwise reciprocal \c 1/d (length \c n), precomputed once by the caller with
 *                    an exact division and reused across the inverse-iteration solves. Entries for a
 *                    vanishing pivot may be \c inf; they are only consumed on the fast path, which is
 *                    not taken for such pivots.
 * \param[in]     du  first super-diagonal of U (length \c n-1).
 * \param[in]     dl  sub-diagonal multipliers of L (length \c n-1).
 * \param[in]     du2 second super-diagonal of U (length \c n-2).
 * \param[in]     piv row-interchange flags from tridiagonal_lagtf() (length \c n).
 * \param[in,out] b   right-hand side on input, solution on output (length \c n).
 * \param[in]     n   dimension of T.
 */
template <typename RealScalar>
void tridiagonal_lagts(const RealScalar* d, const RealScalar* rcp, const RealScalar* du, const RealScalar* dl,
                       const RealScalar* du2, const Index* piv, RealScalar* b, Index n) {
  using std::abs;
  const RealScalar eps = NumTraits<RealScalar>::epsilon();
  const RealScalar sfmin = (std::numeric_limits<RealScalar>::min)();
  const RealScalar bignum = RealScalar(1) / sfmin;

  // Perturbation floor: eps times the largest magnitude entry of U.
  RealScalar tol = abs(d[0]);
  if (n > 1) tol = numext::maxi(tol, numext::maxi(abs(d[1]), abs(du[0])));
  for (Index k = 2; k < n; ++k)
    tol = numext::maxi(tol, numext::maxi(abs(d[k]), numext::maxi(abs(du[k - 1]), abs(du2[k - 2]))));
  tol *= eps;
  if (numext::is_exactly_zero(tol)) tol = eps;

  // Forward substitution: apply P then L^{-1}. Written branchlessly in the (data-dependent, roughly
  // 50/50) pivot-interchange flag so a misprediction cannot stall the serial recurrence; both arms
  // are evaluated and selected, reproducing the two branches bit for bit.
  for (Index k = 1; k < n; ++k) {
    const RealScalar dlk = dl[k - 1];
    const RealScalar bkm1 = b[k - 1];
    const RealScalar bk = b[k];
    const bool interchange = piv[k - 1] != 0;
    b[k - 1] = interchange ? bk : bkm1;
    b[k] = interchange ? (bkm1 - dlk * bk) : (bk - dlk * bkm1);
  }

  // Back substitution: solve U x = b. Away from the singularity the pivot is safe and the quotient is
  // a reciprocal multiply (rcp[k] = 1/d[k], exact and precomputed once for all the solves); only a
  // dangerously small pivot -- the rare near-singular element of the deliberately singular system --
  // falls back to the LAPACK xLAGTS perturbation/rescale with a true division. The guard is written so
  // the fast path is taken exactly when the original would divide by the unperturbed pivot, and is
  // virtually always predicted taken.
  for (Index k = n - 1; k >= 0; --k) {
    RealScalar temp;
    if (k <= n - 3)
      temp = b[k] - du[k] * b[k + 1] - du2[k] * b[k + 2];
    else if (k == n - 2)
      temp = b[k] - du[k] * b[k + 1];
    else
      temp = b[k];
    const RealScalar ak = d[k];
    const RealScalar absak = abs(ak);
    if (EIGEN_PREDICT_TRUE(absak >= RealScalar(1) || (absak >= sfmin && abs(temp) <= absak * bignum))) {
      b[k] = temp * rcp[k];
    } else {
      // Tiny pivot: perturb (or rescale) exactly as LAPACK xLAGTS so the quotient cannot overflow.
      RealScalar a = ak;
      RealScalar t = temp;
      RealScalar pert = (a >= RealScalar(0)) ? tol : -tol;
      while (true) {
        const RealScalar aa = abs(a);
        if (aa < RealScalar(1)) {
          if (aa < sfmin) {
            if (numext::is_exactly_zero(aa) || abs(t) * sfmin > aa) {
              a += pert;
              pert *= RealScalar(2);
              continue;
            } else {
              t *= bignum;
              a *= bignum;
            }
          } else if (abs(t) > aa * bignum) {
            a += pert;
            pert *= RealScalar(2);
            continue;
          }
        }
        break;
      }
      b[k] = t / a;
    }
  }
}

/** \internal
 *
 * Computes eigenvectors of a real symmetric tridiagonal matrix T by inverse iteration, given a set
 * of already-computed eigenvalues (LAPACK's xSTEIN driver, built on tridiagonal_lagtf() /
 * tridiagonal_lagts()). For each requested eigenvalue \f$ \lambda_j \f$ the routine factors
 * \f$ T - \lambda_j I \f$ and applies a few steps of inverse iteration from a deterministic
 * pseudo-random start, reorthogonalizing (modified Gram-Schmidt) against the eigenvectors of any
 * tightly clustered neighbours so that a degenerate cluster yields an orthonormal basis.
 *
 * The eigenvalues are assumed sorted in non-decreasing order (as produced by spectral bisection or
 * the QR algorithm). The whole matrix is treated as a single block: an eigenvalue belonging to a
 * disconnected diagonal block (separated by a zero off-diagonal) factors to a near-singular U only
 * on its own block, so the iterate is automatically supported there; reorthogonalization across
 * blocks is harmless because those eigenvectors are already orthogonal.
 *
 * \a eivals may be an arbitrary subset of the spectrum (e.g. a bisection range): exactly one column
 * is produced per supplied eigenvalue. Reorthogonalization only ever runs among the supplied
 * eigenvalues, so the output columns are mutually orthonormal but are not orthogonalized against
 * cluster members omitted from \a eivals. A subset that splits a numerically degenerate cluster
 * therefore returns an arbitrary (still orthonormal) basis of the requested slice rather than a
 * canonical one; pass the whole cluster when that distinction matters.
 *
 * \param[in]  diag    diagonal of T (length \c n).
 * \param[in]  subdiag sub-diagonal of T (length \c n-1).
 * \param[in]  eivals  the eigenvalues whose eigenvectors are wanted, non-decreasing (length \c m).
 * \param[out] eivecs  filled with the \c m eigenvectors as its columns (must be sized \c n x \c m);
 *                     column \c j is a unit-norm eigenvector for \c eivals[j].
 */
template <typename DiagType, typename SubdiagType, typename EivalType, typename EivecType>
void tridiagonal_inverse_iteration(const DiagType& diag, const SubdiagType& subdiag, const EivalType& eivals,
                                   EivecType& eivecs) {
  typedef typename DiagType::Scalar RealScalar;
  EIGEN_STATIC_ASSERT(NumTraits<RealScalar>::IsInteger == 0 && NumTraits<RealScalar>::IsComplex == 0,
                      THIS_FUNCTION_IS_NOT_FOR_INTEGER_OR_COMPLEX_TYPES)
  using std::abs;
  using std::sqrt;

  const Index n = diag.size();
  const Index m = eivals.size();
  if (n == 0 || m == 0) return;
  if (n == 1) {
    eivecs.setOnes();
    return;
  }

  const RealScalar eps = NumTraits<RealScalar>::epsilon();

  // Normalize T (and the shifts) to O(1) so the deliberately near-singular factor/solve cannot
  // overflow or underflow; eigenvectors are invariant under this uniform scaling. Mirrors the
  // scaling used by tridiagonal_bisection() and SelfAdjointEigenSolver::compute().
  RealScalar scale = diag.cwiseAbs().maxCoeff();
  scale = numext::maxi(scale, subdiag.cwiseAbs().maxCoeff());
  RealScalar inv_scale = RealScalar(1) / scale;
  if (numext::is_exactly_zero(scale) || !(numext::isfinite)(inv_scale)) {
    scale = RealScalar(1);
    inv_scale = RealScalar(1);
  }
  const Matrix<RealScalar, Dynamic, 1> sdiag = diag.array() * inv_scale;
  const Matrix<RealScalar, Dynamic, 1> ssub = subdiag.array() * inv_scale;

  // Infinity norm of the scaled T: max_i (|e_{i-1}| + |d_i| + |e_i|), missing boundary off-diagonals zero.
  RealScalar onenrm = abs(sdiag[0]) + abs(ssub[0]);
  onenrm = numext::maxi(onenrm, abs(sdiag[n - 1]) + abs(ssub[n - 2]));
  for (Index i = 1; i < n - 1; ++i) onenrm = numext::maxi(onenrm, abs(ssub[i - 1]) + abs(sdiag[i]) + abs(ssub[i]));
  if (numext::is_exactly_zero(onenrm)) onenrm = RealScalar(1);  // T == 0: any orthonormal basis works

  // Cluster threshold and convergence threshold (LAPACK xSTEIN constants).
  const RealScalar ortol = RealScalar(1e-3) * onenrm;
  const RealScalar dtpcrt = sqrt(RealScalar(0.1) / RealScalar(n));
  const int maxits = 5;
  const int extra = 2;

  // Work arrays for the LU factors of T - lambda*I (reused across eigenvalues) and the iterate.
  // lu_rcp holds the exact reciprocal of the U diagonal, computed once per factorization and reused
  // by the three inverse-iteration back-solves.
  Matrix<RealScalar, Dynamic, 1> lu_d(n), lu_dl(n), lu_du(n), lu_du2(n), lu_rcp(n), b(n);
  Matrix<Index, Dynamic, 1> piv(n);

  Index gpind = 0;                 // index of the first eigenvector in the current cluster
  RealScalar xjm = RealScalar(0);  // previous (possibly perturbed) shift

  for (Index j = 0; j < m; ++j) {
    RealScalar xj = eivals[j] * inv_scale;
    // Perturb a shift that sits right on top of its predecessor so the two solves stay distinct.
    if (j > 0) {
      const RealScalar pertol = RealScalar(10) * abs(eps * xj);
      if (xj - xjm < pertol) xj = xjm + pertol;
    }
    // A gap larger than ortol starts a new cluster; reorthogonalize only within a cluster.
    if (j == 0 || abs(xj - xjm) > ortol) gpind = j;

    // Factor T - xj*I = P L U once; the inverse-iteration loop reuses it.
    lu_d = sdiag;
    lu_du.head(n - 1) = ssub;
    lu_dl.head(n - 1) = ssub;
    tridiagonal_lagtf<RealScalar>(lu_d.data(), lu_du.data(), lu_dl.data(), lu_du2.data(), piv.data(), xj, n);
    // Exact reciprocals of the U diagonal (pdiv, not the approximate preciprocal of cwiseInverse());
    // a vanishing pivot yields inf, which the back-solve fast path never consumes.
    lu_rcp.array() = RealScalar(1) / lu_d.array();

    // Deterministic pseudo-random start vector (seeded per column for thread independence).
    inverse_iteration_rng rng(numext::uint64_t(j) + 1);
    for (Index i = 0; i < n; ++i) b[i] = rng.template next<RealScalar>();

    int nrmchk = 0;
    for (int its = 0; its < maxits; ++its) {
      // Scale the right-hand side so the near-singular solve neither overflows nor underflows.
      const RealScalar bmax = b.cwiseAbs().maxCoeff();
      if (numext::is_exactly_zero(bmax)) break;  // degenerate iterate; accept as is
      const RealScalar scl = RealScalar(n) * onenrm * numext::maxi(eps, abs(lu_d[n - 1])) / bmax;
      b *= scl;
      tridiagonal_lagts<RealScalar>(lu_d.data(), lu_rcp.data(), lu_du.data(), lu_dl.data(), lu_du2.data(), piv.data(),
                                    b.data(), n);

      // Modified Gram-Schmidt against the already-accepted eigenvectors of this cluster.
      for (Index i = gpind; i < j; ++i) b -= b.dot(eivecs.col(i)) * eivecs.col(i);

      const RealScalar nrm = b.cwiseAbs().maxCoeff();
      if (nrm < dtpcrt) continue;          // not yet grown into the eigenvector; iterate
      if (++nrmchk < extra + 1) continue;  // a couple of safety iterations past the threshold
      break;
    }

    // Normalize to unit 2-norm with a deterministic sign (largest-magnitude entry positive). The
    // iterate can carry huge entries (~1/eps times the start) on a nearly singular solve, so divide
    // by the infinity norm first -- otherwise squaredNorm() would overflow and zero out the vector.
    Index jmax = 0;
    const RealScalar binf = b.cwiseAbs().maxCoeff(&jmax);
    if (!numext::is_exactly_zero(binf)) b /= binf;
    const RealScalar nrm2 = b.norm();
    RealScalar scl = numext::is_exactly_zero(nrm2) ? RealScalar(1) : RealScalar(1) / nrm2;
    if (b[jmax] < RealScalar(0)) scl = -scl;
    eivecs.col(j) = b * scl;

    xjm = xj;
  }
}

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_TRIDIAGONAL_INVERSE_ITERATION_H
