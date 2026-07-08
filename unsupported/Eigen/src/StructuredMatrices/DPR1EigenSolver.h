// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_STRUCTURED_DPR1_EIGEN_SOLVER_H
#define EIGEN_STRUCTURED_DPR1_EIGEN_SOLVER_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

/** \ingroup StructuredMatrices_Module
 * \class DPR1EigenSolver
 * \brief Direct O(n^2) eigensolver for real symmetric diagonal-plus-rank-one
 * matrices \f$ A = D + \rho\, z z^T \f$, via the secular equation.
 *
 * This is the standalone version of the kernel at the heart of the
 * divide-and-conquer symmetric eigensolvers (LAPACK's xLAED2/3/4): after
 * \em deflation -- entries with negligible \f$ |z_i| \f$ are eigenpairs of the
 * diagonal already, and (nearly) equal diagonal entries are combined by Givens
 * rotations whose dropped coupling is below a backward-stability threshold --
 * the surviving eigenvalues are the roots of the secular equation
 * \f[ f(\lambda) = 1 + \rho \sum_i \frac{z_i^2}{d_i - \lambda} = 0, \f]
 * one in each interval between consecutive poles. Each root is bracketed and
 * bisected in coordinates \em shifted to its nearest pole, so every distance
 * \f$ \lambda - d_i \f$ is retained as an exact data difference plus a small
 * offset instead of a cancellation-prone subtraction of close numbers.
 * Eigenvectors are then built not from the original \c z but from the
 * Gu-Eisenstat vector \f$ \hat z \f$ -- the one for which the computed roots
 * are \em exact secular eigenvalues -- which is what makes the computed
 * eigenvector matrix numerically orthogonal without any reorthogonalization.
 *
 * The total cost is O(n^2): O(n log(1/eps)) per bisected root, O(n) per
 * Gu-Eisenstat weight, and O(n) per eigenvector (the deflation rotations are
 * replayed instead of accumulated into a dense matrix).
 *
 * Both signs of \f$ \rho \f$ are supported (negative \f$ \rho \f$ is handled by
 * negating the matrix), as are \f$ \rho = 0 \f$, zero \c z, repeated diagonal
 * entries and any ordering of \c d.
 *
 * \code
 *   DPR1EigenSolver<double> es(d, rho, z);
 *   VectorXd  lambda = es.eigenvalues();   // ascending
 *   MatrixXd  V      = es.eigenvectors();  // orthogonal
 * \endcode
 *
 * \tparam RealScalar_ a real (non-complex) scalar type.
 *
 * References:
 *  - M. Gu and S. C. Eisenstat, "A stable and efficient algorithm for the
 *    rank-one modification of the symmetric eigenproblem," SIAM J. Matrix Anal.
 *    Appl., 15(4):1266-1276, 1994.
 *
 * \sa class DiagonalPlusLowRank, class SelfAdjointEigenSolver
 */
template <typename RealScalar_>
class DPR1EigenSolver {
 public:
  using RealScalar = RealScalar_;
  using Scalar = RealScalar;
  using Index = Eigen::Index;
  using VectorType = Matrix<RealScalar, Dynamic, 1>;
  using MatrixType = Matrix<RealScalar, Dynamic, Dynamic>;

  static_assert(!NumTraits<RealScalar>::IsComplex,
                "DPR1EigenSolver requires a real scalar type; for a Hermitian rank-one "
                "update, factor out the phases of z first (z = Phi*|z| gives "
                "A = Phi (D + rho |z||z|^T) Phi^H).");

  /** Default constructor; call \ref compute before querying results. */
  DPR1EigenSolver() : m_isInitialized(false), m_vectorsComputed(false), m_info(InvalidInput) {}

  /** Computes the eigendecomposition of \c diag(d) + \c rho*z*z^T.
   * \a options is \c ComputeEigenvectors (the default) or \c EigenvaluesOnly. */
  DPR1EigenSolver(const VectorType& d, RealScalar rho, const VectorType& z, int options = ComputeEigenvectors)
      : m_isInitialized(false), m_vectorsComputed(false), m_info(InvalidInput) {
    compute(d, rho, z, options);
  }

  /** Computes the eigendecomposition of \c diag(d) + \c rho*z*z^T. \sa DPR1EigenSolver() */
  DPR1EigenSolver& compute(const VectorType& d, RealScalar rho, const VectorType& z, int options = ComputeEigenvectors);

  /** \returns the eigenvalues, sorted in increasing order. */
  const VectorType& eigenvalues() const {
    eigen_assert(m_isInitialized && "DPR1EigenSolver is not initialized.");
    return m_eivalues;
  }

  /** \returns the orthogonal matrix of eigenvectors; column \c k matches
   * \c eigenvalues()[k]. \pre \ref compute was called with \c ComputeEigenvectors. */
  const MatrixType& eigenvectors() const {
    eigen_assert(m_isInitialized && "DPR1EigenSolver is not initialized.");
    eigen_assert(m_vectorsComputed && "eigenvectors were not computed");
    return m_eivec;
  }

  /** \returns \c Success if the decomposition succeeded, \c NoConvergence if a
   * secular root could not be fully resolved, \c InvalidInput if the input was
   * non-finite or its spectrum is not representable in the scalar type. */
  ComputationInfo info() const {
    eigen_assert(m_isInitialized && "DPR1EigenSolver is not initialized.");
    return m_info;
  }

 private:
  // A deflation-stage Givens rotation acting on working rows (i, j).
  struct Rotation {
    Index i, j;
    RealScalar c, s;
  };

  /** \internal Evaluates the shifted secular function
   * g(tau) = 1 + rho * sum_i zeta_i^2 / (delta_i - tau), with delta_i the pole
   * offsets relative to the chosen shift. */
  static RealScalar secular(const VectorType& delta, const VectorType& zeta2, RealScalar rho, RealScalar tau) {
    RealScalar s(1);
    for (Index i = 0; i < delta.size(); ++i) s += rho * zeta2[i] / (delta[i] - tau);
    return s;
  }

  VectorType m_eivalues;
  MatrixType m_eivec;
  bool m_isInitialized;
  bool m_vectorsComputed;
  ComputationInfo m_info;
};

template <typename RealScalar_>
DPR1EigenSolver<RealScalar_>& DPR1EigenSolver<RealScalar_>::compute(const VectorType& d, RealScalar rho,
                                                                    const VectorType& z, int options) {
  const Index n = d.size();
  eigen_assert(z.size() == n && "d and z must have the same size");
  const bool computeVectors = (options & ComputeEigenvectors) != 0;
  m_vectorsComputed = computeVectors;
  m_info = Success;

  m_eivalues.resize(n);
  if (computeVectors) m_eivec.setIdentity(n, n);
  if (n == 0) {
    m_isInitialized = true;
    return *this;
  }

  // Non-finite input would break the sorting comparator (not a strict weak
  // order under NaN) and silently deflate everything; reject it up front.
  if (!(d.allFinite() && z.allFinite() && (numext::isfinite)(rho))) {
    m_eivalues.setConstant(NumTraits<RealScalar>::quiet_NaN());
    m_info = InvalidInput;
    m_isInitialized = true;
    return *this;
  }

  // ---- reduce to rho >= 0 by negating the matrix if necessary ----
  const bool negated = rho < RealScalar(0);
  VectorType dW = negated ? VectorType(-d) : d;
  RealScalar rhoW = negated ? -rho : rho;

  // ---- sort the diagonal ascending; permutation pi maps working index -> input row ----
  std::vector<Index> pi(static_cast<std::size_t>(n));
  for (Index i = 0; i < n; ++i) pi[static_cast<std::size_t>(i)] = i;
  std::stable_sort(pi.begin(), pi.end(), [&dW](Index a, Index b) { return dW[a] < dW[b]; });
  VectorType ds(n), zs(n);
  for (Index i = 0; i < n; ++i) {
    ds[i] = dW[pi[static_cast<std::size_t>(i)]];
    zs[i] = z[pi[static_cast<std::size_t>(i)]];
  }

  // ---- normalize z, absorbing its norm into rho ----
  const RealScalar znorm = zs.stableNorm();  // overflow/underflow-safe
  if (znorm > RealScalar(0)) zs /= znorm;
  // Multiply rho through one factor at a time: znorm^2 alone can overflow even
  // when rho*||z||^2 -- the only quantity that matters -- is perfectly ordinary.
  rhoW = (rhoW * znorm) * znorm;  // now A~ = diag(ds) + rhoW * zs zs^T with ||zs|| = 1

  // ---- deflation ----
  // Backward-error budget: dropping a coupling of size <= tol perturbs the
  // matrix by O(tol), like LAPACK's xLAED2. Using max(|d|_inf, rho*||z||^2) as
  // the scale is the unscaled-problem generalization of xLAED2's criterion: the
  // perturbation stays O(eps * (||D|| + rho ||z||^2)), i.e. backward stable in
  // the data.
  const RealScalar normScale = numext::maxi(ds.cwiseAbs().maxCoeff(), rhoW);
  if (!((numext::isfinite)(rhoW) && (numext::isfinite)(normScale))) {
    // rho*||z||^2 (or the data) overflows the scalar type: the spectrum is not
    // representable. Pre-scale the problem and try again.
    m_eivalues.setConstant(NumTraits<RealScalar>::quiet_NaN());
    m_info = InvalidInput;
    m_isInitialized = true;
    return *this;
  }
  const RealScalar tol = RealScalar(8) * NumTraits<RealScalar>::epsilon() * normScale;

  std::vector<Rotation> rotations;
  std::vector<bool> deflated(static_cast<std::size_t>(n), false);

  if (rhoW <= tol) {
    // The rank-one update is negligible: everything deflates.
    for (Index i = 0; i < n; ++i) deflated[static_cast<std::size_t>(i)] = true;
  } else {
    // (a) negligible z entries: d_i is an eigenvalue already.
    for (Index i = 0; i < n; ++i)
      if (rhoW * numext::abs(zs[i]) <= tol) deflated[static_cast<std::size_t>(i)] = true;
    // (b) close poles: rotate the weight of the earlier pole onto the later one.
    // The dropped off-diagonal coupling is |c*s*(ds[i]-ds[p])| <= tol.
    Index p = -1;
    for (Index i = 0; i < n; ++i) {
      if (deflated[static_cast<std::size_t>(i)]) continue;
      if (p >= 0) {
        const RealScalar r = numext::hypot(zs[p], zs[i]);
        const RealScalar c = zs[i] / r, s = zs[p] / r;  // zeroes the earlier entry
        if (numext::abs(c * s * (ds[i] - ds[p])) <= tol) {
          // Rotated diagonal: the deflated position keeps one entry, the
          // surviving pole the other; both stay inside [ds[p], ds[i]].
          const RealScalar dp = c * c * ds[p] + s * s * ds[i];
          const RealScalar di = s * s * ds[p] + c * c * ds[i];
          ds[p] = dp;
          ds[i] = di;
          zs[i] = r;
          zs[p] = RealScalar(0);
          deflated[static_cast<std::size_t>(p)] = true;
          if (computeVectors) rotations.push_back(Rotation{p, i, c, s});
        }
      }
      if (!deflated[static_cast<std::size_t>(i)]) p = i;
    }
  }

  // ---- gather the surviving secular subproblem ----
  std::vector<Index> sub;  // working positions of the surviving poles
  for (Index i = 0; i < n; ++i)
    if (!deflated[static_cast<std::size_t>(i)]) sub.push_back(i);
  const Index m = static_cast<Index>(sub.size());

  // Eigenvalues in working (sorted) coordinates; deflated entries are final.
  VectorType lambdaW = ds;

  MatrixType subVectors;  // m x m secular eigenvectors (in subproblem coordinates)

  if (m > 0) {
    VectorType delta(m), zeta(m), zeta2(m);
    std::vector<Index> shiftIndex(static_cast<std::size_t>(m));
    VectorType tau(m);
    for (Index a = 0; a < m; ++a) {
      delta[a] = ds[sub[static_cast<std::size_t>(a)]];
      zeta[a] = zs[sub[static_cast<std::size_t>(a)]];
      zeta2[a] = zeta[a] * zeta[a];
    }
    const RealScalar zeta2sum = zeta2.sum();

    // ---- secular roots by pole-shifted bisection ----
    // The iteration count guarantees bitwise convergence: the surviving pole
    // gaps exceed 16*eps*normScale and the smallest root offsets are bounded
    // below by tol^2/normScale, so the bracket-to-root dynamic range is at most
    // O(1/eps^2), i.e. ~2*digits halvings, plus slack.
    const int maxBisect =
        2 * ((std::numeric_limits<RealScalar>::digits > 0) ? static_cast<int>(std::numeric_limits<RealScalar>::digits)
                                                           : 128) +
        32;
    VectorType lam(m), dsh(m);  // dsh is refilled from scratch each root
    for (Index k = 0; k < m; ++k) {
      // Choose the shift pole and the bracket, entirely in shifted coordinates.
      RealScalar lo, hi;
      Index shift;
      if (k + 1 == m) {
        // Last root: it lies in (delta_m, delta_m + rho*|zeta|^2]; never form
        // the unshifted right end (it can round to the pole itself).
        shift = k;
        lo = RealScalar(0);
        hi = rhoW * zeta2sum;
      } else {
        // Interior root: the secular function is increasing between the poles,
        // so its sign at the midpoint picks the nearer pole as the shift.
        const RealScalar left = delta[k], right = delta[k + 1];
        const RealScalar mid = left + (right - left) / RealScalar(2);
        if (secular(delta, zeta2, rhoW, mid) > RealScalar(0)) {
          shift = k;
          lo = RealScalar(0);
          hi = mid - left;
        } else {
          shift = k + 1;
          lo = mid - right;  // negative
          hi = RealScalar(0);
        }
      }
      const RealScalar shiftVal = delta[shift];
      for (Index a = 0; a < m; ++a) dsh[a] = delta[a] - shiftVal;

      // Bisection: g(lo) < 0 < g(hi) by the pole signs (for shift = k the
      // function tends to -inf as tau -> 0+, for shift = k+1 to +inf as
      // tau -> 0-). The endpoints are never evaluated.
      RealScalar a0 = lo, b0 = hi;
      bool converged = false;
      for (int iter = 0; iter < maxBisect; ++iter) {
        const RealScalar t = a0 + (b0 - a0) / RealScalar(2);
        if (t == a0 || t == b0) {
          converged = true;  // interval fully resolved
          break;
        }
        if (secular(dsh, zeta2, rhoW, t) > RealScalar(0))
          b0 = t;
        else
          a0 = t;
      }
      if (!converged) m_info = NoConvergence;
      const RealScalar t = a0 + (b0 - a0) / RealScalar(2);
      shiftIndex[static_cast<std::size_t>(k)] = shift;
      tau[k] = t;
      lam[k] = shiftVal + t;
    }

    // ---- Gu-Eisenstat weights: the z-vector for which lam are exact roots ----
    // zhat_i^2 = prod_j (lam_j - delta_i) / (rho * prod_{j != i} (delta_j - delta_i)),
    // with every distance lam_j - delta_i formed as (delta_shift(j) - delta_i) + tau_j.
    // Numerator and denominator factors are paired so the running product stays O(1).
    VectorType zhat(m);
    for (Index i = 0; i < m; ++i) {
      RealScalar acc = ((delta[shiftIndex[static_cast<std::size_t>(i)]] - delta[i]) + tau[i]) / rhoW;
      for (Index j = 0; j < m; ++j) {
        if (j == i) continue;
        const RealScalar num = (delta[shiftIndex[static_cast<std::size_t>(j)]] - delta[i]) + tau[j];
        acc *= num / (delta[j] - delta[i]);
      }
      zhat[i] = numext::abs(acc) > RealScalar(0) ? RealScalar(numext::sqrt(numext::abs(acc))) : RealScalar(0);
      if (zeta[i] < RealScalar(0)) zhat[i] = -zhat[i];
    }

    // ---- secular eigenvectors from the Gu-Eisenstat weights ----
    if (computeVectors) {
      subVectors.resize(m, m);
      for (Index j = 0; j < m; ++j) {
        const Index sj = shiftIndex[static_cast<std::size_t>(j)];
        if (tau[j] == RealScalar(0)) {
          // The root coincides with its shift pole to working precision (a
          // fully underflowed bracket): the eigenvector is that pole's axis.
          subVectors.col(j).setZero();
          subVectors(sj, j) = RealScalar(1);
          continue;
        }
        for (Index i = 0; i < m; ++i) {
          const RealScalar dist = (delta[i] - delta[sj]) - tau[j];  // delta_i - lam_j
          subVectors(i, j) = zhat[i] / dist;
        }
        subVectors.col(j).stableNormalize();
      }
    }
    for (Index k = 0; k < m; ++k) lambdaW[sub[static_cast<std::size_t>(k)]] = lam[k];
  }

  // ---- sort all eigenvalues ascending (deflated and secular interleave) ----
  std::vector<Index> order(static_cast<std::size_t>(n));
  for (Index i = 0; i < n; ++i) order[static_cast<std::size_t>(i)] = i;
  std::stable_sort(order.begin(), order.end(), [&lambdaW](Index a, Index b) { return lambdaW[a] < lambdaW[b]; });

  // Map working position -> secular subproblem slot (or -1 when deflated).
  std::vector<Index> subSlot(static_cast<std::size_t>(n), -1);
  for (Index a = 0; a < m; ++a) subSlot[static_cast<std::size_t>(sub[static_cast<std::size_t>(a)])] = a;

  for (Index t = 0; t < n; ++t) {
    const Index w = order[static_cast<std::size_t>(t)];
    const Index outCol = negated ? n - 1 - t : t;
    m_eivalues[outCol] = negated ? -lambdaW[w] : lambdaW[w];
    if (!computeVectors) continue;

    // Assemble the eigenvector in working coordinates, replay the deflation
    // rotations in reverse, and undo the sorting permutation.
    VectorType wvec = VectorType::Zero(n);
    const Index slot = subSlot[static_cast<std::size_t>(w)];
    if (slot < 0) {
      wvec[w] = RealScalar(1);
    } else {
      for (Index a = 0; a < m; ++a) wvec[sub[static_cast<std::size_t>(a)]] = subVectors(a, slot);
    }
    // w <- G*w with G = [[c, s], [-s, c]]: for exactly equal poles this maps the
    // deflated unit vector e_p to (c, -s) = (z_i, -z_p)/r, the exact eigenvector
    // of the 2x2 block orthogonal to the weight vector.
    for (auto it = rotations.rbegin(); it != rotations.rend(); ++it) {
      const RealScalar wi = wvec[it->i], wj = wvec[it->j];
      wvec[it->i] = it->c * wi + it->s * wj;
      wvec[it->j] = -it->s * wi + it->c * wj;
    }
    for (Index i = 0; i < n; ++i) m_eivec(pi[static_cast<std::size_t>(i)], outCol) = wvec[i];
  }

  m_isInitialized = true;
  return *this;
}

}  // namespace Eigen

#endif  // EIGEN_STRUCTURED_DPR1_EIGEN_SOLVER_H
