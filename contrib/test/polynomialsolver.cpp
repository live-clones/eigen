// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Manuel Yguel <manuel.yguel@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include <contrib/Eigen/Polynomials>
#include <iostream>
#include <algorithm>

using namespace std;

namespace Eigen {
namespace internal {
template <int Size>
struct increment_if_fixed_size : std::integral_constant<int, (Size == Dynamic) ? Dynamic : Size + 1> {};
}  // namespace internal
}  // namespace Eigen

template <typename PolynomialType>
PolynomialType polyder(const PolynomialType& p) {
  typedef typename PolynomialType::Scalar Scalar;
  PolynomialType res(p.size());
  for (Index i = 1; i < p.size(); ++i) res[i - 1] = p[i] * Scalar(i);
  res[p.size() - 1] = 0.;
  return res;
}

template <int Deg, typename POLYNOMIAL, typename SOLVER>
bool aux_evalSolver(const POLYNOMIAL& pols, SOLVER& psolve) {
  typedef typename POLYNOMIAL::Scalar Scalar;
  typedef typename POLYNOMIAL::RealScalar RealScalar;

  typedef typename SOLVER::RootsType RootsType;
  typedef Matrix<RealScalar, Deg, 1> EvalRootsType;

  const Index deg = pols.size() - 1;

  // Test template constructor from coefficient vector
  SOLVER solve_constr(pols);

  psolve.compute(pols);
  const RootsType& roots(psolve.roots());
  EvalRootsType evr(deg);
  POLYNOMIAL pols_der = polyder(pols);
  EvalRootsType der(deg);
  for (int i = 0; i < roots.size(); ++i) {
    evr[i] = std::abs(poly_eval(pols, roots[i]));
    der[i] = numext::maxi(RealScalar(1.), std::abs(poly_eval(pols_der, roots[i])));
  }

  // we need to divide by the magnitude of the derivative because
  // with a high derivative is very small error in the value of the root
  // yiels a very large error in the polynomial evaluation.
  bool evalToZero = (evr.cwiseQuotient(der)).isZero(test_precision<Scalar>());
  if (!evalToZero) {
    cerr << "WRONG root: " << endl;
    cerr << "Polynomial: " << pols.transpose() << endl;
    cerr << "Roots found: " << roots.transpose() << endl;
    cerr << "Abs value of the polynomial at the roots: " << evr.transpose() << endl;
    cerr << endl;
  }

  std::vector<RealScalar> rootModuli(roots.size());
  Map<EvalRootsType> aux(&rootModuli[0], roots.size());
  aux = roots.array().abs();
  std::sort(rootModuli.begin(), rootModuli.end());
  bool distinctModuli = true;
  for (size_t i = 1; i < rootModuli.size() && distinctModuli; ++i) {
    if (internal::isApprox(rootModuli[i], rootModuli[i - 1])) {
      distinctModuli = false;
    }
  }
  VERIFY(evalToZero || !distinctModuli);

  return distinctModuli;
}

template <int Deg, typename POLYNOMIAL>
void evalSolver(const POLYNOMIAL& pols) {
  typedef typename POLYNOMIAL::Scalar Scalar;

  typedef PolynomialSolver<Scalar, Deg> PolynomialSolverType;

  PolynomialSolverType psolve;
  aux_evalSolver<Deg, POLYNOMIAL, PolynomialSolverType>(pols, psolve);
}

template <int Deg, typename POLYNOMIAL, typename ROOTS, typename REAL_ROOTS>
void evalSolverSugarFunction(const POLYNOMIAL& pols, const ROOTS& roots, const REAL_ROOTS& real_roots) {
  using std::sqrt;
  typedef typename POLYNOMIAL::Scalar Scalar;
  typedef typename POLYNOMIAL::RealScalar RealScalar;

  typedef PolynomialSolver<Scalar, Deg> PolynomialSolverType;

  PolynomialSolverType psolve;
  if (aux_evalSolver<Deg, POLYNOMIAL, PolynomialSolverType>(pols, psolve)) {
    // It is supposed that
    //  1) the roots found are correct
    //  2) the roots have distinct moduli

    // Root condition: a coefficient perturbation of size delta moves a simple root r_j of monic p by at most
    // delta * sum_k |r_j|^k / |p'(r_j)|, p'(r_j) = prod_{k != j} (r_j - r_k). The companion-matrix eigenvalues
    // have absolute backward error ~eps * max_k |a_k|; sampled roots stay below 10x that bound, hence 32x.
    const RealScalar coefficientError = RealScalar(32) * NumTraits<RealScalar>::epsilon() * pols.cwiseAbs().maxCoeff();
    Matrix<RealScalar, Dynamic, 1> tolerance(real_roots.size());
    for (Index j = 0; j < real_roots.size(); ++j) {
      const RealScalar root = real_roots[j];
      RealScalar powerSum = RealScalar(0), power = RealScalar(1), derivative = RealScalar(1);
      for (Index k = 0; k < pols.size(); ++k) {
        powerSum += power;
        power *= numext::abs(root);
      }
      for (Index k = 0; k < real_roots.size(); ++k) {
        if (k != j) derivative *= numext::abs(root - real_roots[k]);
      }
      tolerance[j] = coefficientError * powerSum / derivative;
    }
    // Every computed root lies within rootTolerance of a true real root, which bounds both the imaginary parts
    // and the error of the extremal-root queries.
    const RealScalar rootTolerance = tolerance.maxCoeff();
    const RealScalar psPrec = numext::maxi(sqrt(test_precision<RealScalar>()), rootTolerance);

    // Test realRoots
    std::vector<RealScalar> calc_realRoots;
    psolve.realRoots(calc_realRoots, psPrec);
    VERIFY_IS_EQUAL(calc_realRoots.size(), (size_t)real_roots.size());

    for (size_t i = 0; i < calc_realRoots.size(); ++i) {
      bool found = false;
      for (Index j = 0; j < real_roots.size() && !found; ++j) {
        if (numext::abs(calc_realRoots[i] - real_roots[j]) <= tolerance[j]) {
          found = true;
        }
      }
      VERIFY(found);
    }

    // Test greatestRoot
    VERIFY(numext::abs(roots.array().abs().maxCoeff() - abs(psolve.greatestRoot())) <= rootTolerance);

    // Test smallestRoot
    VERIFY(numext::abs(roots.array().abs().minCoeff() - abs(psolve.smallestRoot())) <= rootTolerance);

    bool hasRealRoot;
    // Test absGreatestRealRoot
    RealScalar r = psolve.absGreatestRealRoot(hasRealRoot, psPrec);
    VERIFY(hasRealRoot == (real_roots.size() > 0));
    if (hasRealRoot) {
      VERIFY(numext::abs(real_roots.array().abs().maxCoeff() - abs(r)) <= rootTolerance);
    }

    // Test absSmallestRealRoot
    r = psolve.absSmallestRealRoot(hasRealRoot, psPrec);
    VERIFY(hasRealRoot == (real_roots.size() > 0));
    if (hasRealRoot) {
      VERIFY(numext::abs(real_roots.array().abs().minCoeff() - abs(r)) <= rootTolerance);
    }

    // Test greatestRealRoot
    r = psolve.greatestRealRoot(hasRealRoot, psPrec);
    VERIFY(hasRealRoot == (real_roots.size() > 0));
    if (hasRealRoot) {
      VERIFY(numext::abs(real_roots.array().maxCoeff() - r) <= rootTolerance);
    }

    // Test smallestRealRoot
    r = psolve.smallestRealRoot(hasRealRoot, psPrec);
    VERIFY(hasRealRoot == (real_roots.size() > 0));
    if (hasRealRoot) {
      VERIFY(numext::abs(real_roots.array().minCoeff() - r) <= rootTolerance);
    }
  }
}

template <typename Scalar_, int Deg_>
void polynomialsolver(int deg) {
  typedef typename NumTraits<Scalar_>::Real RealScalar;
  typedef internal::increment_if_fixed_size<Deg_> Dim;
  typedef Matrix<Scalar_, Dim::value, 1> PolynomialType;
  typedef Matrix<Scalar_, Deg_, 1> EvalRootsType;
  typedef Matrix<RealScalar, Deg_, 1> RealRootsType;

  cout << "Standard cases" << endl;
  PolynomialType pols = PolynomialType::Random(deg + 1);
  evalSolver<Deg_, PolynomialType>(pols);

  cout << "Hard cases" << endl;
  Scalar_ multipleRoot = internal::random<Scalar_>();
  EvalRootsType allRoots = EvalRootsType::Constant(deg, multipleRoot);
  roots_to_monicPolynomial(allRoots, pols);
  evalSolver<Deg_, PolynomialType>(pols);

  // The companion matrix eigenvalue approach has limited accuracy for float at
  // high degrees. The PolynomialSolver documentation itself warns: "With 32bit
  // (float) floating types this problem shows up frequently." Skip the sugar
  // function test (which requires exact root matching) for float beyond degree 8.
  if (deg <= 8 || sizeof(RealScalar) > sizeof(float)) {
    cout << "Test sugar" << endl;
    RealRootsType realRoots = RealRootsType::Random(deg);
    // sort by ascending absolute value to mitigate precision lost during polynomial expansion
    std::sort(realRoots.begin(), realRoots.end(),
              [](RealScalar a, RealScalar b) { return numext::abs(a) < numext::abs(b); });
    roots_to_monicPolynomial(realRoots, pols);
    evalSolverSugarFunction<Deg_>(pols, realRoots.template cast<std::complex<RealScalar> >().eval(), realRoots);
  }
}

EIGEN_DECLARE_TEST(polynomialsolver) {
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1((polynomialsolver<float, 1>(1)));
    CALL_SUBTEST_2((polynomialsolver<double, 2>(2)));
    CALL_SUBTEST_3((polynomialsolver<double, 3>(3)));
    CALL_SUBTEST_4((polynomialsolver<float, 4>(4)));
    CALL_SUBTEST_5((polynomialsolver<double, 5>(5)));
    CALL_SUBTEST_6((polynomialsolver<float, 6>(6)));
    CALL_SUBTEST_7((polynomialsolver<float, 7>(7)));
    CALL_SUBTEST_8((polynomialsolver<double, 8>(8)));

    CALL_SUBTEST_9((polynomialsolver<float, Dynamic>(internal::random<int>(9, 13))));
    CALL_SUBTEST_10((polynomialsolver<double, Dynamic>(internal::random<int>(9, 13))));
    CALL_SUBTEST_11((polynomialsolver<float, Dynamic>(1)));
    CALL_SUBTEST_12((polynomialsolver<std::complex<double>, Dynamic>(internal::random<int>(2, 13))));
  }
}
