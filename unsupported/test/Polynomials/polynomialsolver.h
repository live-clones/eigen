// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Manuel Yguel <manuel.yguel@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_TEST_POLYNOMIALSOLVER_H
#define EIGEN_TEST_POLYNOMIALSOLVER_H

#include "main.h"
#include <unsupported/Eigen/Polynomials>
#include <iostream>
#include <algorithm>

using namespace std;

namespace Eigen {
namespace internal {
template <int Size>
struct increment_if_fixed_size {
  enum { ret = (Size == Dynamic) ? Dynamic : Size + 1 };
};
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
    std::vector<RealScalar> calc_realRoots;
    psolve.realRoots(calc_realRoots, test_precision<RealScalar>());
    VERIFY_IS_EQUAL(calc_realRoots.size(), (size_t)real_roots.size());

    const RealScalar psPrec = sqrt(test_precision<RealScalar>());

    for (size_t i = 0; i < calc_realRoots.size(); ++i) {
      bool found = false;
      for (size_t j = 0; j < calc_realRoots.size() && !found; ++j) {
        if (internal::isApprox(calc_realRoots[i], real_roots[j], psPrec)) {
          found = true;
        }
      }
      VERIFY(found);
    }

    VERIFY(internal::isApprox(roots.array().abs().maxCoeff(), abs(psolve.greatestRoot()), psPrec));
    VERIFY(internal::isApprox(roots.array().abs().minCoeff(), abs(psolve.smallestRoot()), psPrec));

    bool hasRealRoot;
    RealScalar r = psolve.absGreatestRealRoot(hasRealRoot, test_precision<RealScalar>());
    VERIFY(hasRealRoot == (real_roots.size() > 0));
    if (hasRealRoot) {
      VERIFY(internal::isApprox(real_roots.array().abs().maxCoeff(), abs(r), psPrec));
    }

    r = psolve.absSmallestRealRoot(hasRealRoot, test_precision<RealScalar>());
    VERIFY(hasRealRoot == (real_roots.size() > 0));
    if (hasRealRoot) {
      VERIFY(internal::isApprox(real_roots.array().abs().minCoeff(), abs(r), psPrec));
    }

    r = psolve.greatestRealRoot(hasRealRoot, test_precision<RealScalar>());
    VERIFY(hasRealRoot == (real_roots.size() > 0));
    if (hasRealRoot) {
      VERIFY(internal::isApprox(real_roots.array().maxCoeff(), r, psPrec));
    }

    r = psolve.smallestRealRoot(hasRealRoot, test_precision<RealScalar>());
    VERIFY(hasRealRoot == (real_roots.size() > 0));
    if (hasRealRoot) {
      VERIFY(internal::isApprox(real_roots.array().minCoeff(), r, psPrec));
    }
  }
}

template <typename Scalar_, int Deg_>
void polynomialsolver(int deg) {
  typedef typename NumTraits<Scalar_>::Real RealScalar;
  typedef internal::increment_if_fixed_size<Deg_> Dim;
  typedef Matrix<Scalar_, Dim::ret, 1> PolynomialType;
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

  cout << "Test sugar" << endl;
  RealRootsType realRoots = RealRootsType::Random(deg);
  std::sort(realRoots.begin(), realRoots.end(),
            [](RealScalar a, RealScalar b) { return numext::abs(a) < numext::abs(b); });
  roots_to_monicPolynomial(realRoots, pols);
  evalSolverSugarFunction<Deg_>(pols, realRoots.template cast<std::complex<RealScalar> >().eval(), realRoots);
}

#endif  // EIGEN_TEST_POLYNOMIALSOLVER_H
