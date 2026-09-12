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

    // Test realRoots
    const RealScalar psPrec = sqrt(test_precision<RealScalar>());

    std::vector<RealScalar> calc_realRoots;
    psolve.realRoots(calc_realRoots, psPrec);
    VERIFY_IS_EQUAL(calc_realRoots.size(), (size_t)real_roots.size());

    for (size_t i = 0; i < calc_realRoots.size(); ++i) {
      bool found = false;
      for (Index j = 0; j < real_roots.size() && !found; ++j) {
        if (internal::isApprox(calc_realRoots[i], real_roots[j], psPrec)) {
          found = true;
        }
      }
      VERIFY(found);
    }

    // Test greatestRoot
    VERIFY(internal::isApprox(roots.array().abs().maxCoeff(), abs(psolve.greatestRoot()), psPrec));

    // Test smallestRoot
    VERIFY(internal::isApprox(roots.array().abs().minCoeff(), abs(psolve.smallestRoot()), psPrec));

    bool hasRealRoot;
    // Test absGreatestRealRoot
    RealScalar r = psolve.absGreatestRealRoot(hasRealRoot, psPrec);
    VERIFY(hasRealRoot == (real_roots.size() > 0));
    if (hasRealRoot) {
      VERIFY(internal::isApprox(real_roots.array().abs().maxCoeff(), abs(r), psPrec));
    }

    // Test absSmallestRealRoot
    r = psolve.absSmallestRealRoot(hasRealRoot, psPrec);
    VERIFY(hasRealRoot == (real_roots.size() > 0));
    if (hasRealRoot) {
      VERIFY(internal::isApprox(real_roots.array().abs().minCoeff(), abs(r), psPrec));
    }

    // Test greatestRealRoot
    r = psolve.greatestRealRoot(hasRealRoot, psPrec);
    VERIFY(hasRealRoot == (real_roots.size() > 0));
    if (hasRealRoot) {
      VERIFY(internal::isApprox(real_roots.array().maxCoeff(), r, psPrec));
    }

    // Test smallestRealRoot
    r = psolve.smallestRealRoot(hasRealRoot, psPrec);
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

// Componentwise backward error |p(z)| / sum_k |a_k| |z|^k of a computed root, evaluated in WideReal so that the check
// does not share the solver's rounding.
template <typename WideReal, typename PolynomialType, typename RootType>
WideReal root_backward_error(const PolynomialType& pols, const RootType& root) {
  const std::complex<WideReal> z(WideReal(numext::real(root)), WideReal(numext::imag(root)));
  std::complex<WideReal> value(0);
  WideReal magnitude(0);
  for (Index k = pols.size() - 1; k >= 0; --k) {
    const std::complex<WideReal> coefficient(WideReal(numext::real(pols[k])), WideReal(numext::imag(pols[k])));
    value = value * z + coefficient;
    magnitude = magnitude * numext::abs(z) + numext::abs(coefficient);
  }
  return numext::abs(value) / magnitude;
}

// Every refined root must be an exact root of a nearby polynomial, and, where first-order perturbation theory applies,
// within its condition bound of the same polynomial's roots solved in WideScalar. The roots of a real polynomial must
// be real or exact conjugate pairs.
template <typename Scalar, typename WideScalar, int Deg>
void polynomialsolver_refinement_accuracy(int deg) {
  using RealScalar = typename NumTraits<Scalar>::Real;
  using WideReal = typename NumTraits<WideScalar>::Real;
  using PolynomialType = Matrix<Scalar, internal::increment_if_fixed_size<Deg>::value, 1>;
  using RootsType = Matrix<Scalar, Deg, 1>;
  const RootsType roots = RootsType::Random(deg);
  PolynomialType pols;
  roots_to_monicPolynomial(roots, pols);
  PolynomialSolver<Scalar, Deg> solver(pols);
  const WideReal eps = WideReal(NumTraits<RealScalar>::epsilon());

  // A Newton correction below one ulp bounds |p(z)| by eps |z p'(z)| <= deg eps sum_k |a_k| |z|^k; sampled roots of
  // float and double polynomials up to degree 50 stay below 6 eps.
  for (Index i = 0; i < deg; ++i)
    VERIFY(root_backward_error<WideReal>(pols, solver.roots()[i]) <= WideReal(4 * deg) * eps);

  if (!NumTraits<Scalar>::IsComplex) {
    for (Index i = 0; i < deg; ++i) {
      if (numext::imag(solver.roots()[i]) == RealScalar(0)) continue;
      bool paired = false;
      for (Index j = 0; j < deg && !paired; ++j)
        paired = j != i && solver.roots()[j] == numext::conj(solver.roots()[i]);
      VERIFY(paired);
    }
  }

  // A coefficient perturbation of size delta moves a simple root r_j by at most
  // delta * sum_k |r_j|^k / prod_{k != j} |r_j - r_k| to first order, valid while that is small against the gap to the
  // nearest other root. Each such root must have a computed root within 16 times the bound at delta = eps max_k |a_k|.
  const PolynomialSolver<WideScalar, Deg> reference(pols.template cast<WideScalar>().eval());
  const WideReal delta = eps * WideReal(pols.cwiseAbs().maxCoeff());
  for (Index j = 0; j < deg; ++j) {
    const std::complex<WideReal> r = reference.roots()[j];
    WideReal powerSum(0), power(1), derivative(1), gap = NumTraits<WideReal>::highest();
    for (Index k = 0; k <= deg; ++k) {
      powerSum += power;
      power *= numext::abs(r);
    }
    for (Index k = 0; k < deg; ++k) {
      if (k == j) continue;
      const WideReal d = numext::abs(r - std::complex<WideReal>(reference.roots()[k]));
      derivative *= d;
      gap = numext::mini(gap, d);
    }
    const WideReal bound = WideReal(16) * delta * powerSum / derivative;
    if (!(WideReal(4) * bound < gap)) continue;
    WideReal distance = NumTraits<WideReal>::highest();
    for (Index i = 0; i < deg; ++i) {
      const std::complex<WideReal> z(WideReal(numext::real(solver.roots()[i])),
                                     WideReal(numext::imag(solver.roots()[i])));
      distance = numext::mini(distance, numext::abs(z - r));
    }
    VERIFY(distance <= bound);
  }
}

// At |z| = 3 rounding dominates the residual of the pair 0.1 +- 3i, which is still an accurate root; comparing that
// residual with the one at the real part 0.1 once reported the pair as a double real root.
void polynomialsolver_complex_pair_kept() {
  Matrix<std::complex<float>, 12, 1> roots;
  roots << std::complex<float>(0.1f, 3.0f), std::complex<float>(0.1f, -3.0f), 0.2f, -0.2f, 0.4f, -0.4f, 0.6f, -0.6f,
      0.8f, -0.8f, 1.0f, -1.0f;
  Matrix<std::complex<float>, 13, 1> complexPolynomial;
  roots_to_monicPolynomial(roots, complexPolynomial);
  const Matrix<float, 13, 1> pols = complexPolynomial.real();
  PolynomialSolver<float, 12> solver(pols);
  Index farFromAxis = 0;
  for (Index i = 0; i < 12; ++i) {
    VERIFY(root_backward_error<double>(pols, solver.roots()[i]) <= 48 * double(NumTraits<float>::epsilon()));
    if (numext::abs(numext::imag(solver.roots()[i])) > 1.0f) ++farFromAxis;
  }
  VERIFY_IS_EQUAL(farFromAxis, Index(2));
}

// The companion eigenvalue of a root at 1e-12 carries an absolute error of order eps, a backward error of about
// 3e3 eps; refinement restores the root's full relative accuracy.
void polynomialsolver_tiny_root() {
  Matrix<double, 5, 1> roots;
  roots << 1e-12, 0.25, -0.5, 0.75, -1.0;
  Matrix<double, 6, 1> pols;
  roots_to_monicPolynomial(roots, pols);
  PolynomialSolver<double, 5> solver(pols);
  const double eps = NumTraits<double>::epsilon();
  bool found = false;
  for (Index i = 0; i < 5; ++i) {
    VERIFY(root_backward_error<long double>(pols, solver.roots()[i]) <= static_cast<long double>(20 * eps));
    found = found || numext::abs(solver.roots()[i] - std::complex<double>(1e-12)) <= 16 * eps * 1e-12;
  }
  VERIFY(found);
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

    CALL_SUBTEST_13((polynomialsolver_refinement_accuracy<float, double, 4>(4)));
    CALL_SUBTEST_13((polynomialsolver_refinement_accuracy<float, double, 6>(6)));
    CALL_SUBTEST_13((polynomialsolver_refinement_accuracy<float, double, 7>(7)));
    CALL_SUBTEST_14((polynomialsolver_refinement_accuracy<float, double, Dynamic>(internal::random<int>(8, 13))));
    CALL_SUBTEST_14((polynomialsolver_refinement_accuracy<std::complex<float>, std::complex<double>, Dynamic>(
        internal::random<int>(2, 13))));
    CALL_SUBTEST_14((polynomialsolver_refinement_accuracy<double, long double, Dynamic>(internal::random<int>(2, 20))));
  }
  CALL_SUBTEST_15(polynomialsolver_complex_pair_kept());
  CALL_SUBTEST_15(polynomialsolver_tiny_root());
}
