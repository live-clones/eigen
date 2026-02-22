// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Shared header for split boostmultiprec tests.
//
// Tests that core Eigen decompositions work with boost::multiprecision types.
//
// This file is self-contained rather than #including other test .cpp files,
// because (a) the test functions live in module subdirectories and many define
// identically-named templates that conflict in a single TU, and (b) we only
// need to verify that each decomposition produces correct results for the
// multiprecision type, not re-run the full test suite for each decomposition.

#ifndef EIGEN_TEST_BOOSTMULTIPREC_HELPERS_H
#define EIGEN_TEST_BOOSTMULTIPREC_HELPERS_H

#include <sstream>

#ifdef EIGEN_TEST_MAX_SIZE
#undef EIGEN_TEST_MAX_SIZE
#endif
#define EIGEN_TEST_MAX_SIZE 50

#include "main.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#undef min
#undef max
#undef isnan
#undef isinf
#undef isfinite
#undef I

#include <boost/serialization/nvp.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/number.hpp>
#include <boost/math/special_functions.hpp>
#include <boost/math/complex.hpp>

typedef boost::multiprecision::number<boost::multiprecision::cpp_dec_float<100>, boost::multiprecision::et_on> Real;

namespace Eigen {
template <>
struct NumTraits<Real> : GenericNumTraits<Real> {
  static inline Real dummy_precision() { return 1e-50; }
};

template <typename T1, typename T2, typename T3, typename T4, typename T5>
struct NumTraits<boost::multiprecision::detail::expression<T1, T2, T3, T4, T5> > : NumTraits<Real> {};

template <>
Real test_precision<Real>() {
  return 1e-50;
}

// needed in C++93 mode where number does not support explicit cast.
namespace internal {
template <typename NewType>
struct cast_impl<Real, NewType> {
  static inline NewType run(const Real& x) { return x.template convert_to<NewType>(); }
};

template <>
struct cast_impl<Real, std::complex<Real> > {
  static inline std::complex<Real> run(const Real& x) { return std::complex<Real>(x); }
};
}  // namespace internal
}  // namespace Eigen

namespace boost {
namespace multiprecision {
// to make ADL works as expected:
using boost::math::copysign;
using boost::math::hypot;
using boost::math::isfinite;
using boost::math::isinf;
using boost::math::isnan;

// The following is needed for std::complex<Real>:
Real fabs(const Real& a) { return abs EIGEN_NOT_A_MACRO(a); }
Real fmax(const Real& a, const Real& b) {
  using std::max;
  return max(a, b);
}

// some specialization for the unit tests:
inline bool test_isMuchSmallerThan(const Real& a, const Real& b) {
  return internal::isMuchSmallerThan(a, b, test_precision<Real>());
}

inline bool test_isApprox(const Real& a, const Real& b) { return internal::isApprox(a, b, test_precision<Real>()); }

inline bool test_isApproxOrLessThan(const Real& a, const Real& b) {
  return internal::isApproxOrLessThan(a, b, test_precision<Real>());
}

Real get_test_precision(const Real&) { return test_precision<Real>(); }

Real test_relative_error(const Real& a, const Real& b) {
  using Eigen::numext::abs2;
  return sqrt(abs2<Real>(a - b) / Eigen::numext::mini<Real>(abs2(a), abs2(b)));
}
}  // namespace multiprecision
}  // namespace boost

// ---------------------------------------------------------------------------
// Decomposition test helpers (self-contained, no external test .cpp includes)
// ---------------------------------------------------------------------------

using namespace Eigen;

typedef Matrix<Real, Dynamic, Dynamic> Mat;
typedef Matrix<std::complex<Real>, Dynamic, Dynamic> MatC;

static Real largerEps() { return Real(10) * test_precision<Real>(); }

template <typename MatrixType>
void boostmp_cholesky(const MatrixType& m) {
  Index size = m.rows();
  MatrixType a = MatrixType::Random(size, size);
  MatrixType symm = a * a.adjoint();
  // Make sure diagonal is dominant for numerical stability.
  for (Index i = 0; i < size; i++) symm(i, i) += typename MatrixType::Scalar(size);

  LLT<MatrixType> llt(symm);
  VERIFY(llt.info() == Success);
  MatrixType x = MatrixType::Random(size, size);
  MatrixType b = symm * x;
  VERIFY_IS_APPROX(x, llt.solve(b));

  LDLT<MatrixType> ldlt(symm);
  VERIFY(ldlt.info() == Success);
  VERIFY_IS_APPROX(x, ldlt.solve(b));
}

template <typename MatrixType>
void boostmp_lu(int size) {
  MatrixType m1 = MatrixType::Random(size, size);
  FullPivLU<MatrixType> lu(m1);
  MatrixType x = MatrixType::Random(size, size);
  MatrixType b = m1 * x;
  VERIFY_IS_APPROX(x, lu.solve(b));

  PartialPivLU<MatrixType> plu(m1);
  VERIFY_IS_APPROX(x, plu.solve(b));
}

template <typename MatrixType>
void boostmp_qr(int rows, int cols) {
  MatrixType m = MatrixType::Random(rows, cols);

  // HouseholderQR
  {
    HouseholderQR<MatrixType> qr(m);
    MatrixType q = qr.householderQ();
    VERIFY_IS_UNITARY(q);
  }

  // ColPivHouseholderQR
  {
    ColPivHouseholderQR<MatrixType> qr(m);
    MatrixType q = qr.householderQ();
    VERIFY_IS_UNITARY(q);
    if (rows == cols) {
      MatrixType x = MatrixType::Random(rows, cols);
      MatrixType b = m * x;
      VERIFY(b.isApprox(m * qr.solve(b), largerEps()));
    }
  }

  // FullPivHouseholderQR
  {
    FullPivHouseholderQR<MatrixType> qr(m);
    MatrixType q = qr.matrixQ();
    VERIFY_IS_UNITARY(q);
    if (rows == cols) {
      MatrixType x = MatrixType::Random(rows, cols);
      MatrixType b = m * x;
      VERIFY(b.isApprox(m * qr.solve(b), largerEps()));
    }
  }

  // CompleteOrthogonalDecomposition
  {
    CompleteOrthogonalDecomposition<MatrixType> cod(m);
    MatrixType q = cod.householderQ();
    VERIFY_IS_UNITARY(q);
  }
}

template <typename MatrixType>
void boostmp_eigensolver_selfadjoint(const MatrixType& m) {
  Index size = m.rows();
  MatrixType a = MatrixType::Random(size, size);
  MatrixType symm = a + a.transpose();

  SelfAdjointEigenSolver<MatrixType> es(symm);
  VERIFY(es.info() == Success);
  VERIFY_IS_APPROX(symm * es.eigenvectors(), es.eigenvectors() * es.eigenvalues().asDiagonal());
  VERIFY_IS_UNITARY(es.eigenvectors());
}

template <typename MatrixType>
void boostmp_eigensolver_generic(const MatrixType& m) {
  Index size = m.rows();
  MatrixType a = MatrixType::Random(size, size);

  EigenSolver<MatrixType> es(a);
  VERIFY(es.info() == Success);
  typename EigenSolver<MatrixType>::EigenvalueType eivals = es.eigenvalues();
  // Just verify eigenvalues were computed without error.
  VERIFY(eivals.size() == size);
}

template <typename MatrixType>
void boostmp_generalized_eigensolver(const MatrixType& m) {
  Index size = m.rows();
  MatrixType a = MatrixType::Random(size, size);
  MatrixType b = MatrixType::Random(size, size);
  b = (b + b.transpose()).eval();
  for (Index i = 0; i < size; i++) b(i, i) += typename MatrixType::Scalar(size);

  GeneralizedSelfAdjointEigenSolver<MatrixType> es(a + a.transpose(), b);
  VERIFY(es.info() == Success);
  VERIFY(es.eigenvalues().size() == size);
}

template <typename MatrixType>
void boostmp_jacobisvd(const MatrixType& m) {
  Index rows = m.rows(), cols = m.cols();
  MatrixType a = MatrixType::Random(rows, cols);

  JacobiSVD<MatrixType, ComputeThinU | ComputeThinV> svd(a);
  VERIFY(svd.info() == Success);
  Index diagSize = (std::min)(rows, cols);
  VERIFY(svd.singularValues().size() == diagSize);
  // Verify reconstruction.
  MatrixType recon = svd.matrixU() * svd.singularValues().asDiagonal() * svd.matrixV().adjoint();
  VERIFY_IS_APPROX(a, recon);
}

template <typename MatrixType>
void boostmp_bdcsvd(const MatrixType& m) {
  Index rows = m.rows(), cols = m.cols();
  MatrixType a = MatrixType::Random(rows, cols);

  BDCSVD<MatrixType, ComputeThinU | ComputeThinV> svd(a);
  VERIFY(svd.info() == Success);
  Index diagSize = (std::min)(rows, cols);
  VERIFY(svd.singularValues().size() == diagSize);
  MatrixType recon = svd.matrixU() * svd.singularValues().asDiagonal() * svd.matrixV().adjoint();
  VERIFY_IS_APPROX(a, recon);
}

template <typename T, typename I_, int flag>
void boostmp_simplicial_cholesky() {
  typedef SparseMatrix<T, flag, I_> SparseMatrixType;
  Index size = internal::random<Index>(5, 20);

  // Build a random SPD sparse matrix.
  SparseMatrixType spd(size, size);
  typedef Triplet<T> Trip;
  std::vector<Trip> trips;
  for (Index i = 0; i < size; i++) {
    trips.push_back(Trip(i, i, T(size + internal::random<int>(1, 10))));
    if (i > 0) {
      T val = internal::random<T>();
      trips.push_back(Trip(i, i - 1, val));
      trips.push_back(Trip(i - 1, i, val));
    }
  }
  spd.setFromTriplets(trips.begin(), trips.end());

  SimplicialLLT<SparseMatrixType> llt(spd);
  VERIFY(llt.info() == Success);

  Matrix<T, Dynamic, 1> x = Matrix<T, Dynamic, 1>::Random(size);
  Matrix<T, Dynamic, 1> b = spd * x;
  Matrix<T, Dynamic, 1> sol = llt.solve(b);
  VERIFY_IS_APPROX(x, sol);
}

#endif  // EIGEN_TEST_BOOSTMULTIPREC_HELPERS_H
