// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"

#include <unsupported/Eigen/StructuredMatrices>

using namespace Eigen;

// Reference dense circulant built straight from the generating column, so that the
// fast (FFT) product is validated against an independent construction of the matrix.
template <typename Scalar>
Matrix<Scalar, Dynamic, Dynamic> reference_circulant(const Matrix<Scalar, Dynamic, 1>& c) {
  const Index n = c.size();
  Matrix<Scalar, Dynamic, Dynamic> dense(n, n);
  for (Index j = 0; j < n; ++j)
    for (Index i = 0; i < n; ++i) {
      Index k = i - j;
      if (k < 0) k += n;
      dense(i, j) = c[k];
    }
  return dense;
}

template <typename Scalar>
Matrix<Scalar, Dynamic, Dynamic> reference_toeplitz(const Matrix<Scalar, Dynamic, 1>& c,
                                                    const Matrix<Scalar, Dynamic, 1>& r) {
  const Index m = c.size(), n = r.size();
  Matrix<Scalar, Dynamic, Dynamic> dense(m, n);
  for (Index j = 0; j < n; ++j)
    for (Index i = 0; i < m; ++i) dense(i, j) = (i >= j) ? c[i - j] : r[j - i];
  return dense;
}

template <typename Scalar>
void test_circulant_product(Index n) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Vec c = Vec::Random(n);
  Circulant<Scalar> C(c);
  Mat dense = reference_circulant<Scalar>(c);

  // The operator agrees with the independently-built dense matrix, both through
  // coeff access and assigned to a dense matrix via its evaluator.
  Mat Cd = C;
  VERIFY_IS_APPROX(Cd, dense);
  for (Index t = 0; t < (std::min)(n, Index(5)); ++t) {
    Index i = internal::random<Index>(0, n - 1), j = internal::random<Index>(0, n - 1);
    VERIFY_IS_APPROX(C.coeff(i, j), dense(i, j));
  }

  // Fast matrix-vector and matrix-matrix products.
  Vec x = Vec::Random(n);
  VERIFY_IS_APPROX((C * x).eval(), (dense * x).eval());

  Mat X = Mat::Random(n, 3);
  VERIFY_IS_APPROX((C * X).eval(), (dense * X).eval());

  // Accumulation forms exercised by the iterative solvers.
  Vec y = Vec::Random(n);
  Vec y0 = y;
  y.noalias() += C * x;
  VERIFY_IS_APPROX(y, (y0 + dense * x).eval());
}

template <typename Scalar>
void test_circulant_solve(Index n) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  // Diagonally dominant => well conditioned, so the direct FFT solve is accurate.
  Vec c = Vec::Random(n);
  c[0] += Scalar(RealScalar(2 * n));
  Circulant<Scalar> C(c);
  Mat dense = reference_circulant<Scalar>(c);

  Vec b = Vec::Random(n);
  Vec x = C.solve(b);
  VERIFY_IS_APPROX((dense * x).eval(), b);

  // Multiple right-hand sides at once.
  Mat B = Mat::Random(n, 4);
  Mat Xs = C.solve(B);
  VERIFY_IS_APPROX((dense * Xs).eval(), B);
}

// The precomputed symbol holds the eigenvalues of the circulant matrix:
// C * f_k = symbol[k] * f_k, with f_k the k-th column of the inverse DFT matrix.
template <typename Scalar>
void test_circulant_symbol(Index n) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef std::complex<RealScalar> Complex;
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Complex, Dynamic, 1> ComplexVec;

  Vec c = Vec::Random(n);
  Circulant<Scalar> C(c);
  VERIFY_IS_EQUAL(C.symbol().size(), n);
  Matrix<Complex, Dynamic, Dynamic> denseC = reference_circulant<Scalar>(c).template cast<Complex>();

  const Index step = numext::maxi<Index>(n / 4, 1);
  for (Index k = 0; k < n; k += step) {
    ComplexVec f(n);
    for (Index j = 0; j < n; ++j) f[j] = std::polar(RealScalar(1), RealScalar(2 * EIGEN_PI * j * k) / RealScalar(n));
    VERIFY_IS_APPROX((denseC * f).eval(), (C.symbol()[k] * f).eval());
  }
}

template <typename Scalar>
void test_toeplitz_product(Index m, Index n) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Vec c = Vec::Random(m), r = Vec::Random(n);
  r[0] = c[0];  // diagonal entry; r[0] is ignored anyway
  Toeplitz<Scalar> T(c, r);
  Mat dense = reference_toeplitz<Scalar>(c, r);

  Mat Td = T;
  VERIFY_IS_APPROX(Td, dense);

  Vec x = Vec::Random(n);
  VERIFY_IS_APPROX((T * x).eval(), (dense * x).eval());

  Mat X = Mat::Random(n, 3);
  VERIFY_IS_APPROX((T * X).eval(), (dense * X).eval());
}

// Aliased products: the products carry the default product tag, so assignment
// materializes a temporary exactly like a dense product and x = C * x,
// x += C * x must come out as if the right-hand side had been copied first.
template <typename Scalar>
void test_circulant_aliased_product(Index n) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Vec c = Vec::Random(n);
  Circulant<Scalar> C(c);
  Mat dense = reference_circulant<Scalar>(c);

  Vec x = Vec::Random(n);
  Vec y = x;
  y = C * y;
  VERIFY_IS_APPROX(y, (dense * x).eval());

  y = x;
  y += C * y;
  VERIFY_IS_APPROX(y, (x + dense * x).eval());

  y = x;
  y -= C * y;
  VERIFY_IS_APPROX(y, (x - dense * x).eval());

  Mat X = Mat::Random(n, 3);
  Mat Y = X;
  Y = C * Y;
  VERIFY_IS_APPROX(Y, (dense * X).eval());
}

template <typename Scalar>
void test_toeplitz_aliased_product(Index n) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  // Square so that the destination and the right-hand side can alias.
  Vec c = Vec::Random(n), r = Vec::Random(n);
  r[0] = c[0];
  Toeplitz<Scalar> T(c, r);
  Mat dense = reference_toeplitz<Scalar>(c, r);

  Vec x = Vec::Random(n);
  Vec y = x;
  y = T * y;
  VERIFY_IS_APPROX(y, (dense * x).eval());

  y = x;
  y += T * y;
  VERIFY_IS_APPROX(y, (x + dense * x).eval());
}

// Aliasing beyond the same-object case: the default-product temporary must also
// resolve right-hand-side expressions that reference the destination, overlapping
// views of one buffer, and rectangular self-assignments where the destination is
// resized by the assignment.
template <typename Scalar>
void test_structured_aliased_expression(Index n) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Vec c = Vec::Random(n);
  Circulant<Scalar> C(c);
  Mat dense = reference_circulant<Scalar>(c);

  // Right-hand-side expression referencing the destination.
  Vec x = Vec::Random(n), x0 = x;
  x = C * (x + Vec::Ones(n));
  VERIFY_IS_APPROX(x, (dense * (x0 + Vec::Ones(n))).eval());

  // Overlapping (shifted) segments of one buffer.
  Vec buf = Vec::Random(n + 1);
  Vec expected = dense * buf.tail(n);
  buf.head(n) = C * buf.tail(n);
  VERIFY_IS_APPROX(buf.head(n).eval(), expected);

  // Rectangular self-assignment: x = T * x resizes the destination, so the
  // product must be captured before the destination storage is touched.
  const Index m = n + 3;
  Vec tc = Vec::Random(m), tr = Vec::Random(n);
  tr[0] = tc[0];
  Toeplitz<Scalar> T(tc, tr);
  Mat denseT = reference_toeplitz<Scalar>(tc, tr);
  Vec z = Vec::Random(n), z0 = z;
  z = T * z;
  VERIFY_IS_EQUAL(z.size(), m);
  VERIFY_IS_APPROX(z, (denseT * z0).eval());
}

// Mixed-scalar products: a real operator applied to a complex right-hand side
// (and a complex operator applied to a real one) promotes to the complex product
// scalar, so alpha and the accumulation must run in the promoted type rather than
// the operator scalar.
template <typename RealScalar>
void test_circulant_mixed_scalar(Index n) {
  typedef std::complex<RealScalar> Complex;
  typedef Matrix<RealScalar, Dynamic, 1> RVec;
  typedef Matrix<Complex, Dynamic, 1> CVec;
  typedef Matrix<Complex, Dynamic, Dynamic> CMat;

  RVec c = RVec::Random(n);
  Circulant<RealScalar> C(c);
  CMat dense = reference_circulant<RealScalar>(c).template cast<Complex>();

  CVec x = CVec::Random(n);
  CVec y = C * x;
  VERIFY_IS_APPROX(y, (dense * x).eval());

  CVec y0 = CVec::Random(n);
  y = y0;
  y.noalias() += C * x;
  VERIFY_IS_APPROX(y, (y0 + dense * x).eval());

  CVec cc = CVec::Random(n);
  Circulant<Complex> Cc(cc);
  CMat denseC = reference_circulant<Complex>(cc);
  RVec xr = RVec::Random(n);
  CVec z = Cc * xr;
  VERIFY_IS_APPROX(z, (denseC * xr).eval());
}

template <typename RealScalar>
void test_toeplitz_mixed_scalar(Index m, Index n) {
  typedef std::complex<RealScalar> Complex;
  typedef Matrix<RealScalar, Dynamic, 1> RVec;
  typedef Matrix<Complex, Dynamic, 1> CVec;
  typedef Matrix<Complex, Dynamic, Dynamic> CMat;

  RVec c = RVec::Random(m), r = RVec::Random(n);
  r[0] = c[0];
  Toeplitz<RealScalar> T(c, r);
  CMat dense = reference_toeplitz<RealScalar>(c, r).template cast<Complex>();

  CVec x = CVec::Random(n);
  CVec y = T * x;
  VERIFY_IS_APPROX(y, (dense * x).eval());
}

// The FFT product path accumulates up to p addends per transform, so a finite
// right-hand side near the overflow threshold (or a huge generator) used to
// overflow inside the transforms and return NaN for a representable result. The
// power-of-two scaled path must stay finite and accurate.
template <typename RealScalar>
void test_structured_fft_overflow(Index n) {
  typedef Matrix<RealScalar, Dynamic, 1> Vec;
  // FFT-transform roundoff bound for the identity round trip.
  const RealScalar kFftRoundTripTol = RealScalar(100) * NumTraits<RealScalar>::epsilon();
  const RealScalar huge = (std::numeric_limits<RealScalar>::max)() / RealScalar(16);

  // Identity circulant applied to a huge finite vector returns it unchanged.
  Vec c = Vec::Zero(n);
  c[0] = RealScalar(1);
  Circulant<RealScalar> C(c);
  Vec x = Vec::Constant(n, huge);
  Vec y = C * x;
  VERIFY(y.allFinite());
  VERIFY(((y - x).cwiseAbs() / huge).maxCoeff() <= kFftRoundTripTol);

  // A huge generator applied to a moderate vector: every output entry is huge
  // but representable.
  Circulant<RealScalar> Ch(Vec(huge * c));
  Vec ones = Vec::Ones(n);
  Vec z = Ch * ones;
  VERIFY(z.allFinite());
  VERIFY(((z.array() - huge).abs() / huge).maxCoeff() <= kFftRoundTripTol);

  // Identity Toeplitz, same huge right-hand side.
  Vec r = Vec::Zero(n);
  r[0] = c[0];
  Toeplitz<RealScalar> T(c, r);
  Vec w = T * x;
  VERIFY(w.allFinite());
  VERIFY(((w - x).cwiseAbs() / huge).maxCoeff() <= kFftRoundTripTol);
}

// The scaling exponents are derived from component-wise magnitudes: a finite
// complex value near the overflow threshold has a non-representable modulus,
// which would otherwise disable the scaling and turn an exactly representable
// product into NaN.
template <typename RealScalar>
void test_structured_fft_complex_boundary(Index n) {
  typedef std::complex<RealScalar> Complex;
  typedef Matrix<RealScalar, Dynamic, 1> RVec;
  typedef Matrix<Complex, Dynamic, 1> CVec;
  const RealScalar kFftRoundTripTol = RealScalar(100) * NumTraits<RealScalar>::epsilon();
  const RealScalar big = RealScalar(0.75) * (std::numeric_limits<RealScalar>::max)();

  // Identity circulant with a complex generator: the product returns the
  // right-hand side unchanged even though |x_k| overflows.
  CVec c = CVec::Zero(n);
  c[0] = Complex(1);
  Circulant<Complex> C(c);
  CVec x = CVec::Constant(n, Complex(big, big));
  CVec y = C * x;
  VERIFY(y.allFinite());
  VERIFY(((y - x).cwiseAbs() / big).maxCoeff() <= kFftRoundTripTol);

  // A real identity operator applied to the same complex right-hand side takes
  // the mixed-scalar product path.
  RVec cr = RVec::Zero(n);
  cr[0] = RealScalar(1);
  Circulant<RealScalar> Cr(cr);
  y = Cr * x;
  VERIFY(y.allFinite());
  VERIFY(((y - x).cwiseAbs() / big).maxCoeff() <= kFftRoundTripTol);
}

// Entrywise IEEE comparison for the non-finite tests: NaNs match NaNs,
// infinities match by value (sign included), finite entries match to roundoff.
// VERIFY_IS_APPROX would reject any output containing NaN.
template <typename D1, typename D2>
bool ieee_entrywise_match(const D1& a, const D2& b) {
  if (a.rows() != b.rows() || a.cols() != b.cols()) return false;
  for (Index j = 0; j < a.cols(); ++j)
    for (Index i = 0; i < a.rows(); ++i) {
      const typename D1::Scalar x = a(i, j), y = b(i, j);
      if (x == y) continue;                    // finite match or same-signed infinities
      if ((x != x) && (y != y)) continue;      // both NaN
      if (!test_isApprox(x, y)) return false;  // finite roundoff
    }
  return true;
}

// Scalar-loop product: the mathematically transparent IEEE reference for the
// non-finite tests. Eigen's own vectorized complex kernels can smear a single
// infinity into NaN (Inf - Inf across the split real/imaginary accumulators), so
// the dense product is not a faithful entrywise reference for non-finite data.
template <typename Scalar>
Matrix<Scalar, Dynamic, 1> reference_product_ieee(const Matrix<Scalar, Dynamic, Dynamic>& A,
                                                  const Matrix<Scalar, Dynamic, 1>& x) {
  Matrix<Scalar, Dynamic, 1> y(A.rows());
  for (Index i = 0; i < A.rows(); ++i) {
    Scalar acc(0);
    for (Index j = 0; j < A.cols(); ++j) acc += A(i, j) * x[j];
    y[i] = acc;
  }
  return y;
}

// A single Inf or NaN in the data must propagate like the reference product --
// through the dot products that touch it -- instead of being smeared into NaNs
// across the whole output by the transforms.
template <typename Scalar>
void test_structured_nonfinite_product(Index n) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;
  const RealScalar inf = std::numeric_limits<RealScalar>::infinity();
  const RealScalar nan = std::numeric_limits<RealScalar>::quiet_NaN();

  Vec c = Vec::Random(n);
  Circulant<Scalar> C(c);
  Mat dense = reference_circulant<Scalar>(c);

  // Inf in the right-hand side.
  Vec x = Vec::Random(n);
  x[n / 2] = Scalar(inf);
  VERIFY(ieee_entrywise_match((C * x).eval(), reference_product_ieee(dense, x)));

  // NaN in the right-hand side.
  Vec xn = Vec::Random(n);
  xn[n - 1] = Scalar(nan);
  VERIFY(ieee_entrywise_match((C * xn).eval(), reference_product_ieee(dense, xn)));

  // Inf in the generator: the operator itself is non-finite, whatever the
  // right-hand side.
  Vec c2 = Vec::Random(n);
  c2[1] = Scalar(-inf);
  Circulant<Scalar> C2(c2);
  Mat dense2 = reference_circulant<Scalar>(c2);
  Vec x2 = Vec::Random(n);
  VERIFY(ieee_entrywise_match((C2 * x2).eval(), reference_product_ieee(dense2, x2)));

  // Toeplitz with an Inf in the row generator.
  Vec tc = Vec::Random(n), tr = Vec::Random(n);
  tr[0] = tc[0];
  tr[n / 2] = Scalar(inf);
  Toeplitz<Scalar> T(tc, tr);
  Mat denseT = reference_toeplitz<Scalar>(tc, tr);
  VERIFY(ieee_entrywise_match((T * x2).eval(), reference_product_ieee(denseT, x2)));

  // Non-finite right-hand sides of solve() take the direct inverse application;
  // on the 1x1 operator this is a single scalar multiply by the inverse
  // coefficient, checked against the same multiply on the dense inverse.
  Vec b1(1);
  b1[0] = Scalar(inf);
  Circulant<Scalar> C1(Vec(Vec::Constant(1, Scalar(2))));
  Mat inv1 = Mat::Constant(1, 1, Scalar(1) / Scalar(2));
  VERIFY(ieee_entrywise_match(C1.solve(b1), reference_product_ieee(inv1, b1)));
}

// Fixed-size operators: generators are stored in fixed-size vectors, products and
// solves return fixed-size results, and small sizes go through the coeff-based
// product dispatch.
template <typename Scalar, int N>
void test_circulant_fixed() {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, N, 1> VecN;
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, N, N> MatN;

  VecN c = VecN::Random();
  c[0] += Scalar(RealScalar(2 * N));  // well conditioned for the solve below
  Circulant<Scalar, N> C(c);
  STATIC_CHECK((Circulant<Scalar, N>::RowsAtCompileTime == N));
  STATIC_CHECK((internal::remove_all_t<decltype(makeCirculant(c))>::RowsAtCompileTime == N));

  MatN dense = C;
  VERIFY_IS_APPROX(dense, MatN(reference_circulant<Scalar>(Vec(c))));

  VecN x = VecN::Random();
  VecN y = C * x;
  VERIFY_IS_APPROX(y, (dense * x).eval());

  VecN b = VecN::Random();
  VecN xs = C.solve(b);
  VERIFY_IS_APPROX((dense * xs).eval(), b);
}

template <typename Scalar, int M, int N>
void test_toeplitz_fixed() {
  typedef Matrix<Scalar, M, 1> ColVec;
  typedef Matrix<Scalar, N, 1> RowVec;
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, M, N> MatMN;

  ColVec c = ColVec::Random();
  RowVec r = RowVec::Random();
  r[0] = c[0];
  Toeplitz<Scalar, M, N> T(c, r);
  STATIC_CHECK((Toeplitz<Scalar, M, N>::RowsAtCompileTime == M));
  STATIC_CHECK((Toeplitz<Scalar, M, N>::ColsAtCompileTime == N));
  STATIC_CHECK((internal::remove_all_t<decltype(makeToeplitz(c, r))>::ColsAtCompileTime == N));

  MatMN dense = T;
  VERIFY_IS_APPROX(dense, MatMN(reference_toeplitz<Scalar>(Vec(c), Vec(r))));

  RowVec x = RowVec::Random();
  Matrix<Scalar, M, 1> y = T * x;
  VERIFY_IS_APPROX(y, (dense * x).eval());
}

template <typename Scalar>
void test_matrix_free_cg(Index n) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  // Symmetric, strongly diagonally dominant circulant => SPD.
  Vec c = Vec::Zero(n);
  c[0] = Scalar(4);
  c[1] = Scalar(-1);
  c[n - 1] = Scalar(-1);
  Circulant<Scalar> C(c);
  Mat dense = reference_circulant<Scalar>(c);

  Vec b = Vec::Random(n);
  ConjugateGradient<Circulant<Scalar>, Lower | Upper, IdentityPreconditioner> cg;
  cg.compute(C);
  Vec x = cg.solve(b);
  VERIFY(cg.info() == Success);
  VERIFY_IS_APPROX((dense * x).eval(), b);
}

template <typename Scalar>
void test_matrix_free_gmres(Index n) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  // Strongly diagonally dominant (nonsymmetric) Toeplitz.
  Vec c = Vec::Random(n) * Scalar(RealScalar(0.1));
  Vec r = Vec::Random(n) * Scalar(RealScalar(0.1));
  c[0] = Scalar(3);
  r[0] = Scalar(3);
  Toeplitz<Scalar> T(c, r);
  Mat dense = reference_toeplitz<Scalar>(c, r);

  Vec b = Vec::Random(n);
  GMRES<Toeplitz<Scalar>, IdentityPreconditioner> gmres;
  gmres.compute(T);
  Vec x = gmres.solve(b);
  VERIFY(gmres.info() == Success);
  VERIFY_IS_APPROX((dense * x).eval(), b);
}

// Diagonally dominant (well-conditioned) Toeplitz: the look-ahead solver must agree
// with a dense LU solve.
template <typename Scalar>
void test_levinson_wellcond(Index n) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Vec c = Vec::Random(n), r = Vec::Random(n);
  c[0] = r[0] = Scalar(RealScalar(2 * n));
  Toeplitz<Scalar> T(c, r);
  Mat dense = T;

  Vec b = Vec::Random(n);
  LookAheadLevinson<Scalar> lev(T);
  VERIFY(lev.info() == Success);
  Vec x = lev.solve(b);
  VERIFY_IS_APPROX(x, dense.fullPivLu().solve(b).eval());

  // Multiple right-hand sides.
  Mat B = Mat::Random(n, 3);
  VERIFY_IS_APPROX(lev.solve(B), dense.fullPivLu().solve(B).eval());

  // Transposed and adjoint systems reuse the same factorization (persymmetry).
  Vec xt = lev.transpose().solve(b);
  VERIFY_IS_APPROX(xt, dense.transpose().fullPivLu().solve(b).eval());
  Vec xa = lev.adjoint().solve(b);
  VERIFY_IS_APPROX(xa, dense.adjoint().fullPivLu().solve(b).eval());
  Mat Xt = lev.transpose().solve(B);
  VERIFY_IS_APPROX(Xt, dense.transpose().fullPivLu().solve(B).eval());
}

// Indefinite / ill-conditioned matrices that force look-ahead block steps. The
// generators and required block sizes are from Chan & Hansen's test set; the true
// solution is the all-ones vector.
void test_levinson_lookahead() {
  typedef Matrix<double, Dynamic, 1> Vec;
  // Loose bound for these deliberately ill-conditioned look-ahead cases (~1e-9);
  // the look-ahead Levinson recursion is weakly stable, so the forward error is a
  // large but bounded multiple of epsilon.
  const double tol = 5e6 * NumTraits<double>::epsilon();
  auto check = [tol](const Vec& c, const Vec& r, Index pmax) {
    Toeplitz<double> T(c, r);
    Vec xt = Vec::Ones(c.size());
    Vec b = T * xt;
    LookAheadLevinson<double> lev;
    lev.setMaxBlockSize(pmax).compute(T);
    VERIFY(lev.info() == Success);
    Vec x = lev.solve(b);
    VERIFY((x - xt).norm() <= tol * xt.norm());

    // The transposed solve must track the same look-ahead block steps.
    Matrix<double, Dynamic, Dynamic> dense = T;
    Vec bt = dense.transpose() * xt;
    Vec y = lev.transpose().solve(bt);
    VERIFY((y - xt).norm() <= tol * xt.norm());
  };

  Vec c1(6), r1(6);  // Sweet-1 (block size 2)
  c1 << 4, 6, 71.0 / 15 + 5e-8, 5, 3, 1;
  r1 << 4, 8, 1, 6, 2, 3;
  check(c1, r1, 3);

  Vec c2(6), r2(6);  // Sweet-2 (block size 2)
  c2 << 8, 4, -34 + 5e-13, 5, 3, 1;
  r2 << 8, 4, 1, 6, 2, 3;
  check(c2, r2, 3);

  Vec c3(13), r3(13);  // Sweet-3 (block size 6)
  c3 << 5, 1, -3, 12.755, -19.656, 28.361, -7, -1, 2, 1, -6, 1, -0.5;
  r3 << 5, -1, 6, 2, 5.697, 5.850, 3, -5, -2, -7, 1, 10, -15;
  check(c3, r3, 6);

  // shifted KMS: leading submatrices T_k with k = 1,4,7,... are singular, so the
  // full order n must be a multiple of 3 for T_n itself to be non-singular.
  for (Index n : {15, 30, 60}) {
    Vec c(n), r(n);
    c[0] = r[0] = 1e-14;
    for (Index i = 1; i < n; ++i) c[i] = r[i] = std::pow(0.5, double(i - 1));
    check(c, r, 3);
  }
}

// Fixed-size Toeplitz operators also feed the solver.
void test_levinson_fixed() {
  typedef Matrix<double, 12, 1> Vec12;
  Vec12 c = Vec12::Random(), r = Vec12::Random();
  c[0] = r[0] = 24.0;
  Toeplitz<double, 12, 12> T(c, r);
  Matrix<double, 12, 12> dense = T;
  Vec12 b = Vec12::Random();
  LookAheadLevinson<double> lev(T);
  VERIFY(lev.info() == Success);
  VERIFY_IS_APPROX(lev.solve(b), dense.fullPivLu().solve(b).eval());
}

// A numerically singular Toeplitz must be reported through info().
void test_levinson_singular() {
  typedef Matrix<double, Dynamic, 1> Vec;
  for (Index n : {4, 9}) {
    Vec c = Vec::Ones(n), r = Vec::Ones(n);  // all-ones Toeplitz is rank 1 for n >= 2
    LookAheadLevinson<double> lev(Toeplitz<double>(c, r));
    VERIFY(lev.info() == NumericalIssue);
  }
}

EIGEN_DECLARE_TEST(structured_matrices) {
  for (int i = 0; i < g_repeat; ++i) {
    // Circulant: direct path (small), FFT path (composite and prime sizes), edge cases.
    CALL_SUBTEST_1((test_circulant_product<double>(1)));
    CALL_SUBTEST_1((test_circulant_product<double>(2)));
    CALL_SUBTEST_1((test_circulant_product<double>(8)));
    CALL_SUBTEST_1((test_circulant_product<double>(64)));
    CALL_SUBTEST_1((test_circulant_product<double>(97)));  // prime, FFT path
    CALL_SUBTEST_1((test_circulant_product<float>(48)));
    CALL_SUBTEST_1((test_circulant_product<std::complex<double>>(7)));  // direct path, complex
    CALL_SUBTEST_1((test_circulant_product<std::complex<double>>(50)));
    CALL_SUBTEST_1((test_circulant_product<std::complex<float>>(40)));
    CALL_SUBTEST_1((test_circulant_solve<double>(1)));  // degenerate 1x1 solve
    CALL_SUBTEST_1((test_circulant_solve<double>(8)));
    CALL_SUBTEST_1((test_circulant_solve<double>(50)));
    CALL_SUBTEST_1((test_circulant_solve<std::complex<double>>(40)));
    CALL_SUBTEST_1((test_circulant_solve<float>(32)));
    CALL_SUBTEST_1((test_circulant_symbol<double>(16)));
    CALL_SUBTEST_1((test_circulant_symbol<std::complex<double>>(12)));

    // Toeplitz: square, tall, wide, small (direct), single row/column, real and complex.
    CALL_SUBTEST_2((test_toeplitz_product<double>(1, 1)));
    CALL_SUBTEST_2((test_toeplitz_product<double>(2, 2)));
    CALL_SUBTEST_2((test_toeplitz_product<double>(10, 10)));
    CALL_SUBTEST_2((test_toeplitz_product<double>(64, 64)));
    CALL_SUBTEST_2((test_toeplitz_product<double>(96, 48)));
    CALL_SUBTEST_2((test_toeplitz_product<double>(48, 96)));
    CALL_SUBTEST_2((test_toeplitz_product<double>(1, 40)));  // single row, FFT path
    CALL_SUBTEST_2((test_toeplitz_product<double>(40, 1)));  // single column, FFT path
    CALL_SUBTEST_2((test_toeplitz_product<float>(50, 50)));
    CALL_SUBTEST_2((test_toeplitz_product<std::complex<double>>(5, 7)));  // direct path, complex
    CALL_SUBTEST_2((test_toeplitz_product<std::complex<double>>(48, 64)));
    CALL_SUBTEST_2((test_toeplitz_product<std::complex<float>>(40, 40)));

    // Matrix-free iterative solves through the existing solvers.
    CALL_SUBTEST_3((test_matrix_free_cg<double>(80)));
    CALL_SUBTEST_3((test_matrix_free_gmres<double>(80)));

    // Aliased products across the dispatch tiers, and mixed-scalar promotion.
    CALL_SUBTEST_3((test_circulant_aliased_product<double>(8)));
    CALL_SUBTEST_3((test_circulant_aliased_product<double>(24)));
    CALL_SUBTEST_3((test_circulant_aliased_product<double>(48)));
    CALL_SUBTEST_3((test_circulant_aliased_product<std::complex<double>>(40)));
    CALL_SUBTEST_3((test_toeplitz_aliased_product<double>(8)));
    CALL_SUBTEST_3((test_toeplitz_aliased_product<double>(48)));
    CALL_SUBTEST_3((test_structured_aliased_expression<double>(8)));
    CALL_SUBTEST_3((test_structured_aliased_expression<double>(24)));
    CALL_SUBTEST_3((test_structured_aliased_expression<double>(48)));
    CALL_SUBTEST_3((test_structured_aliased_expression<std::complex<double>>(40)));
    CALL_SUBTEST_3((test_circulant_mixed_scalar<double>(12)));
    CALL_SUBTEST_3((test_circulant_mixed_scalar<double>(48)));
    CALL_SUBTEST_3((test_toeplitz_mixed_scalar<double>(20, 12)));
    CALL_SUBTEST_3((test_toeplitz_mixed_scalar<double>(64, 40)));

    // Fixed-size operators: small (coeff-based dispatch) and above the FFT threshold.
    CALL_SUBTEST_4((test_circulant_fixed<double, 4>()));
    CALL_SUBTEST_4((test_circulant_fixed<std::complex<float>, 4>()));
    CALL_SUBTEST_4((test_circulant_fixed<double, 48>()));
    CALL_SUBTEST_4((test_toeplitz_fixed<double, 4, 6>()));
    CALL_SUBTEST_4((test_toeplitz_fixed<double, 40, 24>()));
    CALL_SUBTEST_4((test_toeplitz_fixed<std::complex<float>, 6, 4>()));

    // Scaled FFT products at the overflow boundary, in double and float, at a
    // 5-smooth and a prime transform size.
    CALL_SUBTEST_4((test_structured_fft_overflow<double>(40)));
    CALL_SUBTEST_4((test_structured_fft_overflow<double>(97)));  // prime, no 5-smooth padding
    CALL_SUBTEST_4((test_structured_fft_overflow<float>(40)));
    CALL_SUBTEST_4((test_structured_fft_complex_boundary<double>(40)));
    CALL_SUBTEST_4((test_structured_fft_complex_boundary<float>(40)));

    // Entrywise Inf/NaN propagation: FFT-sized operators must fall back to the
    // direct kernel; small ones are IEEE-exact already.
    CALL_SUBTEST_4((test_structured_nonfinite_product<double>(40)));
    CALL_SUBTEST_4((test_structured_nonfinite_product<double>(12)));
    CALL_SUBTEST_4((test_structured_nonfinite_product<std::complex<double>>(40)));

    // Look-ahead Levinson direct Toeplitz solver.
    CALL_SUBTEST_5((test_levinson_wellcond<double>(1)));
    CALL_SUBTEST_5((test_levinson_wellcond<double>(2)));
    CALL_SUBTEST_5((test_levinson_wellcond<double>(20)));
    CALL_SUBTEST_5((test_levinson_wellcond<double>(60)));
    CALL_SUBTEST_5((test_levinson_wellcond<float>(40)));
    CALL_SUBTEST_5((test_levinson_wellcond<std::complex<double>>(30)));
    CALL_SUBTEST_5((test_levinson_wellcond<std::complex<float>>(24)));
    CALL_SUBTEST_5(test_levinson_lookahead());
    CALL_SUBTEST_5(test_levinson_fixed());
    CALL_SUBTEST_5(test_levinson_singular());
  }
}
