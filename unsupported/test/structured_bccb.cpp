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

// Reference dense BCCB built entry-wise from the generating array, independently
// of the operator under test: entry (i,j) with i = b1*n2+i2, j = c1*n2+j2 is
// G((i2-j2) mod n2, (b1-c1) mod n1).
template <typename Scalar>
Matrix<Scalar, Dynamic, Dynamic> reference_bccb(const Matrix<Scalar, Dynamic, Dynamic>& G) {
  const Index n2 = G.rows(), n1 = G.cols(), N = n1 * n2;
  Matrix<Scalar, Dynamic, Dynamic> dense(N, N);
  for (Index j = 0; j < N; ++j)
    for (Index i = 0; i < N; ++i) {
      Index k2 = i % n2 - j % n2;
      if (k2 < 0) k2 += n2;
      Index k1 = i / n2 - j / n2;
      if (k1 < 0) k1 += n1;
      dense(i, j) = G(k2, k1);
    }
  return dense;
}

template <typename Scalar>
void test_bccb_product(Index n2, Index n1) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Mat G = Mat::Random(n2, n1);
  Bccb<Scalar> C(G);
  Mat dense = reference_bccb<Scalar>(G);
  const Index N = n1 * n2;
  VERIFY_IS_EQUAL(C.rows(), N);

  Mat Cd = C;
  VERIFY_IS_APPROX(Cd, dense);
  for (Index t = 0; t < 5; ++t) {
    Index i = internal::random<Index>(0, N - 1), j = internal::random<Index>(0, N - 1);
    VERIFY_IS_APPROX(C.coeff(i, j), dense(i, j));
  }

  Vec x = Vec::Random(N);
  VERIFY_IS_APPROX((C * x).eval(), (dense * x).eval());

  Mat X = Mat::Random(N, 3);
  VERIFY_IS_APPROX((C * X).eval(), (dense * X).eval());

  // Accumulation form exercised by the iterative solvers.
  Vec y = Vec::Random(N);
  Vec y0 = y;
  y.noalias() += C * x;
  VERIFY_IS_APPROX(y, (y0 + dense * x).eval());
}

template <typename Scalar>
void test_bccb_transpose(Index n2, Index n1) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Mat G = Mat::Random(n2, n1);
  Bccb<Scalar> C(G);
  Mat dense = reference_bccb<Scalar>(G);
  const Index N = n1 * n2;

  Mat Td = C.transpose();
  VERIFY_IS_APPROX(Td, Mat(dense.transpose()));
  Mat Ad = C.adjoint();
  VERIFY_IS_APPROX(Ad, Mat(dense.adjoint()));
  Mat Kd = C.conjugate();
  VERIFY_IS_APPROX(Kd, Mat(dense.conjugate()));

  Vec x = Vec::Random(N);
  VERIFY_IS_APPROX((C.transpose() * x).eval(), (dense.transpose() * x).eval());
  VERIFY_IS_APPROX((C.adjoint() * x).eval(), (dense.adjoint() * x).eval());

  // Exact round trips: generators and symbols are pure permutations/conjugations.
  Bccb<Scalar> Ctt = C.transpose().transpose();
  VERIFY_IS_EQUAL(Ctt.generator(), G);
  VERIFY_IS_EQUAL(Ctt.symbol(), C.symbol());
  Bccb<Scalar> Caa = C.adjoint().adjoint();
  VERIFY_IS_EQUAL(Caa.generator(), G);
  VERIFY_IS_EQUAL(Caa.symbol(), C.symbol());
}

// transpose()/conjugate()/adjoint() return owning temporaries, so the product
// expression must nest the structured operand by value: a delayed-evaluated
// expression has to outlive the temporary operator it was built from. The static
// check pins the value nesting; the behavioral check would read freed memory if
// the product held a reference instead.
template <typename Scalar>
void test_bccb_delayed_product(Index n2, Index n1) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  STATIC_CHECK(!std::is_reference<typename internal::ref_selector<Bccb<Scalar>>::type>::value);

  Mat G = Mat::Random(n2, n1);
  Bccb<Scalar> C(G);
  Mat dense = reference_bccb<Scalar>(G);
  const Index N = n1 * n2;
  Vec x = Vec::Random(N);

  auto expr = C.adjoint() * x;        // the adjoint temporary dies with the full expression
  Vec scribble = Vec::Random(2 * N);  // reuses the temporary's freed heap storage
  Vec y = expr;
  VERIFY_IS_APPROX(y, (dense.adjoint() * x).eval());
  VERIFY_IS_EQUAL(scribble.size(), 2 * N);  // keep the scribble alive across the evaluation
}

// The products are tagged AliasFreeProduct, so no temporary shields an aliased
// right-hand side: the shared product implementation must copy it instead.
// Without the copy, x = C * x reads a zeroed right-hand side and x += C * x
// interleaves destination writes with right-hand-side reads.
template <typename Scalar>
void test_bccb_aliased_product(Index n2, Index n1) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Mat G = Mat::Random(n2, n1);
  Bccb<Scalar> C(G);
  Mat dense = reference_bccb<Scalar>(G);
  const Index N = n1 * n2;

  Vec x = Vec::Random(N);
  Vec y = x;
  y = C * y;
  VERIFY_IS_APPROX(y, (dense * x).eval());

  y = x;
  y += C * y;
  VERIFY_IS_APPROX(y, (x + dense * x).eval());

  y = x;
  y -= C * y;
  VERIFY_IS_APPROX(y, (x - dense * x).eval());

  Mat X = Mat::Random(N, 3);
  Mat Y = X;
  Y = C * Y;
  VERIFY_IS_APPROX(Y, (dense * X).eval());
}

// Mixed-scalar products: a real operator applied to a complex right-hand side
// (and a complex operator applied to a real one) promotes to the complex product
// scalar, so alpha and the accumulation must run in the promoted type rather than
// the operator scalar.
template <typename RealScalar>
void test_bccb_mixed_scalar(Index n2, Index n1) {
  typedef std::complex<RealScalar> Complex;
  typedef Matrix<RealScalar, Dynamic, 1> RVec;
  typedef Matrix<RealScalar, Dynamic, Dynamic> RMat;
  typedef Matrix<Complex, Dynamic, 1> CVec;
  typedef Matrix<Complex, Dynamic, Dynamic> CMat;

  RMat G = RMat::Random(n2, n1);
  Bccb<RealScalar> C(G);
  CMat dense = reference_bccb<RealScalar>(G).template cast<Complex>();
  const Index N = n1 * n2;

  CVec x = CVec::Random(N);
  CVec y = C * x;
  VERIFY_IS_APPROX(y, (dense * x).eval());

  CVec y0 = CVec::Random(N);
  y = y0;
  y.noalias() += C * x;
  VERIFY_IS_APPROX(y, (y0 + dense * x).eval());

  CMat Gc = CMat::Random(n2, n1);
  Bccb<Complex> Cc(Gc);
  CMat denseC = reference_bccb<Complex>(Gc);
  RVec xr = RVec::Random(N);
  CVec z = Cc * xr;
  VERIFY_IS_APPROX(z, (denseC * xr).eval());
}

// Dynamic dimension mismatches must trip the runtime assertion when the product
// or solve expression is built. (Incompatible *fixed* sizes are rejected at
// compile time by a static assertion in operator* / solve, which a runtime test
// cannot exercise.)
void test_bccb_dimension_asserts() {
  typedef Matrix<double, Dynamic, 1> Vec;
  typedef Matrix<double, Dynamic, Dynamic> Mat;

  Mat G = Mat::Random(3, 4);
  Bccb<double> C(G);
  Vec bad = Vec::Random(11);
  VERIFY_RAISES_ASSERT(C * bad);
  VERIFY_RAISES_ASSERT(C.solve(bad));
}

template <typename Scalar>
void test_bccb_solve(Index n2, Index n1) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  // Diagonal dominance (through the (0,0) generator entry) keeps the symbol away
  // from zero, so the direct 2-D FFT solve is exact up to roundoff.
  Mat G = Mat::Random(n2, n1);
  G(0, 0) += Scalar(RealScalar(2 * n1 * n2));
  Bccb<Scalar> C(G);
  Mat dense = reference_bccb<Scalar>(G);
  const Index N = n1 * n2;

  Vec b = Vec::Random(N);
  Vec x = C.solve(b);
  VERIFY_IS_APPROX((dense * x).eval(), b);

  Mat B = Mat::Random(N, 3);
  Mat Xs = C.solve(B);
  VERIFY_IS_APPROX((dense * Xs).eval(), B);
}

// Rank-deficient BCCB synthesized by zeroing 2-D symbol entries: the rank counts
// the surviving entries and solve() matches the SVD pseudo-inverse.
template <typename Scalar>
void test_bccb_rank_deficient(Index n2, Index n1, Index defect) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef std::complex<RealScalar> Complex;
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;
  typedef Matrix<Complex, Dynamic, Dynamic> CMat;

  const Index N = n1 * n2;
  CMat S = CMat::Random(n2, n1);
  S.array() += Complex(2);  // keep the surviving moduli away from the threshold
  for (Index k = 0; k < defect; ++k) S((2 * k + 1) % n2, (3 * k) % n1) = Complex(0);

  // Build the generator by an inverse 2-D DFT of the symbol (row-column, using
  // the dense DFT matrices for independence from the implementation under test).
  CMat F2(n2, n2), F1(n1, n1);
  for (Index a = 0; a < n2; ++a)
    for (Index c = 0; c < n2; ++c)
      F2(a, c) = std::polar(RealScalar(1) / RealScalar(n2),
                            RealScalar(2 * EIGEN_PI) * RealScalar((a * c) % n2) / RealScalar(n2));
  for (Index a = 0; a < n1; ++a)
    for (Index c = 0; c < n1; ++c)
      F1(a, c) = std::polar(RealScalar(1) / RealScalar(n1),
                            RealScalar(2 * EIGEN_PI) * RealScalar((a * c) % n1) / RealScalar(n1));
  Mat G = (F2 * S * F1.transpose()).eval();  // Scalar is complex here
  Bccb<Scalar> C(G);
  VERIFY_IS_EQUAL(C.rank(), N - defect);

  Mat dense = reference_bccb<Scalar>(G);
  JacobiSVD<Mat> svd(dense, ComputeThinU | ComputeThinV);
  Vec b = Vec::Random(N);
  VERIFY_IS_APPROX(C.solve(b), svd.solve(b).eval());
}

template <typename Scalar>
void test_bccb_zero(Index n2, Index n1) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Bccb<Scalar> C(Mat(Mat::Zero(n2, n1)));
  VERIFY_IS_EQUAL(C.rank(), 0);
  VERIFY(C.singularValues().isZero());
  Vec b = Vec::Random(n1 * n2);
  VERIFY(C.solve(b).isZero());
}

void test_bccb_nan_propagation(Index n2, Index n1) {
  typedef Matrix<double, Dynamic, 1> Vec;
  typedef Matrix<double, Dynamic, Dynamic> Mat;

  Mat G = Mat::Random(n2, n1);
  G(n2 / 2, n1 / 2) = std::numeric_limits<double>::quiet_NaN();
  Bccb<double> C(G);
  VERIFY_IS_EQUAL(C.rank(), n1 * n2);
  Vec b = Vec::Random(n1 * n2);
  Vec x = C.solve(b);
  VERIFY(!(x.array() == x.array()).all());
}

// Inputs at the very top (and bottom) of the exponent range: the transforms sum
// up to N terms, so without scaling the FFT intermediates overflow even though
// every entry of the true result is representable. The apply path rescales the
// symbol and each right-hand-side column by exact powers of two and folds the
// exponent back at the end.
void test_bccb_finite_overflow() {
  typedef Matrix<double, Dynamic, 1> Vec;
  typedef Matrix<double, Dynamic, Dynamic> Mat;

  const double huge = (std::numeric_limits<double>::max)() / 16;
  // The 2-D FFT round trip costs a few ulps per entry; a norm-based comparison
  // would itself overflow at these magnitudes, hence the entrywise bound.
  const double kFftRoundTripTol = 100 * NumTraits<double>::epsilon();

  // Reviewer repro: a 36x36 identity BCCB applied to a vector of DBL_MAX/16
  // must return the vector, not 36 NaNs.
  Mat G = Mat::Zero(6, 6);
  G(0, 0) = 1.0;
  Bccb<double> C(G);
  Vec x = Vec::Constant(36, huge);
  Vec y = C * x;
  VERIFY(y.allFinite());
  VERIFY(((y - x).array().abs() <= kFftRoundTripTol * x.array().abs()).all());

  // A huge generator makes a huge symbol: the scaling must come from the symbol
  // side as well.
  Mat Gh = Mat::Zero(6, 6);
  Gh(0, 0) = huge;  // C == huge * Identity
  Bccb<double> Ch(Gh);
  Vec ones = Vec::Ones(36);
  Vec z = Ch * ones;
  VERIFY(z.allFinite());
  VERIFY(((z.array() - huge).abs() <= kFftRoundTripTol * huge).all());

  // The inverse symbol of a tiny operator is huge; solve() shares the scaled
  // apply path.
  Mat Gt = Mat::Zero(6, 6);
  Gt(0, 0) = 1.0 / huge;
  Bccb<double> Ct(Gt);
  Vec w = Ct.solve(ones);
  VERIFY(w.allFinite());
  VERIFY(((w.array() - huge).abs() <= kFftRoundTripTol * huge).all());

  // An exactly zero column maps to an exactly zero column (short-circuited
  // before any scaling), while nonzero columns are still transformed.
  Mat B(36, 2);
  B.col(0).setConstant(huge);
  B.col(1).setZero();
  Mat Y = C * B;
  VERIFY(Y.col(1).isZero());
  VERIFY(((Y.col(0) - x).array().abs() <= kFftRoundTripTol * x.array().abs()).all());

  // Genuine NaN and infinity inputs keep propagating; they are not laundered by
  // the scaling.
  Vec xn = x;
  xn[7] = std::numeric_limits<double>::quiet_NaN();
  Vec yn = C * xn;
  VERIFY(!(yn.array() == yn.array()).all());
  Vec xi = Vec::Ones(36);
  xi[3] = std::numeric_limits<double>::infinity();
  Vec yi = C * xi;
  VERIFY(!yi.allFinite());
}

// Spectra with a wide dynamic range: the determinant is representable, but a
// plain product of the eigenvalues in FFT order overflows to infinity (or
// underflows to an exact zero) partway through. Pins the balanced accumulation
// in determinant().
void test_bccb_determinant_scaled() {
  typedef std::complex<double> Complex;
  typedef Matrix<Complex, Dynamic, Dynamic> CMat;

  const Index n2 = 25, n1 = 40;  // N = 1000
  const Index nLead = 653;

  // Symbol magnitudes `lead` on the first nLead column-major-flattened indices
  // (the accumulation order of determinant()) and `rest` elsewhere, so the
  // naive running product leaves the representable range partway through while
  // the true determinant lead^nLead * rest^(N - nLead) is representable. The
  // generator is recovered through the dense inverse 2-D DFT matrices,
  // independently of the implementation under test.
  auto makeOperator = [n2, n1](double lead, double rest) {
    CMat S(n2, n1);
    for (Index k1 = 0; k1 < n1; ++k1)
      for (Index k2 = 0; k2 < n2; ++k2) S(k2, k1) = Complex(k1 * n2 + k2 < nLead ? lead : rest);
    CMat F2(n2, n2), F1(n1, n1);
    for (Index a = 0; a < n2; ++a)
      for (Index c = 0; c < n2; ++c)
        F2(a, c) = std::polar(1.0 / double(n2), double(2 * EIGEN_PI) * double((a * c) % n2) / double(n2));
    for (Index a = 0; a < n1; ++a)
      for (Index c = 0; c < n1; ++c)
        F1(a, c) = std::polar(1.0 / double(n1), double(2 * EIGEN_PI) * double((a * c) % n1) / double(n1));
    return Bccb<Complex>((F2 * S * F1.transpose()).eval());
  };

  // The spectrum is only reproduced up to the FFT round trip's forward error,
  // and the determinant multiplies ~1e3 such factors.
  const double kSpectrumRoundTripTol = 1e8 * NumTraits<double>::epsilon();
  {
    // det = 10^653 * 10^-347 = 1e306; the naive partial product reaches 1e327.
    Bccb<Complex> C = makeOperator(10.0, 0.1);
    const Complex det = C.determinant();
    VERIFY((numext::isfinite)(numext::abs(det)));
    VERIFY(numext::abs(det / 1e306 - Complex(1)) <= kSpectrumRoundTripTol);
  }
  {
    // det = 10^-653 * 10^347 = 1e-306; the naive partial product reaches
    // 1e-327, well below the smallest subnormal, and flushes to an exact zero.
    Bccb<Complex> C = makeOperator(0.1, 10.0);
    const Complex det = C.determinant();
    VERIFY(det != Complex(0));
    VERIFY(numext::abs(det / 1e-306 - Complex(1)) <= kSpectrumRoundTripTol);
  }
  {
    // A genuinely overflowing determinant must still saturate to infinity.
    typedef Matrix<double, Dynamic, Dynamic> Mat;
    Mat G = Mat::Zero(6, 6);
    G(0, 0) = 1e308;  // the symbol is 1e308 everywhere: det = 10^11088
    Bccb<double> C(G);
    VERIFY((numext::isinf)(C.determinant()));
  }
}

// The rank decision at the clamped threshold (the smallest normal number): a
// smallest-normal symbol entry is still inverted -- the comparison is strict,
// matching SVDBase::rank(), which likewise reports rank one -- a subnormal entry
// is treated as an exact zero, and non-finite entries count as non-zero.
void test_bccb_rank_boundaries() {
  typedef Matrix<double, Dynamic, 1> Vec;
  typedef Matrix<double, Dynamic, Dynamic> Mat;

  const double mn = (std::numeric_limits<double>::min)();
  const Vec b = Vec::Ones(1);
  {
    Bccb<double> C(Mat(Mat::Constant(1, 1, mn)));
    VERIFY_IS_EQUAL(C.rank(), 1);
    Vec x = C.solve(b);
    VERIFY((numext::isfinite)(x[0]));
    VERIFY_IS_APPROX(x[0], 1.0 / mn);
  }
  {
    Bccb<double> C(Mat(Mat::Constant(1, 1, mn / 2)));  // subnormal
    VERIFY_IS_EQUAL(C.rank(), 0);
    VERIFY(C.solve(b).isZero());
  }
  {
    Bccb<double> C(Mat(Mat::Constant(1, 1, std::numeric_limits<double>::infinity())));
    VERIFY_IS_EQUAL(C.rank(), 1);
    VERIFY((numext::isinf)(C.determinant()));
  }
}

template <typename Scalar>
void test_bccb_eigen(Index n2, Index n1) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef std::complex<RealScalar> Complex;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;
  typedef Matrix<Complex, Dynamic, Dynamic> CMat;

  Mat G = Mat::Random(n2, n1);
  Bccb<Scalar> C(G);
  CMat denseC = reference_bccb<Scalar>(G).template cast<Complex>();
  const Index N = n1 * n2;

  Matrix<Complex, Dynamic, 1> lam = C.eigenvalues();
  CMat V = C.eigenvectors();
  VERIFY_IS_APPROX((denseC * V).eval(), (V * lam.asDiagonal()).eval());
  VERIFY_IS_APPROX((V.adjoint() * V).eval(), CMat(CMat::Identity(N, N)));
}

template <typename Scalar>
void test_bccb_svd(Index n2, Index n1) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef std::complex<RealScalar> Complex;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;
  typedef Matrix<Complex, Dynamic, Dynamic> CMat;

  Mat G = Mat::Random(n2, n1);
  Bccb<Scalar> C(G);
  Mat dense = reference_bccb<Scalar>(G);
  const Index N = n1 * n2;

  Matrix<RealScalar, Dynamic, 1> sv = C.singularValues();
  JacobiSVD<Mat> svd(dense);
  VERIFY_IS_APPROX(sv, svd.singularValues());

  CMat U = C.matrixU(), V = C.matrixV();
  VERIFY_IS_APPROX((U * sv.template cast<Complex>().asDiagonal() * V.adjoint()).eval(),
                   CMat(dense.template cast<Complex>()));
  VERIFY_IS_APPROX((U.adjoint() * U).eval(), CMat(CMat::Identity(N, N)));
  VERIFY_IS_APPROX((V.adjoint() * V).eval(), CMat(CMat::Identity(N, N)));
}

template <typename Scalar>
void test_bccb_inverse(Index n2, Index n1) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Mat G = Mat::Random(n2, n1);
  G(0, 0) += Scalar(RealScalar(2 * n1 * n2));  // safely invertible, tame determinant scale
  Bccb<Scalar> C(G);
  Mat dense = reference_bccb<Scalar>(G);
  const Index N = n1 * n2;

  Mat inv = C.inverse();
  VERIFY_IS_APPROX((inv * dense).eval(), Mat(Mat::Identity(N, N)));

  Vec b = Vec::Random(N);
  VERIFY_IS_APPROX((C.inverse() * b).eval(), C.solve(b));
}

template <typename Scalar>
void test_bccb_determinant(Index n2, Index n1) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Mat G = Mat::Random(n2, n1);
  G(0, 0) += Scalar(RealScalar(2 * n1 * n2));
  Bccb<Scalar> C(G);
  Mat dense = reference_bccb<Scalar>(G);
  VERIFY_IS_APPROX(C.determinant(), dense.determinant());
}

template <typename Scalar>
void test_bccb_matrix_free_cg(Index n2, Index n1) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  // Symmetrizing the generator under the two-dimensional index reversal makes
  // the BCCB matrix symmetric; diagonal dominance then makes it positive
  // definite.
  Mat G = Mat::Random(n2, n1);
  Mat Gs = G;
  for (Index k1 = 0; k1 < n1; ++k1)
    for (Index k2 = 0; k2 < n2; ++k2) Gs(k2, k1) = (G(k2, k1) + G((n2 - k2) % n2, (n1 - k1) % n1)) / Scalar(2);
  Gs(0, 0) += Scalar(RealScalar(2 * n1 * n2));
  Bccb<Scalar> C(Gs);
  Mat dense = reference_bccb<Scalar>(Gs);
  VERIFY_IS_APPROX(dense, Mat(dense.transpose()));

  const Index N = n1 * n2;
  Vec b = Vec::Random(N);
  ConjugateGradient<Bccb<Scalar>, Lower | Upper, IdentityPreconditioner> cg;
  cg.compute(C);
  Vec x = cg.solve(b);
  VERIFY(cg.info() == Success);
  VERIFY_IS_APPROX((dense * x).eval(), b);
}

template <typename Scalar, int N2, int N1>
void test_bccb_fixed() {
  typedef Matrix<Scalar, N2, N1> GenMat;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  GenMat G = GenMat::Random();
  Bccb<Scalar, N2, N1> C(G);
  STATIC_CHECK((Bccb<Scalar, N2, N1>::RowsAtCompileTime == N1 * N2));
  STATIC_CHECK((internal::remove_all_t<decltype(makeBccb(G))>::RowsAtCompileTime == N1 * N2));

  Mat dense = reference_bccb<Scalar>(Mat(G));
  Matrix<Scalar, N1 * N2, N1 * N2> Cd = C;
  VERIFY_IS_APPROX(Mat(Cd), dense);

  Matrix<Scalar, N1 * N2, 1> x = Matrix<Scalar, N1 * N2, 1>::Random();
  Matrix<Scalar, N1 * N2, 1> y = C * x;
  VERIFY_IS_APPROX(y, (dense * x).eval());
}

EIGEN_DECLARE_TEST(structured_bccb) {
  for (int i = 0; i < g_repeat; ++i) {
    // Products, dense assignment, coefficient access: scalar tier (N <= 32) and
    // 2-D FFT tier, including single-row/column-of-blocks degenerate shapes.
    CALL_SUBTEST_1((test_bccb_product<double>(1, 1)));
    CALL_SUBTEST_1((test_bccb_product<double>(3, 4)));
    CALL_SUBTEST_1((test_bccb_product<double>(5, 5)));
    CALL_SUBTEST_1((test_bccb_product<double>(8, 6)));
    CALL_SUBTEST_1((test_bccb_product<double>(7, 5)));  // odd prime dimensions
    CALL_SUBTEST_1((test_bccb_product<double>(1, 40)));
    CALL_SUBTEST_1((test_bccb_product<double>(40, 1)));
    CALL_SUBTEST_1((test_bccb_product<float>(6, 8)));
    CALL_SUBTEST_1((test_bccb_product<std::complex<double>>(4, 3)));
    CALL_SUBTEST_1((test_bccb_product<std::complex<double>>(9, 6)));
    CALL_SUBTEST_1((test_bccb_product<std::complex<float>>(6, 7)));

    // Transposition family with exact symbol round trips.
    CALL_SUBTEST_2((test_bccb_transpose<double>(3, 4)));
    CALL_SUBTEST_2((test_bccb_transpose<double>(8, 6)));
    CALL_SUBTEST_2((test_bccb_transpose<double>(1, 12)));
    CALL_SUBTEST_2((test_bccb_transpose<std::complex<double>>(6, 8)));
    CALL_SUBTEST_2((test_bccb_transpose<std::complex<float>>(5, 7)));

    // Direct and pseudo-inverse solves, degenerate operators.
    CALL_SUBTEST_3((test_bccb_solve<double>(1, 1)));
    CALL_SUBTEST_3((test_bccb_solve<double>(6, 8)));
    CALL_SUBTEST_3((test_bccb_solve<float>(5, 6)));
    CALL_SUBTEST_3((test_bccb_solve<std::complex<double>>(6, 6)));
    CALL_SUBTEST_3((test_bccb_rank_deficient<std::complex<double>>(6, 8, 4)));
    CALL_SUBTEST_3((test_bccb_rank_deficient<std::complex<float>>(4, 5, 2)));
    CALL_SUBTEST_3((test_bccb_zero<double>(4, 5)));
    CALL_SUBTEST_3(test_bccb_nan_propagation(6, 7));

    // Closed-form eigendecomposition, SVD, inverse, determinant.
    CALL_SUBTEST_4((test_bccb_eigen<double>(4, 5)));
    CALL_SUBTEST_4((test_bccb_eigen<double>(1, 8)));
    CALL_SUBTEST_4((test_bccb_eigen<std::complex<double>>(3, 5)));
    CALL_SUBTEST_4((test_bccb_svd<double>(4, 5)));
    CALL_SUBTEST_4((test_bccb_svd<std::complex<double>>(4, 4)));
    CALL_SUBTEST_4((test_bccb_svd<float>(3, 4)));
    CALL_SUBTEST_4((test_bccb_inverse<double>(6, 8)));
    CALL_SUBTEST_4((test_bccb_inverse<std::complex<double>>(5, 5)));
    CALL_SUBTEST_4((test_bccb_determinant<double>(3, 4)));
    CALL_SUBTEST_4((test_bccb_determinant<std::complex<double>>(3, 3)));

    // Matrix-free iterative solve and fixed-size generators.
    CALL_SUBTEST_5((test_bccb_matrix_free_cg<double>(8, 10)));
    CALL_SUBTEST_5((test_bccb_fixed<double, 3, 4>()));
    CALL_SUBTEST_5((test_bccb_fixed<std::complex<float>, 4, 3>()));

    // Product lifetime, aliasing, mixed-scalar promotion, dimension mismatches,
    // across the scalar-loop and FFT dispatch tiers.
    CALL_SUBTEST_6((test_bccb_delayed_product<double>(4, 5)));
    CALL_SUBTEST_6((test_bccb_delayed_product<std::complex<double>>(6, 8)));
    CALL_SUBTEST_6((test_bccb_aliased_product<double>(3, 4)));
    CALL_SUBTEST_6((test_bccb_aliased_product<double>(8, 6)));
    CALL_SUBTEST_6((test_bccb_aliased_product<std::complex<double>>(6, 8)));
    CALL_SUBTEST_6((test_bccb_mixed_scalar<double>(3, 4)));
    CALL_SUBTEST_6((test_bccb_mixed_scalar<double>(6, 8)));
    CALL_SUBTEST_6((test_bccb_mixed_scalar<float>(8, 8)));
    CALL_SUBTEST_6(test_bccb_dimension_asserts());

    // Finite-range robustness: scaled transforms, balanced determinant
    // accumulation, and the rank threshold boundary.
    CALL_SUBTEST_7(test_bccb_finite_overflow());
    CALL_SUBTEST_7(test_bccb_determinant_scaled());
    CALL_SUBTEST_7(test_bccb_rank_boundaries());
  }
}
