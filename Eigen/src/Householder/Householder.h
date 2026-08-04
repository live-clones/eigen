// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_HOUSEHOLDER_H
#define EIGEN_HOUSEHOLDER_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {
template <int N>
struct decrement_size : std::integral_constant<int, N - 1> {};
template <>
struct decrement_size<0> : std::integral_constant<int, 0> {};
template <>
struct decrement_size<Dynamic> : std::integral_constant<int, Dynamic> {};

template <typename RealScalar>
struct householder_norm_accumulator {
  using StableAccumulator = typename stable_norm_accumulator<RealScalar>::type;
  // Widen float before arithmetic so FTZ does not discard representable subnormal coefficients.
  using type = std::conditional_t<std::is_same<RealScalar, float>::value, double, StableAccumulator>;
};

template <typename Scalar, typename Accumulator, bool IsComplex = NumTraits<Scalar>::IsComplex,
          bool HasArrayAccess = complex_array_access<Scalar>::value>
struct householder_rescale;

template <typename Scalar, typename Accumulator, bool HasArrayAccess>
struct householder_rescale<Scalar, Accumulator, false, HasArrayAccess> {
  EIGEN_DEVICE_FUNC static Scalar run(const Scalar& value, const Accumulator& scale) {
    return Scalar(Accumulator(value) / scale);
  }
};

template <typename Scalar, typename Accumulator>
struct householder_rescale<Scalar, Accumulator, true, false> {
  using RealScalar = typename NumTraits<Scalar>::Real;

  EIGEN_DEVICE_FUNC static Scalar run(const Scalar& value, const Accumulator& scale) {
    return Scalar(RealScalar(Accumulator(numext::real(value)) / scale),
                  RealScalar(Accumulator(numext::imag(value)) / scale));
  }
};

template <typename Scalar, typename Accumulator>
struct householder_rescale<Scalar, Accumulator, true, true> {
  using RealScalar = typename NumTraits<Scalar>::Real;

  EIGEN_DEVICE_FUNC static Scalar run(const Scalar& value, const Accumulator& scale) {
    Scalar result = value;
    numext::real_ref(result) = RealScalar(Accumulator(numext::real(value)) / scale);
    numext::imag_ref(result) = RealScalar(Accumulator(numext::imag(value)) / scale);
    return result;
  }
};
}  // namespace internal

/** Computes the elementary reflector H such that:
 * \f$ H *this = [ beta 0 ... 0]^T \f$
 * where the transformation H is:
 * \f$ H = I - tau v v^*\f$
 * and the vector v is:
 * \f$ v^T = [1 essential^T] \f$
 *
 * The essential part of the vector \c v is stored in *this.
 *
 * On output:
 * \param tau the scaling factor of the Householder transformation
 * \param beta the result of H * \c *this
 *
 * \sa MatrixBase::makeHouseholder(), MatrixBase::applyHouseholderOnTheLeft(),
 *     MatrixBase::applyHouseholderOnTheRight()
 */
template <typename Derived>
EIGEN_DEVICE_FUNC void MatrixBase<Derived>::makeHouseholderInPlace(Scalar& tau, RealScalar& beta) {
  VectorBlock<Derived, internal::decrement_size<Base::SizeAtCompileTime>::value> essentialPart(derived(), 1,
                                                                                               size() - 1);
  makeHouseholder(essentialPart, tau, beta);
}

/** Computes the elementary reflector H such that:
 * \f$ H *this = [ beta 0 ... 0]^T \f$
 * where the transformation H is:
 * \f$ H = I - tau v v^*\f$
 * and the vector v is:
 * \f$ v^T = [1 essential^T] \f$
 *
 * On output:
 * \param essential the essential part of the vector \c v
 * \param tau the scaling factor of the Householder transformation
 * \param beta the result of H * \c *this
 *
 * \sa MatrixBase::makeHouseholderInPlace(), MatrixBase::applyHouseholderOnTheLeft(),
 *     MatrixBase::applyHouseholderOnTheRight()
 */
template <typename Derived>
template <typename EssentialPart>
EIGEN_DEVICE_FUNC void MatrixBase<Derived>::makeHouseholder(EssentialPart& essential, Scalar& tau,
                                                            RealScalar& beta) const {
  using numext::conj;

  EIGEN_STATIC_ASSERT_VECTOR_ONLY(EssentialPart)
  const VectorBlock<const Derived, EssentialPart::SizeAtCompileTime> tail(derived(), 1, size() - 1);

  const RealScalar tailSqNorm = size() == 1 ? RealScalar(0) : tail.unwind().squaredNorm();
  Scalar c0 = coeff(0);
  const RealScalar tol = (std::numeric_limits<RealScalar>::min)();
  RealScalar unscaledNormThreshold = tol;
  if (!NumTraits<RealScalar>::IsInteger) {
    const RealScalar precision = RealScalar(NumTraits<RealScalar>::epsilon());
    // With flush-to-zero arithmetic, every tail component square below tol can be lost. Account for every component
    // so the discarded contribution is at most epsilon relative to a squared norm above this threshold.
    const RealScalar componentCount = RealScalar(size() - 1) * RealScalar(NumTraits<Scalar>::IsComplex ? 2 : 1);
    unscaledNormThreshold = (tol / precision) * componentCount;
  }

  if (tailSqNorm <= unscaledNormThreshold && !(numext::isnan)(c0)) {
    using Accumulator = typename internal::householder_norm_accumulator<RealScalar>::type;
    const auto tailView = tail.unwind();
    const auto tailComponents = tailView.realView();
    // Component maxima cannot underflow when a representable tail is nonzero.
    const Accumulator tailMax = tailView.size() == 0
                                    ? Accumulator(0)
                                    : Accumulator(tailComponents.cwiseAbs().template maxCoeff<PropagateNaN>());
    if (numext::is_exactly_zero(tailMax) && numext::is_exactly_zero(numext::imag(c0))) {
      tau = RealScalar(0);
      beta = numext::real(c0);
      essential.setZero();
      return;
    }
    const Accumulator c0RealAbs = numext::abs(Accumulator(numext::real(c0)));
    const Accumulator c0ImagAbs = numext::abs(Accumulator(numext::imag(c0)));
    const Accumulator c0Max = c0RealAbs > c0ImagAbs ? c0RealAbs : c0ImagAbs;
    const Accumulator scale = c0Max > tailMax ? c0Max : tailMax;
    const RealScalar realScale = RealScalar(scale);
    // A target that flushes this scale cannot form meaningful ratios from the entirely subnormal vector.
    if (scale < Accumulator(tol) && numext::is_exactly_zero(realScale + realScale)) {
      tau = RealScalar(0);
      beta = numext::real(c0);
      essential.setZero();
      return;
    }
    // Form the reflector from scale-free ratios to preserve subnormal inputs and avoid overflowing c0 - beta.
    Accumulator scaledTailSqNorm = Accumulator(0);
    for (Index i = 0; i < tailView.size(); ++i) {
      const Accumulator scaledReal = Accumulator(numext::real(tailView.coeff(i))) / scale;
      const Accumulator scaledImag = Accumulator(numext::imag(tailView.coeff(i))) / scale;
      scaledTailSqNorm += scaledReal * scaledReal + scaledImag * scaledImag;
    }
    if (numext::is_exactly_zero(RealScalar(tailMax / scale)) && numext::is_exactly_zero(numext::imag(c0))) {
      tau = RealScalar(0);
      beta = numext::real(c0);
      essential.setZero();
      return;
    }
    const Scalar scaledC0 = internal::householder_rescale<Scalar, Accumulator>::run(c0, scale);
    RealScalar scaledBeta =
        RealScalar(numext::hypot(Accumulator(numext::abs(scaledC0)), numext::sqrt(scaledTailSqNorm)));
    if (numext::real(c0) >= RealScalar(0)) scaledBeta = -scaledBeta;
    beta = RealScalar(scale * Accumulator(scaledBeta));
    for (Index i = 0; i < tailView.size(); ++i) {
      essential.coeffRef(i) =
          internal::householder_rescale<Scalar, Accumulator>::run(tailView.coeff(i), scale) / (scaledC0 - scaledBeta);
    }
    tau = conj(Scalar(RealScalar(1)) - scaledC0 / scaledBeta);
    return;
  }
  beta = numext::sqrt(numext::abs2(c0) + tailSqNorm);
  if (numext::real(c0) >= RealScalar(0)) beta = -beta;
  essential = tail.unwind() / (c0 - beta);
  tau = conj((beta - c0) / beta);
}

/** Apply the elementary reflector H given by
 * \f$ H = I - tau v v^*\f$
 * with
 * \f$ v^T = [1 essential^T] \f$
 * from the left to a vector or matrix.
 *
 * On input:
 * \param essential the essential part of the vector \c v
 * \param tau the scaling factor of the Householder transformation
 * \param workspace a pointer to working space with at least
 *                  this->cols() entries
 *
 * \sa MatrixBase::makeHouseholder(), MatrixBase::makeHouseholderInPlace(),
 *     MatrixBase::applyHouseholderOnTheRight()
 */
template <typename Derived>
template <typename EssentialPart>
EIGEN_DEVICE_FUNC void MatrixBase<Derived>::applyHouseholderOnTheLeft(const EssentialPart& essential, const Scalar& tau,
                                                                      Scalar* workspace) {
  if (rows() == 1) {
    *this *= Scalar(1) - tau;
  } else if (!numext::is_exactly_zero(tau)) {
    Map<typename internal::plain_row_type<PlainObject>::type> tmp(workspace, cols());
    Block<Derived, EssentialPart::SizeAtCompileTime, Derived::ColsAtCompileTime> bottom(derived(), 1, 0, rows() - 1,
                                                                                        cols());
    tmp.noalias() = essential.adjoint() * bottom.unwind();
    tmp = tau * (tmp + this->row(0));
    this->row(0) = this->row(0) - tmp;
    bottom.unwind().noalias() -= essential * tmp;
  }
}

/** Apply the elementary reflector H given by
 * \f$ H = I - tau v v^*\f$
 * with
 * \f$ v^T = [1 essential^T] \f$
 * from the right to a vector or matrix.
 *
 * On input:
 * \param essential the essential part of the vector \c v
 * \param tau the scaling factor of the Householder transformation
 * \param workspace a pointer to working space with at least
 *                  this->rows() entries
 *
 * \sa MatrixBase::makeHouseholder(), MatrixBase::makeHouseholderInPlace(),
 *     MatrixBase::applyHouseholderOnTheLeft()
 */
template <typename Derived>
template <typename EssentialPart>
EIGEN_DEVICE_FUNC void MatrixBase<Derived>::applyHouseholderOnTheRight(const EssentialPart& essential,
                                                                       const Scalar& tau, Scalar* workspace) {
  if (cols() == 1) {
    *this *= Scalar(1) - tau;
  } else if (!numext::is_exactly_zero(tau)) {
    Map<typename internal::plain_col_type<PlainObject>::type> tmp(workspace, rows());
    Block<Derived, Derived::RowsAtCompileTime, EssentialPart::SizeAtCompileTime> right(derived(), 0, 1, rows(),
                                                                                       cols() - 1);
    tmp.noalias() = right.unwind() * essential;
    tmp = tau * (tmp + this->col(0));
    this->col(0) = this->col(0) - tmp;
    right.unwind().noalias() -= tmp * essential.adjoint();
  }
}

}  // end namespace Eigen

#endif  // EIGEN_HOUSEHOLDER_H
