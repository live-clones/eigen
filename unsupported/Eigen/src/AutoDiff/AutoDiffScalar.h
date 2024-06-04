// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_AUTODIFF_SCALAR_H
#define EIGEN_AUTODIFF_SCALAR_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

template <typename DerivativeType, bool Enable>
struct auto_diff_special_op;

template <typename DerivativeType, typename OtherDerivativeType, typename EnableIf = void>
struct maybe_coherent_pad_helper {
  static constexpr int SizeAtCompileTime =
      max_size_prefer_dynamic(DerivativeType::SizeAtCompileTime, OtherDerivativeType::SizeAtCompileTime);
  using type = CoherentPadOp<DerivativeType, SizeAtCompileTime>;
  static type pad(const DerivativeType& x, const OtherDerivativeType& y) {
    // CoherentPadOp uses variable_if_dynamic<SizeAtCompileTime>.  In this case, `SizeAtCompileTime` might
    // by Dynamic, so we need to take the runtime maximum of x, y.
    return CoherentPadOp<DerivativeType, SizeAtCompileTime>(x, numext::maxi(x.size(), y.size()));
  }
};

// Both are fixed-sized and equal, don't need to pad.
// Both are fixed-size and this is larger than other, don't need to pad.
template <typename DerivativeType, typename OtherDerivativeType>
struct maybe_coherent_pad_helper<
    DerivativeType, OtherDerivativeType,
    std::enable_if_t<enum_ge_not_dynamic(DerivativeType::SizeAtCompileTime, OtherDerivativeType::SizeAtCompileTime)>> {
  using type = const DerivativeType&;
  static const DerivativeType& pad(const DerivativeType& x, const OtherDerivativeType& /*y*/) { return x; }
};

template <typename DerivativeType, typename OtherDerivativeType>
constexpr typename maybe_coherent_pad_helper<DerivativeType, OtherDerivativeType>::type MaybeCoherentPad(
    const DerivativeType& x, const OtherDerivativeType& y) {
  return maybe_coherent_pad_helper<DerivativeType, OtherDerivativeType>::pad(x, y);
}

template <typename Op, typename LhsDerivativeType, typename RhsDerivativeType>
constexpr auto MakeCoherentCwiseBinaryOp(const LhsDerivativeType& x, const RhsDerivativeType& y, Op op = Op()) {
  const auto& lhs = MaybeCoherentPad(x, y);
  const auto& rhs = MaybeCoherentPad(y, x);
  return CwiseBinaryOp<Op, remove_all_t<decltype(lhs)>, remove_all_t<decltype(rhs)>>(lhs, rhs, op);
}

}  // namespace internal

template <typename DerivativeType>
class AutoDiffScalar;

template <typename NewDerType>
constexpr AutoDiffScalar<NewDerType> MakeAutoDiffScalar(const typename NewDerType::Scalar& value,
                                                        const NewDerType& der) {
  return AutoDiffScalar<NewDerType>(value, der);
}

/** \class AutoDiffScalar
 * \brief A scalar type replacement with automatic differentiation capability
 *
 * \param DerivativeType the vector type used to store/represent the derivatives. The base scalar type
 *                 as well as the number of derivatives to compute are determined from this type.
 *                 Typical choices include, e.g., \c Vector4f for 4 derivatives, or \c VectorXf
 *                 if the number of derivatives is not known at compile time, and/or, the number
 *                 of derivatives is large.
 *                 Note that DerivativeType can also be a reference (e.g., \c VectorXf&) to wrap a
 *                 existing vector into an AutoDiffScalar.
 *                 Finally, DerivativeType can also be any Eigen compatible expression.
 *
 * This class represents a scalar value while tracking its respective derivatives using Eigen's expression
 * template mechanism.
 *
 * It supports the following list of global math function:
 *  - std::abs, std::sqrt, std::pow, std::exp, std::log, std::sin, std::cos,
 *  - internal::abs, internal::sqrt, numext::pow, internal::exp, internal::log, internal::sin, internal::cos,
 *  - internal::conj, internal::real, internal::imag, numext::abs2.
 *
 * AutoDiffScalar can be used as the scalar type of an Eigen::Matrix object. However,
 * in that case, the expression template mechanism only occurs at the top Matrix level,
 * while derivatives are computed right away.
 *
 */

template <typename DerivativeType>
class AutoDiffScalar
    : public internal::auto_diff_special_op<
          DerivativeType, !internal::is_same<typename internal::traits<internal::remove_all_t<DerivativeType>>::Scalar,
                                             typename NumTraits<typename internal::traits<
                                                 internal::remove_all_t<DerivativeType>>::Scalar>::Real>::value> {
 public:
  typedef internal::auto_diff_special_op<
      DerivativeType,
      !internal::is_same<
          typename internal::traits<internal::remove_all_t<DerivativeType>>::Scalar,
          typename NumTraits<typename internal::traits<internal::remove_all_t<DerivativeType>>::Scalar>::Real>::value>
      Base;
  typedef internal::remove_all_t<DerivativeType> DerType;
  typedef typename internal::traits<DerType>::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real Real;

  using Base::operator+;
  using Base::operator*;

  /** Default constructor without any initialization. */
  constexpr AutoDiffScalar() = default;

  /** Constructs an active scalar from its \a value,
      and initializes the \a nbDer derivatives such that it corresponds to the \a derNumber -th variable */
  constexpr AutoDiffScalar(const Scalar& value, int nbDer, int derNumber)
      : m_value(value), m_derivatives(DerType::Zero(nbDer)) {
    m_derivatives.coeffRef(derNumber) = Scalar(1);
  }

  /** Conversion from a scalar constant to an active scalar.
   * The derivatives are set to zero. */
  constexpr AutoDiffScalar(const Real& value) : m_value(value) {
    if (m_derivatives.size() > 0) m_derivatives.setZero();
  }

  /** Constructs an active scalar from its \a value and derivatives \a der */
  constexpr AutoDiffScalar(const Scalar& value, const DerType& der) : m_value(value), m_derivatives(der) {}

  template <typename OtherDerType>
  constexpr AutoDiffScalar(
      const AutoDiffScalar<OtherDerType>& other
#ifndef EIGEN_PARSED_BY_DOXYGEN
      ,
      std::enable_if_t<
          internal::is_same<Scalar, typename internal::traits<internal::remove_all_t<OtherDerType>>::Scalar>::value &&
              internal::is_convertible<OtherDerType, DerType>::value,
          void*> = 0
#endif
      )
      : m_value(other.value()), m_derivatives(other.derivatives()) {
  }

  friend std::ostream& operator<<(std::ostream& s, const AutoDiffScalar& a) { return s << a.value(); }

  constexpr AutoDiffScalar(const AutoDiffScalar& other) : m_value(other.value()), m_derivatives(other.derivatives()) {}

  template <typename OtherDerType>
  constexpr AutoDiffScalar& operator=(const AutoDiffScalar<OtherDerType>& other) {
    m_value = other.value();
    m_derivatives = other.derivatives();
    return *this;
  }

  constexpr AutoDiffScalar& operator=(const AutoDiffScalar& other) {
    m_value = other.value();
    m_derivatives = other.derivatives();
    return *this;
  }

  constexpr AutoDiffScalar& operator=(const Scalar& other) {
    m_value = other;
    if (m_derivatives.size() > 0) m_derivatives.setZero();
    return *this;
  }

  //     inline operator const Scalar& () const { return m_value; }
  //     inline operator Scalar& () { return m_value; }

  constexpr const Scalar& value() const { return m_value; }
  constexpr Scalar& value() { return m_value; }

  constexpr const DerType& derivatives() const { return m_derivatives; }
  constexpr DerType& derivatives() { return m_derivatives; }

  constexpr bool operator<(const Scalar& other) const { return m_value < other; }
  constexpr bool operator<=(const Scalar& other) const { return m_value <= other; }
  constexpr bool operator>(const Scalar& other) const { return m_value > other; }
  constexpr bool operator>=(const Scalar& other) const { return m_value >= other; }
  constexpr bool operator==(const Scalar& other) const { return m_value == other; }
  constexpr bool operator!=(const Scalar& other) const { return m_value != other; }

  friend constexpr bool operator<(const Scalar& a, const AutoDiffScalar& b) { return a < b.value(); }
  friend constexpr bool operator<=(const Scalar& a, const AutoDiffScalar& b) { return a <= b.value(); }
  friend constexpr bool operator>(const Scalar& a, const AutoDiffScalar& b) { return a > b.value(); }
  friend constexpr bool operator>=(const Scalar& a, const AutoDiffScalar& b) { return a >= b.value(); }
  friend constexpr bool operator==(const Scalar& a, const AutoDiffScalar& b) { return a == b.value(); }
  friend constexpr bool operator!=(const Scalar& a, const AutoDiffScalar& b) { return a != b.value(); }

  template <typename OtherDerType>
  constexpr bool operator<(const AutoDiffScalar<OtherDerType>& b) const {
    return m_value < b.value();
  }
  template <typename OtherDerType>
  constexpr bool operator<=(const AutoDiffScalar<OtherDerType>& b) const {
    return m_value <= b.value();
  }
  template <typename OtherDerType>
  constexpr bool operator>(const AutoDiffScalar<OtherDerType>& b) const {
    return m_value > b.value();
  }
  template <typename OtherDerType>
  constexpr bool operator>=(const AutoDiffScalar<OtherDerType>& b) const {
    return m_value >= b.value();
  }
  template <typename OtherDerType>
  constexpr bool operator==(const AutoDiffScalar<OtherDerType>& b) const {
    return m_value == b.value();
  }
  template <typename OtherDerType>
  constexpr bool operator!=(const AutoDiffScalar<OtherDerType>& b) const {
    return m_value != b.value();
  }

  constexpr AutoDiffScalar<DerType&> operator+(const Scalar& other) const {
    return AutoDiffScalar<DerType&>(m_value + other, m_derivatives);
  }

  friend constexpr AutoDiffScalar<DerType&> operator+(const Scalar& a, const AutoDiffScalar& b) {
    return AutoDiffScalar<DerType&>(a + b.value(), b.derivatives());
  }

  constexpr inline AutoDiffScalar& operator+=(const Scalar& other) {
    value() += other;
    return *this;
  }

  template <typename OtherDerType>
  constexpr auto operator+(const AutoDiffScalar<OtherDerType>& other) const {
    return MakeAutoDiffScalar(
        m_value + other.value(),
        internal::MakeCoherentCwiseBinaryOp<internal::scalar_sum_op<Scalar>>(m_derivatives, other.derivatives()));
  }

  template <typename OtherDerType>
  constexpr AutoDiffScalar& operator+=(const AutoDiffScalar<OtherDerType>& other) {
    (*this) = (*this) + other;
    return *this;
  }

  constexpr AutoDiffScalar<DerType&> operator-(const Scalar& b) const {
    return AutoDiffScalar<DerType&>(m_value - b, m_derivatives);
  }

  friend constexpr AutoDiffScalar<CwiseUnaryOp<internal::scalar_opposite_op<Scalar>, const DerType>> operator-(
      const Scalar& a, const AutoDiffScalar& b) {
    return AutoDiffScalar<CwiseUnaryOp<internal::scalar_opposite_op<Scalar>, const DerType>>(a - b.value(),
                                                                                             -b.derivatives());
  }

  constexpr AutoDiffScalar& operator-=(const Scalar& other) {
    value() -= other;
    return *this;
  }

  template <typename OtherDerType>
  constexpr auto operator-(const AutoDiffScalar<OtherDerType>& other) const {
    return MakeAutoDiffScalar(m_value - other.value(),
                              internal::MakeCoherentCwiseBinaryOp<internal::scalar_difference_op<Scalar>>(
                                  m_derivatives, other.derivatives()));
  }

  template <typename OtherDerType>
  constexpr AutoDiffScalar& operator-=(const AutoDiffScalar<OtherDerType>& other) {
    *this = *this - other;
    return *this;
  }

  constexpr AutoDiffScalar<CwiseUnaryOp<internal::scalar_opposite_op<Scalar>, const DerType>> operator-() const {
    return AutoDiffScalar<CwiseUnaryOp<internal::scalar_opposite_op<Scalar>, const DerType>>(-m_value, -m_derivatives);
  }

  constexpr auto operator*(const Scalar& other) const {
    return MakeAutoDiffScalar(m_value * other, m_derivatives * other);
  }

  friend constexpr auto operator*(const Scalar& other, const AutoDiffScalar& a) {
    return MakeAutoDiffScalar(a.value() * other, a.derivatives() * other);
  }

  constexpr inline auto operator/(const Scalar& other) const {
    return MakeAutoDiffScalar(m_value / other, (m_derivatives * (Scalar(1) / other)));
  }

  friend constexpr auto operator/(const Scalar& other, const AutoDiffScalar& a) {
    return MakeAutoDiffScalar(other / a.value(), a.derivatives() * (Scalar(-other) / (a.value() * a.value())));
  }

  template <typename OtherDerType>
  constexpr auto operator/(const AutoDiffScalar<OtherDerType>& other) const {
    return MakeAutoDiffScalar(m_value / other.value(),
                              internal::MakeCoherentCwiseBinaryOp<internal::scalar_difference_op<Scalar>>(
                                  m_derivatives * other.value(), (other.derivatives() * m_value)) *
                                  (Scalar(1) / (other.value() * other.value())));
  }

  template <typename OtherDerType>
  constexpr auto operator*(const AutoDiffScalar<OtherDerType>& other) const {
    return MakeAutoDiffScalar(m_value * other.value(),
                              internal::MakeCoherentCwiseBinaryOp<internal::scalar_sum_op<Scalar>>(
                                  m_derivatives * other.value(), other.derivatives() * m_value));
  }

  constexpr AutoDiffScalar& operator*=(const Scalar& other) {
    *this = *this * other;
    return *this;
  }

  template <typename OtherDerType>
  constexpr AutoDiffScalar& operator*=(const AutoDiffScalar<OtherDerType>& other) {
    *this = *this * other;
    return *this;
  }

  constexpr AutoDiffScalar& operator/=(const Scalar& other) {
    *this = *this / other;
    return *this;
  }

  template <typename OtherDerType>
  constexpr AutoDiffScalar& operator/=(const AutoDiffScalar<OtherDerType>& other) {
    *this = *this / other;
    return *this;
  }

 protected:
  Scalar m_value;
  DerType m_derivatives;
};

namespace internal {

template <typename DerivativeType>
struct auto_diff_special_op<DerivativeType, true>
//   : auto_diff_scalar_op<DerivativeType, typename NumTraits<Scalar>::Real,
//                            is_same<Scalar,typename NumTraits<Scalar>::Real>::value>
{
  typedef remove_all_t<DerivativeType> DerType;
  typedef typename traits<DerType>::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real Real;

  constexpr const AutoDiffScalar<DerivativeType>& derived() const {
    return *static_cast<const AutoDiffScalar<DerivativeType>*>(this);
  }
  constexpr AutoDiffScalar<DerivativeType>& derived() { return *static_cast<AutoDiffScalar<DerivativeType>*>(this); }

  constexpr AutoDiffScalar<DerType&> operator+(const Real& other) const {
    return AutoDiffScalar<DerType&>(derived().value() + other, derived().derivatives());
  }

  friend constexpr AutoDiffScalar<DerType&> operator+(const Real& a, const AutoDiffScalar<DerivativeType>& b) {
    return AutoDiffScalar<DerType&>(a + b.value(), b.derivatives());
  }

  constexpr AutoDiffScalar<DerivativeType>& operator+=(const Real& other) {
    derived().value() += other;
    return derived();
  }

  constexpr AutoDiffScalar<typename CwiseUnaryOp<bind2nd_op<scalar_product_op<Scalar, Real>>, DerType>::Type> operator*(
      const Real& other) const {
    return AutoDiffScalar<typename CwiseUnaryOp<bind2nd_op<scalar_product_op<Scalar, Real>>, DerType>::Type>(
        derived().value() * other, derived().derivatives() * other);
  }

  friend constexpr AutoDiffScalar<typename CwiseUnaryOp<bind1st_op<scalar_product_op<Real, Scalar>>, DerType>::Type>
  operator*(const Real& other, const AutoDiffScalar<DerivativeType>& a) {
    return AutoDiffScalar<typename CwiseUnaryOp<bind1st_op<scalar_product_op<Real, Scalar>>, DerType>::Type>(
        a.value() * other, a.derivatives() * other);
  }

  constexpr AutoDiffScalar<DerivativeType>& operator*=(const Scalar& other) {
    *this = *this * other;
    return derived();
  }
};

template <typename DerivativeType>
struct auto_diff_special_op<DerivativeType, false> {
  constexpr void operator*() const;
  constexpr void operator-() const;
  constexpr void operator+() const;
};

}  // end namespace internal

template <typename DerType, typename BinOp>
struct ScalarBinaryOpTraits<AutoDiffScalar<DerType>, typename DerType::Scalar, BinOp> {
  typedef AutoDiffScalar<DerType> ReturnType;
};

template <typename DerType, typename BinOp>
struct ScalarBinaryOpTraits<typename DerType::Scalar, AutoDiffScalar<DerType>, BinOp> {
  typedef AutoDiffScalar<DerType> ReturnType;
};

#define EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(FUNC, CODE)                                              \
  template <typename DerType>                                                                        \
  constexpr auto FUNC(const Eigen::AutoDiffScalar<DerType>& x) {                                     \
    using namespace Eigen;                                                                           \
    typedef typename Eigen::internal::traits<Eigen::internal::remove_all_t<DerType>>::Scalar Scalar; \
    EIGEN_UNUSED_VARIABLE(sizeof(Scalar));                                                           \
    CODE;                                                                                            \
  }

template <typename DerType>
struct CleanedUpDerType {
  typedef AutoDiffScalar<typename Eigen::internal::remove_all_t<DerType>::PlainObject> type;
};

template <typename DerType>
constexpr const AutoDiffScalar<DerType>& conj(const AutoDiffScalar<DerType>& x) {
  return x;
}
template <typename DerType>
constexpr const AutoDiffScalar<DerType>& real(const AutoDiffScalar<DerType>& x) {
  return x;
}
template <typename DerType>
constexpr typename DerType::Scalar imag(const AutoDiffScalar<DerType>&) {
  return 0.;
}
template <typename DerType, typename T>
constexpr typename CleanedUpDerType<DerType>::type(min)(const AutoDiffScalar<DerType>& x, const T& y) {
  typedef typename CleanedUpDerType<DerType>::type ADS;
  return (x <= y ? ADS(x) : ADS(y));
}
template <typename DerType, typename T>
constexpr typename CleanedUpDerType<DerType>::type(max)(const AutoDiffScalar<DerType>& x, const T& y) {
  typedef typename CleanedUpDerType<DerType>::type ADS;
  return (x >= y ? ADS(x) : ADS(y));
}
template <typename DerType, typename T>
constexpr typename CleanedUpDerType<DerType>::type(min)(const T& x, const AutoDiffScalar<DerType>& y) {
  typedef typename CleanedUpDerType<DerType>::type ADS;
  return (x < y ? ADS(x) : ADS(y));
}
template <typename DerType, typename T>
constexpr typename CleanedUpDerType<DerType>::type(max)(const T& x, const AutoDiffScalar<DerType>& y) {
  typedef typename CleanedUpDerType<DerType>::type ADS;
  return (x > y ? ADS(x) : ADS(y));
}
template <typename DerType>
inline constexpr
    typename CleanedUpDerType<DerType>::type(min)(const AutoDiffScalar<DerType>& x, const AutoDiffScalar<DerType>& y) {
  return (x.value() < y.value() ? x : y);
}
template <typename DerType>
inline constexpr
    typename CleanedUpDerType<DerType>::type(max)(const AutoDiffScalar<DerType>& x, const AutoDiffScalar<DerType>& y) {
  return (x.value() >= y.value() ? x : y);
}

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(abs, using std::abs;
                                    return Eigen::MakeAutoDiffScalar(abs(x.value()),
                                                                     x.derivatives() * (x.value() < 0 ? -1 : 1));)

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(abs2, using numext::abs2;
                                    return Eigen::MakeAutoDiffScalar(abs2(x.value()),
                                                                     x.derivatives() * (Scalar(2) * x.value()));)

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(sqrt, using std::sqrt; Scalar sqrtx = sqrt(x.value());
                                    return Eigen::MakeAutoDiffScalar(sqrtx, x.derivatives() * (Scalar(0.5) / sqrtx));)

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(cos, using std::cos; using std::sin;
                                    return Eigen::MakeAutoDiffScalar(cos(x.value()),
                                                                     x.derivatives() * (-sin(x.value())));)

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(sin, using std::sin; using std::cos;
                                    return Eigen::MakeAutoDiffScalar(sin(x.value()), x.derivatives() * cos(x.value()));)

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(exp, using std::exp; Scalar expx = exp(x.value());
                                    return Eigen::MakeAutoDiffScalar(expx, x.derivatives() * expx);)

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(log, using std::log;
                                    return Eigen::MakeAutoDiffScalar(log(x.value()),
                                                                     x.derivatives() * (Scalar(1) / x.value()));)

template <typename DerType>
constexpr auto pow(const Eigen::AutoDiffScalar<DerType>& x,
                   const typename internal::traits<internal::remove_all_t<DerType>>::Scalar& y) {
  using namespace Eigen;
  using std::pow;
  return Eigen::MakeAutoDiffScalar(pow(x.value(), y), x.derivatives() * (y * pow(x.value(), y - 1)));
}

template <typename DerTypeA, typename DerTypeB>
constexpr AutoDiffScalar<Matrix<typename internal::traits<internal::remove_all_t<DerTypeA>>::Scalar, Dynamic, 1>> atan2(
    const AutoDiffScalar<DerTypeA>& a, const AutoDiffScalar<DerTypeB>& b) {
  using std::atan2;
  typedef typename internal::traits<internal::remove_all_t<DerTypeA>>::Scalar Scalar;
  typedef AutoDiffScalar<Matrix<Scalar, Dynamic, 1>> PlainADS;
  PlainADS ret;
  ret.value() = atan2(a.value(), b.value());

  Scalar squared_hypot = a.value() * a.value() + b.value() * b.value();

  // if (squared_hypot==0) the derivation is undefined and the following results in a NaN:
  ret.derivatives() = (a.derivatives() * b.value() - a.value() * b.derivatives()) / squared_hypot;

  return ret;
}

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(tan, using std::tan; using std::cos; return Eigen::MakeAutoDiffScalar(
                                        tan(x.value()), x.derivatives() * (Scalar(1) / numext::abs2(cos(x.value()))));)

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(asin, using std::sqrt; using std::asin; return Eigen::MakeAutoDiffScalar(
                                        asin(x.value()),
                                        x.derivatives() * (Scalar(1) / sqrt(1 - numext::abs2(x.value()))));)

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(acos, using std::sqrt; using std::acos; return Eigen::MakeAutoDiffScalar(
                                        acos(x.value()),
                                        x.derivatives() * (Scalar(-1) / sqrt(1 - numext::abs2(x.value()))));)

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(
    tanh, using std::cosh; using std::tanh;
    return Eigen::MakeAutoDiffScalar(tanh(x.value()), x.derivatives() * (Scalar(1) / numext::abs2(cosh(x.value()))));)

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(sinh, using std::sinh; using std::cosh;
                                    return Eigen::MakeAutoDiffScalar(sinh(x.value()),
                                                                     x.derivatives() * cosh(x.value()));)

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(cosh, using std::sinh; using std::cosh;
                                    return Eigen::MakeAutoDiffScalar(cosh(x.value()),
                                                                     x.derivatives() * sinh(x.value()));)

#undef EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY

template <typename DerType>
struct NumTraits<AutoDiffScalar<DerType>>
    : NumTraits<typename NumTraits<typename internal::remove_all_t<DerType>::Scalar>::Real> {
  typedef internal::remove_all_t<DerType> DerTypeCleaned;
  typedef AutoDiffScalar<Matrix<typename NumTraits<typename DerTypeCleaned::Scalar>::Real,
                                DerTypeCleaned::RowsAtCompileTime, DerTypeCleaned::ColsAtCompileTime, 0,
                                DerTypeCleaned::MaxRowsAtCompileTime, DerTypeCleaned::MaxColsAtCompileTime>>
      Real;
  typedef AutoDiffScalar<DerType> NonInteger;
  typedef AutoDiffScalar<DerType> Nested;
  typedef typename NumTraits<typename DerTypeCleaned::Scalar>::Literal Literal;
  enum { RequireInitialization = 1 };
};

namespace internal {
template <typename DerivativeType>
struct is_identically_zero_impl<AutoDiffScalar<DerivativeType>> {
  static constexpr bool run(const AutoDiffScalar<DerivativeType>& s) {
    const DerivativeType& derivatives = s.derivatives();
    for (int i = 0; i < derivatives.size(); ++i) {
      if (!numext::is_exactly_zero(derivatives[i])) {
        return false;
      }
    }
    return numext::is_exactly_zero(s.value());
  }
};
}  // namespace internal
}  // namespace Eigen

namespace std {

template <typename T>
class numeric_limits<Eigen::AutoDiffScalar<T>> : public numeric_limits<typename T::Scalar> {};

template <typename T>
class numeric_limits<Eigen::AutoDiffScalar<T&>> : public numeric_limits<typename T::Scalar> {};

}  // namespace std

#endif  // EIGEN_AUTODIFF_SCALAR_H
