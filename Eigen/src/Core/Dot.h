// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2008, 2010 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_DOT_H
#define EIGEN_DOT_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

template <typename Derived, typename Scalar = typename traits<Derived>::Scalar>
struct squared_norm_impl {
  using Real = typename NumTraits<Scalar>::Real;
  static EIGEN_DEVICE_FUNC constexpr EIGEN_STRONG_INLINE Real run(const Derived& a) {
    return a.realView().cwiseAbs2().sum();
  }
};

template <typename Derived>
struct squared_norm_impl<Derived, bool> {
  static EIGEN_DEVICE_FUNC constexpr EIGEN_STRONG_INLINE bool run(const Derived& a) { return a.any(); }
};

template <typename T>
struct has_direct_data_and_stride {
  static constexpr bool value = bool(traits<T>::Flags & DirectAccessBit);
};

template <typename T, typename Enable = void>
struct clean_dot_xpr_helper {
  using type = T;
  static constexpr bool direct_access = has_direct_data_and_stride<T>::value;
  static constexpr bool linear_access = has_direct_data_and_stride<T>::value;
  static constexpr bool conj_lhs = true;
  static constexpr bool conj_rhs = false;
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE constexpr const T& get(const T& xpr) { return xpr; }
};

template <typename Scalar, typename Xpr>
struct clean_dot_xpr_helper<CwiseUnaryOp<internal::scalar_conjugate_op<Scalar>, Xpr>> {
  using type = Xpr;
  static constexpr bool direct_access = has_direct_data_and_stride<Xpr>::value;
  static constexpr bool linear_access = has_direct_data_and_stride<Xpr>::value;
  static constexpr bool conj_lhs = false;
  static constexpr bool conj_rhs = true;
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE constexpr const Xpr& get(const CwiseUnaryOp<internal::scalar_conjugate_op<Scalar>, Xpr>& xpr) { return xpr.nestedExpression(); }
};

template <typename Scalar, typename Xpr>
struct clean_dot_xpr_helper<const CwiseUnaryOp<internal::scalar_conjugate_op<Scalar>, Xpr>> {
  using type = Xpr;
  static constexpr bool direct_access = has_direct_data_and_stride<Xpr>::value;
  static constexpr bool linear_access = has_direct_data_and_stride<Xpr>::value;
  static constexpr bool conj_lhs = false;
  static constexpr bool conj_rhs = true;
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE constexpr const Xpr& get(const CwiseUnaryOp<internal::scalar_conjugate_op<Scalar>, Xpr>& xpr) { return xpr.nestedExpression(); }
};

template <typename PacketCplx, bool Vectorizable>
struct get_as_real {
  using type = PacketCplx;
};

template <typename PacketCplx>
struct get_as_real<PacketCplx, true> {
  using type = typename unpacket_traits<PacketCplx>::as_real;
};

template <typename LhsScalar, typename RhsScalar, bool ConjLhs, bool ConjRhs, typename Enable = void>
struct generic_dot_impl_helper {
  using ResultType = typename ScalarBinaryOpTraits<LhsScalar, RhsScalar>::ReturnType;
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE ResultType run(const LhsScalar* lhs, const RhsScalar* rhs, Index size) {
    ResultType res(0);
    for (Index i = 0; i < size; ++i) {
      if constexpr (ConjLhs && ConjRhs) {
        res += numext::conj(lhs[i]) * numext::conj(rhs[i]);
      } else if constexpr (ConjLhs) {
        res += numext::conj(lhs[i]) * rhs[i];
      } else if constexpr (ConjRhs) {
        res += lhs[i] * numext::conj(rhs[i]);
      } else {
        res += lhs[i] * rhs[i];
      }
    }
    return res;
  }
};

// Specialization for Same-Type Vectorizable (LhsScalar == RhsScalar)
template <typename LhsScalar, typename RhsScalar, bool ConjLhs, bool ConjRhs>
struct generic_dot_impl_helper<LhsScalar, RhsScalar, ConjLhs, ConjRhs,
    std::enable_if_t<internal::is_same<LhsScalar, RhsScalar>::value>> {
  using ResultType = LhsScalar;
  using Packet = typename packet_traits<ResultType>::type;
  static constexpr bool Vectorizable = packet_traits<ResultType>::Vectorizable &&
                                       packet_traits<ResultType>::HasAdd &&
                                       packet_traits<ResultType>::HasMul;
  static constexpr int PacketSize = unpacket_traits<Packet>::size;

  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE ResultType run(const LhsScalar* lhs, const RhsScalar* rhs, Index size) {
    if constexpr (Vectorizable) {
      if (size < PacketSize) {
        ResultType res(0);
        for (Index i = 0; i < size; ++i) {
          if constexpr (ConjLhs && ConjRhs) {
            res += numext::conj(lhs[i]) * numext::conj(rhs[i]);
          } else if constexpr (ConjLhs) {
            res += numext::conj(lhs[i]) * rhs[i];
          } else if constexpr (ConjRhs) {
            res += lhs[i] * numext::conj(rhs[i]);
          } else {
            res += lhs[i] * rhs[i];
          }
        }
        return res;
      }

      Index numPackets = size / PacketSize;
      Index quadPackets = numPackets / 4;
      Index remPackets = numPackets % 4;

      conj_helper<Packet, Packet, ConjLhs, ConjRhs> cj;
      Packet acc0, acc1, acc2, acc3;
      acc0 = cj.pmul(ploadu<Packet>(lhs + 0 * PacketSize), ploadu<Packet>(rhs + 0 * PacketSize));

      if (numPackets >= 2) {
        acc1 = cj.pmul(ploadu<Packet>(lhs + 1 * PacketSize), ploadu<Packet>(rhs + 1 * PacketSize));
      }
      if (numPackets >= 3) {
        acc2 = cj.pmul(ploadu<Packet>(lhs + 2 * PacketSize), ploadu<Packet>(rhs + 2 * PacketSize));
      }
      if (numPackets >= 4) {
        acc3 = cj.pmul(ploadu<Packet>(lhs + 3 * PacketSize), ploadu<Packet>(rhs + 3 * PacketSize));

        Index i = 4 * PacketSize;
        for (Index q = 1; q < quadPackets; ++q) {
          acc0 = cj.pmadd(ploadu<Packet>(lhs + i + 0 * PacketSize), ploadu<Packet>(rhs + i + 0 * PacketSize), acc0);
          acc1 = cj.pmadd(ploadu<Packet>(lhs + i + 1 * PacketSize), ploadu<Packet>(rhs + i + 1 * PacketSize), acc1);
          acc2 = cj.pmadd(ploadu<Packet>(lhs + i + 2 * PacketSize), ploadu<Packet>(rhs + i + 2 * PacketSize), acc2);
          acc3 = cj.pmadd(ploadu<Packet>(lhs + i + 3 * PacketSize), ploadu<Packet>(rhs + i + 3 * PacketSize), acc3);
          i += 4 * PacketSize;
        }

        if (remPackets >= 1) {
          acc0 = cj.pmadd(ploadu<Packet>(lhs + i + 0 * PacketSize), ploadu<Packet>(rhs + i + 0 * PacketSize), acc0);
        }
        if (remPackets >= 2) {
          acc1 = cj.pmadd(ploadu<Packet>(lhs + i + 1 * PacketSize), ploadu<Packet>(rhs + i + 1 * PacketSize), acc1);
        }
        if (remPackets >= 3) {
          acc2 = cj.pmadd(ploadu<Packet>(lhs + i + 2 * PacketSize), ploadu<Packet>(rhs + i + 2 * PacketSize), acc2);
        }

        acc2 = padd(acc2, acc3);
      }

      if (numPackets >= 3) acc1 = padd(acc1, acc2);
      if (numPackets >= 2) acc0 = padd(acc0, acc1);

      ResultType res = predux(acc0);
      Index packet_end = numPackets * PacketSize;
      for (Index i = packet_end; i < size; ++i) {
        if constexpr (ConjLhs && ConjRhs) {
          res += numext::conj(lhs[i]) * numext::conj(rhs[i]);
        } else if constexpr (ConjLhs) {
          res += numext::conj(lhs[i]) * rhs[i];
        } else if constexpr (ConjRhs) {
          res += lhs[i] * numext::conj(rhs[i]);
        } else {
          res += lhs[i] * rhs[i];
        }
      }
      return res;
    }
    ResultType res(0);
    for (Index i = 0; i < size; ++i) {
      if constexpr (ConjLhs && ConjRhs) {
        res += numext::conj(lhs[i]) * numext::conj(rhs[i]);
      } else if constexpr (ConjLhs) {
        res += numext::conj(lhs[i]) * rhs[i];
      } else if constexpr (ConjRhs) {
        res += lhs[i] * numext::conj(rhs[i]);
      } else {
        res += lhs[i] * rhs[i];
      }
    }
    return res;
  }
};

// Specialization for Real LHS and Complex RHS
template <typename LhsScalar, typename RhsScalar, bool ConjLhs, bool ConjRhs>
struct generic_dot_impl_helper<LhsScalar, RhsScalar, ConjLhs, ConjRhs,
    std::enable_if_t<!NumTraits<LhsScalar>::IsComplex && NumTraits<RhsScalar>::IsComplex>> {
  using ResultType = RhsScalar;
  using PacketCplx = typename packet_traits<ResultType>::type;
  static constexpr bool Vectorizable = packet_traits<ResultType>::Vectorizable &&
                                       packet_traits<ResultType>::HasAdd &&
                                       packet_traits<ResultType>::HasMul;
  using PacketReal = typename get_as_real<PacketCplx, Vectorizable>::type;
  static constexpr int PacketSize = unpacket_traits<PacketCplx>::size;

  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE ResultType run(const LhsScalar* lhs, const RhsScalar* rhs, Index size) {
    if constexpr (Vectorizable) {
      if (size < PacketSize) {
        ResultType res(0);
        for (Index i = 0; i < size; ++i) {
          if constexpr (ConjRhs) {
            res += lhs[i] * numext::conj(rhs[i]);
          } else {
            res += lhs[i] * rhs[i];
          }
        }
        return res;
      }

      Index numPackets = size / PacketSize;
      Index quadPackets = numPackets / 4;
      Index remPackets = numPackets % 4;

      conj_helper<PacketReal, PacketCplx, false, false> cj;
      auto load_rhs = [](const RhsScalar* ptr) EIGEN_LAMBDA_ALWAYS_INLINE {
        if constexpr (ConjRhs) {
          return pconj(ploadu<PacketCplx>(ptr));
        } else {
          return ploadu<PacketCplx>(ptr);
        }
      };

      PacketCplx acc0, acc1, acc2, acc3;
      acc0 = cj.pmul(ploaddup<PacketReal>(lhs + 0 * PacketSize), load_rhs(rhs + 0 * PacketSize));

      if (numPackets >= 2) {
        acc1 = cj.pmul(ploaddup<PacketReal>(lhs + 1 * PacketSize), load_rhs(rhs + 1 * PacketSize));
      }
      if (numPackets >= 3) {
        acc2 = cj.pmul(ploaddup<PacketReal>(lhs + 2 * PacketSize), load_rhs(rhs + 2 * PacketSize));
      }
      if (numPackets >= 4) {
        acc3 = cj.pmul(ploaddup<PacketReal>(lhs + 3 * PacketSize), load_rhs(rhs + 3 * PacketSize));

        Index i = 4 * PacketSize;
        for (Index q = 1; q < quadPackets; ++q) {
          acc0 = cj.pmadd(ploaddup<PacketReal>(lhs + i + 0 * PacketSize), load_rhs(rhs + i + 0 * PacketSize), acc0);
          acc1 = cj.pmadd(ploaddup<PacketReal>(lhs + i + 1 * PacketSize), load_rhs(rhs + i + 1 * PacketSize), acc1);
          acc2 = cj.pmadd(ploaddup<PacketReal>(lhs + i + 2 * PacketSize), load_rhs(rhs + i + 2 * PacketSize), acc2);
          acc3 = cj.pmadd(ploaddup<PacketReal>(lhs + i + 3 * PacketSize), load_rhs(rhs + i + 3 * PacketSize), acc3);
          i += 4 * PacketSize;
        }

        if (remPackets >= 1) {
          acc0 = cj.pmadd(ploaddup<PacketReal>(lhs + i + 0 * PacketSize), load_rhs(rhs + i + 0 * PacketSize), acc0);
        }
        if (remPackets >= 2) {
          acc1 = cj.pmadd(ploaddup<PacketReal>(lhs + i + 1 * PacketSize), load_rhs(rhs + i + 1 * PacketSize), acc1);
        }
        if (remPackets >= 3) {
          acc2 = cj.pmadd(ploaddup<PacketReal>(lhs + i + 2 * PacketSize), load_rhs(rhs + i + 2 * PacketSize), acc2);
        }

        acc2 = padd(acc2, acc3);
      }

      if (numPackets >= 3) acc1 = padd(acc1, acc2);
      if (numPackets >= 2) acc0 = padd(acc0, acc1);

      ResultType res = predux(acc0);
      Index packet_end = numPackets * PacketSize;
      for (Index i = packet_end; i < size; ++i) {
        if constexpr (ConjRhs) {
          res += lhs[i] * numext::conj(rhs[i]);
        } else {
          res += lhs[i] * rhs[i];
        }
      }
      return res;
    }
    ResultType res(0);
    for (Index i = 0; i < size; ++i) {
      if constexpr (ConjRhs) {
        res += lhs[i] * numext::conj(rhs[i]);
      } else {
        res += lhs[i] * rhs[i];
      }
    }
    return res;
  }
};

// Specialization for Complex LHS and Real RHS
template <typename LhsScalar, typename RhsScalar, bool ConjLhs, bool ConjRhs>
struct generic_dot_impl_helper<LhsScalar, RhsScalar, ConjLhs, ConjRhs,
    std::enable_if_t<NumTraits<LhsScalar>::IsComplex && !NumTraits<RhsScalar>::IsComplex>> {
  using ResultType = LhsScalar;
  using PacketCplx = typename packet_traits<ResultType>::type;
  static constexpr bool Vectorizable = packet_traits<ResultType>::Vectorizable &&
                                       packet_traits<ResultType>::HasAdd &&
                                       packet_traits<ResultType>::HasMul;
  using PacketReal = typename get_as_real<PacketCplx, Vectorizable>::type;
  static constexpr int PacketSize = unpacket_traits<PacketCplx>::size;

  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE ResultType run(const LhsScalar* lhs, const RhsScalar* rhs, Index size) {
    if constexpr (Vectorizable) {
      if (size < PacketSize) {
        ResultType res(0);
        for (Index i = 0; i < size; ++i) {
          if constexpr (ConjLhs) {
            res += numext::conj(lhs[i]) * rhs[i];
          } else {
            res += lhs[i] * rhs[i];
          }
        }
        return res;
      }

      Index numPackets = size / PacketSize;
      Index quadPackets = numPackets / 4;
      Index remPackets = numPackets % 4;

      conj_helper<PacketCplx, PacketReal, false, false> cj;
      auto load_lhs = [](const LhsScalar* ptr) EIGEN_LAMBDA_ALWAYS_INLINE {
        if constexpr (ConjLhs) {
          return pconj(ploadu<PacketCplx>(ptr));
        } else {
          return ploadu<PacketCplx>(ptr);
        }
      };

      PacketCplx acc0, acc1, acc2, acc3;
      acc0 = cj.pmul(load_lhs(lhs + 0 * PacketSize), ploaddup<PacketReal>(rhs + 0 * PacketSize));

      if (numPackets >= 2) {
        acc1 = cj.pmul(load_lhs(lhs + 1 * PacketSize), ploaddup<PacketReal>(rhs + 1 * PacketSize));
      }
      if (numPackets >= 3) {
        acc2 = cj.pmul(load_lhs(lhs + 2 * PacketSize), ploaddup<PacketReal>(rhs + 2 * PacketSize));
      }
      if (numPackets >= 4) {
        acc3 = cj.pmul(load_lhs(lhs + 3 * PacketSize), ploaddup<PacketReal>(rhs + 3 * PacketSize));

        Index i = 4 * PacketSize;
        for (Index q = 1; q < quadPackets; ++q) {
          acc0 = cj.pmadd(load_lhs(lhs + i + 0 * PacketSize), ploaddup<PacketReal>(rhs + i + 0 * PacketSize), acc0);
          acc1 = cj.pmadd(load_lhs(lhs + i + 1 * PacketSize), ploaddup<PacketReal>(rhs + i + 1 * PacketSize), acc1);
          acc2 = cj.pmadd(load_lhs(lhs + i + 2 * PacketSize), ploaddup<PacketReal>(rhs + i + 2 * PacketSize), acc2);
          acc3 = cj.pmadd(load_lhs(lhs + i + 3 * PacketSize), ploaddup<PacketReal>(rhs + i + 3 * PacketSize), acc3);
          i += 4 * PacketSize;
        }

        if (remPackets >= 1) {
          acc0 = cj.pmadd(load_lhs(lhs + i + 0 * PacketSize), ploaddup<PacketReal>(rhs + i + 0 * PacketSize), acc0);
        }
        if (remPackets >= 2) {
          acc1 = cj.pmadd(load_lhs(lhs + i + 1 * PacketSize), ploaddup<PacketReal>(rhs + i + 1 * PacketSize), acc1);
        }
        if (remPackets >= 3) {
          acc2 = cj.pmadd(load_lhs(lhs + i + 2 * PacketSize), ploaddup<PacketReal>(rhs + i + 2 * PacketSize), acc2);
        }

        acc2 = padd(acc2, acc3);
      }

      if (numPackets >= 3) acc1 = padd(acc1, acc2);
      if (numPackets >= 2) acc0 = padd(acc0, acc1);

      ResultType res = predux(acc0);
      Index packet_end = numPackets * PacketSize;
      for (Index i = packet_end; i < size; ++i) {
        if constexpr (ConjLhs) {
          res += numext::conj(lhs[i]) * rhs[i];
        } else {
          res += lhs[i] * rhs[i];
        }
      }
      return res;
    }
    ResultType res(0);
    for (Index i = 0; i < size; ++i) {
      if constexpr (ConjLhs) {
        res += numext::conj(lhs[i]) * rhs[i];
      } else {
        res += lhs[i] * rhs[i];
      }
    }
    return res;
  }
};

}  // end namespace internal

/** \fn MatrixBase::dot
 * \returns the dot product of *this with other.
 *
 * \only_for_vectors
 *
 * \note If the scalar type is complex numbers, then this function returns the hermitian
 * (sesquilinear) dot product, conjugate-linear in the first variable and linear in the
 * second variable.
 *
 * \sa squaredNorm(), norm()
 */
template <typename Derived>
template <typename OtherDerived>
EIGEN_DEVICE_FUNC constexpr EIGEN_STRONG_INLINE
    typename ScalarBinaryOpTraits<typename internal::traits<Derived>::Scalar,
                                  typename internal::traits<OtherDerived>::Scalar>::ReturnType
    MatrixBase<Derived>::dot(const MatrixBase<OtherDerived>& other) const {
  using LhsScalar = typename internal::traits<Derived>::Scalar;
  using RhsScalar = typename internal::traits<OtherDerived>::Scalar;
  if constexpr (internal::clean_dot_xpr_helper<Derived>::direct_access &&
                internal::clean_dot_xpr_helper<OtherDerived>::direct_access &&
                internal::clean_dot_xpr_helper<Derived>::linear_access &&
                internal::clean_dot_xpr_helper<OtherDerived>::linear_access &&
                internal::inner_stride_at_compile_time<typename internal::clean_dot_xpr_helper<Derived>::type>::value != 1) {
    const auto& lhs_clean = internal::clean_dot_xpr_helper<Derived>::get(derived());
    const auto& rhs_clean = internal::clean_dot_xpr_helper<OtherDerived>::get(other.derived());
    if (lhs_clean.innerStride() == 1 && rhs_clean.innerStride() == 1 &&
        (bool(Derived::IsVectorAtCompileTime) || lhs_clean.rows() == 1 || lhs_clean.cols() == 1) &&
        (bool(OtherDerived::IsVectorAtCompileTime) || rhs_clean.rows() == 1 || rhs_clean.cols() == 1)) {
      return internal::generic_dot_impl_helper<LhsScalar, RhsScalar, internal::clean_dot_xpr_helper<Derived>::conj_lhs,
                                               internal::clean_dot_xpr_helper<OtherDerived>::conj_rhs>::run(
          lhs_clean.data(), rhs_clean.data(), lhs_clean.size());
    }
  }
  return internal::dot_impl<Derived, OtherDerived>::run(derived(), other.derived());
}

//---------- implementation of L2 norm and related functions ----------

/** \returns, for vectors, the squared \em l2 norm of \c *this, and for matrices the squared Frobenius norm.
 * In both cases, it consists in the sum of the square of all the matrix entries.
 * For vectors, this is also equal to the dot product of \c *this with itself.
 *
 * \sa dot(), norm(), lpNorm()
 */
template <typename Derived>
EIGEN_DEVICE_FUNC constexpr EIGEN_STRONG_INLINE typename NumTraits<typename internal::traits<Derived>::Scalar>::Real
MatrixBase<Derived>::squaredNorm() const {
  return internal::squared_norm_impl<Derived>::run(derived());
}

/** \returns, for vectors, the \em l2 norm of \c *this, and for matrices the Frobenius norm.
 * In both cases, it consists in the square root of the sum of the square of all the matrix entries.
 * For vectors, this is also equal to the square root of the dot product of \c *this with itself.
 *
 * \sa lpNorm(), dot(), squaredNorm()
 */
template <typename Derived>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE typename NumTraits<typename internal::traits<Derived>::Scalar>::Real
MatrixBase<Derived>::norm() const {
  return numext::sqrt(squaredNorm());
}

/** \returns an expression of the quotient of \c *this by its own norm.
 *
 * \warning If the input vector is too small (i.e., this->norm()==0),
 *          then this function returns a copy of the input.
 *
 * \only_for_vectors
 *
 * \sa norm(), normalize()
 */
template <typename Derived>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const typename MatrixBase<Derived>::PlainObject MatrixBase<Derived>::normalized()
    const {
  typedef typename internal::nested_eval<Derived, 2>::type Nested_;
  Nested_ n(derived());
  RealScalar z = n.squaredNorm();
  // NOTE: after extensive benchmarking, this conditional does not impact performance, at least on recent x86 CPU
  if (z > RealScalar(0))
    return n / numext::sqrt(z);
  else
    return n;
}

/** Normalizes the vector, i.e. divides it by its own norm.
 *
 * \only_for_vectors
 *
 * \warning If the input vector is too small (i.e., this->norm()==0), then \c *this is left unchanged.
 *
 * \sa norm(), normalized()
 */
template <typename Derived>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void MatrixBase<Derived>::normalize() {
  RealScalar z = squaredNorm();
  // NOTE: after extensive benchmarking, this conditional does not impact performance, at least on recent x86 CPU
  if (z > RealScalar(0)) derived() /= numext::sqrt(z);
}

/** \returns an expression of the quotient of \c *this by its own norm while avoiding underflow and overflow.
 *
 * \only_for_vectors
 *
 * This method is analogue to the normalized() method, but it reduces the risk of
 * underflow and overflow when computing the norm.
 *
 * \warning If the input vector is too small (i.e., this->norm()==0),
 *          then this function returns a copy of the input.
 *
 * \sa stableNorm(), stableNormalize(), normalized()
 */
template <typename Derived>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const typename MatrixBase<Derived>::PlainObject
MatrixBase<Derived>::stableNormalized() const {
  typedef typename internal::nested_eval<Derived, 3>::type Nested_;
  Nested_ n(derived());
  RealScalar w = n.cwiseAbs().maxCoeff();
  RealScalar z = (n / w).squaredNorm();
  if (z > RealScalar(0))
    return n / (numext::sqrt(z) * w);
  else
    return n;
}

/** Normalizes the vector while avoid underflow and overflow
 *
 * \only_for_vectors
 *
 * This method is analogue to the normalize() method, but it reduces the risk of
 * underflow and overflow when computing the norm.
 *
 * \warning If the input vector is too small (i.e., this->norm()==0), then \c *this is left unchanged.
 *
 * \sa stableNorm(), stableNormalized(), normalize()
 */
template <typename Derived>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void MatrixBase<Derived>::stableNormalize() {
  RealScalar w = cwiseAbs().maxCoeff();
  RealScalar z = (derived() / w).squaredNorm();
  if (z > RealScalar(0)) derived() /= numext::sqrt(z) * w;
}

//---------- implementation of other norms ----------

namespace internal {

template <typename Derived, int p>
struct lpNorm_selector {
  typedef typename NumTraits<typename traits<Derived>::Scalar>::Real RealScalar;
  EIGEN_DEVICE_FUNC static inline RealScalar run(const MatrixBase<Derived>& m) {
    EIGEN_USING_STD(pow)
    return pow(m.cwiseAbs().array().pow(p).sum(), RealScalar(1) / p);
  }
};

template <typename Derived>
struct lpNorm_selector<Derived, 1> {
  EIGEN_DEVICE_FUNC static inline typename NumTraits<typename traits<Derived>::Scalar>::Real run(
      const MatrixBase<Derived>& m) {
    return m.cwiseAbs().sum();
  }
};

template <typename Derived>
struct lpNorm_selector<Derived, 2> {
  EIGEN_DEVICE_FUNC static inline typename NumTraits<typename traits<Derived>::Scalar>::Real run(
      const MatrixBase<Derived>& m) {
    return m.norm();
  }
};

template <typename Derived>
struct lpNorm_selector<Derived, Infinity> {
  typedef typename NumTraits<typename traits<Derived>::Scalar>::Real RealScalar;
  EIGEN_DEVICE_FUNC static inline RealScalar run(const MatrixBase<Derived>& m) {
    if (Derived::SizeAtCompileTime == 0 || (Derived::SizeAtCompileTime == Dynamic && m.size() == 0))
      return RealScalar(0);
    return m.cwiseAbs().maxCoeff();
  }
};

}  // end namespace internal

/** \returns the \b coefficient-wise \f$ \ell^p \f$ norm of \c *this, that is, returns the p-th root of the sum of the
 * p-th powers of the absolute values of the coefficients of \c *this. If \a p is the special value \a Eigen::Infinity,
 * this function returns the \f$ \ell^\infty \f$ norm, that is the maximum of the absolute values of the coefficients of
 * \c *this.
 *
 * In all cases, if \c *this is empty, then the value 0 is returned.
 *
 * \note For matrices, this function does not compute the <a
 * href="https://en.wikipedia.org/wiki/Operator_norm">operator-norm</a>. That is, if \c *this is a matrix, then its
 * coefficients are interpreted as a 1D vector. Nonetheless, you can easily compute the 1-norm and \f$\infty\f$-norm
 * matrix operator norms using \link TutorialReductionsVisitorsBroadcastingReductionsNorm partial reductions \endlink.
 *
 * \sa norm()
 */
template <typename Derived>
template <int p>
#ifndef EIGEN_PARSED_BY_DOXYGEN
EIGEN_DEVICE_FUNC inline typename NumTraits<typename internal::traits<Derived>::Scalar>::Real
#else
EIGEN_DEVICE_FUNC MatrixBase<Derived>::RealScalar
#endif
MatrixBase<Derived>::lpNorm() const {
  return internal::lpNorm_selector<Derived, p>::run(*this);
}

//---------- implementation of isOrthogonal / isUnitary ----------

/** \returns true if *this is approximately orthogonal to \a other,
 *          within the precision given by \a prec.
 *
 * Example: \include MatrixBase_isOrthogonal.cpp
 * Output: \verbinclude MatrixBase_isOrthogonal.out
 */
template <typename Derived>
template <typename OtherDerived>
bool MatrixBase<Derived>::isOrthogonal(const MatrixBase<OtherDerived>& other, const RealScalar& prec) const {
  typename internal::nested_eval<Derived, 2>::type nested(derived());
  typename internal::nested_eval<OtherDerived, 2>::type otherNested(other.derived());
  return numext::abs2(nested.dot(otherNested)) <= prec * prec * nested.squaredNorm() * otherNested.squaredNorm();
}

/** \returns true if *this is approximately an unitary matrix,
 *          within the precision given by \a prec. In the case where the \a Scalar
 *          type is real numbers, a unitary matrix is an orthogonal matrix, whence the name.
 *
 * \note This can be used to check whether a family of vectors forms an orthonormal basis.
 *       Indeed, \c m.isUnitary() returns true if and only if the columns (equivalently, the rows) of m form an
 *       orthonormal basis.
 *
 * Example: \include MatrixBase_isUnitary.cpp
 * Output: \verbinclude MatrixBase_isUnitary.out
 */
template <typename Derived>
bool MatrixBase<Derived>::isUnitary(const RealScalar& prec) const {
  typename internal::nested_eval<Derived, 1>::type self(derived());
  for (Index i = 0; i < cols(); ++i) {
    if (!internal::isApprox(self.col(i).squaredNorm(), static_cast<RealScalar>(1), prec)) return false;
    for (Index j = 0; j < i; ++j)
      if (!internal::isMuchSmallerThan(self.col(i).dot(self.col(j)), static_cast<Scalar>(1), prec)) return false;
  }
  return true;
}

}  // end namespace Eigen

#endif  // EIGEN_DOT_H
