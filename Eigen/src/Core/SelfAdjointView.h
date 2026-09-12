// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_SELFADJOINTMATRIX_H
#define EIGEN_SELFADJOINTMATRIX_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

/** \class SelfAdjointView
 * \ingroup Core_Module
 *
 *
 * \brief Expression of a selfadjoint matrix from a triangular part of a dense matrix
 *
 * \tparam MatrixType the type of the dense matrix storing the coefficients
 * \tparam TriangularPart can be either \c #Lower or \c #Upper
 *
 * This class is an expression of a selfadjoint matrix from a triangular part of a matrix
 * with given dense storage of the coefficients. It is the return type of MatrixBase::selfadjointView()
 * and most of the time this is the only way that it is used.
 *
 * \sa class TriangularBase, MatrixBase::selfadjointView()
 */

namespace internal {

// The column step of the self-adjoint 1-norm: sums[i] += |x_i|, returning the sum of what was
// added, in a single packet pass so that short columns do not pay an expression setup per pass.
// The accumulator has the matrix's scalar type and the column sums are read from the real parts.
// Real scalars go through pabs. std::complex has no packet abs (the functor framework cannot turn
// a complex packet into a real one), so its lanes take sqrt(re^2 + im^2) on the real lanes of the
// complex packet, which leaves |z| in both lanes of its slot: that is the layout of a complex
// accumulator, so it is added as is rather than compressed, which would need an arch-specific
// shuffle.
template <typename Scalar_>
struct selfadjoint_l1norm_real_lanes {
  using Scalar = Scalar_;
  using Real = Scalar_;
  using Packet = typename packet_traits<Scalar>::type;
  using RPacket = Packet;
  static constexpr Index PacketSize = unpacket_traits<Packet>::size;
  // Accumulating a column into the sums of the later ones chains the shortest columns on
  // store-to-load forwarding; up to here reading the mirrored term as a row is cheaper.
  static constexpr Index PerColumnUpTo = 16;
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE RPacket lanes(const Packet& p) { return p; }
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet pack(const RPacket& r) { return r; }
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE RPacket abs(const RPacket& v) { return pabs(v); }
  static EIGEN_DEVICE_FUNC bool isReliable(Real, Index) { return true; }
};

template <typename T>
struct selfadjoint_l1norm_complex_lanes {
  using Scalar = std::complex<T>;
  using Real = T;
  using Packet = typename packet_traits<Scalar>::type;
  using RPacket = typename unpacket_traits<Packet>::as_real;
  static constexpr Index PacketSize = unpacket_traits<Packet>::size;
  static constexpr Index PerColumnUpTo = 0;
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE RPacket lanes(const Packet& p) { return p.v; }
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet pack(const RPacket& r) { return Packet(r); }
  // |z| in both lanes of its slot.
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE RPacket abs(const RPacket& v) {
    RPacket s = pmul(v, v);
    s = padd(s, lanes(pcplxflip(pack(s))));
    return psqrt(s);
  }
  // Squaring the parts overflows and underflows well inside the scalar range: an infinite or a
  // tiny result (elements below sqrt(min) lose precision) is recomputed through numext::abs.
  static EIGEN_DEVICE_FUNC bool isReliable(Real norm, Index n) {
    const Real tiny = Real(n) * numext::sqrt((std::numeric_limits<Real>::min)()) / NumTraits<Real>::epsilon();
    return norm > tiny && (numext::isfinite)(norm);
  }
};

template <typename Lanes>
struct selfadjoint_l1norm_packet_impl : Lanes {
  using Lanes::PacketSize;
  using typename Lanes::Packet;
  using typename Lanes::Real;
  using typename Lanes::RPacket;
  using typename Lanes::Scalar;

  template <typename Derived>
  static EIGEN_DEVICE_FUNC Real accumulate(Scalar* sums, const DenseBase<Derived>& x) {
    using XprEvaluator = evaluator<Derived>;
    constexpr bool Vectorize =
        bool(XprEvaluator::Flags & PacketAccessBit) && bool(XprEvaluator::Flags & LinearAccessBit);
    return accumulate(sums, XprEvaluator(x.derived()), x.size(), bool_constant<Vectorize>());
  }

 private:
  template <typename XprEvaluator>
  static EIGEN_DEVICE_FUNC Real accumulate(Scalar* s, const XprEvaluator& x, Index n, std::false_type) {
    Real r = Real(0);
    for (Index i = 0; i < n; ++i) {
      Real a = numext::abs(x.coeff(i));
      s[i] += a;
      r += a;
    }
    return r;
  }
  template <typename XprEvaluator>
  static EIGEN_DEVICE_FUNC Real accumulate(Scalar* s, const XprEvaluator& x, Index n, std::true_type) {
    RPacket acc = pzero(RPacket());
    Index i = 0;
    for (; i + PacketSize <= n; i += PacketSize) {
      RPacket a = Lanes::abs(Lanes::lanes(x.template packet<Unaligned, Packet>(i)));
      acc = padd(acc, a);
      pstoreu(s + i, Lanes::pack(padd(Lanes::lanes(ploadu<Packet>(s + i)), a)));
    }
    Real r = i > 0 ? numext::real(predux(Lanes::pack(acc))) : Real(0);
    for (; i < n; ++i) {
      Real a = numext::abs(x.coeff(i));
      s[i] += a;
      r += a;
    }
    return r;
  }
};

// Scalar fallback for types without a packet path: custom complex types, or complex packets
// without a real view.
template <typename Scalar, typename Enable = void>
struct selfadjoint_l1norm_impl {
  using Real = typename NumTraits<Scalar>::Real;
  static constexpr Index PerColumnUpTo = 16;
  template <typename Derived>
  static EIGEN_DEVICE_FUNC Real accumulate(Scalar* sums, const DenseBase<Derived>& x) {
    Real r = Real(0);
    for (Index i = 0; i < x.size(); ++i) {
      Real a = numext::abs(x.coeff(i));
      sums[i] += Scalar(a);
      r += a;
    }
    return r;
  }
  static EIGEN_DEVICE_FUNC bool isReliable(Real, Index) { return true; }
};
template <typename Scalar>
struct selfadjoint_l1norm_impl<Scalar, std::enable_if_t<!NumTraits<Scalar>::IsComplex>>
    : selfadjoint_l1norm_packet_impl<selfadjoint_l1norm_real_lanes<Scalar>> {};
template <typename T>
struct selfadjoint_l1norm_impl<std::complex<T>,
                               void_t<typename unpacket_traits<typename packet_traits<std::complex<T>>::type>::as_real>>
    : selfadjoint_l1norm_packet_impl<selfadjoint_l1norm_complex_lanes<T>> {};

template <typename MatrixType, unsigned int UpLo>
struct traits<SelfAdjointView<MatrixType, UpLo>> : traits<MatrixType> {
  using MatrixTypeNested = typename ref_selector<MatrixType>::non_const_type;
  using MatrixTypeNestedCleaned = remove_all_t<MatrixTypeNested>;
  using ExpressionType = MatrixType;
  using FullMatrixType = typename MatrixType::PlainObject;
  enum {
    Mode = UpLo | SelfAdjoint,
    FlagsLvalueBit = is_lvalue<MatrixType>::value ? LvalueBit : 0,
    Flags = MatrixTypeNestedCleaned::Flags & (HereditaryBits | FlagsLvalueBit) &
            (~(PacketAccessBit | DirectAccessBit | LinearAccessBit))  // FIXME these flags should be preserved
  };
};

}  // namespace internal

template <typename MatrixType_, unsigned int UpLo>
class SelfAdjointView : public TriangularBase<SelfAdjointView<MatrixType_, UpLo>> {
 public:
  EIGEN_STATIC_ASSERT(UpLo == Lower || UpLo == Upper, SELFADJOINTVIEW_ACCEPTS_UPPER_AND_LOWER_MODE_ONLY)

  using MatrixType = MatrixType_;
  using Base = TriangularBase<SelfAdjointView>;
  using MatrixTypeNested = typename internal::traits<SelfAdjointView>::MatrixTypeNested;
  using MatrixTypeNestedCleaned = typename internal::traits<SelfAdjointView>::MatrixTypeNestedCleaned;
  using NestedExpression = MatrixTypeNestedCleaned;

  /** \brief The type of coefficients in this matrix */
  using Scalar = typename internal::traits<SelfAdjointView>::Scalar;
  /** Real part of #Scalar */
  using RealScalar = typename NumTraits<Scalar>::Real;
  using StorageIndex = typename MatrixType::StorageIndex;

  enum {
    Mode = internal::traits<SelfAdjointView>::Mode,
    Flags = internal::traits<SelfAdjointView>::Flags,
    TransposeMode = ((int(Mode) & int(Upper)) ? Lower : 0) | ((int(Mode) & int(Lower)) ? Upper : 0)
  };
  using PlainObject = typename MatrixType::PlainObject;

  EIGEN_DEVICE_FUNC explicit inline SelfAdjointView(MatrixType& matrix) : m_matrix(matrix) {}
  using Base::operator*;
  EIGEN_DEFAULT_COPY_CONSTRUCTOR(SelfAdjointView)

  /** Assigns a matrix expression to the referenced triangular part of the selfadjoint matrix. */
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC SelfAdjointView& operator=(const MatrixBase<OtherDerived>& other) {
    m_matrix.template triangularView<UpLo>() = other;
    return *this;
  }

  /** Assigns a triangular or selfadjoint expression without materializing a dense temporary. */
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC SelfAdjointView& operator=(const TriangularBase<OtherDerived>& other) {
    other.evalToLazy(m_matrix);
    return *this;
  }

  EIGEN_DEVICE_FUNC SelfAdjointView& operator=(const SelfAdjointView& other) {
    return *this = static_cast<const Base&>(other);
  }

  /** \sa MatrixBase::operator+=() */
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC SelfAdjointView& operator+=(const DenseBase<OtherDerived>& other) {
    m_matrix.template triangularView<UpLo>() += other;
    return *this;
  }

  /** \sa MatrixBase::operator-=() */
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC SelfAdjointView& operator-=(const DenseBase<OtherDerived>& other) {
    m_matrix.template triangularView<UpLo>() -= other;
    return *this;
  }

  /** \sa MatrixBase::operator*=() */
  EIGEN_DEVICE_FUNC SelfAdjointView& operator*=(const Scalar& other) {
    eigen_assert(numext::imag(other) == typename NumTraits<Scalar>::Real(0) &&
                 "SelfAdjointView in-place scaling requires a real scalar; "
                 "scaling only the stored triangle by a non-real scalar would "
                 "leave conj(other) on the unstored half.");
    m_matrix.template triangularView<UpLo>() *= other;
    return *this;
  }

  /** \sa DenseBase::operator/=() */
  EIGEN_DEVICE_FUNC SelfAdjointView& operator/=(const Scalar& other) {
    eigen_assert(numext::imag(other) == typename NumTraits<Scalar>::Real(0) &&
                 "SelfAdjointView in-place division requires a real scalar; "
                 "dividing only the stored triangle by a non-real scalar would "
                 "leave conj(other) on the unstored half.");
    m_matrix.template triangularView<UpLo>() /= other;
    return *this;
  }

  /** \internal */
  EIGEN_DEVICE_FUNC constexpr const MatrixTypeNestedCleaned& _expression() const noexcept { return m_matrix; }

  EIGEN_DEVICE_FUNC constexpr const MatrixTypeNestedCleaned& nestedExpression() const noexcept { return m_matrix; }
  EIGEN_DEVICE_FUNC constexpr MatrixTypeNestedCleaned& nestedExpression() noexcept { return m_matrix; }

  EIGEN_DEVICE_FUNC const SelfAdjointView<
      const EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE(MatrixType, Scalar, internal::scalar_product_op), UpLo>
  operator*(const Scalar& s) const {
    return (nestedExpression() * s).template selfadjointView<UpLo>();
  }

  friend EIGEN_DEVICE_FUNC const SelfAdjointView<
      const EIGEN_SCALAR_BINARYOP_EXPR_RETURN_TYPE(Scalar, MatrixType, internal::scalar_product_op), UpLo>
  operator*(const Scalar& s, const SelfAdjointView& mat) {
    return (s * mat.nestedExpression()).template selfadjointView<UpLo>();
  }

  /** Perform a symmetric rank 2 update of the selfadjoint matrix \c *this:
   * \f$ this = this + \alpha u v^* + conj(\alpha) v u^* \f$
   * \returns a reference to \c *this
   *
   * The vectors \a u and \c v \b must be column vectors, however they can be
   * an adjoint expression without any overhead. Only the meaningful triangular
   * part of the matrix is updated, the rest is left unchanged.
   *
   * \sa rankUpdate(const MatrixBase<DerivedU>&, Scalar)
   */
  template <typename DerivedU, typename DerivedV>
  EIGEN_DEVICE_FUNC SelfAdjointView& rankUpdate(const MatrixBase<DerivedU>& u, const MatrixBase<DerivedV>& v,
                                                const Scalar& alpha = Scalar(1));

  /** Perform a symmetric rank K update of the selfadjoint matrix \c *this:
   * \f$ this = this + \alpha ( u u^* ) \f$ where \a u is a vector or matrix.
   *
   * \returns a reference to \c *this
   *
   * Note that to perform \f$ this = this + \alpha ( u^* u ) \f$ you can simply
   * call this function with u.adjoint().
   *
   * \sa rankUpdate(const MatrixBase<DerivedU>&, const MatrixBase<DerivedV>&, Scalar)
   */
  template <typename DerivedU>
  EIGEN_DEVICE_FUNC SelfAdjointView& rankUpdate(const MatrixBase<DerivedU>& u, const Scalar& alpha = Scalar(1));

  /** \returns an expression of a triangular view extracted from the current selfadjoint view of a given triangular part
   *
   * The parameter \a TriMode can have the following values: \c #Upper, \c #StrictlyUpper, \c #UnitUpper,
   * \c #Lower, \c #StrictlyLower, \c #UnitLower.
   *
   * If \c TriMode references the same triangular part than \c *this, then this method simply return a \c TriangularView
   * of the nested expression, otherwise, the nested expression is first transposed, thus returning a \c
   * TriangularView<Transpose<MatrixType>> object.
   *
   * \sa MatrixBase::triangularView(), class TriangularView
   */
  template <unsigned int TriMode>
  EIGEN_DEVICE_FUNC
  std::conditional_t<(TriMode & (Upper | Lower)) == (UpLo & (Upper | Lower)), TriangularView<MatrixType, TriMode>,
                     TriangularView<typename MatrixType::AdjointReturnType, TriMode>>
  triangularView() const {
    std::conditional_t<(TriMode & (Upper | Lower)) == (UpLo & (Upper | Lower)), MatrixType&,
                       typename MatrixType::ConstTransposeReturnType>
        tmp1(m_matrix);
    std::conditional_t<(TriMode & (Upper | Lower)) == (UpLo & (Upper | Lower)), MatrixType&,
                       typename MatrixType::AdjointReturnType>
        tmp2(tmp1);
    return std::conditional_t<(TriMode & (Upper | Lower)) == (UpLo & (Upper | Lower)),
                              TriangularView<MatrixType, TriMode>,
                              TriangularView<typename MatrixType::AdjointReturnType, TriMode>>(tmp2);
  }

  /** \returns a const expression of the main diagonal of the matrix \c *this
   *
   * This method simply returns the diagonal of the nested expression, thus by-passing the SelfAdjointView decorator.
   *
   * \sa MatrixBase::diagonal(), class Diagonal */
  EIGEN_DEVICE_FUNC typename MatrixType::ConstDiagonalReturnType diagonal() const {
    return typename MatrixType::ConstDiagonalReturnType(m_matrix);
  }

  /** \returns the matrix 1-norm (maximum absolute column sum) of the implicit
   * full self-adjoint matrix, reading only the stored triangle. For Hermitian
   * (complex) scalars the unstored entries are conjugates of stored ones, and
   * since |conj(x)| = |x| the result matches the L1 norm of the full matrix.
   */
  EIGEN_DEVICE_FUNC RealScalar l1Norm() const {
    const Index n = m_matrix.rows();
#ifdef EIGEN_GPU_COMPILE_PHASE
    // The accumulator below is per-thread local storage on a device.
    return l1NormPerColumn();
#else
    if (n <= L1NormImpl::PerColumnUpTo) return l1NormPerColumn();
    // For a self-adjoint matrix |a_ij| = |a_ji|, so the stored triangle of a row-major matrix is
    // the transposed, column-major, complementary one and yields the same norm read the fast way.
    RealScalar norm;
    EIGEN_IF_CONSTEXPR (bool(MatrixType::IsRowMajor)) {
      norm = l1NormStreaming<TransposeMode>(m_matrix.transpose());
    } else {
      norm = l1NormStreaming<UpLo>(m_matrix);
    }
    return L1NormImpl::isReliable(norm, n) ? norm : l1NormPerColumn();
#endif
  }

 private:
  using L1NormImpl = internal::selfadjoint_l1norm_impl<Scalar>;

  // Reading the mirrored term of column j as a row of the stored triangle costs a stride-n
  // traversal of a column-major matrix. Instead every column is read once, top to bottom, and
  // each |a_ij| is added both to its own column sum and to the sum of column i. Column j is
  // complete once every column holding one of its mirrored terms has been read: the earlier
  // ones for Lower, the later ones for Upper, hence the direction of the walk.
  template <int Mode, typename Mat>
  static RealScalar l1NormStreaming(const Mat& m) {
    const Index n = m.rows();
    ei_declare_aligned_stack_constructed_variable(Scalar, sums, n, 0);
    Map<Matrix<Scalar, Dynamic, 1>>(sums, n).setZero();
    RealScalar norm = RealScalar(0);
    for (Index k = 0; k < n; ++k) {
      const Index j = Mode == Lower ? k : n - 1 - k;
      RealScalar colsum = numext::abs(m.coeff(j, j));
      EIGEN_IF_CONSTEXPR (Mode == Lower) {
        colsum += L1NormImpl::accumulate(sums + j + 1, m.col(j).tail(n - j - 1));
      } else {
        colsum += L1NormImpl::accumulate(sums, m.col(j).head(j));
      }
      norm = numext::maxi(norm, colsum + numext::real(sums[j]));
    }
    return norm;
  }

  // Workspace-free form, one column sum at a time; the mirrored term is read as a row.
  EIGEN_DEVICE_FUNC RealScalar l1NormPerColumn() const {
    RealScalar norm = RealScalar(0);
    const Index n = m_matrix.rows();
    for (Index col = 0; col < n; ++col) {
      RealScalar abs_col_sum;
      EIGEN_IF_CONSTEXPR (UpLo == Lower) {
        abs_col_sum =
            m_matrix.col(col).tail(n - col).template lpNorm<1>() + m_matrix.row(col).head(col).template lpNorm<1>();
      } else {
        abs_col_sum =
            m_matrix.col(col).head(col).template lpNorm<1>() + m_matrix.row(col).tail(n - col).template lpNorm<1>();
      }
      norm = numext::maxi(norm, abs_col_sum);
    }
    return norm;
  }

 public:
  /////////// Cholesky module ///////////

  LLT<PlainObject, UpLo> llt() const;
  LDLT<PlainObject, UpLo> ldlt() const;
  BunchKaufman<PlainObject, UpLo> bunchKaufman() const;

  /////////// Eigenvalue module ///////////

  /** Return type of eigenvalues() */
  using EigenvaluesReturnType = Matrix<RealScalar, internal::traits<MatrixType>::ColsAtCompileTime, 1>;

  EIGEN_DEVICE_FUNC EigenvaluesReturnType eigenvalues() const;
  EIGEN_DEVICE_FUNC RealScalar operatorNorm() const;

 protected:
  MatrixTypeNested m_matrix;
};

// selfadjoint to dense matrix

namespace internal {

// TODO currently a selfadjoint expression has the form SelfAdjointView<.,.>
//      in the future selfadjoint-ness should be defined by the expression traits
//      such that Transpose<SelfAdjointView<.,.> > is valid. (currently TriangularBase::transpose() is overloaded to
//      make it work)
template <typename MatrixType, unsigned int Mode>
struct evaluator_traits<SelfAdjointView<MatrixType, Mode>> {
  using Kind = typename storage_kind_to_evaluator_kind<typename MatrixType::StorageKind>::Kind;
  using Shape = SelfAdjointShape;
};

template <int UpLo, int SetOpposite, typename DstEvaluatorTypeT, typename SrcEvaluatorTypeT, typename Functor,
          int Version>
class triangular_dense_assignment_kernel<UpLo, SelfAdjoint, SetOpposite, DstEvaluatorTypeT, SrcEvaluatorTypeT, Functor,
                                         Version>
    : public generic_dense_assignment_kernel<DstEvaluatorTypeT, SrcEvaluatorTypeT, Functor, Version> {
 protected:
  using Base = generic_dense_assignment_kernel<DstEvaluatorTypeT, SrcEvaluatorTypeT, Functor, Version>;
  using DstXprType = typename Base::DstXprType;
  using SrcXprType = typename Base::SrcXprType;
  using Base::m_dst;
  using Base::m_functor;
  using Base::m_src;

 public:
  using DstEvaluatorType = typename Base::DstEvaluatorType;
  using SrcEvaluatorType = typename Base::SrcEvaluatorType;
  using Scalar = typename Base::Scalar;
  using AssignmentTraits = typename Base::AssignmentTraits;

  EIGEN_DEVICE_FUNC triangular_dense_assignment_kernel(DstEvaluatorType& dst, const SrcEvaluatorType& src,
                                                       const Functor& func, DstXprType& dstExpr)
      : Base(dst, src, func, dstExpr) {}

  EIGEN_DEVICE_FUNC void assignCoeff(Index row, Index col) {
    eigen_internal_assert(row != col);
    Scalar tmp = m_src.coeff(row, col);
    m_functor.assignCoeff(m_dst.coeffRef(row, col), tmp);
    m_functor.assignCoeff(m_dst.coeffRef(col, row), numext::conj(tmp));
  }

  // Override to ensure the SelfAdjoint assignCoeff (which mirrors conjugates) is called,
  // not the base class version (which is a plain copy).
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void assignCoeffByOuterInner(Index outer, Index inner) {
    Index row = Base::rowIndexByOuterInner(outer, inner);
    Index col = Base::colIndexByOuterInner(outer, inner);
    assignCoeff(row, col);
  }

  EIGEN_DEVICE_FUNC void assignDiagonalCoeff(Index id) { Base::assignCoeff(id, id); }

  EIGEN_DEVICE_FUNC void assignOppositeCoeff(Index, Index) { eigen_internal_assert(false && "should never be called"); }
};

}  // end namespace internal

/***************************************************************************
 * Implementation of MatrixBase methods
 ***************************************************************************/

/** This is the const version of MatrixBase::selfadjointView() */
template <typename Derived>
template <unsigned int UpLo>
EIGEN_DEVICE_FUNC constexpr typename MatrixBase<Derived>::template ConstSelfAdjointViewReturnType<UpLo>::Type
MatrixBase<Derived>::selfadjointView() const {
  return typename ConstSelfAdjointViewReturnType<UpLo>::Type(derived());
}

/** \returns an expression of a symmetric/self-adjoint view extracted from the upper or lower triangular part of the
 * current matrix
 *
 * The parameter \a UpLo can be either \c #Upper or \c #Lower
 *
 * Example: \include MatrixBase_selfadjointView.cpp
 * Output: \verbinclude MatrixBase_selfadjointView.out
 *
 * \sa class SelfAdjointView
 */
template <typename Derived>
template <unsigned int UpLo>
EIGEN_DEVICE_FUNC constexpr typename MatrixBase<Derived>::template SelfAdjointViewReturnType<UpLo>::Type
MatrixBase<Derived>::selfadjointView() {
  return typename SelfAdjointViewReturnType<UpLo>::Type(derived());
}

}  // end namespace Eigen

#endif  // EIGEN_SELFADJOINTMATRIX_H
