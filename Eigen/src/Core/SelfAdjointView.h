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

// Column step of the self-adjoint 1-norm, on two columns x0 and x1 sharing the rows [0, n):
// sums[i] += |x0_i| + |x1_i|, returning sum |x0_i| and sum |x1_i|. Walking two columns at once
// halves the traffic on sums, and one packet pass does everything, so short columns pay no
// per-expression setup.
//
// Real scalars use pabs. Complex ones have no packet abs, so |z| = sqrt(re^2 + im^2) is computed
// on the real lanes of the complex packet. That leaves |z|^2 in both lanes of each slot, so one
// square root serves both columns: even lanes from x0 and odd lanes from x1 give |u0| |v0| |u1|
// |v1| ..., whose sum with its flip is the update of sums, and whose reduction as a complex packet
// is (sum |u|, sum |v|). The accumulator has the matrix's scalar type and only its real parts are
// read, so what lands in the imaginary lanes is harmless.
template <typename Scalar_>
struct selfadjoint_l1norm_real_lanes {
  using Scalar = Scalar_;
  using Real = Scalar_;
  using Packet = typename packet_traits<Scalar>::type;
  using RPacket = Packet;
  static constexpr Index PacketSize = unpacket_traits<Packet>::size;
  // Up to this size the per-column form (mirrored term read as a row) beats the column pass, whose
  // accumulator costs more to set up than these columns cost to read.
  static constexpr Index PerColumnUpTo = 4;
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE RPacket lanes(const Packet& p) { return p; }
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Real abs(const Scalar& x) { return numext::abs(x); }
  static EIGEN_DEVICE_FUNC bool isReliable(Real, Index) { return true; }

  // The running sums of a pass over two columns.
  struct Pass {
    RPacket acc0 = pzero(RPacket());
    RPacket acc1 = pzero(RPacket());
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void step(const RPacket& u, const RPacket& v, Scalar* s) {
      RPacket a = pabs(u);
      RPacket b = pabs(v);
      acc0 = padd(acc0, a);
      acc1 = padd(acc1, b);
      pstoreu(s, padd(ploadu<Packet>(s), padd(a, b)));
    }
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Real sum0() const { return predux(acc0); }
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Real sum1() const { return predux(acc1); }
  };
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
  // Same formula as the packets, for the diagonal and the tails: hypot costs more than the packets
  // spend on the rest of a short column.
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Real abs(const Scalar& z) { return numext::sqrt(numext::abs2(z)); }
  // Squaring overflows above sqrt(max) and loses precision below sqrt(min): an overflowed or tiny
  // result is recomputed with numext::abs. The overflow test compares against a finite bound
  // rather than asking isfinite, which -ffinite-math-only folds to true.
  static EIGEN_DEVICE_FUNC bool isReliable(Real norm, Index n) {
    const Real tiny = Real(n) * numext::sqrt((std::numeric_limits<Real>::min)()) / NumTraits<Real>::epsilon();
    const Real huge = NumTraits<Real>::highest() / Real(2);
    return norm > tiny && norm < huge;
  }

  struct Pass {
    RPacket acc = pzero(RPacket());  // |u| in the even lanes, |v| in the odd ones
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void step(const RPacket& u, const RPacket& v, Scalar* s) {
      RPacket r = psqrt(pselect(peven_mask(u), abs2(u), abs2(v)));  // |u0| |v0| |u1| |v1| ...
      acc = padd(acc, r);
      pstoreu(s, Packet(padd(lanes(ploadu<Packet>(s)), padd(r, flip(r)))));
    }
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Real sum0() const { return numext::real(predux(Packet(acc))); }
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Real sum1() const { return numext::imag(predux(Packet(acc))); }
  };

 private:
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE RPacket flip(const RPacket& r) { return pcplxflip(Packet(r)).v; }
  // |z|^2 in both lanes of its slot.
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE RPacket abs2(const RPacket& v) {
    RPacket s = pmul(v, v);
    return padd(s, flip(s));
  }
};

template <typename Lanes>
struct selfadjoint_l1norm_packet_impl : Lanes {
  using Lanes::PacketSize;
  using typename Lanes::Packet;
  using typename Lanes::Real;
  using typename Lanes::RPacket;
  using typename Lanes::Scalar;
  using Sums = std::pair<Real, Real>;

  template <typename Derived0, typename Derived1>
  static EIGEN_DEVICE_FUNC Sums accumulate(Scalar* sums, const DenseBase<Derived0>& x0, const DenseBase<Derived1>& x1) {
    return accumulateCast(sums, x0.derived().template cast<Scalar>(), x1.derived().template cast<Scalar>());
  }

 private:
  template <typename Derived0, typename Derived1>
  static EIGEN_DEVICE_FUNC Sums accumulateCast(Scalar* sums, const DenseBase<Derived0>& x0,
                                               const DenseBase<Derived1>& x1) {
    using Evaluator0 = evaluator<Derived0>;
    using Evaluator1 = evaluator<Derived1>;
    constexpr int Needed = PacketAccessBit | LinearAccessBit;
    constexpr bool Vectorize = (Evaluator0::Flags & Needed) == Needed && (Evaluator1::Flags & Needed) == Needed;
    return accumulate(sums, Evaluator0(x0.derived()), Evaluator1(x1.derived()), x0.size(), bool_constant<Vectorize>());
  }
  template <typename Evaluator>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE RPacket load(const Evaluator& x, Index i) {
    return Lanes::lanes(x.template packet<Unaligned, Packet>(i));
  }
  template <typename Evaluator0, typename Evaluator1>
  static EIGEN_DEVICE_FUNC Sums accumulate(Scalar* s, const Evaluator0& x0, const Evaluator1& x1, Index begin,
                                           Index end) {
    Sums r(Real(0), Real(0));
    for (Index i = begin; i < end; ++i) {
      Real a = Lanes::abs(x0.coeff(i));
      Real b = Lanes::abs(x1.coeff(i));
      s[i] += a + b;
      r.first += a;
      r.second += b;
    }
    return r;
  }
  template <typename Evaluator0, typename Evaluator1>
  static EIGEN_DEVICE_FUNC Sums accumulate(Scalar* s, const Evaluator0& x0, const Evaluator1& x1, Index n,
                                           std::false_type) {
    return accumulate(s, x0, x1, Index(0), n);
  }
  template <typename Evaluator0, typename Evaluator1>
  static EIGEN_DEVICE_FUNC Sums accumulate(Scalar* s, const Evaluator0& x0, const Evaluator1& x1, Index n,
                                           std::true_type) {
    if (n < PacketSize) return accumulate(s, x0, x1, Index(0), n);
    typename Lanes::Pass pass;
    Index i = 0;
    for (; i + PacketSize <= n; i += PacketSize) pass.step(load(x0, i), load(x1, i), s + i);
    Sums tail = accumulate(s, x0, x1, i, n);
    return Sums(pass.sum0() + tail.first, pass.sum1() + tail.second);
  }
};

// Coefficient fallback: custom complex types, or complex packets without a real view.
template <typename Scalar_, typename Enable = void>
struct selfadjoint_l1norm_impl {
  using Scalar = Scalar_;
  using Real = typename NumTraits<Scalar>::Real;
  using Sums = std::pair<Real, Real>;
  static constexpr Index PerColumnUpTo = 16;
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Real abs(const Scalar& x) { return numext::abs(x); }
  template <typename Derived0, typename Derived1>
  static EIGEN_DEVICE_FUNC Sums accumulate(Scalar* sums, const DenseBase<Derived0>& x0, const DenseBase<Derived1>& x1) {
    Sums r(Real(0), Real(0));
    for (Index i = 0; i < x0.size(); ++i) {
      Real a = numext::abs(x0.coeff(i));
      Real b = numext::abs(x1.coeff(i));
      sums[i] += Scalar(a + b);
      r.first += a;
      r.second += b;
    }
    return r;
  }
  static EIGEN_DEVICE_FUNC bool isReliable(Real, Index) { return true; }
};
// half and bfloat16 accumulate in float, as stableNorm does.
template <typename Scalar>
struct selfadjoint_l1norm_impl<Scalar, std::enable_if_t<!NumTraits<Scalar>::IsComplex>>
    : selfadjoint_l1norm_packet_impl<selfadjoint_l1norm_real_lanes<typename stable_norm_accumulator<Scalar>::type>> {};
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
class SelfAdjointView : public TriangularBase<SelfAdjointView<MatrixType_, UpLo> > {
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
                         TriangularView<typename MatrixType::AdjointReturnType, TriMode> >
      triangularView() const {
    std::conditional_t<(TriMode & (Upper | Lower)) == (UpLo & (Upper | Lower)), MatrixType&,
                       typename MatrixType::ConstTransposeReturnType>
        tmp1(m_matrix);
    std::conditional_t<(TriMode & (Upper | Lower)) == (UpLo & (Upper | Lower)), MatrixType&,
                       typename MatrixType::AdjointReturnType>
        tmp2(tmp1);
    return std::conditional_t<(TriMode & (Upper | Lower)) == (UpLo & (Upper | Lower)),
                              TriangularView<MatrixType, TriMode>,
                              TriangularView<typename MatrixType::AdjointReturnType, TriMode> >(tmp2);
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
    // No per-thread accumulator on a device.
    return l1NormPerColumn();
#else
    if (n <= L1NormImpl::PerColumnUpTo) return l1NormPerColumn();
    // The stored triangle of a row-major matrix is the complementary triangle of its column-major
    // transpose, which has the same norm.
    L1NormAccumulator norm;
    EIGEN_IF_CONSTEXPR (bool(MatrixType::IsRowMajor)) {
      norm = l1NormStreaming<TransposeMode>(m_matrix.transpose());
    } else {
      norm = l1NormStreaming<UpLo>(m_matrix);
    }
    return L1NormImpl::isReliable(norm, n) ? RealScalar(norm) : l1NormPerColumn();
#endif
  }

 private:
  using L1NormImpl = internal::selfadjoint_l1norm_impl<Scalar>;
  // float for half and bfloat16, Scalar otherwise.
  using L1NormScalar = typename L1NormImpl::Scalar;
  using L1NormAccumulator = typename L1NormImpl::Real;

  // Each column is read once, top to bottom, two at a time: |a_ij| goes to column j's sum and, as
  // the mirrored a_ji, to sums[i]. Lower walks the columns forward and Upper backward so that
  // sums[j] is complete when column j is reached. Of a pair (j0, j1) only j0's element in row j1
  // lies outside the rows the two share.
  template <int Mode, typename Mat>
  static L1NormAccumulator l1NormStreaming(const Mat& m) {
    const Index n = m.rows();
    using Sums = Matrix<L1NormScalar, Mat::RowsAtCompileTime, 1, 0, Mat::MaxRowsAtCompileTime, 1>;
    Sums sums = Sums::Zero(n);
    L1NormAccumulator norm = L1NormAccumulator(0);
    Index k = 0;
    for (; k + 1 < n; k += 2) {
      const Index j0 = Mode == Lower ? k : n - 1 - k;
      const Index j1 = Mode == Lower ? j0 + 1 : j0 - 1;
      const L1NormAccumulator boundary = L1NormImpl::abs(m.coeff(j1, j0));
      typename L1NormImpl::Sums shared;
      EIGEN_IF_CONSTEXPR (Mode == Lower) {
        shared = L1NormImpl::accumulate(sums.data() + j1 + 1, m.col(j0).tail(n - j1 - 1), m.col(j1).tail(n - j1 - 1));
      } else {
        shared = L1NormImpl::accumulate(sums.data(), m.col(j0).head(j1), m.col(j1).head(j1));
      }
      // Totals are materialized so that maxi compares two accumulators (an integer sum promotes,
      // an autodiff sum is an expression).
      const L1NormAccumulator col0 =
          L1NormImpl::abs(m.coeff(j0, j0)) + boundary + shared.first + numext::real(sums.coeff(j0));
      const L1NormAccumulator col1 =
          L1NormImpl::abs(m.coeff(j1, j1)) + boundary + shared.second + numext::real(sums.coeff(j1));
      norm = numext::maxi(norm, col0);
      norm = numext::maxi(norm, col1);
    }
    if (k < n) {
      const Index j = Mode == Lower ? k : 0;
      const L1NormAccumulator col = L1NormImpl::abs(m.coeff(j, j)) + numext::real(sums.coeff(j));
      norm = numext::maxi(norm, col);
    }
    return norm;
  }

  // One column at a time, the mirrored term read as a row; no workspace.
  EIGEN_DEVICE_FUNC RealScalar l1NormPerColumn() const {
    L1NormAccumulator norm = L1NormAccumulator(0);
    const Index n = m_matrix.rows();
    for (Index col = 0; col < n; ++col) {
      L1NormAccumulator abs_col_sum;
      EIGEN_IF_CONSTEXPR (UpLo == Lower) {
        abs_col_sum = m_matrix.col(col).tail(n - col).template cast<L1NormScalar>().template lpNorm<1>() +
                      m_matrix.row(col).head(col).template cast<L1NormScalar>().template lpNorm<1>();
      } else {
        abs_col_sum = m_matrix.col(col).head(col).template cast<L1NormScalar>().template lpNorm<1>() +
                      m_matrix.row(col).tail(n - col).template cast<L1NormScalar>().template lpNorm<1>();
      }
      norm = numext::maxi(norm, abs_col_sum);
    }
    return RealScalar(norm);
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
struct evaluator_traits<SelfAdjointView<MatrixType, Mode> > {
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
