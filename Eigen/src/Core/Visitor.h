// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_VISITOR_H
#define EIGEN_VISITOR_H

#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

template <typename Visitor, typename Derived, int UnrollCount,
          bool Vectorize = (Derived::PacketAccess && functor_traits<Visitor>::PacketAccess), bool LinearAccess = false,
          bool ShortCircuitEvaluation = false>
struct visitor_impl;

template <typename Visitor, bool ShortCircuitEvaluation = false>
struct short_circuit_eval_impl {
  // if short circuit evaluation is not used, do nothing
  static EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool run(const Visitor&) { return false; }
};
template <typename Visitor>
struct short_circuit_eval_impl<Visitor, true> {
  // if short circuit evaluation is used, check the visitor
  static EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool run(const Visitor& visitor) {
    return visitor.done();
  }
};

// inner-outer unrolled
template <typename Visitor, typename Derived, int UnrollCount, bool LinearAccess, bool ShortCircuitEvaluation>
struct visitor_impl<Visitor, Derived, UnrollCount, /*Vectorize=*/false, /*LinearAccess=*/LinearAccess, ShortCircuitEvaluation> {
  static constexpr int col = Derived::IsRowMajor ? (UnrollCount - 1) % Derived::ColsAtCompileTime
                                                 : (UnrollCount - 1) / Derived::RowsAtCompileTime;
  static constexpr int row = Derived::IsRowMajor ? (UnrollCount - 1) / Derived::ColsAtCompileTime
                                                 : (UnrollCount - 1) % Derived::RowsAtCompileTime;

  using short_circuit = short_circuit_eval_impl<Visitor, ShortCircuitEvaluation>;
  EIGEN_DEVICE_FUNC
  static inline void run(const Derived& mat, Visitor& visitor) {
    visitor_impl<Visitor, Derived, UnrollCount - 1, false, LinearAccess, ShortCircuitEvaluation>::run(mat, visitor);
    if (!short_circuit::run(visitor)) visitor(mat.coeff(row, col), row, col);
  }
};

// inner-outer unrolled
template <typename Visitor, typename Derived, bool LinearAccess, bool ShortCircuitEvaluation>
struct visitor_impl<Visitor, Derived, /*UnrollCount=*/1, /*Vectorize=*/false, /*LinearAccess=*/LinearAccess, ShortCircuitEvaluation> {
  using short_circuit = short_circuit_eval_impl<Visitor, ShortCircuitEvaluation>;
  EIGEN_DEVICE_FUNC
  static inline void run(const Derived& mat, Visitor& visitor) { visitor.init(mat.coeff(0, 0), 0, 0); }
};

// linear unrolled
template <typename Visitor, typename Derived, int UnrollCount, bool ShortCircuitEvaluation>
struct visitor_impl<Visitor, Derived, UnrollCount, /*Vectorize=*/false, /*LinearAccess=*/true, ShortCircuitEvaluation> {
  static constexpr int index = UnrollCount - 1;
  using short_circuit = short_circuit_eval_impl<Visitor, ShortCircuitEvaluation>;
  EIGEN_DEVICE_FUNC
  static inline void run(const Derived& mat, Visitor& visitor) {
    visitor_impl<Visitor, Derived, UnrollCount - 1, false, true, ShortCircuitEvaluation>::run(mat, visitor);
    if (!short_circuit::run(visitor)) visitor(mat.coeff(index), index);
  }
};

// linear unrolled
template <typename Visitor, typename Derived, bool ShortCircuitEvaluation>
struct visitor_impl<Visitor, Derived, /*UnrollCount=*/1, /*Vectorize=*/false, /*LinearAccess=*/true, ShortCircuitEvaluation> {
  using short_circuit = short_circuit_eval_impl<Visitor, ShortCircuitEvaluation>;
  EIGEN_DEVICE_FUNC
  static inline void run(const Derived& mat, Visitor& visitor) { visitor.init(mat.coeff(0), 0); }
};

// This specialization enables visitors on empty matrices at compile-time
template<typename Visitor, typename Derived, bool LinearAccess, bool ShortCircuitEvaluation>
struct visitor_impl<Visitor, Derived, 0, false, LinearAccess, ShortCircuitEvaluation> {
  EIGEN_DEVICE_FUNC
  static inline void run(const Derived &/*mat*/, Visitor& /*visitor*/)
  {}
};

// scalar outer-inner traversal
template <typename Visitor, typename Derived, bool ShortCircuitEvaluation>
struct visitor_impl<Visitor, Derived, Dynamic, /*Vectorize=*/false, /*LinearAccess=*/false, ShortCircuitEvaluation> {
  using short_circuit = short_circuit_eval_impl<Visitor, ShortCircuitEvaluation>;
  static constexpr bool RowMajor = Derived::IsRowMajor;

  EIGEN_DEVICE_FUNC
  static inline void run(const Derived& mat, Visitor& visitor) {
    const Index innerSize = RowMajor ? mat.cols() : mat.rows();
    const Index outerSize = RowMajor ? mat.rows() : mat.cols();
    visitor.init(mat.coeff(0, 0), 0, 0);
    if (short_circuit::run(visitor)) return;
    for (Index j = 0; j < outerSize; j++) {
      for (Index i = j == 0 ? 1 : 0; i < innerSize; ++i) {
        Index r = RowMajor ? j : i;
        Index c = RowMajor ? i : j;
        visitor(mat.coeff(r, c), r, c);
        if (short_circuit::run(visitor)) return;
      }
    }
  }
};

// vectorized outer-inner traversal
template <typename Visitor, typename Derived, int UnrollSize, bool ShortCircuitEvaluation>
struct visitor_impl<Visitor, Derived, UnrollSize, /*Vectorize=*/true, /*LinearAccess=*/false, ShortCircuitEvaluation> {
  using Scalar = typename Derived::Scalar;
  using Packet = typename packet_traits<Scalar>::type;
  static constexpr int PacketSize = packet_traits<Scalar>::size;
  using short_circuit = short_circuit_eval_impl<Visitor, ShortCircuitEvaluation>;
  static constexpr bool RowMajor = Derived::IsRowMajor;

  EIGEN_DEVICE_FUNC
  static inline void run(const Derived& mat, Visitor& visitor) {
    const Index innerSize = RowMajor ? mat.cols() : mat.rows();
    const Index outerSize = RowMajor ? mat.rows() : mat.cols();
    visitor.init(mat.coeff(0, 0), 0, 0);
    if (short_circuit::run(visitor)) return;
    for (Index j = 0; j < outerSize; j++) {
      Index i = j == 0 ? 1 : 0;
      for (; i + PacketSize - 1 < innerSize; i += PacketSize) {
        Index r = RowMajor ? j : i;
        Index c = RowMajor ? i : j;
        Packet p = mat.template packet<Packet>(r, c);
        visitor.packet(p, r, c);
        if (short_circuit::run(visitor)) return;
      }
      for (; i < innerSize; ++i) {
        Index r = RowMajor ? j : i;
        Index c = RowMajor ? i : j;
        visitor(mat.coeff(r, c), r, c);
        if (short_circuit::run(visitor)) return;
      }
    }
  }
};

// scalar linear traversal
template <typename Visitor, typename Derived, bool ShortCircuitEvaluation>
struct visitor_impl<Visitor, Derived, Dynamic, /*Vectorize=*/false, /*LinearAccess=*/true, ShortCircuitEvaluation> {
  using short_circuit = short_circuit_eval_impl<Visitor, ShortCircuitEvaluation>;

  EIGEN_DEVICE_FUNC
  static inline void run(const Derived& mat, Visitor& visitor) {
    const Index size = mat.size();
    visitor.init(mat.coeff(0), 0);
    if (short_circuit::run(visitor)) return;
    for (Index k = 1; k < size; k++) {
      visitor(mat.coeff(k), k);
      if (short_circuit::run(visitor)) return;
    }
  }
};

// vectorized linear traversal
// TODO: implement aligned loads
template <typename Visitor, typename Derived, int UnrollSize, bool ShortCircuitEvaluation>
struct visitor_impl<Visitor, Derived, UnrollSize, /*Vectorize=*/true, /*LinearAccess=*/true, ShortCircuitEvaluation> {
  using Scalar = typename Derived::Scalar;
  using Packet = typename packet_traits<Scalar>::type;
  static constexpr int PacketSize = packet_traits<Scalar>::size;
  using short_circuit = short_circuit_eval_impl<Visitor, ShortCircuitEvaluation>;

  EIGEN_DEVICE_FUNC
  static inline void run(const Derived& mat, Visitor& visitor) {
    const Index size = mat.size();
    visitor.init(mat.coeff(0), 0);
    if (short_circuit::run(visitor)) return;
    Index k = 1;
    for (; k + PacketSize - 1 < size; k += PacketSize) {
      Packet p = mat.template packet<Packet>(k);
      visitor.packet(p, k);
      if (short_circuit::run(visitor)) return;
    }
    for (; k < size; k++) {
      visitor(mat.coeff(k), k);
      if (short_circuit::run(visitor)) return;
    }
  }
};

// evaluator adaptor
template<typename XprType>
class visitor_evaluator
{
public:
  typedef internal::evaluator<XprType> Evaluator;
  typedef typename XprType::Scalar Scalar;
  using Packet = typename packet_traits<Scalar>::type;
  typedef std::remove_const_t<typename XprType::CoeffReturnType> CoeffReturnType;

  enum {
    PacketAccess = Evaluator::Flags & PacketAccessBit,
    LinearAccess = Evaluator::Flags & LinearAccessBit,
    IsRowMajor = XprType::IsRowMajor,
    RowsAtCompileTime = XprType::RowsAtCompileTime,
    ColsAtCompileTime = XprType::ColsAtCompileTime,
    CoeffReadCost = Evaluator::CoeffReadCost
  };


  EIGEN_DEVICE_FUNC
  explicit visitor_evaluator(const XprType &xpr) : m_evaluator(xpr), m_xpr(xpr) { }



  EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR Index rows() const EIGEN_NOEXCEPT { return m_xpr.rows(); }
  EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR Index cols() const EIGEN_NOEXCEPT { return m_xpr.cols(); }
  EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR Index size() const EIGEN_NOEXCEPT { return m_xpr.size(); }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index row, Index col) const { return m_evaluator.coeff(row, col); }
  template <typename Packet, int Alignment = Unaligned>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packet(Index row, Index col) const {
    return m_evaluator.template packet<Alignment, Packet>(row, col);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const { return m_evaluator.coeff(index); }
  template <typename Packet, int Alignment = Unaligned>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packet(Index index) const {
    return m_evaluator.template packet<Alignment, Packet>(index);
  }

protected:
  Evaluator m_evaluator;
  const XprType &m_xpr;
};

template <typename Derived, typename Visitor>
struct short_circuit_visitor_impl {
  using Evaluator = visitor_evaluator<Derived>;
  using Scalar = typename DenseBase<Derived>::Scalar;
  static constexpr int SizeAtCompileTime = DenseBase<Derived>::SizeAtCompileTime;
  static constexpr bool Unroll =
      SizeAtCompileTime != Dynamic && SizeAtCompileTime * int(Evaluator::CoeffReadCost) +
                                              (SizeAtCompileTime - 1) * int(functor_traits<Visitor>::Cost) <=
                                          EIGEN_UNROLLING_LIMIT;
  static constexpr int UnrollCount = Unroll ? int(SizeAtCompileTime) : Dynamic;
  static constexpr bool Vectorize = Evaluator::PacketAccess && functor_traits<Visitor>::PacketAccess;
  static constexpr bool LinearAccess = Evaluator::LinearAccess && functor_traits<Visitor>::LinearAccess;
  using impl = visitor_impl<Visitor, Evaluator, UnrollCount, Vectorize, LinearAccess, true>;

  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void run(const DenseBase<Derived>& mat, Visitor& visitor) {
    if (mat.size() == 0) return;
    Evaluator evaluator(mat.derived());
    impl::run(evaluator, visitor);
  }
};

} // end namespace internal

/** Applies the visitor \a visitor to the whole coefficients of the matrix or vector.
  *
  * The template parameter \a Visitor is the type of the visitor and provides the following interface:
  * \code
  * struct MyVisitor {
  *   // called for the first coefficient
  *   void init(const Scalar& value, Index i, Index j);
  *   // called for all other coefficients
  *   void operator() (const Scalar& value, Index i, Index j);
  * };
  * \endcode
  *
  * \note compared to one or two \em for \em loops, visitors offer automatic
  * unrolling for small fixed size matrix.
  *
  * \note if the matrix is empty, then the visitor is left unchanged.
  *
  * \sa minCoeff(Index*,Index*), maxCoeff(Index*,Index*), DenseBase::redux()
  */
template<typename Derived>
template<typename Visitor>
EIGEN_DEVICE_FUNC
void DenseBase<Derived>::visit(Visitor& visitor) const
{
  using namespace internal;
  using Evaluator = visitor_evaluator<Derived>;
  using Scalar = typename DenseBase<Derived>::Scalar;
  static constexpr int SizeAtCompileTime = DenseBase<Derived>::SizeAtCompileTime;
  static constexpr bool Unroll =
      SizeAtCompileTime != Dynamic && SizeAtCompileTime * int(Evaluator::CoeffReadCost) +
                                              (SizeAtCompileTime - 1) * int(functor_traits<Visitor>::Cost) <=
                                          EIGEN_UNROLLING_LIMIT;
  static constexpr int UnrollCount = Unroll ? int(SizeAtCompileTime) : Dynamic;
  static constexpr bool Vectorize = Evaluator::PacketAccess && functor_traits<Visitor>::PacketAccess;

  if (size() == 0) return;
  Evaluator evaluator(derived());
  visitor_impl<Visitor, Evaluator, UnrollCount, Vectorize, false, false>::run(evaluator, visitor);
}

namespace internal {

/** \internal
  * \brief Base class to implement min and max visitors
  */
template <typename Derived>
struct coeff_visitor
{
  // default initialization to avoid countless invalid maybe-uninitialized warnings by gcc
  EIGEN_DEVICE_FUNC
  coeff_visitor() : row(-1), col(-1), res(0) {}
  typedef typename Derived::Scalar Scalar;
  Index row, col;
  Scalar res;
  EIGEN_DEVICE_FUNC
  inline void init(const Scalar& value, Index i, Index j)
  {
    res = value;
    row = i;
    col = j;
  }
};


template<typename Scalar, int NaNPropagation, bool is_min=true>
struct minmax_compare {
  typedef typename packet_traits<Scalar>::type Packet;
  static EIGEN_DEVICE_FUNC inline bool compare(Scalar a, Scalar b) { return a < b; }
  static EIGEN_DEVICE_FUNC inline Scalar predux(const Packet& p) { return predux_min<NaNPropagation>(p);}
};

template<typename Scalar, int NaNPropagation>
struct minmax_compare<Scalar, NaNPropagation, false> {
  typedef typename packet_traits<Scalar>::type Packet;
  static EIGEN_DEVICE_FUNC inline bool compare(Scalar a, Scalar b) { return a > b; }
  static EIGEN_DEVICE_FUNC inline Scalar predux(const Packet& p) { return predux_max<NaNPropagation>(p);}
};

template <typename Derived, bool is_min, int NaNPropagation>
struct minmax_coeff_visitor : coeff_visitor<Derived>
{
  using Scalar = typename Derived::Scalar;
  using Packet = typename packet_traits<Scalar>::type;
  using Comparator = minmax_compare<Scalar, NaNPropagation, is_min>;

  EIGEN_DEVICE_FUNC inline
  void operator() (const Scalar& value, Index i, Index j)
  {
    if(Comparator::compare(value, this->res)) {
      this->res = value;
      this->row = i;
      this->col = j;
    }
  }

  EIGEN_DEVICE_FUNC inline
  void packet(const Packet& p, Index i, Index j) {
    const Index PacketSize = packet_traits<Scalar>::size;
    Scalar value = Comparator::predux(p);
    if (Comparator::compare(value, this->res)) {
      const Packet range = preverse(plset<Packet>(Scalar(1)));
      Packet mask = pcmp_eq(pset1<Packet>(value), p);
      Index max_idx = PacketSize - static_cast<Index>(predux_max(pand(range, mask)));
      this->res = value;
      this->row = Derived::IsRowMajor ? i : i + max_idx;;
      this->col = Derived::IsRowMajor ? j + max_idx : j;
    }
  }
};

// Suppress NaN. The only case in which we return NaN is if the matrix is all NaN, in which case,
// the row=0, col=0 is returned for the location.
template <typename Derived, bool is_min>
struct minmax_coeff_visitor<Derived, is_min, PropagateNumbers> : coeff_visitor<Derived>
{
  typedef typename Derived::Scalar Scalar;
  using Packet = typename packet_traits<Scalar>::type;
  using Comparator = minmax_compare<Scalar, PropagateNumbers, is_min>;

  EIGEN_DEVICE_FUNC inline
  void operator() (const Scalar& value, Index i, Index j)
  {
    if ((!(numext::isnan)(value) && (numext::isnan)(this->res)) || Comparator::compare(value, this->res)) {
      this->res = value;
      this->row = i;
      this->col = j;
    }
  }

  EIGEN_DEVICE_FUNC inline
  void packet(const Packet& p, Index i, Index j) {
    const Index PacketSize = packet_traits<Scalar>::size;
    Scalar value = Comparator::predux(p);
    if ((!(numext::isnan)(value) && (numext::isnan)(this->res)) || Comparator::compare(value, this->res)) {
      const Packet range = preverse(plset<Packet>(Scalar(1)));
      /* mask will be zero for NaNs, so they will be ignored. */
      Packet mask = pcmp_eq(pset1<Packet>(value), p);
      Index max_idx = PacketSize - static_cast<Index>(predux_max(pand(range, mask)));
      this->res = value;
      this->row = Derived::IsRowMajor ? i : i + max_idx;
      this->col = Derived::IsRowMajor ? j + max_idx : j;
    }
  }

};

// Propagate NaN. If the matrix contains NaN, the location of the first NaN will be returned in
// row and col.
template <typename Derived, bool is_min>
struct minmax_coeff_visitor<Derived, is_min, PropagateNaN> : coeff_visitor<Derived>
{
  typedef typename Derived::Scalar Scalar;
  using Packet = typename packet_traits<Scalar>::type;
  using Comparator = minmax_compare<Scalar, PropagateNaN, is_min>;

  EIGEN_DEVICE_FUNC inline
  void operator() (const Scalar& value, Index i, Index j)
  {
    const bool value_is_nan = (numext::isnan)(value);
    if ((value_is_nan && !(numext::isnan)(this->res)) || Comparator::compare(value, this->res)) {
      this->res = value;
      this->row = i;
      this->col = j;
    }
  }

  EIGEN_DEVICE_FUNC inline
  void packet(const Packet& p, Index i, Index j) {
    const Index PacketSize = packet_traits<Scalar>::size;
    Scalar value = Comparator::predux(p);
    const bool value_is_nan = (numext::isnan)(value);
    if ((value_is_nan && !(numext::isnan)(this->res)) || Comparator::compare(value, this->res)) {
      const Packet range = preverse(plset<Packet>(Scalar(1)));
      // If the value is NaN, pick the first position of a NaN, otherwise pick the first extremal value.
      Packet mask = value_is_nan ? pnot(pcmp_eq(p, p)) : pcmp_eq(pset1<Packet>(value), p);
      Index max_idx = PacketSize - static_cast<Index>(predux_max(pand(range, mask)));
      this->res = value;
      this->row = Derived::IsRowMajor ? i : i + max_idx;;
      this->col = Derived::IsRowMajor ? j + max_idx : j;
    }
  }
};

template<typename Derived, bool is_min, int NaNPropagation>
struct functor_traits<minmax_coeff_visitor<Derived, is_min, NaNPropagation> > {
  using Scalar = typename Derived::Scalar;
  enum {
    Cost = NumTraits<Scalar>::AddCost,
    PacketAccess = packet_traits<Scalar>::HasCmp
  };
};

template <typename Scalar>
struct all_visitor {
  using result_type = bool;
  using Packet = typename packet_traits<Scalar>::type;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void init(const Scalar& value, Index, Index) {
    res = res && (value != Scalar(0));
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void init(const Scalar& value, Index) { res = res && (value != Scalar(0)); }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void operator()(const Scalar& value, Index, Index) {
    res = res && (value != Scalar(0));
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void operator()(const Scalar& value, Index) {
    res = res && (value != Scalar(0));
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void packet(const Packet& p, Index, Index) {
    res = res && !predux_any(pcmp_eq(p, pzero(p)));
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void packet(const Packet& p, Index) {
    res = res && !predux_any(pcmp_eq(p, pzero(p)));
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool done() const { return !res; }
  bool res = true;
};
template <typename Scalar>
struct functor_traits<all_visitor<Scalar>> {
  enum { Cost = NumTraits<Scalar>::ReadCost, LinearAccess = true, PacketAccess = packet_traits<Scalar>::HasCmp };
};

template <typename Scalar>
struct any_visitor {
  using result_type = bool;
  using Packet = typename packet_traits<Scalar>::type;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void init(const Scalar& value, Index, Index) {
    res = res || (value != Scalar(0));
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void init(const Scalar& value, Index) { res = res || (value != Scalar(0)); }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void operator()(const Scalar& value, Index, Index) {
    res = res || (value != Scalar(0));
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void operator()(const Scalar& value, Index) {
    res = res || (value != Scalar(0));
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void packet(const Packet& p, Index, Index) {
    res = res || predux_any(pandnot(ptrue(p), pcmp_eq(p, pzero(p))));
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void packet(const Packet& p, Index) {
    res = res || predux_any(pandnot(ptrue(p), pcmp_eq(p, pzero(p))));
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool done() const { return res; }
  bool res = false;
};
template <typename Scalar>
struct functor_traits<any_visitor<Scalar>> {
  enum { Cost = NumTraits<Scalar>::ReadCost, LinearAccess = true, PacketAccess = packet_traits<Scalar>::HasCmp };
};

template <typename Scalar>
struct count_visitor {
  using result_type = Index;
  using Packet = typename packet_traits<Scalar>::type;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void init(const Scalar& value, Index, Index) {
    if (value != Scalar(0)) res++;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void init(const Scalar& value, Index) {
    if (value != Scalar(0)) res++;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void operator()(const Scalar& value, Index, Index) {
    if (value != Scalar(0)) res++;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void operator()(const Scalar& value, Index) {
    if (value != Scalar(0)) res++;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void packet(const Packet& p, Index, Index) {
    const Packet cst_one = pset1<Packet>(Scalar(1));
    Packet true_vals = pandnot(cst_one, pcmp_eq(p, pzero(p)));
    Scalar num_true = predux(true_vals);
    res += static_cast<Index>(num_true);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void packet(const Packet& p, Index) {
    const Packet cst_one = pset1<Packet>(Scalar(1));
    Packet true_vals = pandnot(cst_one, pcmp_eq(p, pzero(p)));
    Scalar num_true = predux(true_vals);
    res += static_cast<Index>(num_true);
  }
  Index res = 0;
};

template <typename Scalar>
struct functor_traits<count_visitor<Scalar>> {
  enum {
    Cost = NumTraits<Scalar>::ReadCost,
    LinearAccess = true,
    // predux is problematic for bool
    PacketAccess = packet_traits<Scalar>::HasCmp && packet_traits<Scalar>::HasAdd && !is_same<Scalar,bool>::value
  };
};

} // end namespace internal

/** \fn DenseBase<Derived>::minCoeff(IndexType* rowId, IndexType* colId) const
  * \returns the minimum of all coefficients of *this and puts in *row and *col its location.
  *
  * In case \c *this contains NaN, NaNPropagation determines the behavior:
  *   NaNPropagation == PropagateFast : undefined
  *   NaNPropagation == PropagateNaN : result is NaN
  *   NaNPropagation == PropagateNumbers : result is maximum of elements that are not NaN
  * \warning the matrix must be not empty, otherwise an assertion is triggered.
  *
  * \sa DenseBase::minCoeff(Index*), DenseBase::maxCoeff(Index*,Index*), DenseBase::visit(), DenseBase::minCoeff()
  */
template<typename Derived>
template<int NaNPropagation, typename IndexType>
EIGEN_DEVICE_FUNC
typename internal::traits<Derived>::Scalar
DenseBase<Derived>::minCoeff(IndexType* rowId, IndexType* colId) const
{
  eigen_assert(this->rows()>0 && this->cols()>0 && "you are using an empty matrix");

  internal::minmax_coeff_visitor<Derived, true, NaNPropagation> minVisitor;
  this->visit(minVisitor);
  *rowId = minVisitor.row;
  if (colId) *colId = minVisitor.col;
  return minVisitor.res;
}

/** \returns the minimum of all coefficients of *this and puts in *index its location.
  *
  * In case \c *this contains NaN, NaNPropagation determines the behavior:
  *   NaNPropagation == PropagateFast : undefined
  *   NaNPropagation == PropagateNaN : result is NaN
  *   NaNPropagation == PropagateNumbers : result is maximum of elements that are not NaN
  * \warning the matrix must be not empty, otherwise an assertion is triggered.
  *
  * \sa DenseBase::minCoeff(IndexType*,IndexType*), DenseBase::maxCoeff(IndexType*,IndexType*), DenseBase::visit(), DenseBase::minCoeff()
  */
template<typename Derived>
template<int NaNPropagation, typename IndexType>
EIGEN_DEVICE_FUNC
typename internal::traits<Derived>::Scalar
DenseBase<Derived>::minCoeff(IndexType* index) const
{
  eigen_assert(this->rows()>0 && this->cols()>0 && "you are using an empty matrix");

  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
      internal::minmax_coeff_visitor<Derived, true, NaNPropagation> minVisitor;
  this->visit(minVisitor);
  *index = IndexType((RowsAtCompileTime==1) ? minVisitor.col : minVisitor.row);
  return minVisitor.res;
}

/** \fn DenseBase<Derived>::maxCoeff(IndexType* rowId, IndexType* colId) const
  * \returns the maximum of all coefficients of *this and puts in *row and *col its location.
  *
  * In case \c *this contains NaN, NaNPropagation determines the behavior:
  *   NaNPropagation == PropagateFast : undefined
  *   NaNPropagation == PropagateNaN : result is NaN
  *   NaNPropagation == PropagateNumbers : result is maximum of elements that are not NaN
  * \warning the matrix must be not empty, otherwise an assertion is triggered.
  *
  * \sa DenseBase::minCoeff(IndexType*,IndexType*), DenseBase::visit(), DenseBase::maxCoeff()
  */
template<typename Derived>
template<int NaNPropagation, typename IndexType>
EIGEN_DEVICE_FUNC
typename internal::traits<Derived>::Scalar
DenseBase<Derived>::maxCoeff(IndexType* rowPtr, IndexType* colPtr) const
{
  eigen_assert(this->rows()>0 && this->cols()>0 && "you are using an empty matrix");

  internal::minmax_coeff_visitor<Derived, false, NaNPropagation> maxVisitor;
  this->visit(maxVisitor);
  *rowPtr = maxVisitor.row;
  if (colPtr) *colPtr = maxVisitor.col;
  return maxVisitor.res;
}

/** \returns the maximum of all coefficients of *this and puts in *index its location.
  *
  * In case \c *this contains NaN, NaNPropagation determines the behavior:
  *   NaNPropagation == PropagateFast : undefined
  *   NaNPropagation == PropagateNaN : result is NaN
  *   NaNPropagation == PropagateNumbers : result is maximum of elements that are not NaN
  * \warning the matrix must be not empty, otherwise an assertion is triggered.
  *
  * \sa DenseBase::maxCoeff(IndexType*,IndexType*), DenseBase::minCoeff(IndexType*,IndexType*), DenseBase::visitor(), DenseBase::maxCoeff()
  */
template<typename Derived>
template<int NaNPropagation, typename IndexType>
EIGEN_DEVICE_FUNC
typename internal::traits<Derived>::Scalar
DenseBase<Derived>::maxCoeff(IndexType* index) const
{
  eigen_assert(this->rows()>0 && this->cols()>0 && "you are using an empty matrix");

  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
      internal::minmax_coeff_visitor<Derived, false, NaNPropagation> maxVisitor;
  this->visit(maxVisitor);
  *index = (RowsAtCompileTime==1) ? maxVisitor.col : maxVisitor.row;
  return maxVisitor.res;
}

/** \returns true if all coefficients are true
  *
  * Example: \include MatrixBase_all.cpp
  * Output: \verbinclude MatrixBase_all.out
  *
  * \sa any(), Cwise::operator<()
  */
template <typename Derived>
EIGEN_DEVICE_FUNC inline bool DenseBase<Derived>::all() const {
  using Visitor = internal::all_visitor<Scalar>;
  using impl = internal::short_circuit_visitor_impl<Derived, Visitor>;
  Visitor visitor;
  impl::run(derived(), visitor);
  return visitor.res;
}

/** \returns true if at least one coefficient is true
  *
  * \sa all()
  */
template <typename Derived>
EIGEN_DEVICE_FUNC inline bool DenseBase<Derived>::any() const {
  using Visitor = internal::any_visitor<Scalar>;
  using impl = internal::short_circuit_visitor_impl<Derived, Visitor>;
  Visitor visitor;
  impl::run(derived(), visitor);
  return visitor.res;
}

/** \returns the number of coefficients which evaluate to true
  *
  * \sa all(), any()
  */
template<typename Derived>
EIGEN_DEVICE_FUNC
Index DenseBase<Derived>::count() const
{
  using Visitor = internal::count_visitor<Scalar>;
  Visitor visitor;
  visit(visitor);
  return visitor.res;
}

template <typename Derived>
EIGEN_DEVICE_FUNC inline bool DenseBase<Derived>::hasNaN() const {
  return (derived().array() != derived().array()).any();
}

/** \returns true if \c *this contains only finite numbers, i.e., no NaN and no +/-INF values.
  *
  * \sa hasNaN()
  */
template <typename Derived>
EIGEN_DEVICE_FUNC inline bool DenseBase<Derived>::allFinite() const {
  return (derived().array().abs() < NumTraits<Scalar>::infinity()).all();
}

} // end namespace Eigen

#endif // EIGEN_VISITOR_H
