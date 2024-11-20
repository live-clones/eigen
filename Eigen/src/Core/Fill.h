// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_FILL_H
#define EIGEN_FILL_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

template <typename Xpr>
struct eigen_fill_helper : std::false_type {};

template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
struct eigen_fill_helper<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>> : std::true_type {};

template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
struct eigen_fill_helper<Array<Scalar, Rows, Cols, Options, MaxRows, MaxCols>> : std::true_type {};

template <typename Xpr, int BlockRows, int BlockCols>
struct eigen_fill_helper<Block<Xpr, BlockRows, BlockCols, /*InnerPanel*/ true>> : std::true_type {};

template <typename Xpr, bool use_fill = eigen_fill_helper<Xpr>::value>
struct eigen_fill_impl {
  using Scalar = typename Xpr::Scalar;
  using Func = scalar_constant_op<Scalar>;
  using PlainObject = typename Xpr::PlainObject;
  using Constant = CwiseNullaryOp<Func, PlainObject>;
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void run(Xpr& dst, const Scalar& val) {
    dst = Constant(dst.rows(), dst.cols(), Func(val));
  }
};

#if !EIGEN_COMP_MSVC
#ifndef EIGEN_GPU_COMPILE_PHASE
template <typename Xpr>
struct eigen_fill_impl<Xpr, /*use_fill*/ true> {
  using Scalar = typename Xpr::Scalar;
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void run(Xpr& dst, const Scalar& val) {
    EIGEN_USING_STD(fill_n);
    fill_n(dst.data(), dst.size(), val);
  }
};
#endif
#endif

template <typename Xpr>
struct eigen_zero_helper {
  static constexpr bool value = std::is_trivial<typename Xpr::Scalar>::value && eigen_fill_helper<Xpr>::value;
};

template <typename Xpr, bool use_memset = eigen_zero_helper<Xpr>::value>
struct eigen_zero_impl {
  using Scalar = typename Xpr::Scalar;
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void run(Xpr& dst) { eigen_fill_impl<Xpr, false>::run(dst, Scalar(0)); }
};

template <typename Xpr>
struct eigen_zero_impl<Xpr, /*use_memset*/ true> {
  using Scalar = typename Xpr::Scalar;
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void run(Xpr& dst) {
    EIGEN_USING_STD(memset);
    memset(dst.data(), 0, dst.size() * sizeof(Scalar));
  }
};

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_FILL_H
