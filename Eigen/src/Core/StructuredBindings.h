// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Pavel Guzenfeld
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_STRUCTURED_BINDINGS_H
#define EIGEN_STRUCTURED_BINDINGS_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

#if EIGEN_MAX_CPP_VER >= 17 && EIGEN_COMP_CXXVER >= 17

#include <tuple>
#include <type_traits>

// Structured bindings support for fixed-size Eigen vectors and matrices.
//
// Enables:
//   Eigen::Vector3d v(1, 2, 3);
//   auto [x, y, z] = v;
//
//   Eigen::Array3i a(4, 5, 6);
//   auto& [a0, a1, a2] = a;

namespace std {

// std::tuple_size for fixed-size Matrix types.
template <typename Scalar_, int Rows_, int Cols_, int Options_, int MaxRows_, int MaxCols_>
struct tuple_size<Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>>
    : std::enable_if_t<(Rows_ != Eigen::Dynamic && Cols_ != Eigen::Dynamic),
                        std::integral_constant<size_t, static_cast<size_t>(Rows_ * Cols_)>> {};

// std::tuple_element for fixed-size Matrix types.
// Note: uses Idx_ instead of I to avoid conflict with Eigen's test framework macro.
template <size_t Idx_, typename Scalar_, int Rows_, int Cols_, int Options_, int MaxRows_, int MaxCols_>
struct tuple_element<Idx_, Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>> {
  static_assert(Rows_ != Eigen::Dynamic && Cols_ != Eigen::Dynamic,
                "Structured bindings require fixed-size Eigen types.");
  static_assert(Idx_ < static_cast<size_t>(Rows_ * Cols_), "Index out of range.");
  using type = Scalar_;
};

// std::tuple_size for fixed-size Array types.
template <typename Scalar_, int Rows_, int Cols_, int Options_, int MaxRows_, int MaxCols_>
struct tuple_size<Eigen::Array<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>>
    : std::enable_if_t<(Rows_ != Eigen::Dynamic && Cols_ != Eigen::Dynamic),
                        std::integral_constant<size_t, static_cast<size_t>(Rows_ * Cols_)>> {};

// std::tuple_element for fixed-size Array types.
template <size_t Idx_, typename Scalar_, int Rows_, int Cols_, int Options_, int MaxRows_, int MaxCols_>
struct tuple_element<Idx_, Eigen::Array<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>> {
  static_assert(Rows_ != Eigen::Dynamic && Cols_ != Eigen::Dynamic,
                "Structured bindings require fixed-size Eigen types.");
  static_assert(Idx_ < static_cast<size_t>(Rows_ * Cols_), "Index out of range.");
  using type = Scalar_;
};

}  // namespace std

namespace Eigen {

// get<Idx_> free functions for Matrix.
template <size_t Idx_, typename Scalar_, int Rows_, int Cols_, int Options_, int MaxRows_, int MaxCols_>
EIGEN_DEVICE_FUNC constexpr EIGEN_STRONG_INLINE Scalar_& get(
    Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>& m) noexcept {
  static_assert(Rows_ != Dynamic && Cols_ != Dynamic, "Structured bindings require fixed-size Eigen types.");
  static_assert(Idx_ < static_cast<size_t>(Rows_ * Cols_), "Index out of range.");
  return m.coeffRef(static_cast<Index>(Idx_));
}

template <size_t Idx_, typename Scalar_, int Rows_, int Cols_, int Options_, int MaxRows_, int MaxCols_>
EIGEN_DEVICE_FUNC constexpr EIGEN_STRONG_INLINE const Scalar_& get(
    const Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>& m) noexcept {
  static_assert(Rows_ != Dynamic && Cols_ != Dynamic, "Structured bindings require fixed-size Eigen types.");
  static_assert(Idx_ < static_cast<size_t>(Rows_ * Cols_), "Index out of range.");
  return m.coeffRef(static_cast<Index>(Idx_));
}

template <size_t Idx_, typename Scalar_, int Rows_, int Cols_, int Options_, int MaxRows_, int MaxCols_>
EIGEN_DEVICE_FUNC constexpr EIGEN_STRONG_INLINE Scalar_&& get(
    Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>&& m) noexcept {
  static_assert(Rows_ != Dynamic && Cols_ != Dynamic, "Structured bindings require fixed-size Eigen types.");
  static_assert(Idx_ < static_cast<size_t>(Rows_ * Cols_), "Index out of range.");
  return std::move(m.coeffRef(static_cast<Index>(Idx_)));
}

// get<Idx_> free functions for Array.
template <size_t Idx_, typename Scalar_, int Rows_, int Cols_, int Options_, int MaxRows_, int MaxCols_>
EIGEN_DEVICE_FUNC constexpr EIGEN_STRONG_INLINE Scalar_& get(
    Array<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>& a) noexcept {
  static_assert(Rows_ != Dynamic && Cols_ != Dynamic, "Structured bindings require fixed-size Eigen types.");
  static_assert(Idx_ < static_cast<size_t>(Rows_ * Cols_), "Index out of range.");
  return a.coeffRef(static_cast<Index>(Idx_));
}

template <size_t Idx_, typename Scalar_, int Rows_, int Cols_, int Options_, int MaxRows_, int MaxCols_>
EIGEN_DEVICE_FUNC constexpr EIGEN_STRONG_INLINE const Scalar_& get(
    const Array<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>& a) noexcept {
  static_assert(Rows_ != Dynamic && Cols_ != Dynamic, "Structured bindings require fixed-size Eigen types.");
  static_assert(Idx_ < static_cast<size_t>(Rows_ * Cols_), "Index out of range.");
  return a.coeffRef(static_cast<Index>(Idx_));
}

template <size_t Idx_, typename Scalar_, int Rows_, int Cols_, int Options_, int MaxRows_, int MaxCols_>
EIGEN_DEVICE_FUNC constexpr EIGEN_STRONG_INLINE Scalar_&& get(
    Array<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>&& a) noexcept {
  static_assert(Rows_ != Dynamic && Cols_ != Dynamic, "Structured bindings require fixed-size Eigen types.");
  static_assert(Idx_ < static_cast<size_t>(Rows_ * Cols_), "Index out of range.");
  return std::move(a.coeffRef(static_cast<Index>(Idx_)));
}

}  // namespace Eigen

#endif  // C++17

#endif  // EIGEN_STRUCTURED_BINDINGS_H
