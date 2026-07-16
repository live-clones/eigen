// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

/*
 * Name-keyed SFINAE predicates (is_*, require_*) for GPU value and expression
 * classes, plus
 * lazy query aliases over internal::device_expr_traits (scalar_type_t,
 * is_device_expr_v, trans_op). Lives between FwdDecl.h and the class bodies so
 * DeviceMatrix members can be gated on these before the bodies are seen.
 *
 * The predicates need only FwdDecl.h. The device_expr_traits aliases stay
 * lazy: device_expr_traits is forward declared in FwdDecl.h and its
 * specializations live in DeviceExpr.h, so the aliases resolve only once those
 * specializations are complete.
 */

#ifndef EIGEN_GPU_TYPE_TRAITS_H
#define EIGEN_GPU_TYPE_TRAITS_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

#include "./FwdDecl.h"
#include "./Meta.h"

#include <complex>
#include <type_traits>

namespace Eigen {
namespace gpu {

/**
 * @name internal:: partial specializations
 *
 * Each one keys off the *name* of a forward-declared class. None of them
 * inspects a member, so they parse cleanly with FwdDecl.h alone.
 */

namespace internal {

///@{

template <typename Expr>
using scalar_type_t = typename device_expr_traits<Expr>::scalar_type;

template <typename T>
struct is_devicebuffer : Eigen::internal::bool_constant<false> {};
template <>
struct is_devicebuffer<DeviceBuffer> : Eigen::internal::bool_constant<true> {};

template <typename T>
struct is_devicematrix : Eigen::internal::bool_constant<false> {};
template <typename Scalar>
struct is_devicematrix<DeviceMatrix<Scalar>> : Eigen::internal::bool_constant<true> {};

template <typename T>
struct is_gemm_expr : Eigen::internal::bool_constant<false> {};
template <typename Lhs, typename Rhs>
struct is_gemm_expr<GemmExpr<Lhs, Rhs>> : Eigen::internal::bool_constant<true> {};

template <typename T>
struct is_adjoint_view : Eigen::internal::bool_constant<false> {};
template <typename Scalar>
struct is_adjoint_view<AdjointView<Scalar>> : Eigen::internal::bool_constant<true> {};

template <typename T>
struct is_transpose_view : Eigen::internal::bool_constant<false> {};
template <typename Scalar>
struct is_transpose_view<TransposeView<Scalar>> : Eigen::internal::bool_constant<true> {};

template <typename T>
struct is_scaled : Eigen::internal::bool_constant<false> {};
template <typename Inner>
struct is_scaled<Scaled<Inner>> : Eigen::internal::bool_constant<true> {};

/**
 * @brief Detects a Scaled directly over a leaf DeviceMatrix (no view in between).
 * @tparam T The type to test.
 * @note Such a node arriving as a donation (owned rvalue) is materialized by
 * stealing the leaf and applying the scalar in place (scal) instead of a geam
 * into a fresh buffer.
 */
template <typename T>
struct is_scaled_leaf : Eigen::internal::bool_constant<false> {};
template <typename Scalar>
struct is_scaled_leaf<Scaled<DeviceMatrix<Scalar>>> : Eigen::internal::bool_constant<true> {};

/**
 * @brief Detects a Scaled directly over a GemmExpr (a product carrying ONE deferred scalar).
 * @tparam T The type to test.
 * @note Matched by name only, like is_scaled_leaf. Such a summand routes
 * through the GEMM epilogue with its factor as the gemm's alpha_scale (no
 * temporary).
 */
template <typename T>
struct is_scaled_gemm : Eigen::internal::bool_constant<false> {};
template <typename Lhs, typename Rhs>
struct is_scaled_gemm<Scaled<GemmExpr<Lhs, Rhs>>> : Eigen::internal::bool_constant<true> {};

template <typename T>
struct is_triangular_view : Eigen::internal::bool_constant<false> {};
template <typename Inner, int UpLo>
struct is_triangular_view<TriangularView<Inner, UpLo>> : Eigen::internal::bool_constant<true> {};

template <typename T>
struct is_trsm_expr : Eigen::internal::bool_constant<false> {};
template <typename Scalar, int UpLo>
struct is_trsm_expr<TrsmExpr<Scalar, UpLo>> : Eigen::internal::bool_constant<true> {};

template <typename T>
struct is_selfadjoint_view : Eigen::internal::bool_constant<false> {};
template <typename Scalar, int UpLo>
struct is_selfadjoint_view<SelfAdjointView<Scalar, UpLo>> : Eigen::internal::bool_constant<true> {};

template <typename T>
struct is_const_selfadjoint_view : Eigen::internal::bool_constant<false> {};
template <typename Inner, int UpLo>
struct is_const_selfadjoint_view<ConstSelfAdjointView<Inner, UpLo>> : Eigen::internal::bool_constant<true> {};

template <typename T>
struct is_symm_expr : Eigen::internal::bool_constant<false> {};
template <typename Scalar, int UpLo>
struct is_symm_expr<SymmExpr<Scalar, UpLo>> : Eigen::internal::bool_constant<true> {};

template <typename T>
struct is_syrk_expr : Eigen::internal::bool_constant<false> {};
template <typename A, int UpLo>
struct is_syrk_expr<SyrkExpr<A, UpLo>> : Eigen::internal::bool_constant<true> {};

template <typename T>
struct is_llt_solve_expr : Eigen::internal::bool_constant<false> {};
template <typename Scalar, int UpLo>
struct is_llt_solve_expr<LltSolveExpr<Scalar, UpLo>> : Eigen::internal::bool_constant<true> {};

template <typename T>
struct is_lu_solve_expr : Eigen::internal::bool_constant<false> {};
template <typename Scalar>
struct is_lu_solve_expr<LuSolveExpr<Scalar>> : Eigen::internal::bool_constant<true> {};

template <typename T>
struct is_llt_view : Eigen::internal::bool_constant<false> {};
template <typename Scalar, int UpLo>
struct is_llt_view<LLTView<Scalar, UpLo>> : Eigen::internal::bool_constant<true> {};

template <typename T>
struct is_lu_view : Eigen::internal::bool_constant<false> {};
template <typename Scalar>
struct is_lu_view<LUView<Scalar>> : Eigen::internal::bool_constant<true> {};

template <typename T>
struct is_device_add_expr : Eigen::internal::bool_constant<false> {};
template <typename Scalar>
struct is_device_add_expr<DeviceAddExpr<Scalar>> : Eigen::internal::bool_constant<true> {};

template <typename T>
struct is_device_scaled_device : Eigen::internal::bool_constant<false> {};
template <typename Inner>
struct is_device_scaled_device<DeviceScaledDevice<Inner>> : Eigen::internal::bool_constant<true> {};

template <typename T>
struct is_device_scalar : Eigen::internal::bool_constant<false> {};
template <typename S>
struct is_device_scalar<DeviceScalar<S>> : Eigen::internal::bool_constant<true> {};

///@}

}  // namespace internal

/**
 * @name Public is_* / is_*_v / require_* / require_all_* wrappers
 *
 * One block per category, mirroring the previous per-header layout so a grep
 * for "is_gemm_expr" or "require_trsm_expr" lands in the same shape it used to.
 */
///@{

template <typename T>
struct is_devicebuffer : internal::is_devicebuffer<std::decay_t<T>> {};

template <typename T>
constexpr bool is_devicebuffer_v = is_devicebuffer<T>::value;

template <typename T>
using require_devicebuffer = internal::require_t<is_devicebuffer<T>>;

template <typename T>
struct is_devicematrix : internal::is_devicematrix<std::decay_t<T>> {};

template <typename T>
constexpr bool is_devicematrix_v = is_devicematrix<T>::value;

template <typename T>
using require_devicematrix = internal::require_t<is_devicematrix<T>>;

template <typename T>
using require_not_devicematrix = internal::require_not_t<is_devicematrix<T>>;

template <typename... Types>
using require_all_devicematrix = internal::require_all_t<is_devicematrix_v<Types>...>;

template <typename T>
struct is_gemm_expr : internal::is_gemm_expr<std::decay_t<T>> {};

template <typename T>
constexpr bool is_gemm_expr_v = is_gemm_expr<T>::value;

template <typename T>
using require_gemm_expr = internal::require_t<is_gemm_expr<T>>;

template <typename... Types>
using require_all_gemm_expr = internal::require_all_t<is_gemm_expr_v<Types>...>;

template <typename T>
struct is_adjoint_view : internal::is_adjoint_view<std::decay_t<T>> {};

template <typename T>
constexpr bool is_adjoint_view_v = is_adjoint_view<T>::value;

template <typename T>
using require_adjoint_view = internal::require_t<is_adjoint_view<T>>;

template <typename... Types>
using require_all_adjoint_view = internal::require_all_t<is_adjoint_view_v<Types>...>;

template <typename T>
struct is_transpose_view : internal::is_transpose_view<std::decay_t<T>> {};

template <typename T>
constexpr bool is_transpose_view_v = is_transpose_view<T>::value;

template <typename T>
using require_transpose_view = internal::require_t<is_transpose_view<T>>;

template <typename... Types>
using require_all_transpose_view = internal::require_all_t<is_transpose_view_v<Types>...>;

template <typename T>
struct is_scaled : internal::is_scaled<std::decay_t<T>> {};

template <typename T>
constexpr bool is_scaled_v = is_scaled<T>::value;

template <typename T>
using require_scaled = internal::require_t<is_scaled<T>>;

template <typename T>
struct is_scaled_leaf : internal::is_scaled_leaf<std::decay_t<T>> {};

template <typename T>
constexpr bool is_scaled_leaf_v = is_scaled_leaf<T>::value;

template <typename T>
using require_scaled_leaf = internal::require_t<is_scaled_leaf<T>>;

template <typename T>
struct is_scaled_gemm : internal::is_scaled_gemm<std::decay_t<T>> {};

template <typename T>
constexpr bool is_scaled_gemm_v = is_scaled_gemm<T>::value;

template <typename T>
using require_scaled_gemm = internal::require_t<is_scaled_gemm<T>>;

/**
 * @brief True iff a summand routes through the GEMM epilogue.
 * @tparam T The type to test.
 * @note It IS a product, possibly wearing one deferred scalar (which becomes
 * the gemm's alpha_scale).
 */
template <typename T>
constexpr bool is_gemm_like_v = is_gemm_expr_v<T> || is_scaled_gemm_v<T>;

/**
 * @brief Requires an AdjointView or a TransposeView (over any inner operand).
 * @tparam T The type to constrain.
 * @note Used by the materialize driver to select the geam-based view lowering.
 */
template <typename T>
using require_adjoint_or_transpose_view =
    internal::require_t<Eigen::internal::bool_constant<is_adjoint_view_v<T> || is_transpose_view_v<T>>>;

template <typename T>
struct is_triangular_view : internal::is_triangular_view<std::decay_t<T>> {};

template <typename T>
constexpr bool is_triangular_view_v = is_triangular_view<T>::value;

template <typename T>
using require_triangular_view = internal::require_t<is_triangular_view<T>>;

template <typename... Types>
using require_all_triangular_view = internal::require_all_t<is_triangular_view_v<Types>...>;

template <typename T>
struct is_trsm_expr : internal::is_trsm_expr<std::decay_t<T>> {};

template <typename T>
constexpr bool is_trsm_expr_v = is_trsm_expr<T>::value;

template <typename T>
using require_trsm_expr = internal::require_t<is_trsm_expr<T>>;

template <typename... Types>
using require_all_trsm_expr = internal::require_all_t<is_trsm_expr_v<Types>...>;

template <typename T>
struct is_selfadjoint_view : internal::is_selfadjoint_view<std::decay_t<T>> {};

template <typename T>
constexpr bool is_selfadjoint_view_v = is_selfadjoint_view<T>::value;

template <typename T>
using require_selfadjoint_view = internal::require_t<is_selfadjoint_view<T>>;

template <typename... Types>
using require_all_selfadjoint_view = internal::require_all_t<is_selfadjoint_view_v<Types>...>;

template <typename T>
struct is_const_selfadjoint_view : internal::is_const_selfadjoint_view<std::decay_t<T>> {};

template <typename T>
constexpr bool is_const_selfadjoint_view_v = is_const_selfadjoint_view<T>::value;

template <typename T>
using require_const_selfadjoint_view = internal::require_t<is_const_selfadjoint_view<T>>;

template <typename... Types>
using require_all_const_selfadjoint_view = internal::require_all_t<is_const_selfadjoint_view_v<Types>...>;

template <typename T>
struct is_symm_expr : internal::is_symm_expr<std::decay_t<T>> {};

template <typename T>
constexpr bool is_symm_expr_v = is_symm_expr<T>::value;

template <typename T>
using require_symm_expr = internal::require_t<is_symm_expr<T>>;

template <typename... Types>
using require_all_symm_expr = internal::require_all_t<is_symm_expr_v<Types>...>;

template <typename T>
struct is_syrk_expr : internal::is_syrk_expr<std::decay_t<T>> {};

template <typename T>
constexpr bool is_syrk_expr_v = is_syrk_expr<T>::value;

template <typename T>
using require_syrk_expr = internal::require_t<is_syrk_expr<T>>;

template <typename... Types>
using require_all_syrk_expr = internal::require_all_t<is_syrk_expr_v<Types>...>;

template <typename T>
struct is_llt_solve_expr : internal::is_llt_solve_expr<std::decay_t<T>> {};

template <typename T>
constexpr bool is_llt_solve_expr_v = is_llt_solve_expr<T>::value;

template <typename T>
using require_llt_solve_expr = internal::require_t<is_llt_solve_expr<T>>;

template <typename... Types>
using require_all_llt_solve_expr = internal::require_all_t<is_llt_solve_expr_v<Types>...>;

template <typename T>
struct is_lu_solve_expr : internal::is_lu_solve_expr<std::decay_t<T>> {};

template <typename T>
constexpr bool is_lu_solve_expr_v = is_lu_solve_expr<T>::value;

template <typename T>
using require_lu_solve_expr = internal::require_t<is_lu_solve_expr<T>>;

template <typename... Types>
using require_all_lu_solve_expr = internal::require_all_t<is_lu_solve_expr_v<Types>...>;

template <typename T>
struct is_llt_view : internal::is_llt_view<std::decay_t<T>> {};

template <typename T>
constexpr bool is_llt_view_v = is_llt_view<T>::value;

template <typename T>
using require_llt_view = internal::require_t<is_llt_view<T>>;

template <typename... Types>
using require_all_llt_view = internal::require_all_t<is_llt_view_v<Types>...>;

template <typename T>
struct is_lu_view : internal::is_lu_view<std::decay_t<T>> {};

template <typename T>
constexpr bool is_lu_view_v = is_lu_view<T>::value;

template <typename T>
using require_lu_view = internal::require_t<is_lu_view<T>>;

template <typename... Types>
using require_all_lu_view = internal::require_all_t<is_lu_view_v<Types>...>;

/**
 * @brief True for a factorization handle (d_A.lu() / d_A.llt()), which is not a matrix operand.
 * @tparam T The type to test.
 * @note The operator overloads (*, +, -) exclude these so `d_A.lu() * d_C` is
 * rejected at overload resolution instead of silently lowering the handle's
 * inner matrix. A future QR handle slots in here.
 */
template <typename T>
constexpr bool is_factor_expr_v = is_lu_view_v<T> || is_llt_view_v<T>;

template <typename T>
struct is_device_add_expr : internal::is_device_add_expr<std::decay_t<T>> {};

template <typename T>
constexpr bool is_device_add_expr_v = is_device_add_expr<T>::value;

template <typename T>
using require_device_add_expr = internal::require_t<is_device_add_expr<T>>;

template <typename... Types>
using require_all_device_add_expr = internal::require_all_t<is_device_add_expr_v<Types>...>;

template <typename T>
struct is_device_scaled_device : internal::is_device_scaled_device<std::decay_t<T>> {};

template <typename T>
constexpr bool is_device_scaled_device_v = is_device_scaled_device<T>::value;

template <typename T>
using require_device_scaled_device = internal::require_t<is_device_scaled_device<T>>;

template <typename... Types>
using require_all_device_scaled_device = internal::require_all_t<is_device_scaled_device_v<Types>...>;

template <typename T>
struct is_device_scalar : internal::is_device_scalar<std::decay_t<T>> {};

template <typename T>
constexpr bool is_device_scalar_v = is_device_scalar<T>::value;

template <typename T>
using require_device_scalar = internal::require_t<is_device_scalar<T>>;

template <typename... Types>
using require_all_device_scalar = internal::require_all_t<is_device_scalar_v<Types>...>;

/**
 * @name device_expr_traits query aliases
 *
 * Lazy readers of the member-inspecting internal::device_expr_traits, whose
 * specializations live in DeviceExpr.h. Declared from the FwdDecl.h forward
 * declaration; they resolve only when instantiated on a complete specialization.
 */
///@{

template <typename T>
using scalar_type_t = internal::scalar_type_t<std::decay_t<T>>;

template <typename T>
struct is_device_expr : Eigen::internal::bool_constant<internal::device_expr_traits<std::decay_t<T>>::is_device_expr> {
};

template <typename T>
constexpr bool is_device_expr_v = is_device_expr<T>::value;

template <typename T>
using require_device_expr = internal::require_t<is_device_expr<T>>;

template <typename... Types>
using require_all_device_expr = internal::require_all_t<is_device_expr_v<Types>...>;

template <typename T>
constexpr GpuOp trans_op = internal::device_expr_traits<std::decay_t<T>>::op;

///@}

namespace internal {
template <typename T>
struct is_complex : Eigen::internal::bool_constant<false> {};

template <typename T>
struct is_complex<std::complex<T>> : Eigen::internal::bool_constant<true> {};
}  // namespace internal

template <typename T>
struct is_complex : internal::is_complex<std::decay_t<T>> {};

template <typename T>
constexpr bool is_complex_v = is_complex<T>::value;
template <typename T>
struct is_host_scalar : Eigen::internal::bool_constant<std::is_floating_point<std::decay_t<T>>::value ||
                                                       std::is_integral<std::decay_t<T>>::value || is_complex_v<T>> {};

template <typename T>
using require_host_scalar = internal::require_t<is_host_scalar<T>>;

///@}

}  // namespace gpu
}  // namespace Eigen

#endif  // EIGEN_GPU_TYPE_TRAITS_H
