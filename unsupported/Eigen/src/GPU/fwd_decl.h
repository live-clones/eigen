// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Steve Bronder <sbronder@flatironinstitute.org>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Shared forward declarations for the GPU module. Included by GpuSupport.h
// and other headers to break circular includes between DeviceMatrix,
// DeviceBuffer, and the expression / view / solver class templates.

#ifndef EIGEN_GPU_FWD_DECL_H
#define EIGEN_GPU_FWD_DECL_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {
namespace gpu {

// Device-resident types.
template <typename Scalar_>
class DeviceMatrix;

// Solvers.
template <typename, int>
class LLT;
template <typename>
class LU;

// Views.
template <typename>
class AdjointView;
template <typename>
class TransposeView;
template <typename, int>
class LLTView;
template <typename>
class LUView;
template <typename, int>
class TriangularView;
template <typename, int>
class SelfAdjointView;
template <typename, int>
class ConstSelfAdjointView;

// Expressions.
template <typename>
class Assignment;
template <typename, typename>
class GemmExpr;
template <typename, int>
class LltSolveExpr;
template <typename>
class LuSolveExpr;
template <typename, int>
class TrsmExpr;
template <typename, int>
class SymmExpr;
template <typename, int>
class SyrkExpr;

// Execution context.
class Context;

namespace internal {
class DeviceBuffer;
}  // namespace internal

}  // namespace gpu
}  // namespace Eigen

#endif  // EIGEN_GPU_FWD_DECL_H
