/* 
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 * Assign_AOCL.h - AOCL Dispatch Layer for Eigen
 *
 * Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Description:
 * ------------
 * This file implements the dispatch layer that routes Eigen’s vectorized math operations
 * (e.g., exp, sin, cos, sqrt, log, log10, add, pow) to AOCL MathLib functions for double
 * precision, with scalar fallbacks for float precision. Requires EIGEN_AOCL_VML_THRESHOLD
 * from AOCL_support.h.
 *
 * Example Usage:
 * --------------
 * Eigen expressions like x.array().exp() or x.array().pow(y.array()) are routed to
 * AOCL MathLib functions for double, or scalar ops for float.
 *
 * Developer:
 * ----------
 * Name: Sharad Saurabh Bhaskar
 * Email: shbhaska@amd.com
 */

#ifndef EIGEN_ASSIGN_AOCL_H
#define EIGEN_ASSIGN_AOCL_H

#include <Eigen/Core>
#include <cmath>
#include <cassert>
#include <type_traits>
#include "AOCL_Support.h"

#include "amdlibm_vec.h"

// Define the SIMD width for AOCL MathLib (for AVX-512, typically 8 doubles).
#ifndef AOCL_SIMD_WIDTH
#define AOCL_SIMD_WIDTH 8
#endif

namespace Eigen {
namespace internal {

// Traits for unary operations.
template <typename Dst, typename Src>
class aocl_assign_traits {
 private:
  enum {
    DstHasDirectAccess = Dst::Flags & DirectAccessBit,
    SrcHasDirectAccess = Src::Flags & DirectAccessBit,
    StorageOrdersAgree = (int(Dst::IsRowMajor) == int(Src::IsRowMajor)),
    InnerSize = Dst::IsVectorAtCompileTime   ? int(Dst::SizeAtCompileTime)
                : (Dst::Flags & RowMajorBit) ? int(Dst::ColsAtCompileTime)
                                             : int(Dst::RowsAtCompileTime),
    LargeEnough = (InnerSize == Dynamic) || (InnerSize >= EIGEN_AOCL_VML_THRESHOLD)
  };

 public:
  enum {
    EnableAoclVML = DstHasDirectAccess && SrcHasDirectAccess && StorageOrdersAgree && LargeEnough,
    Traversal = LinearTraversal
  };
};

// Traits for binary operations (e.g., add, pow).
template <typename Dst, typename Lhs, typename Rhs>
class aocl_assign_binary_traits {
 private:
  enum {
    DstHasDirectAccess = Dst::Flags & DirectAccessBit,
    LhsHasDirectAccess = Lhs::Flags & DirectAccessBit,
    RhsHasDirectAccess = Rhs::Flags & DirectAccessBit,
    StorageOrdersAgree =
        (int(Dst::IsRowMajor) == int(Lhs::IsRowMajor)) && (int(Dst::IsRowMajor) == int(Rhs::IsRowMajor)),
    InnerSize = Dst::IsVectorAtCompileTime   ? int(Dst::SizeAtCompileTime)
                : (Dst::Flags & RowMajorBit) ? int(Dst::ColsAtCompileTime)
                                             : int(Dst::RowsAtCompileTime),
    LargeEnough = (InnerSize == Dynamic) || (InnerSize >= EIGEN_AOCL_VML_THRESHOLD)
  };

 public:
  enum {
    EnableAoclVML = DstHasDirectAccess && LhsHasDirectAccess && RhsHasDirectAccess && StorageOrdersAgree && LargeEnough
  };
};

// Unary operation dispatch for float (scalar fallback).
#define EIGEN_AOCL_VML_UNARY_CALL_FLOAT(EIGENOP)                                                                   \
  template <typename DstXprType, typename SrcXprNested>                                                            \
  struct Assignment<DstXprType, CwiseUnaryOp<scalar_##EIGENOP##_op<float>, SrcXprNested>, assign_op<float, float>, \
                    Dense2Dense, std::enable_if_t<aocl_assign_traits<DstXprType, SrcXprNested>::EnableAoclVML> > { \
    typedef CwiseUnaryOp<scalar_##EIGENOP##_op<float>, SrcXprNested> SrcXprType;                                   \
    static void run(DstXprType& dst, const SrcXprType& src, const assign_op<float, float>&) {                      \
      eigen_assert(dst.rows() == src.rows() && dst.cols() == src.cols());                                          \
      int n = internal::convert_index<int>(dst.size());                                                            \
      if (n <= 0) return;                                                                                          \
      const float* input = reinterpret_cast<const float*>(src.nestedExpression().data());                          \
      float* output = reinterpret_cast<float*>(dst.data());                                                        \
      for (int i = 0; i < n; ++i) {                                                                                \
        output[i] = std::EIGENOP(input[i]);                                                                        \
      }                                                                                                            \
    }                                                                                                              \
  };

// Unary operation dispatch for double (AOCL vectorized).
#define EIGEN_AOCL_VML_UNARY_CALL_DOUBLE(EIGENOP, AOCLOP)                                                             \
  template <typename DstXprType, typename SrcXprNested>                                                               \
  struct Assignment<DstXprType, CwiseUnaryOp<scalar_##EIGENOP##_op<double>, SrcXprNested>, assign_op<double, double>, \
                    Dense2Dense, std::enable_if_t<aocl_assign_traits<DstXprType, SrcXprNested>::EnableAoclVML> > {    \
    typedef CwiseUnaryOp<scalar_##EIGENOP##_op<double>, SrcXprNested> SrcXprType;                                     \
    static void run(DstXprType& dst, const SrcXprType& src, const assign_op<double, double>&) {                       \
      eigen_assert(dst.rows() == src.rows() && dst.cols() == src.cols());                                             \
      int n = internal::convert_index<int>(dst.size());                                                               \
      if (n <= 0) return;                                                                                             \
      const double* input = reinterpret_cast<const double*>(src.nestedExpression().data());                           \
      double* output = reinterpret_cast<double*>(dst.data());                                                         \
      int simdBlocks = n / AOCL_SIMD_WIDTH;                                                                           \
      int remainder = n % AOCL_SIMD_WIDTH;                                                                            \
      if (simdBlocks > 0) {                                                                                           \
        AOCLOP(simdBlocks* AOCL_SIMD_WIDTH, const_cast<double*>(input), output);                                      \
      }                                                                                                               \
      if (remainder > 0) {                                                                                            \
        int offset = simdBlocks * AOCL_SIMD_WIDTH;                                                                    \
        for (int i = 0; i < remainder; ++i) {                                                                         \
          output[offset + i] = std::EIGENOP(input[offset + i]);                                                       \
        }                                                                                                             \
      }                                                                                                               \
    }                                                                                                                 \
  };

// Instantiate unary calls for float (scalar).
// EIGEN_AOCL_VML_UNARY_CALL_FLOAT(exp)
// EIGEN_AOCL_VML_UNARY_CALL_FLOAT(sin)
// EIGEN_AOCL_VML_UNARY_CALL_FLOAT(cos)
// EIGEN_AOCL_VML_UNARY_CALL_FLOAT(sqrt)
// EIGEN_AOCL_VML_UNARY_CALL_FLOAT(log)
// EIGEN_AOCL_VML_UNARY_CALL_FLOAT(log10)

// Instantiate unary calls for double (AOCL vectorized).
EIGEN_AOCL_VML_UNARY_CALL_DOUBLE(exp, amd_vrda_exp)
EIGEN_AOCL_VML_UNARY_CALL_DOUBLE(sin, amd_vrda_sin)
EIGEN_AOCL_VML_UNARY_CALL_DOUBLE(cos, amd_vrda_cos)
EIGEN_AOCL_VML_UNARY_CALL_DOUBLE(sqrt, amd_vrda_sqrt)
EIGEN_AOCL_VML_UNARY_CALL_DOUBLE(log, amd_vrda_log)
EIGEN_AOCL_VML_UNARY_CALL_DOUBLE(log10, amd_vrda_log10)
// EIGEN_AOCL_VML_UNARY_CALL_DOUBLE(asin, amd_vrda_asin)
// EIGEN_AOCL_VML_UNARY_CALL_DOUBLE(sinh, amd_vrda_sinh)
// EIGEN_AOCL_VML_UNARY_CALL_DOUBLE(acos, amd_vrda_acos)
// EIGEN_AOCL_VML_UNARY_CALL_DOUBLE(cosh, amd_vrda_cosh)
// EIGEN_AOCL_VML_UNARY_CALL_DOUBLE(tan, amd_vrda_tan)
// EIGEN_AOCL_VML_UNARY_CALL_DOUBLE(atan, amd_vrda_atan)
// EIGEN_AOCL_VML_UNARY_CALL_DOUBLE(tanh, amd_vrda_tanh)
EIGEN_AOCL_VML_UNARY_CALL_DOUBLE(log2, amd_vrda_log2)
// Binary operation dispatch for float (scalar fallback).
#define EIGEN_AOCL_VML_BINARY_CALL_FLOAT(EIGENOP, STDFUNC)                                                   \
  template <typename DstXprType, typename LhsXprNested, typename RhsXprNested>                               \
  struct Assignment<                                                                                         \
      DstXprType, CwiseBinaryOp<scalar_##EIGENOP##_op<float, float>, LhsXprNested, RhsXprNested>,            \
      assign_op<float, float>, Dense2Dense,                                                                  \
      std::enable_if_t<aocl_assign_binary_traits<DstXprType, LhsXprNested, RhsXprNested>::EnableAoclVML> > { \
    typedef CwiseBinaryOp<scalar_##EIGENOP##_op<float, float>, LhsXprNested, RhsXprNested> SrcXprType;       \
    static void run(DstXprType& dst, const SrcXprType& src, const assign_op<float, float>&) {                \
      eigen_assert(dst.rows() == src.rows() && dst.cols() == src.cols());                                    \
      int n = internal::convert_index<int>(dst.size());                                                      \
      if (n <= 0) return;                                                                                    \
      const float* lhs = reinterpret_cast<const float*>(src.lhs().data());                                   \
      const float* rhs = reinterpret_cast<const float*>(src.rhs().data());                                   \
      float* output = reinterpret_cast<float*>(dst.data());                                                  \
      for (int i = 0; i < n; ++i) {                                                                          \
        output[i] = STDFUNC(lhs[i], rhs[i]);                                                                 \
      }                                                                                                      \
    }                                                                                                        \
  };

// Binary operation dispatch for double (AOCL vectorized).
#define EIGEN_AOCL_VML_BINARY_CALL_DOUBLE(EIGENOP, AOCLOP)                                                   \
  template <typename DstXprType, typename LhsXprNested, typename RhsXprNested>                               \
  struct Assignment<                                                                                         \
      DstXprType, CwiseBinaryOp<scalar_##EIGENOP##_op<double, double>, LhsXprNested, RhsXprNested>,          \
      assign_op<double, double>, Dense2Dense,                                                                \
      std::enable_if_t<aocl_assign_binary_traits<DstXprType, LhsXprNested, RhsXprNested>::EnableAoclVML> > { \
    typedef CwiseBinaryOp<scalar_##EIGENOP##_op<double, double>, LhsXprNested, RhsXprNested> SrcXprType;     \
    static void run(DstXprType& dst, const SrcXprType& src, const assign_op<double, double>&) {              \
      eigen_assert(dst.rows() == src.rows() && dst.cols() == src.cols());                                    \
      int n = internal::convert_index<int>(dst.size());                                                      \
      if (n <= 0) return;                                                                                    \
      const double* lhs = reinterpret_cast<const double*>(src.lhs().data());                                 \
      const double* rhs = reinterpret_cast<const double*>(src.rhs().data());                                 \
      double* output = reinterpret_cast<double*>(dst.data());                                                \
      AOCLOP(n, const_cast<double*>(lhs), const_cast<double*>(rhs), output);                                 \
    }                                                                                                        \
  };

// Instantiate binary calls for float (scalar).
// EIGEN_AOCL_VML_BINARY_CALL_FLOAT(sum, std::plus<float>)  // Using scalar_sum_op for addition
// EIGEN_AOCL_VML_BINARY_CALL_FLOAT(pow, std::pow)

// Instantiate binary calls for double (AOCL vectorized).
EIGEN_AOCL_VML_BINARY_CALL_DOUBLE(sum, amd_vrda_add)  // Using scalar_sum_op for addition
EIGEN_AOCL_VML_BINARY_CALL_DOUBLE(pow, amd_vrda_pow)

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_ASSIGN_AOCL_H
