// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// A failed CUDA runtime or library call has to reach EIGEN_GPU_CHECK_FAILED in a
// release build as much as in a debug one (runtime_checks_ndebug.cpp compiles
// this file with EIGEN_NO_DEBUG). The hook is overridden here to record the
// failure instead of stopping the process. The failures are rejected at the call
// and leave the context usable: none is a sticky error.

#include <string>

struct RecordedGpuFailure {
  std::string error;
  std::string expression;
  std::string file;
  int line = 0;
};
static RecordedGpuFailure g_last_failure;
static int g_num_failures = 0;

static void record_gpu_failure(const char* error, const char* expression, const char* file, int line) {
  ++g_num_failures;
  g_last_failure.error = error;
  g_last_failure.expression = expression;
  g_last_failure.file = file;
  g_last_failure.line = line;
}
#define EIGEN_GPU_CHECK_FAILED(error, expression, file, line) ::record_gpu_failure(error, expression, file, line)

#define EIGEN_USE_GPU
#include "main.h"
#include <contrib/Eigen/GPU>

#include "./gpu_test_helpers.h"

using namespace Eigen;

static bool starts_with(const std::string& s, const std::string& prefix) {
  return s.compare(0, prefix.size(), prefix) == 0;
}

static bool ends_with(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// A runtime call that fails inside the module is reported with the call's own
// text and location: downloading a view of a null pointer copies from null.
void test_runtime_call() {
  gpu::Context ctx;
  g_num_failures = 0;
  gpu::DeviceMatrix<double> bogus = gpu::DeviceMatrix<double>::view(ctx, nullptr, 4, 1);
  (void)bogus.toHost(ctx);
  VERIFY_IS_EQUAL(g_num_failures, 1);
  VERIFY_IS_EQUAL(g_last_failure.error, std::string("cudaErrorInvalidValue"));
  VERIFY(starts_with(g_last_failure.expression, "cudaMemcpyAsync"));
  VERIFY(ends_with(g_last_failure.file, "DeviceDispatch.h"));
  VERIFY(g_last_failure.line > 0);

  // The context is still usable, and a successful call reports nothing.
  const VectorXd x = VectorXd::Random(8);
  VERIFY_IS_APPROX(VectorXd(gpu::DeviceMatrix<double>::fromHost(ctx, x).toHost(ctx)), x);
  VERIFY_IS_EQUAL(g_num_failures, 1);
}

// Each library's check macro reports a failed status, named where the library
// can name it.
void test_library_checks() {
  gpu::Context ctx;
  g_num_failures = 0;

  EIGEN_CUBLAS_CHECK(cublasSetStream(nullptr, ctx.stream()));
  VERIFY_IS_EQUAL(g_num_failures, 1);
  VERIFY(starts_with(g_last_failure.error, "CUBLAS_STATUS_"));
  VERIFY(starts_with(g_last_failure.expression, "cublasSetStream"));

  EIGEN_CUBLASLT_CHECK(cublasLtMatmulDescCreate(nullptr, CUBLAS_COMPUTE_64F, CUDA_R_64F));
  VERIFY_IS_EQUAL(g_num_failures, 2);
  VERIFY(starts_with(g_last_failure.error, "CUBLAS_STATUS_"));

  // cuSOLVER dereferences a null handle, so pass a negative size instead.
  int lwork = 0;
  EIGEN_CUSOLVER_CHECK(
      cusolverDnDpotrf_bufferSize(ctx.cusolverHandle(), CUBLAS_FILL_MODE_LOWER, -1, nullptr, 1, &lwork));
  VERIFY_IS_EQUAL(g_num_failures, 3);
  VERIFY_IS_EQUAL(g_last_failure.error, std::string("CUSOLVER_STATUS_INVALID_VALUE"));

  EIGEN_CUSPARSE_CHECK(cusparseSetStream(nullptr, ctx.stream()));
  VERIFY_IS_EQUAL(g_num_failures, 4);
  VERIFY(starts_with(g_last_failure.error, "CUSPARSE_STATUS_"));

  EIGEN_CUFFT_CHECK(cufftSetStream(cufftHandle(-1), ctx.stream()));
  VERIFY_IS_EQUAL(g_num_failures, 5);
  VERIFY(starts_with(g_last_failure.error, "cuFFT status "));

  EIGEN_NPP_CHECK(nppsSqrt_64f_I_Ctx(nullptr, 1, gpu::internal::make_npp_stream_ctx(ctx.stream())));
  VERIFY_IS_EQUAL(g_num_failures, 6);
  VERIFY(starts_with(g_last_failure.error, "NPP status "));
  VERIFY(ends_with(g_last_failure.file, "runtime_checks.cpp"));
}

EIGEN_DECLARE_TEST(gpu_runtime_checks) {
  gpu_test::require_cuda_device();
  CALL_SUBTEST(test_runtime_call());
  CALL_SUBTEST(test_library_checks());
}
