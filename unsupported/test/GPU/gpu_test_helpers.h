// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// Helpers shared across GPU library tests:
//   * runtime probes that exit(77) (CI skip) when a library is unavailable
//   * a small make_test_value() that constructs a Scalar with an imaginary
//     component for complex types so the complex code paths are genuinely
//     exercised — without this, Scalar(real_value) silently zeros the imag.

#ifndef EIGEN_UNSUPPORTED_TEST_GPU_TEST_HELPERS_H
#define EIGEN_UNSUPPORTED_TEST_GPU_TEST_HELPERS_H

#include <Eigen/Core>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <type_traits>

namespace gpu_test {

template <typename Scalar, typename RealScalar>
inline std::enable_if_t<Eigen::NumTraits<Scalar>::IsComplex, Scalar> make_test_value(RealScalar re, RealScalar im) {
  return Scalar(re, im);
}
template <typename Scalar, typename RealScalar>
inline std::enable_if_t<!Eigen::NumTraits<Scalar>::IsComplex, Scalar> make_test_value(RealScalar re,
                                                                                      RealScalar /*im*/) {
  return Scalar(re);
}

#ifdef CUDSS_VERSION
inline void require_cudss_context() {
  cudssHandle_t handle = nullptr;
  const cudssStatus_t status = cudssCreate(&handle);
  if (status != CUDSS_STATUS_SUCCESS) {
    std::cout << "SKIP: cuDSS tests require an initialized cuDSS context. cudssCreate failed with status "
              << static_cast<int>(status) << std::endl;
    std::exit(77);
  }
  EIGEN_CUDSS_CHECK(cudssDestroy(handle));
}
#endif

#ifdef CUSPARSE_VERSION
inline void require_cusparse_context() {
  cusparseHandle_t handle = nullptr;
  const cusparseStatus_t status = cusparseCreate(&handle);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    std::cout << "SKIP: cuSPARSE tests require an initialized cuSPARSE context. cusparseCreate failed with status "
              << static_cast<int>(status) << std::endl;
    std::exit(77);
  }
  EIGEN_CUSPARSE_CHECK(cusparseDestroy(handle));
}
#endif

#ifdef CUFFT_VERSION
inline void require_cufft_context() {
  cufftHandle plan = 0;
  // cufftCreate allocates a plan handle without configuring it; succeeds only
  // when the cuFFT runtime is loadable.
  const cufftResult status = cufftCreate(&plan);
  if (status != CUFFT_SUCCESS) {
    std::cout << "SKIP: cuFFT tests require a working cuFFT runtime. cufftCreate failed with status "
              << static_cast<int>(status) << std::endl;
    std::exit(77);
  }
  EIGEN_CUFFT_CHECK(cufftDestroy(plan));
}
#endif

#ifdef CUDART_VERSION
// A host-function gate that holds a CUDA stream closed until opened from the
// host, so tests can keep enqueued work "in flight" at a deterministic point:
//   EIGEN_CUDA_RUNTIME_CHECK(cudaLaunchHostFunc(stream, gpu_test::wait_for_stream_gate, &gate));
struct StreamGate {
  std::atomic<bool> open{false};
};

inline void CUDART_CB wait_for_stream_gate(void* user_data) {
  auto* gate = static_cast<StreamGate*>(user_data);
  while (!gate->open.load(std::memory_order_acquire)) std::this_thread::yield();
}

// Opens \p gate from a helper thread after \p delay_ms, so a host thread that
// then blocks on the gated stream (a draining destructor, HostTransfer::get(),
// cudaStreamSynchronize) always makes progress. Joins on destruction.
class StreamGateOpener {
 public:
  StreamGateOpener(StreamGate& gate, int delay_ms)
      : thread_([&gate, delay_ms] {
          std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
          gate.open.store(true, std::memory_order_release);
        }) {}
  ~StreamGateOpener() {
    if (thread_.joinable()) thread_.join();
  }
  StreamGateOpener(const StreamGateOpener&) = delete;
  StreamGateOpener& operator=(const StreamGateOpener&) = delete;

 private:
  std::thread thread_;
};
#endif

}  // namespace gpu_test

#endif  // EIGEN_UNSUPPORTED_TEST_GPU_TEST_HELPERS_H
