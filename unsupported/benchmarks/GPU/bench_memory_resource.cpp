// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// What each gpu::MemoryResource costs, measured through DeviceMatrix.
//
// bench_unified_memory measures the same storage kinds on raw CUDA calls, which
// is a ceiling. This one runs them through the module, and answers three things
// the ceiling cannot:
//
//   Alloc_*       does routing allocation through a virtual MemoryResource cost
//                 anything against the direct device_malloc path it replaced?
//   RoundTrip_*   operands in, GEMM, result readable on the host -- per resource
//   HostReadLoop_*  the host reads device output every iteration, so whatever
//                 syncHost() has to do is charged per iteration rather than once
//   Gemm_*        GEMM throughput with operands in each storage kind
//   DotRead_*     the same question one type down: a reduction whose
//                 DeviceScalar result is read on the host every iteration, which
//                 is the inner loop of every convergence check
//
// Build (standalone project, see CMakeLists.txt in this directory):
//   cmake -G Ninja -B build-bench-gpu -S unsupported/benchmarks/GPU \
//         -DCMAKE_CUDA_ARCHITECTURES=87
//   cmake --build build-bench-gpu --target bench_memory_resource

#ifndef EIGEN_USE_GPU
#define EIGEN_USE_GPU
#endif
#include <Eigen/Core>
#include <unsupported/Eigen/GPU>

#include <benchmark/benchmark.h>

#ifndef SCALAR
#define SCALAR float
#endif
using Scalar = SCALAR;

using Eigen::Index;
namespace gpu = Eigen::gpu;
using HostMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
using HostMap = Eigen::Map<HostMatrix>;

namespace {

enum class Kind { Device, Managed, Mapped, Registered };

gpu::MemoryResource& resourceFor(Kind k) {
  switch (k) {
    case Kind::Managed:
      return gpu::managedMemoryResource();
    case Kind::Mapped:
      return gpu::mappedHostMemoryResource();
    case Kind::Registered:
      return gpu::registeredHostMemoryResource();
    case Kind::Device:
    default:
      return gpu::deviceMemoryResource();
  }
}

void setFlopsCounter(benchmark::State& state, Index n) {
  const double flops = 2.0 * double(n) * double(n) * double(n);
  state.counters["GFLOPS"] = benchmark::Counter(flops * double(state.iterations()), benchmark::Counter::kIsRate,
                                                benchmark::Counter::OneK::kIs1000);
}

}  // namespace

// ---------------------------------------------------------------------------
// Allocation. The direct case is the pre-abstraction baseline. The default and
// explicitly named device-resource cases show the wrapper and virtual-call
// overhead separately.
// ---------------------------------------------------------------------------

static void BM_Alloc_Direct(benchmark::State& state) {
  const Index n = state.range(0);
  const size_t bytes = static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(Scalar);
  for (auto _ : state) {
    void* p = gpu::internal::device_malloc(bytes);
    benchmark::DoNotOptimize(p);
    gpu::internal::device_free(p);
  }
}
BENCHMARK(BM_Alloc_Direct)->Arg(64)->Arg(1024)->Unit(benchmark::kMicrosecond)->UseRealTime()->MinWarmUpTime(0.5);

static void BM_Alloc_Default(benchmark::State& state) {
  const Index n = state.range(0);
  for (auto _ : state) {
    gpu::DeviceMatrix<Scalar> m(n, n);
    benchmark::DoNotOptimize(m.data());
  }
}
BENCHMARK(BM_Alloc_Default)->Arg(64)->Arg(1024)->Unit(benchmark::kMicrosecond)->UseRealTime()->MinWarmUpTime(0.5);

static void allocBench(benchmark::State& state, Kind k) {
  const Index n = state.range(0);
  gpu::MemoryResource& r = resourceFor(k);
  for (auto _ : state) {
    gpu::DeviceMatrix<Scalar> m(n, n, r);
    benchmark::DoNotOptimize(m.data());
  }
}

#define ALLOC_CASE(Name)                                                                  \
  static void BM_Alloc_##Name(benchmark::State& state) { allocBench(state, Kind::Name); } \
  BENCHMARK(BM_Alloc_##Name)->Arg(64)->Arg(1024)->Unit(benchmark::kMicrosecond)->UseRealTime()->MinWarmUpTime(0.5)

ALLOC_CASE(Device);
ALLOC_CASE(Managed);
ALLOC_CASE(Mapped);
ALLOC_CASE(Registered);

// ---------------------------------------------------------------------------
// Round trip: two operands in, GEMM, result readable on the host.
// ---------------------------------------------------------------------------

// Today's path: device-only storage with explicit upload and download.
static void BM_RoundTrip_Device(benchmark::State& state) {
  const Index n = state.range(0);
  const HostMatrix ha = HostMatrix::Random(n, n);
  const HostMatrix hb = HostMatrix::Random(n, n);
  HostMatrix hc(n, n);
  gpu::Context ctx;
  gpu::DeviceMatrix<Scalar> a(n, n), b(n, n), c(n, n);
  const size_t bytes = static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(Scalar);
  for (auto _ : state) {
    EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(a.data(), ha.data(), bytes, cudaMemcpyHostToDevice, ctx.stream()));
    EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(b.data(), hb.data(), bytes, cudaMemcpyHostToDevice, ctx.stream()));
    c.device(ctx) = a * b;
    EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(hc.data(), c.data(), bytes, cudaMemcpyDeviceToHost, ctx.stream()));
    EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(ctx.stream()));
    benchmark::DoNotOptimize(hc.data());
  }
  setFlopsCounter(state, n);
}
BENCHMARK(BM_RoundTrip_Device)
    ->Arg(256)
    ->Arg(1024)
    ->Arg(4096)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->MinWarmUpTime(0.5);

// Host-accessible storage: written and read in place, no transfer anywhere.
static void roundTripBench(benchmark::State& state, Kind k) {
  const Index n = state.range(0);
  const HostMatrix ha = HostMatrix::Random(n, n);
  const HostMatrix hb = HostMatrix::Random(n, n);
  gpu::MemoryResource& r = resourceFor(k);
  gpu::Context ctx;
  gpu::DeviceMatrix<Scalar> a(n, n, r), b(n, n, r), c(n, n, r);
  for (auto _ : state) {
    HostMap(a.hostData(), n, n) = ha;
    HostMap(b.hostData(), n, n) = hb;
    c.device(ctx) = a * b;
    c.syncHost(ctx);
    Scalar observed = HostMap(c.hostData(), n, n).coeff(0, 0);
    benchmark::DoNotOptimize(observed);
  }
  setFlopsCounter(state, n);
}

#define ROUNDTRIP_CASE(Name)                                                                      \
  static void BM_RoundTrip_##Name(benchmark::State& state) { roundTripBench(state, Kind::Name); } \
  BENCHMARK(BM_RoundTrip_##Name)                                                                  \
      ->Arg(256)                                                                                  \
      ->Arg(1024)                                                                                 \
      ->Arg(4096)                                                                                 \
      ->Unit(benchmark::kMicrosecond)                                                             \
      ->UseRealTime()                                                                             \
      ->MinWarmUpTime(0.5)

ROUNDTRIP_CASE(Managed);
ROUNDTRIP_CASE(Mapped);
ROUNDTRIP_CASE(Registered);

// ---------------------------------------------------------------------------
// Host-read loop: the host inspects device output every iteration, the shape an
// iterative solver has when it tests a residual. Managed storage on a device
// without concurrentManagedAccess pays a device-wide synchronize here; the
// page-locked kinds only wait on the stream.
// ---------------------------------------------------------------------------

static void BM_HostReadLoop_DeviceOnly(benchmark::State& state) {
  const Index n = state.range(0);
  const HostMatrix ha = HostMatrix::Random(n, n);
  const HostMatrix hb = HostMatrix::Random(n, n);
  gpu::Context ctx;
  auto a = gpu::DeviceMatrix<Scalar>::fromHost(ha, ctx.stream());
  auto b = gpu::DeviceMatrix<Scalar>::fromHost(hb, ctx.stream());
  gpu::DeviceMatrix<Scalar> c(n, n);
  for (auto _ : state) {
    c.device(ctx) = a * b;
    EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(ctx.stream()));
  }
  setFlopsCounter(state, n);
}
BENCHMARK(BM_HostReadLoop_DeviceOnly)
    ->Arg(64)
    ->Arg(256)
    ->Arg(1024)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->MinWarmUpTime(0.5);

static void hostReadLoopBench(benchmark::State& state, Kind k) {
  const Index n = state.range(0);
  const HostMatrix ha = HostMatrix::Random(n, n);
  const HostMatrix hb = HostMatrix::Random(n, n);
  gpu::MemoryResource& r = resourceFor(k);
  gpu::Context ctx;
  gpu::DeviceMatrix<Scalar> a(n, n, r), b(n, n, r), c(n, n, r);
  HostMap(a.hostData(), n, n) = ha;
  HostMap(b.hostData(), n, n) = hb;
  for (auto _ : state) {
    c.device(ctx) = a * b;
    c.syncHost(ctx);
    Scalar observed = HostMap(c.hostData(), n, n).coeff(0, 0);
    benchmark::DoNotOptimize(observed);
  }
  setFlopsCounter(state, n);
}

#define HOSTREAD_CASE(Name)                                                                             \
  static void BM_HostReadLoop_##Name(benchmark::State& state) { hostReadLoopBench(state, Kind::Name); } \
  BENCHMARK(BM_HostReadLoop_##Name)                                                                     \
      ->Arg(64)                                                                                         \
      ->Arg(256)                                                                                        \
      ->Arg(1024)                                                                                       \
      ->Unit(benchmark::kMicrosecond)                                                                   \
      ->UseRealTime()                                                                                   \
      ->MinWarmUpTime(0.5)

HOSTREAD_CASE(Managed);
HOSTREAD_CASE(Mapped);
HOSTREAD_CASE(Registered);

// ---------------------------------------------------------------------------
// Reduce-then-read: dot() into each storage kind, read on the host every
// iteration. Device-only pays cudaMemcpyAsync + cudaStreamSynchronize; a
// host-accessible result drops the copy and keeps only the synchronization.
// This is what an iterative solver does per iteration, so the difference is
// charged per iteration rather than once.
// ---------------------------------------------------------------------------

static void BM_DotRead_DeviceOnly(benchmark::State& state) {
  const Index n = state.range(0);
  const HostMatrix hx = HostMatrix::Random(n, 1);
  gpu::Context ctx;
  auto x = gpu::DeviceMatrix<Scalar>::fromHost(hx, ctx.stream());
  auto y = gpu::DeviceMatrix<Scalar>::fromHost(hx, ctx.stream());
  for (auto _ : state) {
    benchmark::DoNotOptimize(Scalar(x.dot(ctx, y)));
  }
}
BENCHMARK(BM_DotRead_DeviceOnly)
    ->Arg(1024)
    ->Arg(65536)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->MinWarmUpTime(0.5);

static void dotReadBench(benchmark::State& state, Kind k) {
  const Index n = state.range(0);
  const HostMatrix hx = HostMatrix::Random(n, 1);
  gpu::MemoryResource& r = resourceFor(k);
  gpu::Context ctx;
  auto x = gpu::DeviceMatrix<Scalar>::fromHost(hx, ctx.stream());
  auto y = gpu::DeviceMatrix<Scalar>::fromHost(hx, ctx.stream());
  for (auto _ : state) {
    benchmark::DoNotOptimize(Scalar(x.dot(ctx, y, r)));
  }
}

#define DOTREAD_CASE(Name)                                                                    \
  static void BM_DotRead_##Name(benchmark::State& state) { dotReadBench(state, Kind::Name); } \
  BENCHMARK(BM_DotRead_##Name)->Arg(1024)->Arg(65536)->Unit(benchmark::kMicrosecond)->UseRealTime()->MinWarmUpTime(0.5)

DOTREAD_CASE(Managed);
DOTREAD_CASE(Mapped);
DOTREAD_CASE(Registered);

// ---------------------------------------------------------------------------
// GEMM throughput by operand storage: does the kernel care where the bytes are?
// ---------------------------------------------------------------------------

static void gemmBench(benchmark::State& state, Kind k) {
  const Index n = state.range(0);
  gpu::MemoryResource& r = resourceFor(k);
  gpu::Context ctx;
  gpu::DeviceMatrix<Scalar> a(n, n, r), b(n, n, r);
  gpu::DeviceMatrix<Scalar> c(n, n);  // destination always device-only
  if (a.isHostAccessible()) {
    HostMap(a.hostData(), n, n) = HostMatrix::Random(n, n);
    HostMap(b.hostData(), n, n) = HostMatrix::Random(n, n);
  }
  for (auto _ : state) {
    c.device(ctx) = a * b;
    EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(ctx.stream()));
  }
  setFlopsCounter(state, n);
}

#define GEMM_CASE(Name)                                                                 \
  static void BM_Gemm_##Name(benchmark::State& state) { gemmBench(state, Kind::Name); } \
  BENCHMARK(BM_Gemm_##Name)                                                             \
      ->Arg(256)                                                                        \
      ->Arg(1024)                                                                       \
      ->Arg(4096)                                                                       \
      ->Unit(benchmark::kMicrosecond)                                                   \
      ->UseRealTime()                                                                   \
      ->MinWarmUpTime(0.5)

GEMM_CASE(Device);
GEMM_CASE(Managed);
GEMM_CASE(Mapped);
GEMM_CASE(Registered);

// ---------------------------------------------------------------------------

static void BM_ReportCapabilities(benchmark::State& state) {
  for (auto _ : state) benchmark::DoNotOptimize(state.iterations());
  state.counters["integrated"] = gpu::deviceIsIntegrated() ? 1 : 0;
  state.counters["concurrentManagedAccess"] = gpu::deviceSupportsConcurrentManagedAccess() ? 1 : 0;
}
BENCHMARK(BM_ReportCapabilities)->Iterations(1);
