// Benchmarks for the SVE PacketMath implementations of `plset`, `ploaddup`,
// `ploadquad`, `predux_mul`, and `ptranspose`. To compare against a prior
// implementation, build and run this same file against the Eigen checkout
// in question -- it only calls the public `Eigen::internal` packet API, so
// it is source-compatible with whatever PacketMath.h happens to provide.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>

#if defined(EIGEN_VECTORIZE_SVE)

#include <cstdint>

namespace Eigen {
namespace {

using internal::packet_traits;
using internal::PacketBlock;
using internal::PacketXd;
using internal::PacketXf;
using internal::PacketXi;
using internal::pfirst;

constexpr int kNI = packet_traits<numext::int32_t>::size;
constexpr int kNF = packet_traits<float>::size;
constexpr int kND = packet_traits<double>::size;

template <int N>
__attribute__((noinline)) void call_ptranspose(PacketBlock<PacketXi, N>& kernel) {
  internal::ptranspose(kernel);
}
template <int N>
__attribute__((noinline)) void call_ptranspose(PacketBlock<PacketXf, N>& kernel) {
  internal::ptranspose(kernel);
}
template <int N>
__attribute__((noinline)) void call_ptranspose(PacketBlock<PacketXd, N>& kernel) {
  internal::ptranspose(kernel);
}

// ---- plset<PacketXi> ----

void BM_Plset(benchmark::State& state) {
  numext::int32_t a = 7;
  numext::int32_t out[kNI];
  for (auto _ : state) {
    svst1_s32(svptrue_b32(), out, internal::plset<PacketXi>(a));
    benchmark::DoNotOptimize(out);
  }
}
BENCHMARK(BM_Plset)->Name("Plset_int32");

// ---- ploaddup ----

void BM_Ploaddup_int32(benchmark::State& state) {
  numext::int32_t in[kNI];
  for (int i = 0; i < kNI; ++i) in[i] = i;
  numext::int32_t out[kNI];
  for (auto _ : state) {
    svst1_s32(svptrue_b32(), out, internal::ploaddup<PacketXi>(in));
    benchmark::DoNotOptimize(out);
  }
}
BENCHMARK(BM_Ploaddup_int32)->Name("Ploaddup_int32");

void BM_Ploaddup_float(benchmark::State& state) {
  float in[kNF];
  for (int i = 0; i < kNF; ++i) in[i] = static_cast<float>(i);
  float out[kNF];
  for (auto _ : state) {
    svst1_f32(svptrue_b32(), out, internal::ploaddup<PacketXf>(in));
    benchmark::DoNotOptimize(out);
  }
}
BENCHMARK(BM_Ploaddup_float)->Name("Ploaddup_float");

// ---- ploadquad ----

void BM_Ploadquad_int32(benchmark::State& state) {
  numext::int32_t in[kNI];
  for (int i = 0; i < kNI; ++i) in[i] = i;
  numext::int32_t out[kNI];
  for (auto _ : state) {
    svst1_s32(svptrue_b32(), out, internal::ploadquad<PacketXi>(in));
    benchmark::DoNotOptimize(out);
  }
}
BENCHMARK(BM_Ploadquad_int32)->Name("Ploadquad_int32");

void BM_Ploadquad_float(benchmark::State& state) {
  float in[kNF];
  for (int i = 0; i < kNF; ++i) in[i] = static_cast<float>(i);
  float out[kNF];
  for (auto _ : state) {
    svst1_f32(svptrue_b32(), out, internal::ploadquad<PacketXf>(in));
    benchmark::DoNotOptimize(out);
  }
}
BENCHMARK(BM_Ploadquad_float)->Name("Ploadquad_float");

// ---- predux_mul ----

void BM_ReduxMul_int32(benchmark::State& state) {
  numext::int32_t in[kNI];
  for (int i = 0; i < kNI; ++i) in[i] = (i % 5) + 1;
  PacketXi a = svld1_s32(svptrue_b32(), in);
  for (auto _ : state) {
    benchmark::DoNotOptimize(internal::predux_mul<PacketXi>(a));
  }
}
BENCHMARK(BM_ReduxMul_int32)->Name("ReduxMul_int32");

void BM_ReduxMul_float(benchmark::State& state) {
  float in[kNF];
  for (int i = 0; i < kNF; ++i) in[i] = 1.0f + 0.01f * static_cast<float>(i);
  PacketXf a = svld1_f32(svptrue_b32(), in);
  for (auto _ : state) {
    benchmark::DoNotOptimize(internal::predux_mul<PacketXf>(a));
  }
}
BENCHMARK(BM_ReduxMul_float)->Name("ReduxMul_float");

void BM_ReduxMul_double(benchmark::State& state) {
  double in[kND];
  for (int i = 0; i < kND; ++i) in[i] = 1.0 + 0.01 * static_cast<double>(i);
  PacketXd a = svld1_f64(svptrue_b64(), in);
  for (auto _ : state) {
    benchmark::DoNotOptimize(internal::predux_mul<PacketXd>(a));
  }
}
BENCHMARK(BM_ReduxMul_double)->Name("ReduxMul_double");

// ---- ptranspose ----
// Benchmarked at N == the type's packet width, i.e. a full square transpose.

void BM_Ptranspose_int32(benchmark::State& state) {
  PacketBlock<PacketXi, kNI> kernel;
  numext::int32_t in[kNI * kNI];
  for (int i = 0; i < kNI * kNI; ++i) in[i] = i;
  for (int i = 0; i < kNI; ++i) kernel.packet[i] = svld1_s32(svptrue_b32(), in + i * kNI);
  for (auto _ : state) {
    call_ptranspose<kNI>(kernel);
    benchmark::DoNotOptimize(pfirst<PacketXi>(kernel.packet[0]));
  }
}
BENCHMARK(BM_Ptranspose_int32)->Name("Ptranspose_int32");

void BM_Ptranspose_float(benchmark::State& state) {
  PacketBlock<PacketXf, kNF> kernel;
  float in[kNF * kNF];
  for (int i = 0; i < kNF * kNF; ++i) in[i] = static_cast<float>(i);
  for (int i = 0; i < kNF; ++i) kernel.packet[i] = svld1_f32(svptrue_b32(), in + i * kNF);
  for (auto _ : state) {
    call_ptranspose<kNF>(kernel);
    benchmark::DoNotOptimize(pfirst<PacketXf>(kernel.packet[0]));
  }
}
BENCHMARK(BM_Ptranspose_float)->Name("Ptranspose_float");

void BM_Ptranspose_double(benchmark::State& state) {
  PacketBlock<PacketXd, kND> kernel;
  double in[kND * kND];
  for (int i = 0; i < kND * kND; ++i) in[i] = static_cast<double>(i);
  for (int i = 0; i < kND; ++i) kernel.packet[i] = svld1_f64(svptrue_b64(), in + i * kND);
  for (auto _ : state) {
    call_ptranspose<kND>(kernel);
    benchmark::DoNotOptimize(pfirst<PacketXd>(kernel.packet[0]));
  }
}
BENCHMARK(BM_Ptranspose_double)->Name("Ptranspose_double");

}  // namespace
}  // namespace Eigen

#endif  // defined(EIGEN_VECTORIZE_SVE)
