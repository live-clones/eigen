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

// `ploadquad` reads one value per group of four lanes; at the smallest vector
// length that would round down to zero groups, so at least one is read.
constexpr int kQuarterI = (kNI / 4 > 0) ? kNI / 4 : 1;
constexpr int kQuarterF = (kNF / 4 > 0) ? kNF / 4 : 1;

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
  const numext::int32_t a = 7;
  numext::int32_t out[kNI];

  svst1_s32(svptrue_b32(), out, internal::plset<PacketXi>(a));
  for (int i = 0; i < kNI; ++i) {
    if (out[i] != a + i) {
      state.SkipWithError("Plset: materialized result does not match scalar reference");
      return;
    }
  }

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

  svst1_s32(svptrue_b32(), out, internal::ploaddup<PacketXi>(in));
  for (int i = 0; i < kNI / 2; ++i) {
    if (out[2 * i] != in[i] || out[2 * i + 1] != in[i]) {
      state.SkipWithError("Ploaddup: materialized result does not match scalar reference");
      return;
    }
  }

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

  svst1_f32(svptrue_b32(), out, internal::ploaddup<PacketXf>(in));
  for (int i = 0; i < kNF / 2; ++i) {
    if (out[2 * i] != in[i] || out[2 * i + 1] != in[i]) {
      state.SkipWithError("Ploaddup: materialized result does not match scalar reference");
      return;
    }
  }

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

  svst1_s32(svptrue_b32(), out, internal::ploadquad<PacketXi>(in));
  for (int j = 0; j < kQuarterI; ++j) {
    for (int k = 0; k < 4; ++k) {
      if (out[4 * j + k] != in[j]) {
        state.SkipWithError("Ploadquad: materialized result does not match scalar reference");
        return;
      }
    }
  }

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

  svst1_f32(svptrue_b32(), out, internal::ploadquad<PacketXf>(in));
  for (int j = 0; j < kQuarterF; ++j) {
    for (int k = 0; k < 4; ++k) {
      if (out[4 * j + k] != in[j]) {
        state.SkipWithError("Ploadquad: materialized result does not match scalar reference");
        return;
      }
    }
  }

  for (auto _ : state) {
    svst1_f32(svptrue_b32(), out, internal::ploadquad<PacketXf>(in));
    benchmark::DoNotOptimize(out);
  }
}
BENCHMARK(BM_Ploadquad_float)->Name("Ploadquad_float");

// ---- predux_mul ----
// Inputs are chosen so the true product is exactly representable regardless
// of the order the reduction folds lanes together: powers of two multiply
// without rounding for float/double, and multiplying by 1 is exact and
// overflow-free for int32.

void BM_ReduxMul_int32(benchmark::State& state) {
  numext::int32_t in[kNI];
  for (int i = 0; i < kNI; ++i) in[i] = 1;
  in[0] = -3;
  in[kNI - 1] = 2;
  const numext::int32_t expected = -6;
  PacketXi a = svld1_s32(svptrue_b32(), in);

  if (internal::predux_mul<PacketXi>(a) != expected) {
    state.SkipWithError("ReduxMul: materialized result does not match scalar reference");
    return;
  }

  for (auto _ : state) {
    benchmark::DoNotOptimize(internal::predux_mul<PacketXi>(a));
  }
}
BENCHMARK(BM_ReduxMul_int32)->Name("ReduxMul_int32");

void BM_ReduxMul_float(benchmark::State& state) {
  float in[kNF];
  float expected = 1.0f;
  for (int i = 0; i < kNF; ++i) {
    in[i] = static_cast<float>(1 << i);
    expected *= in[i];
  }
  PacketXf a = svld1_f32(svptrue_b32(), in);

  if (internal::predux_mul<PacketXf>(a) != expected) {
    state.SkipWithError("ReduxMul: materialized result does not match scalar reference");
    return;
  }

  for (auto _ : state) {
    benchmark::DoNotOptimize(internal::predux_mul<PacketXf>(a));
  }
}
BENCHMARK(BM_ReduxMul_float)->Name("ReduxMul_float");

void BM_ReduxMul_double(benchmark::State& state) {
  double in[kND];
  double expected = 1.0;
  for (int i = 0; i < kND; ++i) {
    in[i] = static_cast<double>(std::int64_t(1) << i);
    expected *= in[i];
  }
  PacketXd a = svld1_f64(svptrue_b64(), in);

  if (internal::predux_mul<PacketXd>(a) != expected) {
    state.SkipWithError("ReduxMul: materialized result does not match scalar reference");
    return;
  }

  for (auto _ : state) {
    benchmark::DoNotOptimize(internal::predux_mul<PacketXd>(a));
  }
}
BENCHMARK(BM_ReduxMul_double)->Name("ReduxMul_double");

// ---- ptranspose ----
// Benchmarked at N == the type's packet width, i.e. a full square transpose,
// so the expected result is simply new_packet[i][k] == old_packet[k][i].
// Correctness is checked once on a scratch kernel -- ptranspose applied
// repeatedly toggles between the original and transposed state, so checking
// after the timed loop would depend on the (unpredictable) iteration count.

void BM_Ptranspose_int32(benchmark::State& state) {
  numext::int32_t in[kNI * kNI];
  for (int i = 0; i < kNI * kNI; ++i) in[i] = i;

  PacketBlock<PacketXi, kNI> check;
  for (int i = 0; i < kNI; ++i) check.packet[i] = svld1_s32(svptrue_b32(), in + i * kNI);
  call_ptranspose<kNI>(check);
  for (int i = 0; i < kNI; ++i) {
    numext::int32_t row[kNI];
    svst1_s32(svptrue_b32(), row, check.packet[i]);
    for (int k = 0; k < kNI; ++k) {
      if (row[k] != in[k * kNI + i]) {
        state.SkipWithError("Ptranspose: materialized result does not match scalar reference");
        return;
      }
    }
  }

  PacketBlock<PacketXi, kNI> kernel;
  for (int i = 0; i < kNI; ++i) kernel.packet[i] = svld1_s32(svptrue_b32(), in + i * kNI);
  for (auto _ : state) {
    call_ptranspose<kNI>(kernel);
    benchmark::DoNotOptimize(pfirst<PacketXi>(kernel.packet[0]));
  }
}
BENCHMARK(BM_Ptranspose_int32)->Name("Ptranspose_int32");

void BM_Ptranspose_float(benchmark::State& state) {
  float in[kNF * kNF];
  for (int i = 0; i < kNF * kNF; ++i) in[i] = static_cast<float>(i);

  PacketBlock<PacketXf, kNF> check;
  for (int i = 0; i < kNF; ++i) check.packet[i] = svld1_f32(svptrue_b32(), in + i * kNF);
  call_ptranspose<kNF>(check);
  for (int i = 0; i < kNF; ++i) {
    float row[kNF];
    svst1_f32(svptrue_b32(), row, check.packet[i]);
    for (int k = 0; k < kNF; ++k) {
      if (row[k] != in[k * kNF + i]) {
        state.SkipWithError("Ptranspose: materialized result does not match scalar reference");
        return;
      }
    }
  }

  PacketBlock<PacketXf, kNF> kernel;
  for (int i = 0; i < kNF; ++i) kernel.packet[i] = svld1_f32(svptrue_b32(), in + i * kNF);
  for (auto _ : state) {
    call_ptranspose<kNF>(kernel);
    benchmark::DoNotOptimize(pfirst<PacketXf>(kernel.packet[0]));
  }
}
BENCHMARK(BM_Ptranspose_float)->Name("Ptranspose_float");

void BM_Ptranspose_double(benchmark::State& state) {
  double in[kND * kND];
  for (int i = 0; i < kND * kND; ++i) in[i] = static_cast<double>(i);

  PacketBlock<PacketXd, kND> check;
  for (int i = 0; i < kND; ++i) check.packet[i] = svld1_f64(svptrue_b64(), in + i * kND);
  call_ptranspose<kND>(check);
  for (int i = 0; i < kND; ++i) {
    double row[kND];
    svst1_f64(svptrue_b64(), row, check.packet[i]);
    for (int k = 0; k < kND; ++k) {
      if (row[k] != in[k * kND + i]) {
        state.SkipWithError("Ptranspose: materialized result does not match scalar reference");
        return;
      }
    }
  }

  PacketBlock<PacketXd, kND> kernel;
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
