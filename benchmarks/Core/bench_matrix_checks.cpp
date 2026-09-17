// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>

template <typename Scalar, int Operation, int FailurePosition = 0>
static void matrix_checks(benchmark::State& state) {
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> a(state.range(0), state.range(0));
  if (Operation == 0) a.setZero();
  if (Operation == 1) a.setConstant(Scalar(0.5));
  if (Operation >= 2) a.setIdentity();
  if (FailurePosition != 0) a(FailurePosition == 1 ? 0 : a.size() - 1) = Scalar(2);
  if ((Operation == 0 && a.isZero() != (FailurePosition == 0)) ||
      (Operation == 1 && a.isApproxToConstant(Scalar(0.5)) != (FailurePosition == 0))) {
    state.SkipWithError("incorrect fuzzy comparison");
    return;
  }
  for (auto _ : state) {
    benchmark::ClobberMemory();
    bool result;
    if (Operation == 0) result = a.isZero();
    if (Operation == 1) result = a.isApproxToConstant(Scalar(0.5));
    if (Operation == 3) result = a.isUpperTriangular();
    if (Operation == 4) result = a.isLowerTriangular();
    benchmark::DoNotOptimize(result);
  }
}

BENCHMARK_TEMPLATE(matrix_checks, double, 0)->Arg(3)->Arg(4)->Arg(8)->Arg(16)->Arg(32)->Arg(256)->Arg(1024)->Arg(2048);
BENCHMARK_TEMPLATE(matrix_checks, double, 1)->Arg(3)->Arg(4)->Arg(8)->Arg(16)->Arg(32)->Arg(256)->Arg(1024)->Arg(2048);
BENCHMARK_TEMPLATE(matrix_checks, double, 3)->Arg(3)->Arg(4)->Arg(8)->Arg(16)->Arg(32)->Arg(256);
BENCHMARK_TEMPLATE(matrix_checks, double, 4)->Arg(3)->Arg(4)->Arg(8)->Arg(16)->Arg(32)->Arg(256);
BENCHMARK_TEMPLATE(matrix_checks, float, 0)->Arg(3)->Arg(32)->Arg(256)->Arg(1024)->Arg(2048);
BENCHMARK_TEMPLATE(matrix_checks, float, 1)->Arg(3)->Arg(32)->Arg(256)->Arg(1024)->Arg(2048);

BENCHMARK_TEMPLATE(matrix_checks, double, 0, 1)->Arg(3)->Arg(32)->Arg(256);
BENCHMARK_TEMPLATE(matrix_checks, double, 0, 2)->Arg(3)->Arg(32)->Arg(256);
BENCHMARK_TEMPLATE(matrix_checks, double, 1, 1)->Arg(3)->Arg(32)->Arg(256);
BENCHMARK_TEMPLATE(matrix_checks, double, 1, 2)->Arg(3)->Arg(32)->Arg(256);
