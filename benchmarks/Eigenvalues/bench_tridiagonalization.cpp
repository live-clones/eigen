// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>

#include <cstdint>
#include <complex>
#include <string>
#include <vector>

#include <Eigen/Eigenvalues>

namespace {

enum class Algorithm { Unblocked, Blocked8, Blocked16, Blocked32, Dispatched };

template <typename Scalar, Algorithm algorithm>
void RunTridiagonalization(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& matrix,
                           Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& h_coeffs) {
  if (algorithm == Algorithm::Unblocked) {
    Eigen::internal::tridiagonalization_inplace_unblocked(matrix, h_coeffs);
  } else if (algorithm == Algorithm::Blocked8) {
    Eigen::internal::tridiagonalization_inplace_blocked(matrix, h_coeffs, 8);
  } else if (algorithm == Algorithm::Blocked16) {
    Eigen::internal::tridiagonalization_inplace_blocked(matrix, h_coeffs, 16);
  } else if (algorithm == Algorithm::Blocked32) {
    Eigen::internal::tridiagonalization_inplace_blocked(matrix, h_coeffs, 32);
  } else {
    Eigen::internal::tridiagonalization_inplace(matrix, h_coeffs);
  }
}

template <typename Scalar, Algorithm algorithm>
void BM_Tridiagonalization(benchmark::State& state) {
  using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using RealScalar = typename MatrixType::RealScalar;
  using CoeffVectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using SequenceType = typename Eigen::Tridiagonalization<MatrixType>::HouseholderSequenceType;

  const Eigen::Index size = state.range(0);
  const MatrixType random = MatrixType::Random(size, size);
  MatrixType input = random.template selfadjointView<Eigen::Lower>();
  input.diagonal() = input.diagonal().real().template cast<Scalar>();
  MatrixType matrix = input;
  CoeffVectorType h_coeffs(size - 1);

  RunTridiagonalization<Scalar, algorithm>(matrix, h_coeffs);
  const MatrixType q = SequenceType(matrix, h_coeffs.conjugate()).setLength(size - 1).setShift(1);
  MatrixType t = MatrixType::Zero(size, size);
  t.diagonal() = matrix.diagonal().real().template cast<Scalar>();
  t.diagonal(-1) = matrix.diagonal(-1).real().template cast<Scalar>();
  t.diagonal(1) = matrix.diagonal(-1).real().template cast<Scalar>();
  const RealScalar relative_error = (input - q * t * q.adjoint()).norm() / input.norm();
  const RealScalar tolerance = RealScalar(100) * Eigen::NumTraits<RealScalar>::epsilon() * RealScalar(size);
  if (!(relative_error <= tolerance)) {
    state.SkipWithError("tridiagonalization reconstruction failed");
    return;
  }

  for (auto _ : state) {
    state.PauseTiming();
    matrix = input;
    state.ResumeTiming();

    RunTridiagonalization<Scalar, algorithm>(matrix, h_coeffs);
    benchmark::ClobberMemory();
  }
}

template <typename Scalar>
void RegisterBenchmarks(const char* scalar_name) {
  const std::vector<std::vector<std::int64_t>> sizes = {{64, 96, 128, 192, 256, 384, 512, 768, 1024}};
  benchmark::RegisterBenchmark((std::string(scalar_name) + "/unblocked").c_str(),
                               &BM_Tridiagonalization<Scalar, Algorithm::Unblocked>)
      ->ArgsProduct(sizes);
  benchmark::RegisterBenchmark((std::string(scalar_name) + "/blocked8").c_str(),
                               &BM_Tridiagonalization<Scalar, Algorithm::Blocked8>)
      ->ArgsProduct(sizes);
  benchmark::RegisterBenchmark((std::string(scalar_name) + "/blocked16").c_str(),
                               &BM_Tridiagonalization<Scalar, Algorithm::Blocked16>)
      ->ArgsProduct(sizes);
  benchmark::RegisterBenchmark((std::string(scalar_name) + "/blocked32").c_str(),
                               &BM_Tridiagonalization<Scalar, Algorithm::Blocked32>)
      ->ArgsProduct(sizes);
  benchmark::RegisterBenchmark((std::string(scalar_name) + "/dispatched").c_str(),
                               &BM_Tridiagonalization<Scalar, Algorithm::Dispatched>)
      ->ArgsProduct(sizes);
}

const bool registered = [] {
  RegisterBenchmarks<float>("float");
  RegisterBenchmarks<double>("double");
  RegisterBenchmarks<std::complex<float>>("complex_float");
  RegisterBenchmarks<std::complex<double>>("complex_double");
  return true;
}();

}  // namespace
