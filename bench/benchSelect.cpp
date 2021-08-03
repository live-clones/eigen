// Build with: g++ -O3 -DNDEBUG benchSelect.cpp -I ../ -o benchSelect

#include <iostream>

#include <Eigen/Core>

#include "BenchUtil.h"

namespace impl {
template<typename LhsExpr, typename RhsExpr>
const Eigen::CwiseBinaryOp<
    Eigen::internal::scalar_typed_cmp_op<typename LhsExpr::Scalar, typename RhsExpr::Scalar, Eigen::internal::cmp_EQ>,
    const LhsExpr,
    const RhsExpr>
equal(const LhsExpr& A, const RhsExpr& B) {
    return Eigen::CwiseBinaryOp<
        Eigen::internal::scalar_typed_cmp_op<typename LhsExpr::Scalar, typename RhsExpr::Scalar, Eigen::internal::cmp_EQ>,
        const LhsExpr,
        const RhsExpr>(A, B);
}
}  // namespace impl

template <typename T>
double BenchmarkSelect(int rows, int cols, int iterations) {
  typedef Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> Array;
  Array x = Array::Random(rows, cols);
  Array x_transpose = x.transpose();

  Array zero = Array::Zero(rows, cols);
  Array ones = Array::Ones(rows, cols);

  BenchTimer timer;
  timer.reset();
  timer.start();
  for (int i = 0; i < iterations; i++) {
    Array result = impl::equal(x, x_transpose).select(ones, zero);
  }
  timer.stop();

  return timer.value();
}

int main(int argc, char *argv[])
{
  constexpr int rows = 512;
  constexpr int cols = 512;
  constexpr int iterations = 10000;

  #define RUN_BENCHMARK(TYPE)                                                \
  {                                                                          \
    double runtime = BenchmarkSelect<TYPE>(rows, cols, iterations);          \
    std::cout << #TYPE << ": Ran " << iterations << " in " << runtime        \
              << " seconds.\n";                                              \
  }

  RUN_BENCHMARK(int8_t);
  RUN_BENCHMARK(int16_t);
  RUN_BENCHMARK(int32_t);
  RUN_BENCHMARK(int64_t);
  RUN_BENCHMARK(uint8_t);
  RUN_BENCHMARK(uint16_t);
  RUN_BENCHMARK(uint32_t);
  RUN_BENCHMARK(uint64_t);
  RUN_BENCHMARK(float);
  RUN_BENCHMARK(double);

  return 0;
}
