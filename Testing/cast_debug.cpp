// Targeted reproducer for casting_all failure on loongarch64 LSX.
// Tests each SrcScalar -> float cast individually, many iterations.
#include <Eigen/Core>
#include "test/random_without_cast_overflow.h"
#include <iostream>
#include <cstdlib>

using namespace Eigen;
using namespace Eigen::internal;

template <typename SrcScalar, typename TgtScalar>
bool test_cast_pair(int seed_offset) {
  const int kIters = 10000;
  for (int iter = 0; iter < kIters; ++iter) {
    Matrix<SrcScalar, 4, 4> m;
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        m(i, j) = random_without_cast_overflow<SrcScalar, TgtScalar>::value();

    Matrix<TgtScalar, 4, 4> n = m.template cast<TgtScalar>();
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        TgtScalar expected = cast<SrcScalar, TgtScalar>(m(i, j));
        TgtScalar actual = n(i, j);
        // Bitwise comparison to catch NaN mismatches too
        bool bitwise_equal;
        if constexpr (sizeof(TgtScalar) == sizeof(uint32_t)) {
          uint32_t a, b;
          std::memcpy(&a, &actual, sizeof(a));
          std::memcpy(&b, &expected, sizeof(b));
          bitwise_equal = (a == b);
        } else if constexpr (sizeof(TgtScalar) == sizeof(uint16_t)) {
          uint16_t a, b;
          std::memcpy(&a, &actual, sizeof(a));
          std::memcpy(&b, &expected, sizeof(b));
          bitwise_equal = (a == b);
        } else {
          bitwise_equal = (actual == expected) ||
                          (numext::isnan(actual) && numext::isnan(expected));
        }
        if (!bitwise_equal) {
          std::cerr << "MISMATCH: cast<" << typeid(SrcScalar).name()
                    << ", " << typeid(TgtScalar).name() << ">"
                    << " at (" << i << "," << j << ")"
                    << " iter=" << iter
                    << std::endl;
          // Print as float for readability
          std::cerr << "  src value = " << static_cast<double>(m(i, j)) << std::endl;
          std::cerr << "  vectorized result = " << static_cast<double>(actual) << std::endl;
          std::cerr << "  scalar result     = " << static_cast<double>(expected) << std::endl;

          // Also print raw bits
          if constexpr (sizeof(SrcScalar) <= 8) {
            uint64_t src_bits = 0;
            std::memcpy(&src_bits, &m(i, j), sizeof(SrcScalar));
            std::cerr << "  src bits = 0x" << std::hex << src_bits << std::dec << std::endl;
          }
          if constexpr (sizeof(TgtScalar) <= 8) {
            uint64_t act_bits = 0, exp_bits = 0;
            std::memcpy(&act_bits, &actual, sizeof(TgtScalar));
            std::memcpy(&exp_bits, &expected, sizeof(TgtScalar));
            std::cerr << "  actual bits   = 0x" << std::hex << act_bits << std::dec << std::endl;
            std::cerr << "  expected bits = 0x" << std::hex << exp_bits << std::dec << std::endl;
          }
          return false;
        }
      }
    }
  }
  return true;
}

#define TEST_CAST(Src, Tgt)                                                \
  do {                                                                     \
    std::cout << "Testing " #Src " -> " #Tgt " ... " << std::flush;       \
    if (test_cast_pair<Src, Tgt>(0))                                       \
      std::cout << "OK" << std::endl;                                      \
    else {                                                                 \
      std::cout << "FAILED" << std::endl;                                  \
      failures++;                                                          \
    }                                                                      \
  } while (0)

int main(int argc, char* argv[]) {
  int seed = argc > 1 ? std::atoi(argv[1]) : 42;
  std::srand(seed);
  std::cout << "Seed: " << seed << std::endl;

  int failures = 0;

  // Test all SrcScalar -> float casts (the failing TgtScalar)
  TEST_CAST(bool, float);
  TEST_CAST(int8_t, float);
  TEST_CAST(uint8_t, float);
  TEST_CAST(int16_t, float);
  TEST_CAST(uint16_t, float);
  TEST_CAST(int32_t, float);
  TEST_CAST(uint32_t, float);
  TEST_CAST(int64_t, float);
  TEST_CAST(uint64_t, float);
  TEST_CAST(Eigen::half, float);
  TEST_CAST(Eigen::bfloat16, float);
  TEST_CAST(float, float);
  TEST_CAST(double, float);

  // Also test -> half and -> bfloat16 since those are tricky
  TEST_CAST(float, Eigen::half);
  TEST_CAST(float, Eigen::bfloat16);
  TEST_CAST(double, Eigen::half);
  TEST_CAST(double, Eigen::bfloat16);
  TEST_CAST(int32_t, Eigen::half);
  TEST_CAST(int32_t, Eigen::bfloat16);
  TEST_CAST(int64_t, Eigen::half);
  TEST_CAST(int64_t, Eigen::bfloat16);
  TEST_CAST(Eigen::half, Eigen::bfloat16);
  TEST_CAST(Eigen::bfloat16, Eigen::half);

  std::cout << "\n" << (failures ? "SOME TESTS FAILED" : "ALL TESTS PASSED") << std::endl;
  return failures;
}
