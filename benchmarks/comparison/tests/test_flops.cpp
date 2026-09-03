// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Checks the flop formulas in benchmarks/bench_common.h against independently
// known values.
//
// A miscounted multiplier rescales every rate the benchmark reports without
// making anything fail, and in the comparison harness it would rescale every
// published Eigen-versus-vendor ratio identically on both arms, so no ratio
// would look wrong either. These are the only assertions that catch it.
//
// Deliberately free of GoogleTest: the standalone benchmarks project must not
// grow that dependency. CHECK is used rather than <cassert> because benchmark
// targets compile with NDEBUG, which turns assert into nothing.

#include <complex>
#include <cmath>
#include <cstdio>
#include <string>

#include "benchmarks/bench_common.h"

namespace {

int failures = 0;

void reportFailure(const char* file, int line, const std::string& what) {
  std::fprintf(stderr, "%s:%d: FAILED %s\n", file, line, what.c_str());
  ++failures;
}

void checkImpl(bool condition, const char* expression, const char* file, int line) {
  if (!condition) reportFailure(file, line, expression);
}

void checkNearImpl(double actual, double expected, double relative_tolerance, const char* expression, const char* file,
                   int line) {
  const double tolerated = relative_tolerance * std::fabs(expected);
  if (!(std::fabs(actual - expected) <= tolerated)) {
    reportFailure(
        file, line,
        std::string(expression) + ": got " + std::to_string(actual) + ", expected " + std::to_string(expected));
  }
}

#define CHECK(cond) checkImpl((cond), #cond, __FILE__, __LINE__)
#define CHECK_EQ(actual, expected) \
  checkNearImpl((actual), (expected), 0.0, #actual " == " #expected, __FILE__, __LINE__)
#define CHECK_NEAR(actual, expected, tol) \
  checkNearImpl((actual), (expected), (tol), #actual " ~= " #expected, __FILE__, __LINE__)

using Eigen::Index;
using c32 = std::complex<float>;
using c64 = std::complex<double>;

// The loop that Cholesky/bench_cholesky.cpp and Cholesky/bench_bunchkaufman.cpp
// evaluate, transcribed here so the closed form in bench_common.h is checked
// against the summation it replaced rather than against itself.
double symmetricFactorizationLoop(int n) {
  double cost = 0;
  for (int j = 0; j < n; ++j) {
    const int rem = n - j - 1 > 0 ? n - j - 1 : 0;
    cost += 2 * (double(rem) * j + rem + j);
  }
  return cost;
}

void testScales() {
  CHECK_EQ(eigen_bench::flopScale<float>(), 2.0);
  CHECK_EQ(eigen_bench::flopScale<double>(), 2.0);
  CHECK_EQ(eigen_bench::flopScale<c32>(), 8.0);
  CHECK_EQ(eigen_bench::flopScale<c64>(), 8.0);

  CHECK_EQ(eigen_bench::complexFactor<float>(), 1.0);
  CHECK_EQ(eigen_bench::complexFactor<double>(), 1.0);
  CHECK_EQ(eigen_bench::complexFactor<c32>(), 4.0);
  CHECK_EQ(eigen_bench::complexFactor<c64>(), 4.0);

  // The invariant that lets ops.toml state one real flop expression per op and
  // have both real and complex arms derive from it.
  CHECK_EQ(eigen_bench::flopScale<double>(), 2.0 * eigen_bench::complexFactor<double>());
  CHECK_EQ(eigen_bench::flopScale<c64>(), 2.0 * eigen_bench::complexFactor<c64>());

  // Usable in a constant expression, which is what makes the scale free at the
  // call site.
  static_assert(eigen_bench::flopScale<double>() == 2.0, "flopScale must be constexpr");
  static_assert(eigen_bench::complexFactor<c64>() == 4.0, "complexFactor must be constexpr");
}

void testLevel1And2() {
  // n multiply-adds.
  CHECK_EQ(eigen_bench::dotFlops<double>(1000), 2000.0);
  CHECK_EQ(eigen_bench::dotFlops<c64>(1000), 8000.0);
  CHECK_EQ(eigen_bench::axpyFlops<float>(7), 14.0);
  CHECK_EQ(eigen_bench::axpyFlops<c32>(7), 56.0);

  // m*n multiply-adds; non-square on purpose, and m != n catches a transposed
  // argument order that a square case cannot see.
  CHECK_EQ(eigen_bench::gemvFlops<double>(10000, 100), 2.0e6);
  CHECK_EQ(eigen_bench::gemvFlops<double>(100, 10000), 2.0e6);
  CHECK_EQ(eigen_bench::gemvFlops<c64>(10000, 100), 8.0e6);

  CHECK_EQ(eigen_bench::symvFlops<double>(512), 2.0 * 512.0 * 512.0);
  CHECK_EQ(eigen_bench::symvFlops<c32>(512), 8.0 * 512.0 * 512.0);

  // Deliberately the same formula as symv: Core/bench_trmv.cpp has always
  // published scale*n*n, roughly 2x the exact scale*n*(n+1)/2, and the helper
  // preserves that value so adopting it moves no published number.
  CHECK_EQ(eigen_bench::trmvFlops<double>(512), eigen_bench::symvFlops<double>(512));
  CHECK_EQ(eigen_bench::trmvFlops<double>(512), 2.0 * 512.0 * 512.0);

  // n*(n+1)/2 multiply-adds for the triangle SYR touches.
  CHECK_EQ(eigen_bench::syrFlops<double>(4), 2.0 * 10.0);
  CHECK_EQ(eigen_bench::syrFlops<c64>(4), 8.0 * 10.0);
  // SYR2 touches the same triangle twice.
  CHECK_EQ(eigen_bench::syr2Flops<double>(4), 2.0 * 20.0);
  CHECK_EQ(eigen_bench::syr2Flops<double>(4), 2.0 * eigen_bench::syrFlops<double>(4));
  // Odd n, where the /2 must not truncate.
  CHECK_EQ(eigen_bench::syrFlops<double>(3), 12.0);
}

void testLevel3() {
  // 2*m*n*k for real, 8*m*n*k for complex.
  CHECK_EQ(eigen_bench::gemmFlops<double>(1024, 1024, 1024), 2.0 * 1024.0 * 1024.0 * 1024.0);
  CHECK_EQ(eigen_bench::gemmFlops<c64>(1024, 1024, 1024), 8.0 * 1024.0 * 1024.0 * 1024.0);
  CHECK_EQ(eigen_bench::gemmFlops<c64>(1024, 1024, 1024), 4.0 * eigen_bench::gemmFlops<double>(1024, 1024, 1024));

  // Non-square, all three extents distinct: this is the case that fails if two
  // arguments are swapped or one is dropped.
  CHECK_EQ(eigen_bench::gemmFlops<float>(4096, 96, 128), 2.0 * 4096.0 * 96.0 * 128.0);
  CHECK_EQ(eigen_bench::gemmFlops<float>(96, 128, 4096), 2.0 * 4096.0 * 96.0 * 128.0);
  CHECK_EQ(eigen_bench::gemmFlops<c32>(10000, 8, 4000), 8.0 * 10000.0 * 8.0 * 4000.0);

  // The product must be formed in double: floor(cbrt(2^64 - 1)) cubed overflows
  // the signed 64-bit Index that state.range() feeds in, so an Index-valued
  // intermediate would wrap.
  const Index big = 2642245;
  CHECK_NEAR(eigen_bench::gemmFlops<double>(big, big, big), 2.0 * double(big) * double(big) * double(big), 1e-12);
  CHECK(eigen_bench::gemmFlops<double>(big, big, big) > 3.6e19);

  CHECK_EQ(eigen_bench::trsmFlops<double>(2048, 16), 2048.0 * 2048.0 * 16.0);
  CHECK_EQ(eigen_bench::trsmFlops<c64>(2048, 16), 4.0 * 2048.0 * 2048.0 * 16.0);
}

void testFactorizations() {
  // Against the summation loop the closed form replaced, including the small n
  // where the cubic term is negative or zero.
  for (int n : {0, 1, 2, 3, 4, 5, 16, 100, 513, 2048}) {
    CHECK_NEAR(eigen_bench::symmetricFactorizationFlops<double>(Index(n)), symmetricFactorizationLoop(n), 1e-12);
  }
  // Hand-evaluated: j=0 gives 2*(0*2+2+0)=4, j=1 gives 2*(1*1+1+1)=6, j=2 gives
  // 2*(0*2+0+2)=4.
  CHECK_EQ(eigen_bench::symmetricFactorizationFlops<double>(3), 14.0);
  CHECK_EQ(eigen_bench::symmetricFactorizationFlops<c64>(3), 56.0);
  // Leading term n^3/3.
  CHECK_NEAR(eigen_bench::symmetricFactorizationFlops<double>(2048), 2048.0 * 2048.0 * 2048.0 / 3.0, 2e-3);

  // LAWN 41 (DGETRF): m*n^2 - n^3/3, i.e. 2*n^3/3 for a square LU.
  // Not exact: the helper and this expectation associate the divisions
  // differently, so the two doubles differ in the last bit.
  CHECK_NEAR(eigen_bench::getrfFlops<double>(512, 512), 2.0 * 512.0 * 512.0 * 512.0 / 3.0, 1e-15);
  CHECK_EQ(eigen_bench::getrfFlops<double>(1000, 100), 1000.0 * 100.0 * 100.0 - 100.0 * 100.0 * 100.0 / 3.0);
  // Householder QR costs twice an LU of the same order; if these two ever agree,
  // one of them has been copied from the other.
  CHECK_EQ(eigen_bench::geqrfFlops<double>(512, 512), 2.0 * eigen_bench::getrfFlops<double>(512, 512));

  // An LU is NOT two symmetric factorizations, however close the two get for
  // large n. bench_bunchkaufman once counted BM_PartialPivLU as
  // 2 * symmetricFactorizationFlops, which overstates the work by 31% at n=8 and
  // 4.6% at n=64 -- and that benchmark sweeps from n=8, so the small end of its
  // curve was published materially faster than it ran. These must stay distinct.
  CHECK(2.0 * eigen_bench::symmetricFactorizationFlops<double>(8) != eigen_bench::getrfFlops<double>(8, 8));
  CHECK(2.0 * eigen_bench::symmetricFactorizationFlops<double>(64) != eigen_bench::getrfFlops<double>(64, 64));
  // and the LU helper is what an LU of order n costs
  CHECK_EQ(eigen_bench::getrfFlops<double>(8, 8), 8.0 * 64.0 - 512.0 / 3.0);
  CHECK_EQ(eigen_bench::getrfFlops<c64>(512, 512), 4.0 * eigen_bench::getrfFlops<double>(512, 512));

  CHECK_EQ(eigen_bench::geqrfFlops<double>(1000, 100),
           2.0 * 1000.0 * 100.0 * 100.0 - 2.0 * 100.0 * 100.0 * 100.0 / 3.0);
  CHECK_EQ(eigen_bench::geqrfFlops<c32>(1000, 100), 4.0 * eigen_bench::geqrfFlops<float>(1000, 100));

  // Nominal counts: fixed conventions, not operation counts, so the test pins
  // the convention rather than deriving it.
  CHECK_EQ(eigen_bench::gesddFlops<double>(10000, 1000),
           8.0 * 10000.0 * 1000.0 * 1000.0 + 4.0 * 1000.0 * 1000.0 * 1000.0 / 3.0);
  CHECK_EQ(eigen_bench::gesddFlops<c64>(10000, 1000), 4.0 * eigen_bench::gesddFlops<double>(10000, 1000));
  CHECK_EQ(eigen_bench::syevFlops<double>(512), 9.0 * 512.0 * 512.0 * 512.0);
  CHECK_EQ(eigen_bench::syevFlops<c64>(512), 4.0 * 9.0 * 512.0 * 512.0 * 512.0);
}

// Degenerate extents must produce zero work rather than a negative or NaN rate.
void testDegenerate() {
  CHECK_EQ(eigen_bench::dotFlops<double>(0), 0.0);
  CHECK_EQ(eigen_bench::gemvFlops<double>(0, 1000), 0.0);
  CHECK_EQ(eigen_bench::gemmFlops<double>(1024, 1024, 0), 0.0);
  CHECK_EQ(eigen_bench::syrFlops<double>(0), 0.0);
  CHECK_EQ(eigen_bench::symmetricFactorizationFlops<double>(0), 0.0);
  CHECK_EQ(eigen_bench::symmetricFactorizationFlops<double>(1), 0.0);
  CHECK_EQ(eigen_bench::getrfFlops<double>(0, 0), 0.0);
}

void testCounterName() {
  // The reducer joins on this exact string.
  CHECK(std::string(eigen_bench::kFlopCounterName) == "GFLOPS");
}

}  // namespace

int main() {
  testScales();
  testLevel1And2();
  testLevel3();
  testFactorizations();
  testDegenerate();
  testCounterName();

  if (failures != 0) {
    std::fprintf(stderr, "test_flops: %d check(s) failed\n", failures);
    return 1;
  }
  std::printf("test_flops: all checks passed\n");
  return 0;
}
