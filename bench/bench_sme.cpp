
#ifdef EIGEN_ARM64_USE_SME
#include <arm_neon.h>
#endif

#include <iostream>

#include <vector>
#include <string>
#include <bench/BenchTimer.h>
#include <Eigen/Core>

using namespace std;
using namespace Eigen;

#ifndef SCALAR
#define SCALAR float
#endif

typedef SCALAR Scalar;
typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

EIGEN_DONT_INLINE void gemm(const Mat& a, const Mat& b, Mat& c) { c.noalias() += a * b; }

int main(int argc, char** argv) {
  int s = 1024;
  int m = s;
  int n = s;
  int k = s;
  int rep = 1;

  if (argc > 1) {
    s = atoi(argv[1]);
    m = n = k = s;
  }
  if (argc > 2) rep = atoi(argv[2]);

  cout << "Pattern: C += A*B" << endl;
  cout << "Scalar: " << (sizeof(Scalar) == 4 ? "float" : "double") << endl;
  cout << "Dimensions: " << m << "x" << n << " (" << k << ")" << endl;
  cout << "Threads: " << Eigen::nbThreads() << endl;

#ifdef EIGEN_VECTORIZE_SME
  cout << "EIGEN_VECTORIZE_SME defined" << endl;
#else
  cout << "EIGEN_VECTORIZE_SME NOT defined" << endl;
#endif

#ifdef EIGEN_ARM64_USE_SME
  cout << "EIGEN_ARM64_USE_SME defined" << endl;
#else
  cout << "EIGEN_ARM64_USE_SME NOT defined" << endl;
#endif

  typedef internal::gebp_traits<Scalar, Scalar> Traits;
  cout << "Register blocking = " << Traits::mr << " x " << Traits::nr << "\n";

  Mat A(m, k);
  Mat B(k, n);
  Mat C(m, n);

  A.setRandom();
  B.setRandom();
  C.setZero();

  BenchTimer t;

  // Warmup
  gemm(A, B, C);

  BENCH(t, 1, rep, gemm(A, B, C));

  cout << "Best Time: " << t.best() << " s" << endl;
  double flops = 2.0 * double(m) * double(n) * double(k);
  cout << "GFLOPS: " << (flops * 1e-9) / t.best() << endl;

  // Prevent dead code elimination
  cout << "Result check: " << C(0, 0) << endl;

  return 0;
}
