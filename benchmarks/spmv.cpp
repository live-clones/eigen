#include <benchmark/benchmark.h>
#include <Eigen/Sparse>
#include <set>

using namespace Eigen;

typedef double Scalar;
typedef SparseMatrix<Scalar> SpMat;
typedef Matrix<Scalar, Dynamic, 1> DenseVec;

static void fillMatrix(int nnzPerCol, int rows, int cols, SpMat& dst) {
  dst.resize(rows, cols);
  dst.reserve(VectorXi::Constant(cols, nnzPerCol));
  for (int j = 0; j < cols; ++j) {
    std::set<int> used;
    for (int i = 0; i < nnzPerCol; ++i) {
      int row;
      do { row = internal::random<int>(0, rows - 1); } while (used.count(row));
      used.insert(row);
      dst.insert(row, j) = internal::random<Scalar>();
    }
  }
  dst.makeCompressed();
}

static void BM_SpMV_Normal(benchmark::State& state) {
  int rows = state.range(0);
  int cols = state.range(1);
  int nnzPerCol = state.range(2);
  SpMat sm(rows, cols);
  fillMatrix(nnzPerCol, rows, cols, sm);
  DenseVec dv = DenseVec::Random(cols);
  DenseVec res(rows);
  for (auto _ : state) {
    res.noalias() += sm * dv;
    benchmark::DoNotOptimize(res.data());
  }
  state.counters["nnz"] = sm.nonZeros();
}

static void BM_SpMV_Trans(benchmark::State& state) {
  int rows = state.range(0);
  int cols = state.range(1);
  int nnzPerCol = state.range(2);
  SpMat sm(rows, cols);
  fillMatrix(nnzPerCol, rows, cols, sm);
  DenseVec dv = DenseVec::Random(rows);
  DenseVec res(cols);
  for (auto _ : state) {
    res.noalias() += sm.transpose() * dv;
    benchmark::DoNotOptimize(res.data());
  }
  state.counters["nnz"] = sm.nonZeros();
}

static void SpMVSizes(::benchmark::Benchmark* b) {
  for (int nnz : {4, 10, 20, 40}) {
    b->Args({10000, 10000, nnz});
    b->Args({100000, 100000, nnz});
  }
}

BENCHMARK(BM_SpMV_Normal)->Apply(SpMVSizes);
BENCHMARK(BM_SpMV_Trans)->Apply(SpMVSizes);
