// Standalone benchmark for PMR allocator primitives and DenseStorage integration.
// Not part of CI — run manually to measure overhead and compare strategies.
//
// Build: g++ -std=c++17 -O2 -DNDEBUG -I.. -o allocator_bench allocator_bench.cpp
// Disasm: g++ -std=c++17 -O2 -DNDEBUG -I.. -S -fverbose-asm -o allocator_bench.s allocator_bench.cpp
// Run:   ./allocator_bench

#include <Eigen/Core>

#include <chrono>
#include <cstdio>
#include <cstring>

using Clock = std::chrono::high_resolution_clock;

static double ns_per_op(Clock::duration d, int ops) {
  return std::chrono::duration<double, std::nano>(d).count() / ops;
}

template <typename T>
__attribute__((noinline)) void do_not_optimize(T* p) {
  asm volatile("" : : "r"(p) : "memory");
}

// =========================================================================
// Part 1: Raw allocation throughput
// =========================================================================

static void bench_throughput() {
  constexpr int N = 100000;
  constexpr std::size_t sz = 256;

  {
    auto t0 = Clock::now();
    for (int i = 0; i < N; ++i) {
      void* p = Eigen::internal::handmade_aligned_malloc(sz);
      do_not_optimize(p);
      Eigen::internal::handmade_aligned_free(p);
    }
    auto t1 = Clock::now();
    std::printf("  direct handmade_aligned:     %6.1f ns/op\n", ns_per_op(t1 - t0, N));
  }

  {
    Eigen::memory_resource* r = Eigen::new_delete_resource();
    auto t0 = Clock::now();
    for (int i = 0; i < N; ++i) {
      void* p = r->allocate(sz);
      do_not_optimize(p);
      r->deallocate(p, sz);
    }
    auto t1 = Clock::now();
    std::printf("  virtual new_delete_resource: %6.1f ns/op\n", ns_per_op(t1 - t0, N));
  }

  {
    Eigen::byte_allocator alloc;
    auto t0 = Clock::now();
    for (int i = 0; i < N; ++i) {
      void* p = alloc.allocate(sz);
      do_not_optimize(p);
      alloc.deallocate(p, sz);
    }
    auto t1 = Clock::now();
    std::printf("  byte_allocator (default):    %6.1f ns/op\n", ns_per_op(t1 - t0, N));
  }

  {
    Eigen::monotonic_buffer_resource arena(N * sz * 2);
    auto t0 = Clock::now();
    for (int i = 0; i < N; ++i) {
      void* p = arena.allocate(sz);
      do_not_optimize(p);
    }
    auto t1 = Clock::now();
    std::printf("  monotonic_buffer_resource:   %6.1f ns/op\n", ns_per_op(t1 - t0, N));
  }
}

// =========================================================================
// Part 2: Arena reuse
// =========================================================================

static void bench_arena_reuse() {
  constexpr int N = 1000;
  constexpr int M = 1000;
  constexpr std::size_t sz = 128;

  {
    Eigen::monotonic_buffer_resource arena(N * sz * 2);
    auto t0 = Clock::now();
    for (int j = 0; j < M; ++j) {
      for (int i = 0; i < N; ++i) {
        void* p = arena.allocate(sz);
        do_not_optimize(p);
      }
      arena.release();
    }
    auto t1 = Clock::now();
    std::printf("  arena alloc+release:         %6.1f ns/op\n", ns_per_op(t1 - t0, N * M));
  }

  {
    auto t0 = Clock::now();
    for (int j = 0; j < M; ++j) {
      for (int i = 0; i < N; ++i) {
        void* p = Eigen::internal::handmade_aligned_malloc(sz);
        do_not_optimize(p);
        Eigen::internal::handmade_aligned_free(p);
      }
    }
    auto t1 = Clock::now();
    std::printf("  individual malloc/free:      %6.1f ns/op\n", ns_per_op(t1 - t0, N * M));
  }
}

// =========================================================================
// Part 3: sizeof overhead
// =========================================================================

static void bench_sizeof() {
  std::printf("  MatrixXd:              %2zu bytes\n", sizeof(Eigen::MatrixXd));
  std::printf("  VectorXd:              %2zu bytes\n", sizeof(Eigen::VectorXd));
  std::printf("  Matrix3d (fixed):      %2zu bytes  (unaffected)\n", sizeof(Eigen::Matrix3d));
  std::printf("  Matrix<d,-1,3>:        %2zu bytes\n", sizeof(Eigen::Matrix<double, Eigen::Dynamic, 3>));
  std::printf("  byte_allocator:        %2zu bytes\n", sizeof(Eigen::byte_allocator));
  std::printf("  memory_resource*:      %2zu bytes  (the per-object cost)\n", sizeof(Eigen::memory_resource*));
}

// =========================================================================
// Part 4: MatrixXd construction latency — default vs byte_allocator vs arena
// =========================================================================

__attribute__((noinline)) static double bench_matrix_default(int N, int dim) {
  auto t0 = Clock::now();
  for (int i = 0; i < N; ++i) {
    Eigen::MatrixXd m(dim, dim);
    do_not_optimize(m.data());
  }
  auto t1 = Clock::now();
  return ns_per_op(t1 - t0, N);
}

__attribute__((noinline)) static double bench_matrix_byte_alloc(int N, int dim) {
  Eigen::byte_allocator alloc;  // uses default resource
  auto t0 = Clock::now();
  for (int i = 0; i < N; ++i) {
    Eigen::MatrixXd m(dim, dim, alloc);
    do_not_optimize(m.data());
  }
  auto t1 = Clock::now();
  return ns_per_op(t1 - t0, N);
}

__attribute__((noinline)) static double bench_matrix_arena(int N, int dim) {
  std::size_t arena_sz = static_cast<std::size_t>(N) * dim * dim * sizeof(double) + 65536;
  if (arena_sz > 128 * 1024 * 1024) arena_sz = 128 * 1024 * 1024;
  Eigen::monotonic_buffer_resource arena(arena_sz);
  Eigen::byte_allocator alloc(&arena);
  auto t0 = Clock::now();
  for (int i = 0; i < N; ++i) {
    Eigen::MatrixXd m(dim, dim, alloc);
    do_not_optimize(m.data());
  }
  auto t1 = Clock::now();
  return ns_per_op(t1 - t0, N);
}

static void bench_matrix_construction() {
  for (int dim : {4, 16, 64}) {
    int N = dim <= 16 ? 200000 : 50000;
    std::printf("\n  %dx%d (%zu bytes):\n", dim, dim, dim * dim * sizeof(double));

    double def = bench_matrix_default(N, dim);
    double ba = bench_matrix_byte_alloc(N, dim);
    double ar = bench_matrix_arena(N, dim);

    std::printf("    default (no alloc member): %6.1f ns\n", def);
    std::printf("    byte_allocator (default):  %6.1f ns  (%+.1f%%)\n", ba, (ba - def) / def * 100);
    std::printf("    byte_allocator (arena):    %6.1f ns  (%.1fx vs default)\n", ar, def / ar);
  }
}

// =========================================================================
// Part 5: Matrix multiply — compute overhead (should be zero)
// =========================================================================

__attribute__((noinline)) static double bench_matmul(int N, int dim, Eigen::byte_allocator* alloc) {
  Eigen::MatrixXd A = alloc ? Eigen::MatrixXd(dim, dim, *alloc) : Eigen::MatrixXd(dim, dim);
  Eigen::MatrixXd B = alloc ? Eigen::MatrixXd(dim, dim, *alloc) : Eigen::MatrixXd(dim, dim);
  Eigen::MatrixXd C = alloc ? Eigen::MatrixXd(dim, dim, *alloc) : Eigen::MatrixXd(dim, dim);
  A.setRandom();
  B.setRandom();

  auto t0 = Clock::now();
  for (int i = 0; i < N; ++i) {
    C.noalias() = A * B;
    do_not_optimize(C.data());
  }
  auto t1 = Clock::now();
  return ns_per_op(t1 - t0, N);
}

static void bench_matmul() {
  for (int dim : {4, 16, 64}) {
    int N = dim <= 16 ? 500000 : 10000;
    double def = bench_matmul(N, dim, nullptr);

    Eigen::monotonic_buffer_resource arena(dim * dim * sizeof(double) * 4 + 4096);
    Eigen::byte_allocator alloc(&arena);
    double ar = bench_matmul(N, dim, &alloc);

    std::printf("  %3dx%-3d  default: %8.1f ns   arena: %8.1f ns   diff: %+.1f%%\n", dim, dim, def, ar,
                (ar - def) / def * 100);
  }
}

// =========================================================================
// Main
// =========================================================================

int main() {
  std::printf("=== PMR Allocator Benchmarks ===\n\n");

  std::printf("--- 1. Raw allocation throughput (256B, alloc+dealloc) ---\n");
  bench_throughput();

  std::printf("\n--- 2. Arena reuse (128B, 1000 x 1000) ---\n");
  bench_arena_reuse();

  std::printf("\n--- 3. sizeof overhead ---\n");
  bench_sizeof();

  std::printf("\n--- 4. MatrixXd construction latency ---\n");
  bench_matrix_construction();

  std::printf("\n--- 5. Matrix multiply (compute, should be ~0%% diff) ---\n");
  bench_matmul();

  return 0;
}
