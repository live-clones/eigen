// Standalone benchmark for PMR allocator primitives.
// Not part of CI — run manually to compare allocation strategies.
//
// Build: g++ -std=c++17 -O2 -I.. -o allocator_bench allocator_bench.cpp
// Run:   ./allocator_bench

#include <Eigen/Core>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

static double ns_per_op(Clock::duration d, int ops) {
  return std::chrono::duration<double, std::nano>(d).count() / ops;
}

// Benchmark 1: Throughput — allocate+deallocate cycles.
static void bench_throughput() {
  constexpr int N = 100000;
  constexpr std::size_t sz = 256;

  // (a) Raw handmade_aligned_malloc/free.
  {
    auto t0 = Clock::now();
    for (int i = 0; i < N; ++i) {
      void* p = Eigen::internal::handmade_aligned_malloc(sz);
      Eigen::internal::handmade_aligned_free(p);
    }
    auto t1 = Clock::now();
    std::printf("handmade_aligned_malloc/free:  %8.1f ns/op\n", ns_per_op(t1 - t0, N));
  }

  // (b) new_delete_resource (virtual call overhead).
  {
    Eigen::memory_resource* r = Eigen::new_delete_resource();
    auto t0 = Clock::now();
    for (int i = 0; i < N; ++i) {
      void* p = r->allocate(sz);
      r->deallocate(p, sz);
    }
    auto t1 = Clock::now();
    std::printf("new_delete_resource:           %8.1f ns/op\n", ns_per_op(t1 - t0, N));
  }

  // (c) monotonic_buffer_resource (allocate only, deallocate is no-op).
  {
    Eigen::monotonic_buffer_resource arena(N * sz * 2);
    auto t0 = Clock::now();
    for (int i = 0; i < N; ++i) {
      arena.allocate(sz);
    }
    auto t1 = Clock::now();
    std::printf("monotonic_buffer_resource:     %8.1f ns/op\n", ns_per_op(t1 - t0, N));
  }
}

// Benchmark 2: Arena reuse — allocate N blocks, release, repeat M times.
static void bench_arena_reuse() {
  constexpr int N = 1000;
  constexpr int M = 1000;
  constexpr std::size_t sz = 128;

  // (a) Arena with release.
  {
    Eigen::monotonic_buffer_resource arena(N * sz * 2);
    auto t0 = Clock::now();
    for (int j = 0; j < M; ++j) {
      for (int i = 0; i < N; ++i) {
        arena.allocate(sz);
      }
      arena.release();
    }
    auto t1 = Clock::now();
    std::printf("arena alloc+release:           %8.1f ns/op  (%d x %d)\n", ns_per_op(t1 - t0, N * M), N, M);
  }

  // (b) Individual malloc/free.
  {
    auto t0 = Clock::now();
    for (int j = 0; j < M; ++j) {
      for (int i = 0; i < N; ++i) {
        void* p = Eigen::internal::handmade_aligned_malloc(sz);
        Eigen::internal::handmade_aligned_free(p);
      }
    }
    auto t1 = Clock::now();
    std::printf("individual malloc/free:         %8.1f ns/op  (%d x %d)\n", ns_per_op(t1 - t0, N * M), N, M);
  }
}

// Benchmark 3: Mixed-size allocations.
static void bench_mixed_sizes() {
  constexpr int N = 50000;
  static const std::size_t sizes[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
  constexpr int nsizes = sizeof(sizes) / sizeof(sizes[0]);

  // (a) monotonic.
  {
    std::size_t total = 0;
    for (int i = 0; i < N; ++i) total += sizes[i % nsizes];
    Eigen::monotonic_buffer_resource arena(total + 1024 * 1024);
    auto t0 = Clock::now();
    for (int i = 0; i < N; ++i) {
      arena.allocate(sizes[i % nsizes]);
    }
    auto t1 = Clock::now();
    std::printf("monotonic mixed-size:          %8.1f ns/op\n", ns_per_op(t1 - t0, N));
  }

  // (b) malloc/free.
  {
    auto t0 = Clock::now();
    for (int i = 0; i < N; ++i) {
      void* p = Eigen::internal::handmade_aligned_malloc(sizes[i % nsizes]);
      Eigen::internal::handmade_aligned_free(p);
    }
    auto t1 = Clock::now();
    std::printf("malloc/free mixed-size:         %8.1f ns/op\n", ns_per_op(t1 - t0, N));
  }
}

int main() {
  std::printf("=== PMR Allocator Benchmarks ===\n\n");

  std::printf("--- Throughput (256 bytes, alloc+dealloc) ---\n");
  bench_throughput();

  std::printf("\n--- Arena Reuse (128 bytes, 1000 allocs x 1000 cycles) ---\n");
  bench_arena_reuse();

  std::printf("\n--- Mixed Sizes (64B-64KB) ---\n");
  bench_mixed_sizes();

  return 0;
}
