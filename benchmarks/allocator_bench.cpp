// Standalone benchmark for PMR allocator primitives.
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
// Main
// =========================================================================

int main() {
  std::printf("=== PMR Allocator Benchmarks ===\n\n");

  std::printf("--- 1. Raw allocation throughput (256B, alloc+dealloc) ---\n");
  bench_throughput();

  std::printf("\n--- 2. Arena reuse (128B, 1000 x 1000) ---\n");
  bench_arena_reuse();

  return 0;
}
