// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Pavel Guzenfeld <pavelguzenfeld@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#include <cstdint>
#include <vector>

// ---------------------------------------------------------------------------
// Test helper: counting_resource tracks upstream allocations/deallocations.
// ---------------------------------------------------------------------------
class counting_resource : public Eigen::memory_resource {
  Eigen::memory_resource* m_upstream;
  int m_alloc_count = 0;
  int m_dealloc_count = 0;
  std::size_t m_total_bytes = 0;

  void* do_allocate(std::size_t bytes, std::size_t alignment) override {
    ++m_alloc_count;
    m_total_bytes += bytes;
    return m_upstream->allocate(bytes, alignment);
  }
  void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) override {
    ++m_dealloc_count;
    m_upstream->deallocate(p, bytes, alignment);
  }
  bool do_is_equal(const
#if EIGEN_HAS_CXX17_PMR
                   std::pmr::memory_resource
#else
                   Eigen::memory_resource
#endif
                       & other) const noexcept override {
    return this == &other;
  }

 public:
  explicit counting_resource(Eigen::memory_resource* upstream = Eigen::new_delete_resource()) : m_upstream(upstream) {}
  int alloc_count() const { return m_alloc_count; }
  int dealloc_count() const { return m_dealloc_count; }
  std::size_t total_bytes() const { return m_total_bytes; }
  void reset_counts() {
    m_alloc_count = 0;
    m_dealloc_count = 0;
    m_total_bytes = 0;
  }
};

// ---------------------------------------------------------------------------
// Helper: check pointer alignment.
// ---------------------------------------------------------------------------
static bool is_aligned(const void* ptr, std::size_t alignment) {
  return (reinterpret_cast<std::uintptr_t>(ptr) % alignment) == 0;
}

// ---------------------------------------------------------------------------
// 1. new_delete_resource basics
// ---------------------------------------------------------------------------
static void test_new_delete_resource() {
  Eigen::memory_resource* r = Eigen::new_delete_resource();
  VERIFY(r != nullptr);

  // Singleton: repeated calls return the same pointer.
  VERIFY(Eigen::new_delete_resource() == r);

  // Allocate, write, deallocate.
  void* p = r->allocate(128);
  VERIFY(p != nullptr);
  VERIFY(is_aligned(p, Eigen::internal::pmr_default_alignment));
  std::memset(p, 0xAB, 128);
  r->deallocate(p, 128);

  // Various alignments.
  for (std::size_t align : {8u, 16u, 32u, 64u}) {
    void* q = r->allocate(256, align);
    VERIFY(q != nullptr);
    VERIFY(is_aligned(q, align));
    r->deallocate(q, 256, align);
  }

  // Zero-size allocation.
  void* z = r->allocate(0);
  VERIFY(z == nullptr);
}

// ---------------------------------------------------------------------------
// 2. Default resource management
// ---------------------------------------------------------------------------
static void test_default_resource() {
  // Initially returns new_delete_resource.
  VERIFY(Eigen::get_default_resource() == Eigen::new_delete_resource());

  // Set a custom resource.
  counting_resource custom;
  Eigen::memory_resource* prev = Eigen::set_default_resource(&custom);
  VERIFY(prev == Eigen::new_delete_resource());
  VERIFY(Eigen::get_default_resource() == &custom);

  // Restore default.
  prev = Eigen::set_default_resource(nullptr);
  VERIFY(prev == &custom);
  VERIFY(Eigen::get_default_resource() == Eigen::new_delete_resource());
}

// ---------------------------------------------------------------------------
// 3. monotonic_buffer_resource with stack buffer
// ---------------------------------------------------------------------------
static void test_monotonic_stack_buffer() {
  alignas(64) char buf[1024];
  Eigen::monotonic_buffer_resource arena(buf, sizeof(buf));

  // Small allocations should come from the stack buffer.
  std::vector<void*> ptrs;
  for (int i = 0; i < 10; ++i) {
    void* p = arena.allocate(64);
    VERIFY(p != nullptr);
    VERIFY(p >= buf && p < buf + sizeof(buf));
    ptrs.push_back(p);
  }

  // Verify no overlap between consecutive allocations.
  for (std::size_t i = 1; i < ptrs.size(); ++i) {
    VERIFY(static_cast<char*>(ptrs[i]) >= static_cast<char*>(ptrs[i - 1]) + 64);
  }
}

// ---------------------------------------------------------------------------
// 4. monotonic_buffer_resource heap growth + release/reuse
// ---------------------------------------------------------------------------
static void test_monotonic_heap_growth() {
  counting_resource counter;
  {
    Eigen::monotonic_buffer_resource arena(64, &counter);

    // Allocate more than 64 bytes to force block growth.
    arena.allocate(128);
    VERIFY(counter.alloc_count() >= 1);

    int allocs_before = counter.alloc_count();
    // Many small allocations should batch into existing blocks.
    for (int i = 0; i < 100; ++i) {
      arena.allocate(8);
    }
    // Should NOT have allocated 100 new blocks.
    VERIFY(counter.alloc_count() - allocs_before < 10);
  }
  // Destructor should free all upstream blocks.
  VERIFY(counter.alloc_count() == counter.dealloc_count());
}

// ---------------------------------------------------------------------------
// 5. monotonic_buffer_resource release and reuse
// ---------------------------------------------------------------------------
static void test_monotonic_release() {
  counting_resource counter;
  Eigen::monotonic_buffer_resource arena(256, &counter);

  arena.allocate(128);
  arena.allocate(128);
  int allocs = counter.alloc_count();

  arena.release();
  VERIFY(counter.dealloc_count() == allocs);

  // Allocate again after release.
  void* p = arena.allocate(64);
  VERIFY(p != nullptr);

  // Multiple release calls are idempotent.
  arena.release();
  arena.release();
}

// ---------------------------------------------------------------------------
// 6. Alignment verification
// ---------------------------------------------------------------------------
static void test_alignment() {
  Eigen::monotonic_buffer_resource arena(4096);

  for (std::size_t align : {8u, 16u, 32u, 64u}) {
    void* p = arena.allocate(100, align);
    VERIFY(p != nullptr);
    VERIFY(is_aligned(p, align));
  }
}

// ---------------------------------------------------------------------------
// 7. polymorphic_allocator basics
// ---------------------------------------------------------------------------
static void test_polymorphic_allocator() {
  // Default construction uses default resource.
  Eigen::polymorphic_allocator alloc;
  VERIFY(alloc.resource() == Eigen::get_default_resource());

  // Construction from resource.
  Eigen::monotonic_buffer_resource arena(1024);
  Eigen::polymorphic_allocator arena_alloc(&arena);
  VERIFY(arena_alloc.resource() == &arena);

  // Allocate and deallocate.
  void* p = arena_alloc.allocate(256);
  VERIFY(p != nullptr);
  arena_alloc.deallocate(p, 256);

  // Copy construction.
  Eigen::polymorphic_allocator copy(arena_alloc);
  VERIFY(copy.resource() == arena_alloc.resource());
}

// ---------------------------------------------------------------------------
// 8. Resource equality semantics
// ---------------------------------------------------------------------------
static void test_resource_equality() {
  // Same singleton compares equal.
  VERIFY(*Eigen::new_delete_resource() == *Eigen::new_delete_resource());

  // Two different monotonic_buffer_resources compare unequal (identity).
  Eigen::monotonic_buffer_resource a(256);
  Eigen::monotonic_buffer_resource b(256);
  VERIFY(a != b);

  // Self-equality.
  VERIFY(a == a);

  // Allocator equality follows resource equality.
  Eigen::polymorphic_allocator alloc_a(&a);
  Eigen::polymorphic_allocator alloc_b(&b);
  Eigen::polymorphic_allocator alloc_a2(&a);
  VERIFY(alloc_a == alloc_a2);
  VERIFY(alloc_a != alloc_b);
}

// ---------------------------------------------------------------------------
// 9. Edge cases: zero-size, large allocation, nested resources
// ---------------------------------------------------------------------------
static void test_edge_cases() {
  Eigen::monotonic_buffer_resource arena(64);

  // Zero-size allocation.
  void* p = arena.allocate(0);
  VERIFY(p == nullptr);

  // Allocation larger than initial size triggers growth.
  void* big = arena.allocate(1024);
  VERIFY(big != nullptr);
  VERIFY(is_aligned(big, Eigen::internal::pmr_default_alignment));

  // Nested resources: monotonic using monotonic as upstream.
  Eigen::monotonic_buffer_resource outer(4096);
  Eigen::monotonic_buffer_resource inner(256, &outer);
  void* q = inner.allocate(128);
  VERIFY(q != nullptr);
}

// ---------------------------------------------------------------------------
// 10. Leak detection via counting_resource
// ---------------------------------------------------------------------------
static void test_no_leaks() {
  counting_resource counter;
  {
    Eigen::monotonic_buffer_resource arena(128, &counter);
    for (int i = 0; i < 50; ++i) {
      arena.allocate(64);
    }
    // Many allocations, several blocks allocated upstream.
    VERIFY(counter.alloc_count() > 0);
  }
  // After destruction, all upstream allocations should be freed.
  VERIFY_IS_EQUAL(counter.alloc_count(), counter.dealloc_count());
}

// ---------------------------------------------------------------------------
// 11. Non-overlapping allocations
// ---------------------------------------------------------------------------
static void test_no_overlap() {
  Eigen::monotonic_buffer_resource arena(4096);

  struct region {
    char* start;
    std::size_t size;
  };
  std::vector<region> regions;

  for (int i = 0; i < 20; ++i) {
    std::size_t sz = 32 + (i * 17) % 200;  // varying sizes
    void* p = arena.allocate(sz);
    VERIFY(p != nullptr);
    regions.push_back({static_cast<char*>(p), sz});
  }

  // Verify no overlap between any two regions.
  for (std::size_t i = 0; i < regions.size(); ++i) {
    for (std::size_t j = i + 1; j < regions.size(); ++j) {
      char* a_start = regions[i].start;
      char* a_end = a_start + regions[i].size;
      char* b_start = regions[j].start;
      char* b_end = b_start + regions[j].size;
      VERIFY(a_end <= b_start || b_end <= a_start);
    }
  }
}

// ---------------------------------------------------------------------------
// 12. Allocate-release-allocate cycle
// ---------------------------------------------------------------------------
static void test_release_reuse_cycle() {
  counting_resource counter;
  Eigen::monotonic_buffer_resource arena(512, &counter);

  for (int cycle = 0; cycle < 5; ++cycle) {
    for (int i = 0; i < 10; ++i) {
      void* p = arena.allocate(32);
      VERIFY(p != nullptr);
    }
    arena.release();
  }
  // After release cycles, should have reasonable upstream usage.
  VERIFY(counter.alloc_count() == counter.dealloc_count());
}

// ---------------------------------------------------------------------------
// 13. C++17 interop
// ---------------------------------------------------------------------------
#if EIGEN_HAS_CXX17_PMR
static void test_cpp17_interop() {
  // Eigen resource is-a std::pmr::memory_resource.
  Eigen::monotonic_buffer_resource arena(4096);
  std::pmr::memory_resource* std_r = &arena;
  void* p = std_r->allocate(128, 16);
  VERIFY(p != nullptr);
  VERIFY(is_aligned(p, 16));
  std_r->deallocate(p, 128, 16);

  // Eigen resource with std::pmr::vector.
  Eigen::monotonic_buffer_resource vec_arena(4096);
  std::pmr::vector<double> v(&vec_arena);
  v.resize(100);
  for (int i = 0; i < 100; ++i) v[i] = static_cast<double>(i);
  VERIFY_IS_APPROX(v[42], 42.0);

  // Eigen resource with std::pmr::polymorphic_allocator.
  std::pmr::polymorphic_allocator<int> std_alloc(&arena);
  int* arr = std_alloc.allocate(10);
  VERIFY(arr != nullptr);
  std_alloc.deallocate(arr, 10);
}
#endif

// ---------------------------------------------------------------------------
// Main test entry point
// ---------------------------------------------------------------------------
EIGEN_DECLARE_TEST(allocator) {
  CALL_SUBTEST(test_new_delete_resource());
  CALL_SUBTEST(test_default_resource());
  CALL_SUBTEST(test_monotonic_stack_buffer());
  CALL_SUBTEST(test_monotonic_heap_growth());
  CALL_SUBTEST(test_monotonic_release());
  CALL_SUBTEST(test_alignment());
  CALL_SUBTEST(test_polymorphic_allocator());
  CALL_SUBTEST(test_resource_equality());
  CALL_SUBTEST(test_edge_cases());
  CALL_SUBTEST(test_no_leaks());
  CALL_SUBTEST(test_no_overlap());
  CALL_SUBTEST(test_release_reuse_cycle());
#if EIGEN_HAS_CXX17_PMR
  CALL_SUBTEST(test_cpp17_interop());
#endif
}
