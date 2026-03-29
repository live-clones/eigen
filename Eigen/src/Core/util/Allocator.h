// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2024 Steve Bronder <sbronder@flatironinstitute.org>
// Copyright (C) 2026 Pavel Guzenfeld <pavelguzenfeld@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_ALLOCATOR_H
#define EIGEN_ALLOCATOR_H

// IWYU pragma: private
#include "../InternalHeaderCheck.h"

// Detect C++17 PMR support.
#ifndef EIGEN_HAS_CXX17_PMR
#if EIGEN_MAX_CPP_VER >= 17 && EIGEN_COMP_CXXVER >= 17
#ifdef __has_include
#if __has_include(<memory_resource>)
#define EIGEN_HAS_CXX17_PMR 1
#endif
#endif
#endif
#ifndef EIGEN_HAS_CXX17_PMR
#define EIGEN_HAS_CXX17_PMR 0
#endif
#endif

#if EIGEN_HAS_CXX17_PMR
#include <memory_resource>
#endif

#include <atomic>
#include <cstddef>
#include <memory>

#ifndef EIGEN_GPU_COMPILE_PHASE

namespace Eigen {

namespace internal {

// Minimum alignment for PMR allocations. When EIGEN_DEFAULT_ALIGN_BYTES is 0
// (alignment disabled), fall back to max_align_t.
static constexpr std::size_t pmr_default_alignment =
    EIGEN_DEFAULT_ALIGN_BYTES > 0 ? static_cast<std::size_t>(EIGEN_DEFAULT_ALIGN_BYTES) : alignof(std::max_align_t);

}  // namespace internal

/** \ingroup Core_Module
 * \brief Abstract interface for memory resources.
 *
 * This class mirrors std::pmr::memory_resource. When compiling with C++17 and
 * \<memory_resource\> is available, Eigen::memory_resource inherits from
 * std::pmr::memory_resource, allowing Eigen resources to be used with standard
 * PMR containers (e.g. std::pmr::vector).
 *
 * The default alignment is EIGEN_DEFAULT_ALIGN_BYTES (SIMD-friendly), unlike
 * std::pmr which defaults to alignof(max_align_t).
 *
 * \note This class is not thread-safe. External synchronization is required
 * when a single resource is shared across threads.
 *
 * \sa polymorphic_allocator, monotonic_buffer_resource, new_delete_resource()
 */
class memory_resource
#if EIGEN_HAS_CXX17_PMR
    : public std::pmr::memory_resource
#endif
{
 public:
  memory_resource() = default;
  memory_resource(const memory_resource&) = default;
  memory_resource& operator=(const memory_resource&) = default;
  virtual ~memory_resource() = default;

  /** Allocate \a bytes bytes of memory with the given \a alignment. */
  void* allocate(std::size_t bytes, std::size_t alignment = internal::pmr_default_alignment) {
#if EIGEN_HAS_CXX17_PMR
    return std::pmr::memory_resource::allocate(bytes, alignment);
#else
    return do_allocate(bytes, alignment);
#endif
  }

  /** Deallocate memory previously obtained from allocate(). */
  void deallocate(void* p, std::size_t bytes, std::size_t alignment = internal::pmr_default_alignment) {
#if EIGEN_HAS_CXX17_PMR
    std::pmr::memory_resource::deallocate(p, bytes, alignment);
#else
    do_deallocate(p, bytes, alignment);
#endif
  }

  /** Returns true if memory allocated from \c *this can be deallocated from \a other and vice versa. */
  bool is_equal(const memory_resource& other) const noexcept {
#if EIGEN_HAS_CXX17_PMR
    return std::pmr::memory_resource::is_equal(other);
#else
    return do_is_equal(other);
#endif
  }

#if !EIGEN_HAS_CXX17_PMR
 private:
  virtual void* do_allocate(std::size_t bytes, std::size_t alignment) = 0;
  virtual void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) = 0;
  virtual bool do_is_equal(const memory_resource& other) const noexcept = 0;
#endif
};

inline bool operator==(const memory_resource& a, const memory_resource& b) noexcept {
  return &a == &b || a.is_equal(b);
}
inline bool operator!=(const memory_resource& a, const memory_resource& b) noexcept { return !(a == b); }

namespace internal {

/** Default memory resource using Eigen's handmade aligned malloc/free. */
class new_delete_memory_resource final : public memory_resource {
  void* do_allocate(std::size_t bytes, std::size_t alignment) override {
    if (bytes == 0) return nullptr;
    // Ensure alignment meets handmade_aligned_malloc requirements.
    if (alignment < sizeof(void*)) alignment = sizeof(void*);
    void* p = handmade_aligned_malloc(bytes, alignment);
    if (!p) internal::throw_std_bad_alloc();
    return p;
  }

  void do_deallocate(void* p, std::size_t /*bytes*/, std::size_t /*alignment*/) override {
    handmade_aligned_free(p);
  }

  bool do_is_equal(const
#if EIGEN_HAS_CXX17_PMR
                   std::pmr::memory_resource
#else
                   memory_resource
#endif
                       & other) const noexcept override {
    return this == &other;
  }
};

inline std::atomic<memory_resource*>& default_resource_instance() noexcept {
  static std::atomic<memory_resource*> instance{nullptr};
  return instance;
}

}  // namespace internal

/** Returns a pointer to the default new/delete memory resource using Eigen's aligned allocation. */
inline memory_resource* new_delete_resource() noexcept {
  static internal::new_delete_memory_resource instance;
  return &instance;
}

/** Returns the current default memory resource. If none has been set, returns new_delete_resource(). */
inline memory_resource* get_default_resource() noexcept {
  memory_resource* r = internal::default_resource_instance().load(std::memory_order_acquire);
  return r ? r : new_delete_resource();
}

/** Sets the default memory resource to \a r. Returns the previous default resource.
 * Pass nullptr to restore new_delete_resource() as the default. */
inline memory_resource* set_default_resource(memory_resource* r) noexcept {
  memory_resource* prev = internal::default_resource_instance().exchange(r, std::memory_order_acq_rel);
  return prev ? prev : new_delete_resource();
}

/** \ingroup Core_Module
 * \brief Arena-style memory resource that allocates monotonically.
 *
 * Memory is allocated from contiguous blocks. Individual deallocations are
 * no-ops; all memory is released at once via release() or on destruction.
 *
 * Blocks are managed via an intrusive linked list (no std::vector overhead).
 * When the current block is exhausted, a new block of at least 2x the previous
 * size is allocated from the upstream resource.
 *
 * An optional initial buffer (e.g. stack-allocated) can be provided to avoid
 * any upstream allocations for small workloads.
 *
 * \note Not thread-safe. Not copyable. Not movable.
 *
 * \sa memory_resource, polymorphic_allocator
 */
class monotonic_buffer_resource : public memory_resource {
 public:
  /** Construct with default upstream resource and no initial buffer. */
  monotonic_buffer_resource() noexcept : monotonic_buffer_resource(0, get_default_resource()) {}

  /** Construct with explicit upstream resource. */
  explicit monotonic_buffer_resource(memory_resource* upstream) noexcept
      : monotonic_buffer_resource(0, upstream) {}

  /** Construct with an initial size hint. The first upstream allocation will be at least \a initial_size bytes. */
  explicit monotonic_buffer_resource(std::size_t initial_size) noexcept
      : monotonic_buffer_resource(initial_size, get_default_resource()) {}

  /** Construct with initial size hint and explicit upstream resource. */
  monotonic_buffer_resource(std::size_t initial_size, memory_resource* upstream) noexcept
      : m_upstream(upstream), m_initial_size(initial_size > 0 ? initial_size : 256) {}

  /** Construct with a user-provided initial buffer. */
  monotonic_buffer_resource(void* buffer, std::size_t buffer_size) noexcept
      : monotonic_buffer_resource(buffer, buffer_size, get_default_resource()) {}

  /** Construct with a user-provided initial buffer and explicit upstream resource. */
  monotonic_buffer_resource(void* buffer, std::size_t buffer_size, memory_resource* upstream) noexcept
      : m_upstream(upstream),
        m_initial_buffer(static_cast<char*>(buffer)),
        m_initial_size(buffer_size),
        m_current_buffer(static_cast<char*>(buffer)),
        m_current_size(buffer_size) {}

  ~monotonic_buffer_resource() override { release(); }

  monotonic_buffer_resource(const monotonic_buffer_resource&) = delete;
  monotonic_buffer_resource& operator=(const monotonic_buffer_resource&) = delete;

  /** Release all memory obtained from the upstream resource. Pointers previously
   * returned by allocate() become invalid. The initial user-provided buffer (if any)
   * is NOT freed but is made available for new allocations. */
  void release() noexcept {
    block_header* b = m_block_head;
    while (b) {
      block_header* next = b->next;
      m_upstream->deallocate(b, b->size, alignof(block_header));
      b = next;
    }
    m_block_head = nullptr;
    m_current_buffer = m_initial_buffer;
    m_current_size = m_initial_buffer ? m_initial_size : 0;
    m_current_offset = 0;
  }

  /** Returns the upstream memory resource. */
  memory_resource* upstream_resource() const noexcept { return m_upstream; }

 private:
  struct block_header {
    block_header* next;
    std::size_t size;
  };

  // Round up to alignment boundary.
  static constexpr std::size_t header_aligned_size(std::size_t align) {
    return (sizeof(block_header) + align - 1) & ~(align - 1);
  }

  void* do_allocate(std::size_t bytes, std::size_t alignment) override {
    if (bytes == 0) return nullptr;

    // Try to allocate from the current buffer.
    if (m_current_buffer) {
      std::size_t space = m_current_size - m_current_offset;
      void* ptr = static_cast<void*>(m_current_buffer + m_current_offset);
      if (std::align(alignment, bytes, ptr, space)) {
        m_current_offset = m_current_size - space + bytes;
        return ptr;
      }
    }
    return allocate_from_new_block(bytes, alignment);
  }

  void do_deallocate(void* /*p*/, std::size_t /*bytes*/, std::size_t /*alignment*/) override {
    // Monotonic: individual deallocation is a no-op.
  }

  bool do_is_equal(const
#if EIGEN_HAS_CXX17_PMR
                   std::pmr::memory_resource
#else
                   memory_resource
#endif
                       & other) const noexcept override {
    return this == &other;
  }

  void* allocate_from_new_block(std::size_t bytes, std::size_t alignment) {
    const std::size_t hdr_size = header_aligned_size(alignment > alignof(block_header) ? alignment : alignof(block_header));
    const std::size_t min_block = hdr_size + bytes + alignment;

    // Growth: at least 2x previous, at least min_block, at least m_initial_size.
    std::size_t grow = m_current_size > 0 ? m_current_size * 2 : m_initial_size;
    if (grow < min_block) grow = min_block;
    if (grow < 256) grow = 256;

    // Check for overflow.
    if (grow < bytes) internal::throw_std_bad_alloc();

    void* raw = m_upstream->allocate(grow, alignof(block_header));

    // Link the new block into the list.
    block_header* header = static_cast<block_header*>(raw);
    header->next = m_block_head;
    header->size = grow;
    m_block_head = header;

    // Usable region starts after the header.
    m_current_buffer = static_cast<char*>(raw) + hdr_size;
    m_current_size = grow - hdr_size;
    m_current_offset = 0;

    // Allocate from the new block.
    std::size_t space = m_current_size;
    void* ptr = static_cast<void*>(m_current_buffer);
    if (std::align(alignment, bytes, ptr, space)) {
      m_current_offset = m_current_size - space + bytes;
      return ptr;
    }

    // Should not happen given min_block calculation.
    eigen_assert(false && "monotonic_buffer_resource: allocation from new block failed");
    internal::throw_std_bad_alloc();
    return nullptr;
  }

  memory_resource* m_upstream = nullptr;
  char* m_initial_buffer = nullptr;
  std::size_t m_initial_size = 256;
  char* m_current_buffer = nullptr;
  std::size_t m_current_size = 0;
  std::size_t m_current_offset = 0;
  block_header* m_block_head = nullptr;
};

/** \ingroup Core_Module
 * \brief Type-erased polymorphic allocator for Eigen types.
 *
 * Unlike std::pmr::polymorphic_allocator\<T\>, this allocator is not templated
 * on a value type. It operates on raw bytes, with the caller responsible for
 * computing sizes. This avoids type-mismatch issues when mixing scalar types
 * in expression templates.
 *
 * Following PMR semantics, copy assignment is deleted (allocators do not
 * propagate on container assignment).
 *
 * \sa memory_resource, monotonic_buffer_resource, new_delete_resource()
 */
class polymorphic_allocator {
 public:
  /** Default-construct using get_default_resource(). */
  polymorphic_allocator() noexcept : m_resource(get_default_resource()) {}

  /** Construct from a memory resource. \a r must not be null. */
  polymorphic_allocator(memory_resource* r) : m_resource(r) { eigen_assert(r != nullptr); }

  polymorphic_allocator(const polymorphic_allocator&) = default;

  // PMR semantics: allocator does not propagate on assignment.
  polymorphic_allocator& operator=(const polymorphic_allocator&) = delete;

  /** Allocate \a bytes bytes with the given \a alignment. */
  void* allocate(std::size_t bytes, std::size_t alignment = internal::pmr_default_alignment) {
    return m_resource->allocate(bytes, alignment);
  }

  /** Deallocate memory previously obtained from allocate(). */
  void deallocate(void* p, std::size_t bytes, std::size_t alignment = internal::pmr_default_alignment) {
    m_resource->deallocate(p, bytes, alignment);
  }

  /** Returns the underlying memory resource. */
  memory_resource* resource() const noexcept { return m_resource; }

  friend bool operator==(const polymorphic_allocator& a, const polymorphic_allocator& b) noexcept {
    return *a.m_resource == *b.m_resource;
  }
  friend bool operator!=(const polymorphic_allocator& a, const polymorphic_allocator& b) noexcept {
    return !(a == b);
  }

 private:
  memory_resource* m_resource;
};

}  // namespace Eigen

#endif  // EIGEN_GPU_COMPILE_PHASE

#endif  // EIGEN_ALLOCATOR_H
