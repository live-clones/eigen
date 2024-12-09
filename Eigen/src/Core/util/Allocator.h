// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2024 Steve Bronder <sbronder@flatironinstitute.org>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*****************************************************************************
*** Platform checks for aligned malloc functions                           ***
*****************************************************************************/

#ifndef EIGEN_ALLOCATOR_H
#define EIGEN_ALLOCATOR_H

#if (EIGEN_COMP_CXXVER >= 17)
#include <memory_resource>
#define EIGEN_PUBLIC_INHERIT_MEMORY_RESOURCE : public std::pmr::memory_resource
#define EIGEN_MEMORY_RESOURCE_COMPATIBLE
#else
#include <memory>
#define EIGEN_PUBLIC_INHERIT_MEMORY_RESOURCE
#endif
namespace Eigen {

#ifdef EIGEN_MEMORY_RESOURCE_COMPATIBLE
using memory_resource = std::pmr::memory_resource;
#else
/**
 * An abstract interface for encapsulating memory resources.
 */
class memory_resource {
 public:
  /**
   * Allocates storage with a size of at least `bytes` bytes,
   * aligned to the specified alignment.
   * Equivalent to `do_allocate(bytes, alignment);`.
   * @param bytes Number of bytes to allocate.
   * @param alignment Alignment to use.
   * @throw Throws an exception if storage of the requested size and alignment cannot be obtained.
   */
  void* allocate(std::size_t bytes, std::size_t alignment = EIGEN_DEFAULT_ALIGN_BYTES) {
    return do_allocate(bytes, alignment);
  }

  /**
   * Deallocates the storage pointed to by p. p shall have been
   * returned by a prior call to `allocate(bytes, alignment)`
   * on a `memory_resource`` that compares equal to `*this`,
   * and the storage it points to shall not yet have been
   * deallocated.
   * Equivalent to do_deallocate(p, bytes, alignment);
   * @param p Pointer to memory to deallocate.
   * @param bytes Number of bytes originally allocated.
   * @param alignment Alignment originally allocated.
   */
  void deallocate(void* p, std::size_t bytes, std::size_t alignment = EIGEN_DEFAULT_ALIGN_BYTES) {
    do_deallocate(p, bytes, alignment);
  }

  /**
   * Compares `*this` with `other` for equality.
   * Two memory_resources compare equal if and only if memory allocated
   * from one memory_resource can be deallocated from the other
   * and vice versa. Equivalent to `do_is_equal(other);`.
   * @param other The memory_resource to compare with.
   * @return true if the memory_resources are equal, false otherwise.
   */
  bool is_equal(const Eigen::memory_resource& other) const noexcept { return do_is_equal(other); }
  virtual ~memory_resource() = default;

 private:
  /**
   * Allocates storage with a size of at least bytes bytes, aligned to * the specified alignment. Alignment shall be a
   * power of two.
   * @param bytes Number of bytes to allocate.
   * @param alignment Alignment to use.
   * @throw Throws an exception if storage of the requested size and alignment cannot be obtained.
   */
  virtual void* do_allocate(std::size_t bytes, std::size_t alignment) = 0;
  /**
   * Deallocates the storage pointed to by p.
   * p must have been returned by a prior call to
   * `allocate(bytes, alignment)` on a memory_resource that compares
   * equal to *this, and the storage it points to must not yet have
   * been deallocated, otherwise the behavior is undefined.
   * @param p Pointer to memory to deallocate.
   * @param bytes Number of bytes originally allocated.
   * @param alignment Alignment originally allocated.
   */
  virtual void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) = 0;

  /**
   * Compares *this for equality with other.
   * Two memory_resources compare equal if and only if memory allocated
   * from one memory_resource can be deallocated from the other and vice
   * versa.
   * @param other The memory_resource to compare with.
   */
  virtual bool do_is_equal(const Eigen::memory_resource& other) const noexcept = 0;
};

/**
 * Compares the memory_resources a and b for equality.
 * Two memory_resources compare equal if and only if memory allocated
 * from one memory_resource can be deallocated from the other
 * and vice versa.
 */
bool operator==(const Eigen::memory_resource& a, const Eigen::memory_resource& b) noexcept {
  return &a == &b || a.is_equal(b);
}
bool operator!=(const Eigen::memory_resource& a, const Eigen::memory_resource& b) noexcept { return !(a == b); }
#endif

class new_delete_resource final : public Eigen::memory_resource {
 public:
  virtual void* do_allocate(std::size_t bytes, std::size_t alignment) {
    return internal::handmade_aligned_malloc(bytes, alignment);
  }
  virtual void do_deallocate(void* p, std::size_t /* bytes */, std::size_t /* alignment */) {
    internal::handmade_aligned_free(p);
  }
  virtual bool do_is_equal(const Eigen::memory_resource& other) const noexcept {
    return dynamic_cast<const new_delete_resource*>(&other) != nullptr;
  }
  virtual ~new_delete_resource(){};
};

static memory_resource* get_default_resource() {
  static new_delete_resource new_delete_resource_instance;
  return &new_delete_resource_instance;
}

using byte_t = unsigned char;
namespace internal {
static constexpr size_t DEFAULT_INITIAL_NBYTES = 1 << 2;
}

class monotonic_buffer_resource final : public memory_resource {
 private:
  // resource used to request memory
  memory_resource* upstream_;
  // storage for blocks
  std::vector<byte_t*> blocks_;
  // size of each block
  std::vector<size_t> sizes_;
  // end of current block
  byte_t* block_end_;
  // next available spot
  byte_t* next_mem_;
  // current block in use
  size_t block_idx_{0};
  // factor by which to grow the buffer
  float growth_factor_{1.5};
  // true if we are using a user-provided buffer as the first block
  bool user_buffer_{false};

 public:
  /**
   * Sets the current buffer to an implementation-defined size.
   */
  monotonic_buffer_resource()
      : upstream_(get_default_resource()),
        blocks_(1, static_cast<byte_t*>(upstream_->allocate(internal::DEFAULT_INITIAL_NBYTES))),
        sizes_(1, internal::DEFAULT_INITIAL_NBYTES),
        block_end_(blocks_[0] + internal::DEFAULT_INITIAL_NBYTES),
        next_mem_(blocks_[0]) {}

  /**
   * Sets the current buffer to an implementation-defined size.
   */
  explicit monotonic_buffer_resource(Eigen::memory_resource* upstream)
      : upstream_(upstream),
        blocks_(1, static_cast<byte_t*>(upstream_->allocate(internal::DEFAULT_INITIAL_NBYTES))),
        sizes_(1, internal::DEFAULT_INITIAL_NBYTES),
        block_end_(blocks_[0] + internal::DEFAULT_INITIAL_NBYTES),
        next_mem_(blocks_[0]) {}

  /**
   * Sets the current buffer to a user size.
   */
  explicit monotonic_buffer_resource(std::size_t initial_size)
      : upstream_(get_default_resource()),
        blocks_(1, static_cast<byte_t*>(upstream_->allocate(initial_size))),
        sizes_(1, initial_size),
        block_end_(blocks_[0] + initial_size),
        next_mem_(blocks_[0]) {}

  /**
   * Sets the current buffer to a user size.
   */
  monotonic_buffer_resource(std::size_t initial_size, Eigen::memory_resource* upstream)
      : upstream_(upstream),
        blocks_(1, static_cast<byte_t*>(upstream_->allocate(initial_size))),
        sizes_(1, initial_size),
        block_end_(blocks_[0] + initial_size),
        next_mem_(blocks_[0]) {}

  /**
   * Sets the current buffer to `buffer` and the next buffer size to buffer_size
   * (but not less than 1). Then increase the next buffer size by a
   * growth factor (which does not have to be integral).
   */
  monotonic_buffer_resource(void* buffer, std::size_t initial_size)
      : upstream_(get_default_resource()),
        blocks_(1, static_cast<byte_t*>(buffer)),
        sizes_(1, initial_size),
        block_end_(blocks_[0] + initial_size),
        next_mem_(blocks_[0]),
        block_idx_(0),
        growth_factor_(1.5),
        user_buffer_(true) {}

  /**
   * Sets the current buffer to `buffer` and the next buffer size to buffer_size
   * (but not less than 1). Then increase the next buffer size by a
   * growth factor (which does not have to be integral).
   */
  monotonic_buffer_resource(void* buffer, std::size_t initial_size, Eigen::memory_resource* upstream)
      : upstream_(upstream),
        blocks_(1, static_cast<byte_t*>(buffer)),
        sizes_(1, initial_size),
        block_end_(blocks_[0] + initial_size),
        next_mem_(blocks_[0]),
        block_idx_(0),
        growth_factor_(1.5),
        user_buffer_(true) {}

  monotonic_buffer_resource(const monotonic_buffer_resource&) = delete;
  monotonic_buffer_resource& operator=(const monotonic_buffer_resource&) = delete;
  // TODO(SteveBronder): Do we want to allow merging of monotonic_buffer_resources?
  monotonic_buffer_resource(monotonic_buffer_resource&& other) = delete;
  monotonic_buffer_resource& operator=(monotonic_buffer_resource&&) = delete;

  /**
   * Destroy this memory allocator.
   */
  virtual ~monotonic_buffer_resource() {
    std::size_t start_free = user_buffer_ ? 1 : 0;
    for (std::size_t i = start_free; i < blocks_.size(); ++i) {
      if (blocks_[i]) {
        upstream_->deallocate(blocks_[i], sizes_[i]);
      }
    }
  }

  EIGEN_STRONG_INLINE memory_resource* upstream_resource() noexcept { return upstream_; }
  EIGEN_STRONG_INLINE const memory_resource* upstream_resource() const noexcept { return upstream_; }

  EIGEN_STRONG_INLINE void set_growth_factor(std::size_t growth_factor) noexcept { growth_factor_ = growth_factor; }

 private:
  /**
   * Moves to the next block of memory, allocating that block
   * if necessary, and allocates bytes of memory within that
   * block.
   *
   * @param bytes Number of bytes to allocate.
   * @return A pointer to the allocated memory.
   */
  EIGEN_STRONG_INLINE byte_t* move_to_next_block(size_t bytes, std::size_t alignment) {
    ++block_idx_;
    // Find the next block (if any) containing at least `bytes` bytes.
    while ((block_idx_ < blocks_.size()) && (sizes_[block_idx_] < bytes + alignment)) {
      ++block_idx_;
    }
    if (EIGEN_PREDICT_TRUE(block_idx_ < blocks_.size())) {
      uint8_t offset =
          static_cast<uint8_t>(alignment - (reinterpret_cast<std::size_t>(blocks_[block_idx_]) & (alignment - 1)));
      byte_t* result = blocks_[block_idx_] + offset;
      // Get the object's state back in order.
      next_mem_ = blocks_[block_idx_] + bytes + offset;
      block_end_ = blocks_[block_idx_] + sizes_[block_idx_];
      return result;
    } else {
      // Allocate a new block if necessary.
      // Cold lambda since this is a rare event
      return [this, bytes, alignment]() mutable EIGEN_COLD {
        size_t newsize = this->sizes_.back() * this->growth_factor_;
        if (newsize < bytes) {
          newsize = bytes;
        }
        this->blocks_.push_back(static_cast<byte_t*>(this->upstream_->allocate(newsize, alignment)));
        if (!this->blocks_.back()) {
          throw std::bad_alloc();
        }
        this->sizes_.push_back(newsize);
        byte_t* result = this->blocks_[this->block_idx_];
        // Get the object's state back in order.
        this->next_mem_ = result + bytes;
        this->block_end_ = result + this->sizes_[this->block_idx_];
        return result;
      }();
    }
  }

 protected:
  /**
   * Allocates storage with a size of at least bytes bytes, aligned to * the specified alignment. Alignment shall be a
   * power of two.
   * @throw Throws an exception if storage of the requested size and alignment cannot be obtained.
   */
  virtual void* do_allocate(std::size_t bytes, std::size_t alignment) {
    uint8_t offset = static_cast<uint8_t>(alignment - (reinterpret_cast<std::size_t>(next_mem_) & (alignment - 1)));
    byte_t* result = next_mem_ + offset;
    next_mem_ += bytes + offset;
    if (EIGEN_PREDICT_FALSE(next_mem_ >= block_end_)) {
      result = move_to_next_block(bytes, alignment);
    }
    return reinterpret_cast<void*>(result);
  }

  /**
   * Deallocates the storage pointed to by p.
   * p must have been returned by a prior call to
   * `allocate(bytes, alignment)` on a memory_resource that compares
   * equal to *this, and the storage it points to must not yet have
   * been deallocated, otherwise the behavior is undefined.
   */
  virtual void do_deallocate(void* /* p */, std::size_t /* bytes */, std::size_t /* alignment */){};

  /**
   * Compares *this for equality with other.
   * Two memory_resources compare equal if and only if memory allocated
   * from one memory_resource can be deallocated from the other and vice
   * versa.
   */
  virtual bool do_is_equal(const Eigen::memory_resource& other) const noexcept {
    return dynamic_cast<const monotonic_buffer_resource*>(&other) != nullptr;
  }

 public:
  /**
   * Recover all the memory used by the stack allocator.  The stack
   * of memory blocks allocated so far will be available for further
   * allocations.  To free memory back to the system, use the
   * function free_all().
   */
  inline void release() {
    block_idx_ = 0;
    next_mem_ = blocks_[0];
    block_end_ = next_mem_ + sizes_[0];
  }
  /**
   * Return number of bytes allocated to this instance by the heap.
   * This is not the same as the number of bytes allocated through
   * calls to memalloc_.  The latter number is not calculatable
   * because space is wasted at the end of blocks if the next
   * alloc request doesn't fit.  (Perhaps we could trim down to
   * what is actually used?)
   *
   * @return number of bytes allocated to this instance
   */
  inline size_t bytes_allocated() const {
    size_t sum = 0;
    for (size_t i = 0; i <= block_idx_; ++i) {
      sum += sizes_[i];
    }
    return sum;
  }

  EIGEN_STRONG_INLINE const auto& blocks() const noexcept { return blocks_; }
  EIGEN_STRONG_INLINE const auto& sizes() const noexcept { return sizes_; }
};

#if (EIGEN_COMP_CXXVER >= 17)
class polymorphic_allocator : public std::pmr::polymorphic_allocator<byte_t> {
 public:
  /**
   * Constructs a polymorphic_allocator using the return
   * value of Eigen::get_default_resource() as the
   * underlying memory resource.
   */
  polymorphic_allocator() noexcept : std::pmr::polymorphic_allocator<byte_t>(get_default_resource()){};

  /**
   * Constructs a polymorphic_allocator using
   * `other.resource()` as the underlying memory resource.
   */
  polymorphic_allocator(const polymorphic_allocator& other) = default;

  /**
   * Constructs a polymorphic_allocator using `r`` as the
   * underlying memory resource. This constructor
   * provides an implicit conversion from
   * `Eigen::memory_resource*`.
   */
  polymorphic_allocator(Eigen::memory_resource* resource) noexcept
      : std::pmr::polymorphic_allocator<byte_t>(resource) {}

  /**
   * Allocates storage for n objects of type T using the underlying memory resource.
   * @param n	the number of objects to allocate storage for
   * @return A pointer to the allocated storage.
   */
  template <typename T>
  EIGEN_STRONG_INLINE T* allocate(std::size_t n, std::size_t alignment = EIGEN_DEFAULT_ALIGN_BYTES) {
    return static_cast<T*>(resource()->allocate(n * sizeof(T), alignment));
  }

  /**
   * Deallocates the storage pointed to by p, which must have
   * been allocated from a Eigen::memory_resource x that
   * compares equal to `*resource()`` using
   *  `x.allocate(n * sizeof(T), alignof(T))`.
   */
  template <typename T>
  EIGEN_STRONG_INLINE void deallocate(T* p, std::size_t n, std::size_t alignment = EIGEN_DEFAULT_ALIGN_BYTES) {
    resource()->deallocate(p, n * sizeof(T), alignment);
  }
};

#else
class polymorphic_allocator {
  Eigen::memory_resource* resource_;

 public:
  /**
   * Returns the underlying memory resource.
   */
  Eigen::memory_resource* resource() noexcept { return resource_; }

  /**
   * Returns the underlying memory resource.
   */
  const Eigen::memory_resource* resource() const noexcept { return resource_; }

  polymorphic_allocator select_on_container_copy_construction() const { return *this; }

  /**
   * Constructs a polymorphic_allocator using the return
   * value of Eigen::get_default_resource() as the
   * underlying memory resource.
   */
  polymorphic_allocator() noexcept : resource_(get_default_resource()){};

  /**
   * Constructs a polymorphic_allocator using `other.resource()` as the
   * underlying memory resource.
   */
  polymorphic_allocator(const polymorphic_allocator& other) = default;

  /**
   * Constructs a polymorphic_allocator using `r`` as the
   * underlying memory resource. This constructor
   * provides an implicit conversion from
   * `Eigen::memory_resource*`.
   */
  polymorphic_allocator(Eigen::memory_resource* resource) noexcept : resource_(resource) {}

  /**
   * Allocates storage for n objects of type T using the underlying memory resource.
   * @param n	the number of objects to allocate storage for
   * @return A pointer to the allocated storage.
   */
  template <typename T>
  T* allocate(std::size_t n, std::size_t alignment = EIGEN_DEFAULT_ALIGN_BYTES) {
    return static_cast<T*>(resource()->allocate(n * sizeof(T), alignment));
  }

  /**
   * Deallocates the storage pointed to by p, which must have
   * been allocated from a Eigen::memory_resource x that
   * compares equal to `*resource()`` using
   *  `x.allocate(n * sizeof(T), alignof(T))`.
   */
  template <typename T>
  void deallocate(T* p, std::size_t n, std::size_t alignment = EIGEN_DEFAULT_ALIGN_BYTES) {
    resource()->deallocate(p, n * sizeof(T), alignment);
  }

  /**
   * Creates an object of the given type U by means of
   * uses-allocator construction at the uninitialized memory
   * location indicated by p, using *this as the allocator.
   * @tparam U type of allocated storage
   * @tparam Args Types of constructor arguments for `U`
   * @param p pointer to allocated, but not initialized storage
   * @param args the constructor arguments to pass to the
   * constructor of T
   */
  template <typename U, typename... Args>
  void construct(U* p, std::size_t alignment = EIGEN_DEFAULT_ALIGN_BYTES, Args&&... args) {
    internal::construct_at(p, std::forward<Args>(args)...);
  }

  /**
   * Destroys the object pointed to by p, as if by calling p->~U().
   * @tparam U type of allocated storage
   * @tparam Args Types of constructor arguments for `U`
   * @param p	pointer to the object being destroyed
   */
  template <class U>
  void destroy(U* p) {
    internal::destroy_at(p);
  }

  /**
   * Allocates nbytes bytes of storage at specified alignment alignment using the
   * underlying memory resource. Equivalent to
   * `return resource()->allocate(nbytes, alignment);`
   * @param nbytes The number of bytes to allocate
   * @param alignment The alignment to use
   */
  void* allocate_bytes(std::size_t nbytes, std::size_t alignment = EIGEN_DEFAULT_ALIGN_BYTES) {
    return resource()->allocate(nbytes, alignment);
  }

  /**
   * Deallocates the storage pointed to by p, which must have been
   * allocated from a `Eigen::memory_resource` x that compares equal
   * to `*resource()`, using `x.allocate(nbytes, alignment)`,
   * typically through a call to `allocate_bytes(nbytes, alignment)`.
   * `Equivalent to resource()->deallocate(p, nbytes, alignment);`.
   *
   * @param p pointer to memory to deallocate
   * @param nbytes The number of bytes originally allocated
   * @param alignment The alignment originally allocated
   */
  void deallocate_bytes(void* p, std::size_t nbytes, std::size_t alignment = EIGEN_DEFAULT_ALIGN_BYTES) {
    resource()->deallocate(p, nbytes, alignment);
  }
};
#endif

}  // namespace Eigen

#endif
