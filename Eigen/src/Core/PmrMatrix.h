// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Pavel Guzenfeld <pavelguzenfeld@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_PMR_MATRIX_H
#define EIGEN_PMR_MATRIX_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

#ifndef EIGEN_GPU_COMPILE_PHASE

#include <new>          // placement new
#include <type_traits>  // is_trivially_destructible

namespace Eigen {

namespace internal {

// Map alignment option matching the storage alignment PmrMatrix requests from
// its memory_resource. Kept in lockstep so the Map base never claims more
// alignment than the allocation actually guarantees.
constexpr int pmr_map_options() {
#if EIGEN_DEFAULT_ALIGN_BYTES >= 64
  return Aligned64;
#elif EIGEN_DEFAULT_ALIGN_BYTES >= 32
  return Aligned32;
#elif EIGEN_DEFAULT_ALIGN_BYTES >= 16
  return Aligned16;
#elif EIGEN_DEFAULT_ALIGN_BYTES >= 8
  return Aligned8;
#else
  return Unaligned;
#endif
}

}  // namespace internal

/** \ingroup Core_Module
 * \brief Allocator-aware matrix/array wrapper backed by an Eigen::memory_resource.
 *
 * \b This \b is \b a \b prototype. The API (name, Map-derived design, conflicting-resource
 * handling) is under active review on MR !2359 and may change.
 *
 * PmrMatrix wraps an Eigen \c Map over storage obtained from a memory_resource.
 * Unlike plugging an allocator into DenseStorage, only this opt-in type carries
 * the allocator pointer, so regular Eigen matrices pay no size or dispatch cost.
 * It behaves like the underlying Map for all dense operations:
 *
 * \code
 * Eigen::monotonic_buffer_resource arena(1 << 20);
 * Eigen::PmrMatrix<Eigen::MatrixXd> A(&arena, Eigen::MatrixXd::Random(10, 10));
 * Eigen::PmrMatrix<Eigen::MatrixXd> B(&arena, Eigen::MatrixXd::Random(10, 10));
 * Eigen::PmrMatrix<Eigen::MatrixXd> C(&arena, 10, 10);
 * C = A * B;  // result evaluated into C's arena-backed storage
 * \endcode
 *
 * Ownership: PmrMatrix owns the storage and deallocates it (back to the resource)
 * on destruction and before any reallocation. For a monotonic_buffer_resource the
 * deallocate is a no-op; for new_delete_resource it actually frees, so the wrapper
 * does not leak regardless of which resource backs it.
 *
 * Assignment keeps the destination's resource (PMR semantics): assigning between
 * PmrMatrix instances with different resources copies the data and leaves each
 * instance bound to its own resource.
 *
 * \tparam PlainObjectType the wrapped dense type, e.g. \c Eigen::MatrixXd. The scalar
 * type must be trivially destructible (element destructors are not run).
 *
 * \sa memory_resource, monotonic_buffer_resource, byte_allocator
 */
template <typename PlainObjectType>
class PmrMatrix : public Map<PlainObjectType, internal::pmr_map_options()> {
 public:
  using Base = Map<PlainObjectType, internal::pmr_map_options()>;
  using Scalar = typename PlainObjectType::Scalar;
  using Base::cols;
  using Base::rows;

  static_assert(std::is_trivially_destructible<Scalar>::value,
                "PmrMatrix requires a trivially destructible scalar type; element destructors are not run.");

  /** Construct a \a rows x \a cols matrix backed by \a resource (uninitialized, like Eigen). */
  PmrMatrix(memory_resource* resource, Index rows, Index cols)
      : Base(allocate_raw(resource, rows, cols), rows, cols),
        m_resource(resource),
        m_data(this->data()),
        m_capacity_bytes(byte_count(rows, cols)) {}

  /** Construct from a dense expression, evaluating it into \a resource-backed storage. */
  template <typename OtherDerived>
  PmrMatrix(memory_resource* resource, const DenseBase<OtherDerived>& expr)
      : Base(allocate_raw(resource, expr.rows(), expr.cols()), expr.rows(), expr.cols()),
        m_resource(resource),
        m_data(this->data()),
        m_capacity_bytes(byte_count(expr.rows(), expr.cols())) {
    Base::operator=(expr.derived());
  }

  /** Copy construction: allocates fresh storage from \a other's resource and copies the data. */
  PmrMatrix(const PmrMatrix& other)
      : Base(allocate_raw(other.m_resource, other.rows(), other.cols()), other.rows(), other.cols()),
        m_resource(other.m_resource),
        m_data(this->data()),
        m_capacity_bytes(byte_count(other.rows(), other.cols())) {
    Base::operator=(other);
  }

  /** Move construction: steals storage; the moved-from object is left empty. */
  PmrMatrix(PmrMatrix&& other) noexcept
      : Base(other.m_data, other.rows(), other.cols()),
        m_resource(other.m_resource),
        m_data(other.m_data),
        m_capacity_bytes(other.m_capacity_bytes) {
    other.m_data = nullptr;
    other.m_capacity_bytes = 0;
  }

  ~PmrMatrix() { release_storage(); }

  /** Assignment from another PmrMatrix. Keeps \c *this's resource (PMR semantics):
   * the data is copied; the allocators do not propagate. This is the explicit
   * resolution for assigning between instances with different resources. */
  PmrMatrix& operator=(const PmrMatrix& other) {
    if (this != &other) assign_from(other);
    return *this;
  }

  PmrMatrix& operator=(PmrMatrix&& other) noexcept {
    if (this != &other) {
      release_storage();
      m_resource = other.m_resource;
      m_data = other.m_data;
      m_capacity_bytes = other.m_capacity_bytes;
      reseat(m_data, other.rows(), other.cols());
      other.m_data = nullptr;
      other.m_capacity_bytes = 0;
    }
    return *this;
  }

  /** Assignment from any dense expression. Reallocates from \c *this's resource if the
   * size changes, then evaluates the expression into the storage. */
  template <typename OtherDerived>
  PmrMatrix& operator=(const DenseBase<OtherDerived>& other) {
    assign_from(other.derived());
    return *this;
  }

  /** Resize the storage (contents are not preserved, matching Eigen's resize()). */
  void resize(Index rows, Index cols) {
    if (rows != this->rows() || cols != this->cols()) reallocate(rows, cols);
  }

  /** Returns the backing memory resource. */
  memory_resource* resource() const noexcept { return m_resource; }

 private:
  // Storage alignment requested from the resource — matched to the Map base option.
  static constexpr std::size_t kAllocAlign = internal::pmr_map_options() == Unaligned
                                                 ? internal::pmr_default_alignment
                                                 : static_cast<std::size_t>(internal::pmr_map_options());

  static std::size_t byte_count(Index rows, Index cols) {
    return static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols) * sizeof(Scalar);
  }

  static Scalar* allocate_raw(memory_resource* resource, Index rows, Index cols) {
    eigen_assert(resource != nullptr);
    return static_cast<Scalar*>(resource->allocate(byte_count(rows, cols), kAllocAlign));
  }

  void release_storage() noexcept {
    if (m_data) m_resource->deallocate(m_data, m_capacity_bytes, kAllocAlign);
    m_data = nullptr;
    m_capacity_bytes = 0;
  }

  // Re-point the Map base at new storage. Map is trivially destructible, so
  // placement-new over the base subobject is well defined.
  void reseat(Scalar* data, Index rows, Index cols) { ::new (static_cast<Base*>(this)) Base(data, rows, cols); }

  void reallocate(Index rows, Index cols) {
    Scalar* p = allocate_raw(m_resource, rows, cols);
    release_storage();
    m_data = p;
    m_capacity_bytes = byte_count(rows, cols);
    reseat(p, rows, cols);
  }

  template <typename OtherDerived>
  void assign_from(const DenseBase<OtherDerived>& other) {
    if (other.rows() != this->rows() || other.cols() != this->cols()) reallocate(other.rows(), other.cols());
    Base::operator=(other.derived());
  }

  memory_resource* m_resource;
  Scalar* m_data;
  std::size_t m_capacity_bytes;
};

}  // namespace Eigen

#endif  // EIGEN_GPU_COMPILE_PHASE

#endif  // EIGEN_PMR_MATRIX_H
