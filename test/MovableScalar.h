// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2020 Sebastien Boisvert <seb@boisvert.info>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_MISC_MOVABLE_SCALAR_H
#define EIGEN_MISC_MOVABLE_SCALAR_H

#include <vector>

namespace Eigen {
template <typename Scalar, typename Base = std::vector<Scalar>>
struct MovableScalar : public Base {
  MovableScalar() = default;
  ~MovableScalar() = default;
  MovableScalar(const MovableScalar&) = default;
  MovableScalar(MovableScalar&& other) = default;
  MovableScalar& operator=(const MovableScalar&) = default;
  MovableScalar& operator=(MovableScalar&& other) = default;
  MovableScalar(Scalar scalar) : Base(100, scalar) {}

  operator Scalar() const { return this->size() > 0 ? this->back() : Scalar(); }
};

template <>
struct NumTraits<MovableScalar<float>> : GenericNumTraits<float> {};

namespace internal {
template <typename T>
struct random_impl<MovableScalar<T>> {
  using MoveableT = MovableScalar<T>;
  using Impl = random_impl<T>;
  static EIGEN_DEVICE_FUNC inline MoveableT run(const MoveableT& x, const MoveableT& y) {
    MoveableT result = Impl::run(x, y);
    return result;
  }
  static EIGEN_DEVICE_FUNC inline MoveableT run() {
    MoveableT result = Impl::run();
    return result;
  }
};
}  // namespace internal

}  // namespace Eigen

#endif
