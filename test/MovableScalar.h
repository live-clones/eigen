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
template <typename Scalar>
struct MovableScalar {
  MovableScalar() : m_data(new Scalar) {}
  ~MovableScalar() { delete m_data; }
  MovableScalar(const MovableScalar& other) : m_data(new Scalar) { *m_data = *other.m_data; }
  MovableScalar(MovableScalar&& other) : m_data(other.m_data) { other.m_data = nullptr; }
  MovableScalar& operator=(const MovableScalar& other) {
    *m_data = *other.m_data;
    return *this;
  }
  MovableScalar& operator=(MovableScalar&& other) {
    m_data = other.m_data;
    other.m_data = nullptr;
    return *this;
  }
  MovableScalar(Scalar scalar) : m_data(new Scalar) { *m_data = scalar; }

  operator Scalar() const { return m_data == nullptr ? Scalar() : *m_data; }
  Scalar* m_data = nullptr;
};

template <typename Scalar>
struct NumTraits<MovableScalar<Scalar>> : GenericNumTraits<Scalar> {
  enum { RequireInitialization = 1 };
};

}  // namespace Eigen

#endif
