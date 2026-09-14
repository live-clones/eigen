// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Keep the release-mode optimizer path that originally produced -Warray-bounds.
#define EIGEN_NO_DEBUG
#include "main.h"
#include <Eigen/Core>

using BoundedVector4d = Matrix<double, Dynamic, 1, ColMajor, 4, 1>;
using BoundedMatrix4d = Matrix<double, Dynamic, Dynamic, ColMajor, 4, 4>;

EIGEN_DONT_INLINE MatrixXd bounded_transform(const MatrixXd& vertices, const BoundedMatrix4d& transform) {
  MatrixXd transformed(vertices.rows(), vertices.cols());
  for (Index col = 0; col < vertices.cols(); ++col) {
    BoundedVector4d homogeneous = BoundedVector4d::Ones(vertices.rows() + 1);
    homogeneous.head(vertices.rows()) = vertices.col(col);
    transformed.col(col) = (transform * homogeneous).head(vertices.rows());
  }
  return transformed;
}

struct BoundedFirst {
  int operator()(int first, int) const { return first; }
};

template <int Order>
void mixed_bounded_products() {
  using BoundedVector = Matrix<double, Dynamic, 1, ColMajor, 3, 1>;
  using DynamicMatrix = Matrix<double, Dynamic, Dynamic, Order>;
  for (Index size = 1; size <= 3; ++size) {
    const DynamicMatrix matrix = DynamicMatrix::Random(size, size);
    const BoundedVector vector = BoundedVector::Random(size);
    const VectorXd dynamic = vector;
    const auto left = dynamic.cwiseProduct(vector);
    const auto right = vector.cwiseProduct(dynamic);
    STATIC_CHECK(decltype(left)::MaxSizeAtCompileTime == 3);
    STATIC_CHECK(decltype(right)::MaxSizeAtCompileTime == 3);
    VERIFY_IS_EQUAL(left.sum(), right.sum());
    const BoundedVector result = matrix * vector;
    const auto reversed = (vector.transpose() * matrix.transpose()).eval();
    for (Index row = 0; row < size; ++row) {
      double expected = 0;
      for (Index col = 0; col < size; ++col) expected += matrix(row, col) * vector(col);
      VERIFY(numext::abs(result(row) - expected) <= 8 * NumTraits<double>::epsilon());
      VERIFY(numext::abs(reversed(row) - expected) <= 8 * NumTraits<double>::epsilon());
    }
  }
}

template <int Order>
void bounded_reductions() {
  for (Index size : {1, 4, 15, 16, 31, 32, 33, 64}) {
    Matrix<int, Dynamic, Dynamic, Order, 64, 64> matrix(size, size);
    for (Index row = 0; row < size; ++row)
      for (Index col = 0; col < size; ++col) matrix(row, col) = int(row + col + 1);
    VERIFY_IS_EQUAL(matrix.sum(), size * size * size);
    VERIFY_IS_EQUAL(matrix.redux(BoundedFirst()), 1);
    VERIFY_IS_EQUAL(matrix.topLeftCorner(size, size).sum(), size * size * size);
    VERIFY_IS_EQUAL(matrix.topLeftCorner(size, size).redux(BoundedFirst()), 1);
  }
}

EIGEN_DECLARE_TEST(redux_bounded) {
  for (Index size = 0; size <= 3; ++size) {
    const MatrixXd vertices = MatrixXd::Random(size, 10);
    const BoundedMatrix4d transform = BoundedMatrix4d::Random(size + 1, size + 1);
    const MatrixXd result = bounded_transform(vertices, transform);
    for (Index row = 0; row < size; ++row) {
      for (Index col = 0; col < vertices.cols(); ++col) {
        double expected = transform(row, size);
        for (Index k = 0; k < size; ++k) expected += transform(row, k) * vertices(k, col);
        VERIFY(numext::abs(result(row, col) - expected) <= 32 * NumTraits<double>::epsilon());
      }
    }
  }
  bounded_reductions<ColMajor>();
  bounded_reductions<RowMajor>();
  mixed_bounded_products<ColMajor>();
  mixed_bounded_products<RowMajor>();
}
