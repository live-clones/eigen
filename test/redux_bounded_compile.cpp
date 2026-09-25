// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <Eigen/Core>
#include <functional>

using BoundedVector4d = Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor, 4, 1>;
using BoundedMatrix4d = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 4, 4>;

Eigen::MatrixXd bounded_transform(const Eigen::MatrixXd& vertices, const BoundedMatrix4d& transform) {
  Eigen::MatrixXd transformed(vertices.rows(), vertices.cols());
  for (Eigen::Index col = 0; col < vertices.cols(); ++col) {
    BoundedVector4d homogeneous = BoundedVector4d::Ones(vertices.rows() + 1);
    homogeneous.head(vertices.rows()) = vertices.col(col);
    transformed.col(col) = (transform * homogeneous).head(vertices.rows());
  }
  return transformed;
}

using BoundedVector3d = Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor, 3, 1>;

std::function<Eigen::VectorXd(Eigen::Index)> evaluate_position;

EIGEN_ALWAYS_INLINE double bounded_residual(const BoundedVector3d& target, Eigen::Index size) {
  const Eigen::VectorXd position = evaluate_position(size);
  return (position - target.head(size)).norm();
}

Eigen::VectorXd bounded_residuals(const Eigen::MatrixXd& points, Eigen::Index size) {
  Eigen::VectorXd result(points.cols());
  for (Eigen::Index i = 0; i < points.cols(); ++i) result(i) = bounded_residual(BoundedVector3d(points.col(i)), size);
  return result;
}

int main() {
  evaluate_position = [](Eigen::Index size) -> Eigen::VectorXd { return Eigen::VectorXd::Zero(size); };
  for (Eigen::Index size = 1; size <= 3; ++size) {
    const Eigen::VectorXd result = bounded_residuals(Eigen::MatrixXd::Ones(size, 2), size);
    for (Eigen::Index i = 0; i < result.size(); ++i)
      if (Eigen::numext::abs(result(i) - Eigen::numext::sqrt(double(size))) > 1e-12) return 1;
  }
}
