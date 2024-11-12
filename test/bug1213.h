// SPDX-License-Identifier: MPL-2.0

#include <Eigen/Core>

template <typename T, int dim>
bool bug1213_2(const Eigen::Matrix<T, dim, 1>& x);

bool bug1213_1(const Eigen::Vector3f& x);
