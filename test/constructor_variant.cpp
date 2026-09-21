// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include <Eigen/Core>
#include <variant>

EIGEN_DECLARE_TEST(constructor_variant) {
  const Vector3d source(1, 2, 3);
  const Vector3i integers(1, 2, 3);
  std::variant<Vector3d, Vector3i> variant(source);
  VERIFY_IS_EQUAL(variant.index(), std::size_t(0));
  VERIFY_IS_EQUAL(std::get<Vector3d>(variant), source);
  variant = integers;
  VERIFY_IS_EQUAL(std::get<Vector3i>(variant), integers);
  std::variant<Vector3i, Vector3d> reversed(source);
  VERIFY_IS_EQUAL(std::get<Vector3d>(reversed), source);
  std::variant<VectorXd, VectorXi> dynamic(source + source);
  VERIFY_IS_EQUAL(std::get<VectorXd>(dynamic), 2 * source);

  const Vector3cd complex = source.cast<std::complex<double>>() * std::complex<double>(1, 2);
  std::variant<Vector3d, Vector3cd> complex_variant(complex);
  VERIFY_IS_EQUAL(std::get<Vector3cd>(complex_variant), complex);
  complex_variant = source;
  VERIFY_IS_EQUAL(std::get<Vector3d>(complex_variant), source);
  complex_variant = complex;
  VERIFY_IS_EQUAL(std::get<Vector3cd>(complex_variant), complex);
  std::variant<Vector3cd, Vector3d> complex_reversed(complex);
  VERIFY_IS_EQUAL(std::get<Vector3cd>(complex_reversed), complex);
  std::variant<VectorXd, VectorXcd> complex_expression(complex + complex);
  VERIFY_IS_EQUAL(std::get<VectorXcd>(complex_expression), 2 * complex);
  complex_expression = complex + complex;
  VERIFY_IS_EQUAL(std::get<VectorXcd>(complex_expression), 2 * complex);
}
