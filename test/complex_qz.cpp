// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Alexey Korepanov <kaikaikai@yandex.ru>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_RUNTIME_NO_MALLOC
#include "main.h"

#include <Eigen/Eigenvalues>


template <typename MatrixType>
void complex_qz(const Index dim) {
  /* this test covers the following files:
     ComplexQZ.h
  */
  using std::abs;


	MatrixType A = MatrixType::Random(dim, dim), B = MatrixType::Random(dim, dim);

	// Set each row of B with a probability of 10% to 0
	for (int i = 0; i < dim; i++) {

		if (internal::random<int>(0, 10) == 0)
			B.row(i).setZero();
	}

	ComplexQZ<MatrixType> qz(A, B);

	VERIFY_IS_EQUAL(qz.info(), Success);

	using RealScalar = typename ComplexQZ<MatrixType>::RealScalar;

	//RealScalar max_T(0), max_S(0);

	auto T = qz.matrixT(),
			 S = qz.matrixS();

	bool is_all_zero_T = true, is_all_zero_S = true;

	for (Index j = 0; j < dim; j++) {
		for (Index i = j+1; i < dim; i++){


			if (std::abs(T(i, j)) > 1e-14)
			{
				std::cerr << std::abs(T(i, j)) << std::endl;
				is_all_zero_T = false;
			}

			if (std::abs(S(i, j)) > 1e-14)
			{
				std::cerr << std::abs(S(i, j)) << std::endl;
				is_all_zero_S = false;
			}

		}
	}

	VERIFY_IS_EQUAL(is_all_zero_T, true);
	VERIFY_IS_EQUAL(is_all_zero_S, true);

	//VERIFY_IS_EQUAL(all_zeros, true);
	VERIFY_IS_APPROX(qz.matrixQ() * qz.matrixS() * qz.matrixZ(), A);
	VERIFY_IS_APPROX(qz.matrixQ() * qz.matrixT() * qz.matrixZ(), B);
	VERIFY_IS_APPROX(qz.matrixQ() * qz.matrixQ().adjoint(), MatrixType::Identity(dim, dim));
	VERIFY_IS_APPROX(qz.matrixZ() * qz.matrixZ().adjoint(), MatrixType::Identity(dim, dim));

}

EIGEN_DECLARE_TEST(complex_qz) {

	const Index dim = 80;
	using MatrixType = Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>;

	for (int i = 0; i < g_repeat; i++) {

		CALL_SUBTEST(complex_qz<MatrixType>(dim));

	}

}
