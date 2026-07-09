// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) European Organization for Nuclear Research (CERN)
//
// Adapted from ROOT::Math::CholeskyDecomp and CERNLIB cholesky inversion routines.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_CHOLESKY_INVERSE_IMPL_H
#define EIGEN_CHOLESKY_INVERSE_IMPL_H

// #include <iostream>
namespace Eigen {
namespace internal {

/********************************
*** Fixed size implementation ***
********************************/

template<typename MatrixType, typename ResultType, int Size = MatrixType::ColsAtCompileTime>
struct compute_inverse_cholesky {
  EIGEN_DEVICE_FUNC static inline void run(const MatrixType& matrix, ResultType& result) {
    // std::cout << "chol fixed" << "\n";
    using T = typename ResultType::Scalar;
    T a[Size][Size];

    for (int i = 0; i < Size; ++i) {
      a[i][i] = matrix.coeff(i, i);
      for (int j = i + 1; j < Size; ++j) {
        a[j][i] = matrix.coeff(i, j);
      }
    }

    for (int j = 0; j < Size; ++j) {
      a[j][j] = T(1.) / a[j][j];
      int jp1 = j + 1;
      for (int l = jp1; l < Size; ++l) {
        a[j][l] = a[j][j] * a[l][j];
        T s1 = -a[l][jp1];
        for (int i = 0; i < jp1; ++i) {
          s1 += a[l][i] * a[i][jp1];
        }
        a[l][jp1] = -s1;
      }
    }

    if constexpr (Size == 1) {
      result.coeffRef(0, 0) = a[0][0];
      return;
    }

    a[0][1] = -a[0][1];
    a[1][0] = a[0][1] * a[1][1];
    for (int j = 2; j < Size; ++j) {
      int jm1 = j - 1;
      for (int k = 0; k < jm1; ++k) {
        T s31 = a[k][j];
        for (int i = k; i < jm1; ++i) {
          s31 += a[k][i + 1] * a[i + 1][j];
        }
        a[k][j] = -s31;
        a[j][k] = -s31 * a[j][j];
      }
      a[jm1][j] = -a[jm1][j];
      a[j][jm1] = a[jm1][j] * a[j][j];
    }

    int j = 0;
    while (j < Size - 1) {
      T s33 = a[j][j];
      for (int i = j + 1; i < Size; ++i) {
        s33 += a[j][i] * a[i][j];
      }
      result.coeffRef(j, j) = s33;

      ++j;
      for (int k = 0; k < j; ++k) {
        T s32 = 0;
        for (int i = j; i < Size; ++i) {
          s32 += a[k][i] * a[i][j];
        }
        result.coeffRef(k, j) = result.coeffRef(j, k) = s32;
      }
    }
    result.coeffRef(j, j) = a[j][j];
  }
};

/**********************************
*** Dynamic size implementation ***
**********************************/

template<typename MatrixType, typename ResultType>
struct compute_inverse_cholesky<MatrixType, ResultType, Eigen::Dynamic> {
  EIGEN_DEVICE_FUNC static inline void run(const MatrixType& matrix, ResultType& result) {
    // std::cout << "chol dynamic" << "\n";
    result = matrix.llt().solve(ResultType::Identity(matrix.rows(), matrix.cols()));
  }
};

/****************************
*** Size 1 implementation ***
****************************/

template<typename MatrixType, typename ResultType>
struct compute_inverse_cholesky<MatrixType, ResultType, 1> {
  EIGEN_DEVICE_FUNC static inline void run(const MatrixType& matrix, ResultType& result) {
    // std::cout << "chol1" << "\n";
    using F = decltype(matrix(0, 0));
    result(0, 0) = F(1.0) / matrix(0, 0);
  }
};

/****************************
*** Size 2 implementation ***
****************************/

template <typename MatrixType, typename ResultType>
EIGEN_DEVICE_FUNC inline void compute_inverse_cholesky_size2_helper(const MatrixType& matrix,
                                                                    ResultType& result) {
  using F = decltype(matrix.coeff(0, 0));
  auto luc0 = F(1.0) / matrix.coeff(0, 0);
  auto luc1 = matrix.coeff(1, 0) * matrix.coeff(1, 0) * luc0;
  auto luc2 = F(1.0) / (matrix.coeff(1, 1) - luc1);

  auto li21 = luc1 * luc0 * luc2;

  result.coeffRef(0, 0) = li21 + luc0;
  result.coeffRef(1, 0) = -matrix.coeff(1, 0) * luc0 * luc2;
  result.coeffRef(1, 1) = luc2;
}

template <typename MatrixType>
EIGEN_DEVICE_FUNC inline void symmetrize_size2_helper(MatrixType& matrix) {
  matrix.coeffRef(0, 1) = matrix.coeff(1, 0);
}

template<typename MatrixType, typename ResultType>
struct compute_inverse_cholesky<MatrixType, ResultType, 2> {
  EIGEN_DEVICE_FUNC static inline void run(const MatrixType& matrix, ResultType& result) {
    // std::cout << "chol2" << "\n";
    compute_inverse_cholesky_size2_helper(matrix, result);
    symmetrize_size2_helper(result);
  }
};

/****************************
*** Size 3 implementation ***
****************************/

template <typename MatrixType, typename ResultType>
EIGEN_DEVICE_FUNC inline void compute_inverse_cholesky_size3_helper(MatrixType const& matrix,
                                                                    ResultType& result) {
  using F = decltype(matrix.coeff(0, 0));
  auto luc0 = F(1.0) / matrix.coeff(0, 0);
  auto luc1 = matrix.coeff(1, 0);
  auto luc2 = matrix.coeff(1, 1) - luc0 * luc1 * luc1;
  luc2 = F(1.0) / luc2;
  auto luc3 = matrix.coeff(2, 0);
  auto luc4 = (matrix.coeff(2, 1) - luc0 * luc1 * luc3);
  auto luc5 = matrix.coeff(2, 2) - (luc0 * luc3 * luc3 + (luc2 * luc4) * luc4);
  luc5 = F(1.0) / luc5;

  auto li21 = -luc0 * luc1;
  auto li32 = -(luc2 * luc4);
  auto li31 = (luc1 * (luc2 * luc4) - luc3) * luc0;

  result.coeffRef(0, 0) = luc5 * li31 * li31 + li21 * li21 * luc2 + luc0;
  result.coeffRef(1, 0) = luc5 * li31 * li32 + li21 * luc2;
  result.coeffRef(1, 1) = luc5 * li32 * li32 + luc2;
  result.coeffRef(2, 0) = luc5 * li31;
  result.coeffRef(2, 1) = luc5 * li32;
  result.coeffRef(2, 2) = luc5;
}

template <typename MatrixType>
EIGEN_DEVICE_FUNC inline void symmetrize_size3_helper(MatrixType& matrix) {
  symmetrize_size2_helper(matrix);
  matrix.coeffRef(0, 2) = matrix.coeff(2, 0);
  matrix.coeffRef(1, 2) = matrix.coeff(2, 1);
}

template<typename MatrixType, typename ResultType>
struct compute_inverse_cholesky<MatrixType, ResultType, 3> {
  EIGEN_DEVICE_FUNC static inline void run(const MatrixType& matrix, ResultType& result) {
    // std::cout << "chol3" << "\n";
    compute_inverse_cholesky_size3_helper(matrix, result);
    symmetrize_size3_helper(result);
  }
};

/****************************
*** Size 4 implementation ***
****************************/

template <typename MatrixType, typename ResultType>
EIGEN_DEVICE_FUNC inline void compute_inverse_cholesky_size4_helper(MatrixType const& matrix,
                                                                    ResultType& result) {
  using F = decltype(matrix.coeff(0, 0));
  auto luc0 = F(1.0) / matrix.coeff(0, 0);
  auto luc1 = matrix.coeff(1, 0);
  auto luc2 = matrix.coeff(1, 1) - luc0 * luc1 * luc1;
  luc2 = F(1.0) / luc2;
  auto luc3 = matrix.coeff(2, 0);
  auto luc4 = (matrix.coeff(2, 1) - luc0 * luc1 * luc3);
  auto luc5 = matrix.coeff(2, 2) - (luc0 * luc3 * luc3 + luc2 * luc4 * luc4);
  luc5 = F(1.0) / luc5;
  auto luc6 = matrix.coeff(3, 0);
  auto luc7 = (matrix.coeff(3, 1) - luc0 * luc1 * luc6);
  auto luc8 = (matrix.coeff(3, 2) - luc0 * luc3 * luc6 - luc2 * luc4 * luc7);
  auto luc9 = matrix.coeff(3, 3) - (luc0 * luc6 * luc6 + luc2 * luc7 * luc7 + luc8 * (luc8 * luc5));
  luc9 = F(1.0) / luc9;

  auto li21 = -luc1 * luc0;
  auto li32 = -luc2 * luc4;
  auto li31 = (luc1 * (luc2 * luc4) - luc3) * luc0;
  auto li43 = -(luc8 * luc5);
  auto li42 = (luc4 * luc8 * luc5 - luc7) * luc2;
  auto li41 = (-luc1 * (luc2 * luc4) * (luc8 * luc5) + luc1 * (luc2 * luc7) + luc3 * (luc8 * luc5) - luc6) * luc0;

  result.coeffRef(0, 0) = luc9 * li41 * li41 + luc5 * li31 * li31 + luc2 * li21 * li21 + luc0;
  result.coeffRef(1, 0) = luc9 * li41 * li42 + luc5 * li31 * li32 + luc2 * li21;
  result.coeffRef(1, 1) = luc9 * li42 * li42 + luc5 * li32 * li32 + luc2;
  result.coeffRef(2, 0) = luc9 * li41 * li43 + luc5 * li31;
  result.coeffRef(2, 1) = luc9 * li42 * li43 + luc5 * li32;
  result.coeffRef(2, 2) = luc9 * li43 * li43 + luc5;
  result.coeffRef(3, 0) = luc9 * li41;
  result.coeffRef(3, 1) = luc9 * li42;
  result.coeffRef(3, 2) = luc9 * li43;
  result.coeffRef(3, 3) = luc9;
}

template <typename MatrixType>
EIGEN_DEVICE_FUNC inline void symmetrize_size4_helper(MatrixType& matrix) {
  symmetrize_size3_helper(matrix);
  matrix.coeffRef(0, 3) = matrix.coeff(3, 0);
  matrix.coeffRef(1, 3) = matrix.coeff(3, 1);
  matrix.coeffRef(2, 3) = matrix.coeff(3, 2);
}

template<typename MatrixType, typename ResultType>
struct compute_inverse_cholesky<MatrixType, ResultType, 4> {
  EIGEN_DEVICE_FUNC static inline void run(const MatrixType& matrix, ResultType& result) {
    // std::cout << "chol4" << "\n";
    compute_inverse_cholesky_size4_helper(matrix, result);
    symmetrize_size4_helper(result);
  }
};

/****************************
*** Size 5 implementation ***
****************************/

template <typename MatrixType, typename ResultType>
EIGEN_DEVICE_FUNC inline void compute_inverse_cholesky_size5_helper(MatrixType const& matrix,
                                                                    ResultType& result) {
  using F = decltype(matrix.coeff(0, 0));
  auto luc0 = F(1.0) / matrix.coeff(0, 0);
  auto luc1 = matrix.coeff(1, 0);
  auto luc2 = matrix.coeff(1, 1) - luc0 * luc1 * luc1;
  luc2 = F(1.0) / luc2;
  auto luc3 = matrix.coeff(2, 0);
  auto luc4 = (matrix.coeff(2, 1) - luc0 * luc1 * luc3);
  auto luc5 = matrix.coeff(2, 2) - (luc0 * luc3 * luc3 + luc2 * luc4 * luc4);
  luc5 = F(1.0) / luc5;
  auto luc6 = matrix.coeff(3, 0);
  auto luc7 = (matrix.coeff(3, 1) - luc0 * luc1 * luc6);
  auto luc8 = (matrix.coeff(3, 2) - luc0 * luc3 * luc6 - luc2 * luc4 * luc7);
  auto luc9 = matrix.coeff(3, 3) - (luc0 * luc6 * luc6 + luc2 * luc7 * luc7 + luc8 * (luc8 * luc5));
  luc9 = F(1.0) / luc9;
  auto luc10 = matrix.coeff(4, 0);
  auto luc11 = (matrix.coeff(4, 1) - luc0 * luc1 * luc10);
  auto luc12 = (matrix.coeff(4, 2) - luc0 * luc3 * luc10 - luc2 * luc4 * luc11);
  auto luc13 = (matrix.coeff(4, 3) - luc0 * luc6 * luc10 - luc2 * luc7 * luc11 - luc5 * luc8 * luc12);
  auto luc14 = matrix.coeff(4, 4) - (luc0 * luc10 * luc10 + luc2 * luc11 * luc11 + luc5 * luc12 * luc12 + luc9 * luc13 * luc13);
  luc14 = F(1.0) / luc14;

  auto li21 = -luc1 * luc0;
  auto li32 = -luc2 * luc4;
  auto li31 = (luc1 * (luc2 * luc4) - luc3) * luc0;
  auto li43 = -(luc8 * luc5);
  auto li42 = (luc4 * luc8 * luc5 - luc7) * luc2;
  auto li41 = (-luc1 * (luc2 * luc4) * (luc8 * luc5) + luc1 * (luc2 * luc7) + luc3 * (luc8 * luc5) - luc6) * luc0;
  auto li54 = -luc13 * luc9;
  auto li53 = (luc13 * luc8 * luc9 - luc12) * luc5;
  auto li52 = (-luc4 * luc8 * luc13 * luc5 * luc9 + luc4 * luc12 * luc5 + luc7 * luc13 * luc9 - luc11) * luc2;
  auto li51 = (luc1 * luc4 * luc8 * luc13 * luc2 * luc5 * luc9 - luc13 * luc8 * luc3 * luc9 * luc5 -
               luc12 * luc4 * luc1 * luc2 * luc5 - luc13 * luc7 * luc1 * luc9 * luc2 + luc11 * luc1 * luc2 +
               luc12 * luc3 * luc5 + luc13 * luc6 * luc9 - luc10) * luc0;

  result.coeffRef(0, 0) = luc14 * li51 * li51 + luc9 * li41 * li41 + luc5 * li31 * li31 + luc2 * li21 * li21 + luc0;
  result.coeffRef(1, 0) = luc14 * li51 * li52 + luc9 * li41 * li42 + luc5 * li31 * li32 + luc2 * li21;
  result.coeffRef(1, 1) = luc14 * li52 * li52 + luc9 * li42 * li42 + luc5 * li32 * li32 + luc2;
  result.coeffRef(2, 0) = luc14 * li51 * li53 + luc9 * li41 * li43 + luc5 * li31;
  result.coeffRef(2, 1) = luc14 * li52 * li53 + luc9 * li42 * li43 + luc5 * li32;
  result.coeffRef(2, 2) = luc14 * li53 * li53 + luc9 * li43 * li43 + luc5;
  result.coeffRef(3, 0) = luc14 * li51 * li54 + luc9 * li41;
  result.coeffRef(3, 1) = luc14 * li52 * li54 + luc9 * li42;
  result.coeffRef(3, 2) = luc14 * li53 * li54 + luc9 * li43;
  result.coeffRef(3, 3) = luc14 * li54 * li54 + luc9;
  result.coeffRef(4, 0) = luc14 * li51;
  result.coeffRef(4, 1) = luc14 * li52;
  result.coeffRef(4, 2) = luc14 * li53;
  result.coeffRef(4, 3) = luc14 * li54;
  result.coeffRef(4, 4) = luc14;
}

template <typename MatrixType>
EIGEN_DEVICE_FUNC inline void symmetrize_size5_helper(MatrixType& matrix) {
  symmetrize_size4_helper(matrix);
  matrix.coeffRef(0, 4) = matrix.coeff(4, 0);
  matrix.coeffRef(1, 4) = matrix.coeff(4, 1);
  matrix.coeffRef(2, 4) = matrix.coeff(4, 2);
  matrix.coeffRef(3, 4) = matrix.coeff(4, 3);
}

template<typename MatrixType, typename ResultType>
struct compute_inverse_cholesky<MatrixType, ResultType, 5> {
  EIGEN_DEVICE_FUNC static inline void run(const MatrixType& matrix, ResultType& result) {
    // std::cout << "chol5" << "\n";
    compute_inverse_cholesky_size5_helper(matrix, result);
    symmetrize_size5_helper(result);
  }
};

/****************************
*** Size 6 implementation ***
****************************/

template <typename MatrixType, typename ResultType>
EIGEN_DEVICE_FUNC inline void compute_inverse_cholesky_size6_helper(MatrixType const& matrix,
                                                                    ResultType& result) {
  using F = decltype(matrix.coeff(0, 0));
  auto luc0 = F(1.0) / matrix.coeff(0, 0);
  auto luc1 = matrix.coeff(1, 0);
  auto luc2 = matrix.coeff(1, 1) - luc0 * luc1 * luc1;
  luc2 = F(1.0) / luc2;
  auto luc3 = matrix.coeff(2, 0);
  auto luc4 = (matrix.coeff(2, 1) - luc0 * luc1 * luc3);
  auto luc5 = matrix.coeff(2, 2) - (luc0 * luc3 * luc3 + luc2 * luc4 * luc4);
  luc5 = F(1.0) / luc5;
  auto luc6 = matrix.coeff(3, 0);
  auto luc7 = (matrix.coeff(3, 1) - luc0 * luc1 * luc6);
  auto luc8 = (matrix.coeff(3, 2) - luc0 * luc3 * luc6 - luc2 * luc4 * luc7);
  auto luc9 = matrix.coeff(3, 3) - (luc0 * luc6 * luc6 + luc2 * luc7 * luc7 + luc8 * (luc8 * luc5));
  luc9 = F(1.0) / luc9;
  auto luc10 = matrix.coeff(4, 0);
  auto luc11 = (matrix.coeff(4, 1) - luc0 * luc1 * luc10);
  auto luc12 = (matrix.coeff(4, 2) - luc0 * luc3 * luc10 - luc2 * luc4 * luc11);
  auto luc13 = (matrix.coeff(4, 3) - luc0 * luc6 * luc10 - luc2 * luc7 * luc11 - luc5 * luc8 * luc12);
  auto luc14 = matrix.coeff(4, 4) - (luc0 * luc10 * luc10 + luc2 * luc11 * luc11 + luc5 * luc12 * luc12 + luc9 * luc13 * luc13);
  luc14 = F(1.0) / luc14;
  auto luc15 = matrix.coeff(5, 0);
  auto luc16 = (matrix.coeff(5, 1) - luc0 * luc1 * luc15);
  auto luc17 = (matrix.coeff(5, 2) - luc0 * luc3 * luc15 - luc2 * luc4 * luc16);
  auto luc18 = (matrix.coeff(5, 3) - luc0 * luc6 * luc15 - luc2 * luc7 * luc16 - luc5 * luc8 * luc17);
  auto luc19 = (matrix.coeff(5, 4) - luc0 * luc10 * luc15 - luc2 * luc11 * luc16 - luc5 * luc12 * luc17 - luc9 * luc13 * luc18);
  auto luc20 = matrix.coeff(5, 5) - (luc0 * luc15 * luc15 + luc2 * luc16 * luc16 + luc5 * luc17 * luc17 +
                            luc9 * luc18 * luc18 + luc14 * luc19 * luc19);
  luc20 = F(1.0) / luc20;

  auto li21 = -luc1 * luc0;
  auto li32 = -luc2 * luc4;
  auto li31 = (luc1 * (luc2 * luc4) - luc3) * luc0;
  auto li43 = -(luc8 * luc5);
  auto li42 = (luc4 * luc8 * luc5 - luc7) * luc2;
  auto li41 = (-luc1 * (luc2 * luc4) * (luc8 * luc5) + luc1 * (luc2 * luc7) + luc3 * (luc8 * luc5) - luc6) * luc0;
  auto li54 = -luc13 * luc9;
  auto li53 = (luc13 * luc8 * luc9 - luc12) * luc5;
  auto li52 = (-luc4 * luc8 * luc13 * luc5 * luc9 + luc4 * luc12 * luc5 + luc7 * luc13 * luc9 - luc11) * luc2;
  auto li51 = (luc1 * luc4 * luc8 * luc13 * luc2 * luc5 * luc9 - luc13 * luc8 * luc3 * luc9 * luc5 -
               luc12 * luc4 * luc1 * luc2 * luc5 - luc13 * luc7 * luc1 * luc9 * luc2 + luc11 * luc1 * luc2 +
               luc12 * luc3 * luc5 + luc13 * luc6 * luc9 - luc10) *
              luc0;

  auto li65 = -luc19 * luc14;
  auto li64 = (luc19 * luc14 * luc13 - luc18) * luc9;
  auto li63 = (-luc8 * luc13 * (luc19 * luc14) * luc9 + luc8 * luc9 * luc18 + luc12 * (luc19 * luc14) - luc17) * luc5;
  auto li62 = (luc4 * (luc8 * luc9) * luc13 * luc5 * (luc19 * luc14) - luc18 * luc4 * (luc8 * luc9) * luc5 -
               luc19 * luc12 * luc4 * luc14 * luc5 - luc19 * luc13 * luc7 * luc14 * luc9 + luc17 * luc4 * luc5 +
               luc18 * luc7 * luc9 + luc19 * luc11 * luc14 - luc16) * luc2;
  auto li61 = (-luc19 * luc13 * luc8 * luc4 * luc1 * luc2 * luc5 * luc9 * luc14 +
               luc18 * luc8 * luc4 * luc1 * luc2 * luc5 * luc9 + luc19 * luc12 * luc4 * luc1 * luc2 * luc5 * luc14 +
               luc19 * luc13 * luc7 * luc1 * luc2 * luc9 * luc14 + luc19 * luc13 * luc8 * luc3 * luc5 * luc9 * luc14 -
               luc17 * luc4 * luc1 * luc2 * luc5 - luc18 * luc7 * luc1 * luc2 * luc9 - luc19 * luc11 * luc1 * luc2 * luc14 -
               luc18 * luc8 * luc3 * luc5 * luc9 - luc19 * luc12 * luc3 * luc5 * luc14 -
               luc19 * luc13 * luc6 * luc9 * luc14 + luc16 * luc1 * luc2 + luc17 * luc3 * luc5 + luc18 * luc6 * luc9 +
               luc19 * luc10 * luc14 - luc15) * luc0;

  result.coeffRef(0, 0) = luc20 * li61 * li61 + luc14 * li51 * li51 + luc9 * li41 * li41 + luc5 * li31 * li31 + luc2 * li21 * li21 + luc0;
  result.coeffRef(1, 0) = luc20 * li61 * li62 + luc14 * li51 * li52 + luc9 * li41 * li42 + luc5 * li31 * li32 + luc2 * li21;
  result.coeffRef(1, 1) = luc20 * li62 * li62 + luc14 * li52 * li52 + luc9 * li42 * li42 + luc5 * li32 * li32 + luc2;
  result.coeffRef(2, 0) = luc20 * li61 * li63 + luc14 * li51 * li53 + luc9 * li41 * li43 + luc5 * li31;
  result.coeffRef(2, 1) = luc20 * li62 * li63 + luc14 * li52 * li53 + luc9 * li42 * li43 + luc5 * li32;
  result.coeffRef(2, 2) = luc20 * li63 * li63 + luc14 * li53 * li53 + luc9 * li43 * li43 + luc5;
  result.coeffRef(3, 0) = luc20 * li61 * li64 + luc14 * li51 * li54 + luc9 * li41;
  result.coeffRef(3, 1) = luc20 * li62 * li64 + luc14 * li52 * li54 + luc9 * li42;
  result.coeffRef(3, 2) = luc20 * li63 * li64 + luc14 * li53 * li54 + luc9 * li43;
  result.coeffRef(3, 3) = luc20 * li64 * li64 + luc14 * li54 * li54 + luc9;
  result.coeffRef(4, 0) = luc20 * li61 * li65 + luc14 * li51;
  result.coeffRef(4, 1) = luc20 * li62 * li65 + luc14 * li52;
  result.coeffRef(4, 2) = luc20 * li63 * li65 + luc14 * li53;
  result.coeffRef(4, 3) = luc20 * li64 * li65 + luc14 * li54;
  result.coeffRef(4, 4) = luc20 * li65 * li65 + luc14;
  result.coeffRef(5, 0) = luc20 * li61;
  result.coeffRef(5, 1) = luc20 * li62;
  result.coeffRef(5, 2) = luc20 * li63;
  result.coeffRef(5, 3) = luc20 * li64;
  result.coeffRef(5, 4) = luc20 * li65;
  result.coeffRef(5, 5) = luc20;
}

template <typename MatrixType>
EIGEN_DEVICE_FUNC inline void symmetrize_size6_helper(MatrixType& matrix) {
  symmetrize_size5_helper(matrix);
  matrix.coeffRef(0, 5) = matrix.coeff(5, 0);
  matrix.coeffRef(1, 5) = matrix.coeff(5, 1);
  matrix.coeffRef(2, 5) = matrix.coeff(5, 2);
  matrix.coeffRef(3, 5) = matrix.coeff(5, 3);
  matrix.coeffRef(4, 5) = matrix.coeff(5, 4);
}

template<typename MatrixType, typename ResultType>
struct compute_inverse_cholesky<MatrixType, ResultType, 6> {
  EIGEN_DEVICE_FUNC static inline void run(const MatrixType& matrix, ResultType& result) {
    // std::cout << "chol6" << "\n";
    compute_inverse_cholesky_size6_helper(matrix, result);
    symmetrize_size6_helper(result);
  }
};

}  // end namespace internal

/**********************************
*** Public Internal Dispatcher  ***
**********************************/

namespace internal {

template <typename MatrixType, typename ResultType>
EIGEN_DEVICE_FUNC inline void inverse_cholesky(const MatrixType& matrix, ResultType& result) {
  compute_inverse_cholesky<MatrixType, ResultType, MatrixType::ColsAtCompileTime>::run(matrix, result);
}

}  // end namespace internal

} // namespace Eigen

#endif // EIGEN_CHOLESKY_INVERSE_IMPL_H
