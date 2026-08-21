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

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

/**
 * \internal
 * Computes the inverse of a positive-definite Hermitian/symmetric matrix using
 * Cholesky decomposition on the data in the LOWER triangle.
 */

/********************************
*** Fixed size implementation ***
********************************/

template <typename MatrixType, typename ResultType, int Size = MatrixType::ColsAtCompileTime>
struct compute_inverse_cholesky {
  template <typename Dummy = MatrixType,
            std::enable_if_t<NumTraits<typename Dummy::Scalar>::IsComplex && (Size > 6), int> = 0>
  EIGEN_DEVICE_FUNC static inline void run(const MatrixType& m, ResultType& result) {
    result = m.template selfadjointView<Eigen::Lower>().llt().solve(ResultType::Identity());
  }

  template <typename Dummy = MatrixType,
            std::enable_if_t<!(NumTraits<typename Dummy::Scalar>::IsComplex && (Size > 6)) && (Size < 32), int> = 0>
  EIGEN_DEVICE_FUNC static inline void run(const MatrixType& m, ResultType& result) {
    using T = typename ResultType::Scalar;
    T a[Size][Size];

    for (int i = 0; i < Size; ++i) {
      a[i][i] = m.coeff(i, i);
      for (int j = i + 1; j < Size; ++j) {
        a[j][i] = m.coeff(j, i);
      }
    }

    for (int j = 0; j < Size; ++j) {
      a[j][j] = T(1) / a[j][j];
      const int jp1 = j + 1;
      for (int l = jp1; l < Size; ++l) {
        a[j][l] = a[j][j] * a[l][j];
        T sum = -a[l][jp1];
        for (int i = 0; i < jp1; ++i) {
          sum += a[l][i] * a[i][jp1];
        }
        a[l][jp1] = -sum;
      }
    }

    for (int j = 1; j < Size; ++j) {
      const int jm1 = j - 1;
      for (int k = 0; k < jm1; ++k) {
        T sum = a[k][j];
        for (int i = k; i < jm1; ++i) {
          sum += a[k][i + 1] * a[i + 1][j];
        }
        a[k][j] = -sum;
        a[j][k] = -sum * a[j][j];
      }
      a[jm1][j] = -a[jm1][j];
      a[j][jm1] = a[jm1][j] * a[j][j];
    }

    for (int j = 0; j < Size - 1; ++j) {
      T sum = a[j][j];
      for (int i = j + 1; i < Size; ++i) {
        sum += a[j][i] * a[i][j];
      }
      result.coeffRef(j, j) = sum;

      const int jp1 = j + 1;
      for (int k = 0; k < jp1; ++k) {
        T off_diag_sum = T(0);
        for (int i = jp1; i < Size; ++i) {
          off_diag_sum += a[k][i] * a[i][jp1];
        }
        result.coeffRef(k, jp1) = result.coeffRef(jp1, k) = off_diag_sum;
      }
    }
    result.coeffRef(Size - 1, Size - 1) = a[Size - 1][Size - 1];
  }

    template <typename Dummy = MatrixType,
            std::enable_if_t<!(NumTraits<typename Dummy::Scalar>::IsComplex && (Size > 6)) && (Size >= 32), int> = 0>
  EIGEN_DEVICE_FUNC static inline void run(const MatrixType& m, ResultType& result) {
    using T = typename ResultType::Scalar;

    for (int i = 0; i < Size; ++i) {
      result.coeffRef(i, i) = m.coeff(i, i);
      for (int j = i + 1; j < Size; ++j) {
        result.coeffRef(j, i) = m.coeff(j, i);
      }
    }

    for (int j = 0; j < Size; ++j) {
      result.coeffRef(j, j) = T(1) / result.coeff(j, j);
      const int jp1 = j + 1;
      for (int l = jp1; l < Size; ++l) {
        result.coeffRef(j, l) = result.coeff(j, j) * result.coeff(l, j);
        T sum = -result.coeff(l, jp1);
        for (int i = 0; i < jp1; ++i) {
          sum += result.coeff(l, i) * result.coeff(i, jp1);
        }
        result.coeffRef(l, jp1) = -sum;
      }
    }

    for (int j = 1; j < Size; ++j) {
      const int jm1 = j - 1;
      for (int k = 0; k < jm1; ++k) {
        T sum = result.coeff(k, j);
        for (int i = k; i < jm1; ++i) {
          sum += result.coeff(k, i + 1) * result.coeff(i + 1, j);
        }
        result.coeffRef(k, j) = -sum;
        result.coeffRef(j, k) = -sum * result.coeff(j, j);
      }
      result.coeffRef(jm1, j) = -result.coeff(jm1, j);
      result.coeffRef(j, jm1) = result.coeff(jm1, j) * result.coeff(j, j);
    }

    for (int j = 0; j < Size - 1; ++j) {
      T sum = result.coeff(j, j);
      for (int i = j + 1; i < Size; ++i) {
        sum += result.coeff(j, i) * result.coeff(i, j);
      }

      const int jp1 = j + 1;
      for (int k = 0; k < jp1; ++k) {
        T off_diag_sum = T(0);
        for (int i = jp1; i < Size; ++i) {
          off_diag_sum += result.coeff(k, i) * result.coeff(i, jp1);
        }
        result.coeffRef(k, jp1) = result.coeffRef(jp1, k) = off_diag_sum;
      }
      result.coeffRef(j, j) = sum;
    }
  }
};

/**********************************
*** Dynamic size implementation ***
**********************************/

template <typename MatrixType, typename ResultType>
struct compute_inverse_cholesky<MatrixType, ResultType, Eigen::Dynamic> {
  EIGEN_DEVICE_FUNC static inline void run(const MatrixType& matrix, ResultType& result) {
    result =
        matrix.template selfadjointView<Eigen::Lower>().llt().solve(ResultType::Identity(matrix.rows(), matrix.cols()));
  }
};

/****************************
*** Size 1 implementation ***
****************************/

template <typename MatrixType, typename ResultType>
struct compute_inverse_cholesky<MatrixType, ResultType, 1> {
  EIGEN_DEVICE_FUNC static inline void run(const MatrixType& matrix, ResultType& result) {
    using F = typename ResultType::Scalar;
    result.coeffRef(0, 0) = F(1.0) / matrix.coeff(0, 0);
  }
};

/****************************
*** Size 2 implementation ***
****************************/

template <typename MatrixType, typename ResultType>
EIGEN_DEVICE_FUNC inline void compute_inverse_cholesky_size2_helper(const MatrixType& m, ResultType& result) {
  using F = typename ResultType::Scalar;
  using numext::abs2;

  auto luc0 = F(1.0) / m.coeff(0, 0);
  auto luc1 = abs2(m.coeff(1, 0)) * luc0;
  auto luc2 = F(1.0) / (m.coeff(1, 1) - luc1);

  auto li21 = luc1 * luc0 * luc2;

  result.coeffRef(0, 0) = li21 + luc0;
  result.coeffRef(1, 0) = -m.coeff(1, 0) * luc0 * luc2;
  result.coeffRef(1, 1) = luc2;
}

template <typename MatrixType>
EIGEN_DEVICE_FUNC inline void symmetrize_size2_helper(MatrixType& matrix) {
  using numext::conj;
  matrix.coeffRef(0, 1) = conj(matrix.coeff(1, 0));
}

template <typename MatrixType, typename ResultType>
struct compute_inverse_cholesky<MatrixType, ResultType, 2> {
  EIGEN_DEVICE_FUNC static inline void run(const MatrixType& matrix, ResultType& result) {
    compute_inverse_cholesky_size2_helper(matrix, result);
    symmetrize_size2_helper(result);
  }
};

/****************************
*** Size 3 implementation ***
****************************/

template <typename MatrixType, typename ResultType>
EIGEN_DEVICE_FUNC inline void compute_inverse_cholesky_size3_helper(MatrixType const& m, ResultType& result) {
  using F = typename ResultType::Scalar;
  using numext::abs2;
  using numext::conj;

  auto luc0 = F(1.0) / m.coeff(0, 0);
  auto luc1 = m.coeff(1, 0);
  auto luc2 = F(1.0) / (m.coeff(1, 1) - luc0 * abs2(luc1));
  auto luc3 = m.coeff(2, 0);
  auto luc4 = m.coeff(2, 1) - luc0 * conj(luc1) * luc3;
  auto luc5 = F(1.0) / (m.coeff(2, 2) - (luc0 * abs2(luc3) + luc2 * abs2(luc4)));

  auto li21 = -luc0 * luc1;
  auto li32 = -luc2 * luc4;
  auto li31 = -luc0 * luc3 + li32 * li21;

  result.coeffRef(0, 0) = luc5 * abs2(li31) + abs2(li21) * luc2 + luc0;
  result.coeffRef(1, 0) = luc5 * li31 * conj(li32) + luc2 * li21;
  result.coeffRef(1, 1) = luc5 * abs2(li32) + luc2;
  result.coeffRef(2, 0) = luc5 * li31;
  result.coeffRef(2, 1) = luc5 * li32;
  result.coeffRef(2, 2) = luc5;
}

template <typename MatrixType>
EIGEN_DEVICE_FUNC inline void symmetrize_size3_helper(MatrixType& matrix) {
  using numext::conj;
  symmetrize_size2_helper(matrix);
  matrix.coeffRef(0, 2) = conj(matrix.coeff(2, 0));
  matrix.coeffRef(1, 2) = conj(matrix.coeff(2, 1));
}

template <typename MatrixType, typename ResultType>
struct compute_inverse_cholesky<MatrixType, ResultType, 3> {
  EIGEN_DEVICE_FUNC static inline void run(const MatrixType& matrix, ResultType& result) {
    compute_inverse_cholesky_size3_helper(matrix, result);
    symmetrize_size3_helper(result);
  }
};

/****************************
*** Size 4 implementation ***
****************************/

template <typename MatrixType, typename ResultType>
EIGEN_DEVICE_FUNC inline void compute_inverse_cholesky_size4_helper(MatrixType const& m, ResultType& result) {
  using F = typename ResultType::Scalar;
  using numext::abs2;
  using numext::conj;

  auto luc0 = F(1.0) / m.coeff(0, 0);
  auto luc1 = m.coeff(1, 0);
  auto luc2 = F(1.0) / (m.coeff(1, 1) - luc0 * abs2(luc1));
  auto luc3 = m.coeff(2, 0);
  auto luc4 = m.coeff(2, 1) - luc0 * conj(luc1) * luc3;
  auto luc5 = F(1.0) / (m.coeff(2, 2) - (luc0 * abs2(luc3) + luc2 * abs2(luc4)));
  auto luc6 = m.coeff(3, 0);
  auto luc7 = m.coeff(3, 1) - luc0 * conj(luc1) * luc6;
  auto luc8 = m.coeff(3, 2) - luc0 * conj(luc3) * luc6 - luc2 * conj(luc4) * luc7;
  auto luc9 = F(1.0) / (m.coeff(3, 3) - luc0 * abs2(luc6) - luc2 * abs2(luc7) - luc5 * abs2(luc8));

  auto li21 = -luc0 * luc1;
  auto li32 = -luc2 * luc4;
  auto li31 = -luc0 * luc3 + li32 * li21;
  auto li43 = -luc5 * luc8;
  auto li42 = -luc2 * luc7 + li43 * li32;
  auto li41 = -luc0 * luc6 - luc2 * luc7 * li21 + li31 * li43;

  result.coeffRef(0, 0) = luc9 * abs2(li41) + luc5 * abs2(li31) + luc2 * abs2(li21) + luc0;
  result.coeffRef(1, 0) = luc9 * conj(li42) * li41 + luc5 * conj(li32) * li31 + luc2 * li21;
  result.coeffRef(1, 1) = luc9 * abs2(li42) + luc5 * abs2(li32) + luc2;
  result.coeffRef(2, 0) = luc9 * conj(li43) * li41 + luc5 * li31;
  result.coeffRef(2, 1) = luc9 * conj(li43) * li42 + luc5 * li32;
  result.coeffRef(2, 2) = luc9 * abs2(li43) + luc5;
  result.coeffRef(3, 0) = luc9 * li41;
  result.coeffRef(3, 1) = luc9 * li42;
  result.coeffRef(3, 2) = luc9 * li43;
  result.coeffRef(3, 3) = luc9;
}

template <typename MatrixType>
EIGEN_DEVICE_FUNC inline void symmetrize_size4_helper(MatrixType& matrix) {
  using numext::conj;
  symmetrize_size3_helper(matrix);
  matrix.coeffRef(0, 3) = conj(matrix.coeff(3, 0));
  matrix.coeffRef(1, 3) = conj(matrix.coeff(3, 1));
  matrix.coeffRef(2, 3) = conj(matrix.coeff(3, 2));
}

template <typename MatrixType, typename ResultType>
struct compute_inverse_cholesky<MatrixType, ResultType, 4> {
  EIGEN_DEVICE_FUNC static inline void run(const MatrixType& matrix, ResultType& result) {
    compute_inverse_cholesky_size4_helper(matrix, result);
    symmetrize_size4_helper(result);
  }
};

/****************************
*** Size 5 implementation ***
****************************/

template <typename MatrixType, typename ResultType>
EIGEN_DEVICE_FUNC inline void compute_inverse_cholesky_size5_helper(MatrixType const& m, ResultType& result) {
  using F = typename ResultType::Scalar;
  using numext::abs2;
  using numext::conj;

  auto luc0 = F(1.0) / m.coeff(0, 0);
  auto luc1 = m.coeff(1, 0);
  auto luc2 = F(1.0) / (m.coeff(1, 1) - luc0 * abs2(luc1));
  auto luc3 = m.coeff(2, 0);
  auto luc4 = m.coeff(2, 1) - luc0 * conj(luc1) * luc3;
  auto luc5 = F(1.0) / (m.coeff(2, 2) - (luc0 * abs2(luc3) + luc2 * abs2(luc4)));
  auto luc6 = m.coeff(3, 0);
  auto luc7 = m.coeff(3, 1) - luc0 * conj(luc1) * luc6;
  auto luc8 = m.coeff(3, 2) - luc0 * conj(luc3) * luc6 - luc2 * conj(luc4) * luc7;
  auto luc9 = F(1.0) / (m.coeff(3, 3) - (luc0 * abs2(luc6) + luc2 * abs2(luc7) + luc5 * abs2(luc8)));
  auto luc10 = m.coeff(4, 0);
  auto luc11 = m.coeff(4, 1) - luc0 * conj(luc1) * luc10;
  auto luc12 = m.coeff(4, 2) - luc0 * conj(luc3) * luc10 - luc2 * conj(luc4) * luc11;
  auto luc13 = m.coeff(4, 3) - luc0 * conj(luc6) * luc10 - luc2 * conj(luc7) * luc11 - luc5 * conj(luc8) * luc12;
  auto luc14 =
      F(1.0) / (m.coeff(4, 4) - (luc0 * abs2(luc10) + luc2 * abs2(luc11) + luc5 * abs2(luc12) + luc9 * abs2(luc13)));

  auto li21 = -luc0 * luc1;
  auto li32 = -luc2 * luc4;
  auto li31 = -luc0 * luc3 + li32 * li21;

  auto li43 = -luc5 * luc8;
  auto li42 = -luc2 * luc7 + li43 * li32;
  auto li41 = -luc0 * luc6 - luc2 * luc7 * li21 + li43 * li31;

  auto li54 = -luc9 * luc13;
  auto li53 = -luc5 * luc12 + li54 * li43;
  auto li52 = -luc2 * luc11 - luc5 * luc12 * li32 + li54 * li42;
  auto li51 = -luc0 * luc10 - luc2 * luc11 * li21 - luc5 * luc12 * li31 + li54 * li41;

  result.coeffRef(0, 0) = luc14 * abs2(li51) + luc9 * abs2(li41) + luc5 * abs2(li31) + luc2 * abs2(li21) + luc0;
  result.coeffRef(1, 0) = luc14 * conj(li52) * li51 + luc9 * conj(li42) * li41 + luc5 * conj(li32) * li31 + luc2 * li21;
  result.coeffRef(1, 1) = luc14 * abs2(li52) + luc9 * abs2(li42) + luc5 * abs2(li32) + luc2;
  result.coeffRef(2, 0) = luc14 * conj(li53) * li51 + luc9 * conj(li43) * li41 + luc5 * li31;
  result.coeffRef(2, 1) = luc14 * conj(li53) * li52 + luc9 * conj(li43) * li42 + luc5 * li32;
  result.coeffRef(2, 2) = luc14 * abs2(li53) + luc9 * abs2(li43) + luc5;
  result.coeffRef(3, 0) = luc14 * conj(li54) * li51 + luc9 * li41;
  result.coeffRef(3, 1) = luc14 * conj(li54) * li52 + luc9 * li42;
  result.coeffRef(3, 2) = luc14 * conj(li54) * li53 + luc9 * li43;
  result.coeffRef(3, 3) = luc14 * abs2(li54) + luc9;
  result.coeffRef(4, 0) = luc14 * li51;
  result.coeffRef(4, 1) = luc14 * li52;
  result.coeffRef(4, 2) = luc14 * li53;
  result.coeffRef(4, 3) = luc14 * li54;
  result.coeffRef(4, 4) = luc14;
}

template <typename MatrixType>
EIGEN_DEVICE_FUNC inline void symmetrize_size5_helper(MatrixType& matrix) {
  using numext::conj;
  symmetrize_size4_helper(matrix);
  matrix.coeffRef(0, 4) = conj(matrix.coeff(4, 0));
  matrix.coeffRef(1, 4) = conj(matrix.coeff(4, 1));
  matrix.coeffRef(2, 4) = conj(matrix.coeff(4, 2));
  matrix.coeffRef(3, 4) = conj(matrix.coeff(4, 3));
}

template <typename MatrixType, typename ResultType>
struct compute_inverse_cholesky<MatrixType, ResultType, 5> {
  EIGEN_DEVICE_FUNC static inline void run(const MatrixType& matrix, ResultType& result) {
    compute_inverse_cholesky_size5_helper(matrix, result);
    symmetrize_size5_helper(result);
  }
};

/****************************
*** Size 6 implementation ***
****************************/

template <typename MatrixType, typename ResultType>
EIGEN_DEVICE_FUNC inline void compute_inverse_cholesky_size6_helper(MatrixType const& m, ResultType& result) {
  using F = typename ResultType::Scalar;
  using numext::abs2;
  using numext::conj;

  // D diagonals inverse (luc0, luc2, luc5, luc9, luc14, luc20)
  auto luc0 = F(1.0) / m.coeff(0, 0);
  auto luc1 = m.coeff(1, 0);
  auto luc2 = F(1.0) / (m.coeff(1, 1) - luc0 * abs2(luc1));

  auto luc3 = m.coeff(2, 0);
  auto luc4 = m.coeff(2, 1) - luc0 * conj(luc1) * luc3;
  auto luc5 = F(1.0) / (m.coeff(2, 2) - (luc0 * abs2(luc3) + luc2 * abs2(luc4)));

  auto luc6 = m.coeff(3, 0);
  auto luc7 = m.coeff(3, 1) - luc0 * conj(luc1) * luc6;
  auto luc8 = m.coeff(3, 2) - (luc0 * conj(luc3) * luc6 + luc2 * conj(luc4) * luc7);
  auto luc9 = F(1.0) / (m.coeff(3, 3) - (luc0 * abs2(luc6) + luc2 * abs2(luc7) + luc5 * abs2(luc8)));

  auto luc10 = m.coeff(4, 0);
  auto luc11 = m.coeff(4, 1) - luc0 * conj(luc1) * luc10;
  auto luc12 = m.coeff(4, 2) - luc0 * conj(luc3) * luc10 - luc2 * conj(luc4) * luc11;
  auto luc13 = m.coeff(4, 3) - luc0 * conj(luc6) * luc10 - luc2 * conj(luc7) * luc11 - luc5 * conj(luc8) * luc12;
  auto luc14 =
      F(1.0) / (m.coeff(4, 4) - (luc0 * abs2(luc10) + luc2 * abs2(luc11) + luc5 * abs2(luc12) + luc9 * abs2(luc13)));

  auto luc15 = m.coeff(5, 0);
  auto luc16 = m.coeff(5, 1) - luc0 * conj(luc1) * luc15;
  auto luc17 = m.coeff(5, 2) - luc0 * conj(luc3) * luc15 - luc2 * conj(luc4) * luc16;
  auto luc18 = m.coeff(5, 3) - luc0 * conj(luc6) * luc15 - luc2 * conj(luc7) * luc16 - luc5 * conj(luc8) * luc17;
  auto luc19 = m.coeff(5, 4) - luc0 * conj(luc10) * luc15 - luc2 * conj(luc11) * luc16 - luc5 * conj(luc12) * luc17 -
               luc9 * conj(luc13) * luc18;
  auto luc20 = F(1.0) / (m.coeff(5, 5) - luc0 * abs2(luc15) - luc2 * abs2(luc16) - luc5 * abs2(luc17) -
                         luc9 * abs2(luc18) - luc14 * abs2(luc19));

  auto li21 = -luc0 * luc1;
  auto li32 = -luc2 * luc4;
  auto li31 = -luc0 * luc3 + li21 * li32;

  auto li43 = -luc5 * luc8;
  auto li42 = -luc2 * luc7 + li32 * li43;
  auto li41 = -luc0 * luc6 - luc2 * luc7 * li21 + li31 * li43;

  auto li54 = -luc9 * luc13;
  auto li53 = -luc5 * luc12 + li54 * li43;
  auto li52 = -luc2 * luc11 - luc5 * luc12 * li32 + li42 * li54;
  auto li51 = -luc0 * luc10 - luc2 * luc11 * li21 - luc5 * luc12 * li31 + li41 * li54;

  auto li65 = -luc14 * luc19;
  auto li64 = -luc9 * luc18 + li65 * li54;
  auto li63 = -luc5 * luc17 - luc9 * luc18 * li43 + li65 * li53;
  auto li62 = -luc2 * luc16 - luc5 * luc17 * li32 - luc9 * luc18 * li42 + li52 * li65;
  auto li61 = -luc0 * luc15 - luc2 * luc16 * li21 - luc5 * luc17 * li31 - luc9 * luc18 * li41 + li51 * li65;

  result.coeffRef(0, 0) =
      luc20 * abs2(li61) + luc14 * abs2(li51) + luc9 * abs2(li41) + luc5 * abs2(li31) + luc2 * abs2(li21) + luc0;
  result.coeffRef(1, 0) = luc20 * conj(li62) * li61 + luc14 * conj(li52) * li51 + luc9 * conj(li42) * li41 +
                          luc5 * conj(li32) * li31 + luc2 * li21;
  result.coeffRef(1, 1) = luc20 * abs2(li62) + luc14 * abs2(li52) + luc9 * abs2(li42) + luc5 * abs2(li32) + luc2;
  result.coeffRef(2, 0) =
      luc20 * conj(li63) * li61 + luc14 * conj(li53) * li51 + luc9 * conj(li43) * li41 + luc5 * li31;
  result.coeffRef(2, 1) =
      luc20 * conj(li63) * li62 + luc14 * conj(li53) * li52 + luc9 * conj(li43) * li42 + luc5 * li32;
  result.coeffRef(2, 2) = luc20 * abs2(li63) + luc14 * abs2(li53) + luc9 * abs2(li43) + luc5;
  result.coeffRef(3, 0) = luc20 * conj(li64) * li61 + luc14 * conj(li54) * li51 + luc9 * li41;
  result.coeffRef(3, 1) = luc20 * conj(li64) * li62 + luc14 * conj(li54) * li52 + luc9 * li42;
  result.coeffRef(3, 2) = luc20 * conj(li64) * li63 + luc14 * conj(li54) * li53 + luc9 * li43;
  result.coeffRef(3, 3) = luc20 * abs2(li64) + luc14 * abs2(li54) + luc9;
  result.coeffRef(4, 0) = luc20 * conj(li65) * li61 + luc14 * li51;
  result.coeffRef(4, 1) = luc20 * conj(li65) * li62 + luc14 * li52;
  result.coeffRef(4, 2) = luc20 * conj(li65) * li63 + luc14 * li53;
  result.coeffRef(4, 3) = luc20 * conj(li65) * li64 + luc14 * li54;
  result.coeffRef(4, 4) = luc20 * abs2(li65) + luc14;
  result.coeffRef(5, 0) = luc20 * li61;
  result.coeffRef(5, 1) = luc20 * li62;
  result.coeffRef(5, 2) = luc20 * li63;
  result.coeffRef(5, 3) = luc20 * li64;
  result.coeffRef(5, 4) = luc20 * li65;
  result.coeffRef(5, 5) = luc20;
}

template <typename MatrixType>
EIGEN_DEVICE_FUNC inline void symmetrize_size6_helper(MatrixType& matrix) {
  using numext::conj;
  symmetrize_size5_helper(matrix);
  matrix.coeffRef(0, 5) = conj(matrix.coeff(5, 0));
  matrix.coeffRef(1, 5) = conj(matrix.coeff(5, 1));
  matrix.coeffRef(2, 5) = conj(matrix.coeff(5, 2));
  matrix.coeffRef(3, 5) = conj(matrix.coeff(5, 3));
  matrix.coeffRef(4, 5) = conj(matrix.coeff(5, 4));
}

template <typename MatrixType, typename ResultType>
struct compute_inverse_cholesky<MatrixType, ResultType, 6> {
  EIGEN_DEVICE_FUNC static inline void run(const MatrixType& matrix, ResultType& result) {
    compute_inverse_cholesky_size6_helper(matrix, result);
    symmetrize_size6_helper(result);
  }
};

}  // end namespace internal

/*********************************
*** Public Internal Dispatcher ***
*********************************/

namespace internal {

/****************************
*** Fixed size dispatcher ***
****************************/

namespace detail {
template <typename MatrixType>
EIGEN_DEVICE_FUNC inline typename MatrixType::PlainObject inverse_cholesky_impl(const MatrixType& matrix,
                                                                                std::true_type) {
  typename MatrixType::PlainObject result(matrix.rows(), matrix.cols());
  compute_inverse_cholesky<MatrixType, typename MatrixType::PlainObject, MatrixType::ColsAtCompileTime>::run(matrix,
                                                                                                             result);
  return result;
}

/******************************
*** Dynamic size dispatcher ***
******************************/

template <typename MatrixType, size_t... Is>
EIGEN_DEVICE_FUNC inline bool inverse_cholesky_dynamic_dispatch(const MatrixType& matrix,
                                                                typename MatrixType::PlainObject& result,
                                                                std::index_sequence<Is...>) {
  bool matched = false;
  std::initializer_list<bool>{
      ((!matched && matrix.rows() == (1 + Is))
           ? (compute_inverse_cholesky<MatrixType, typename MatrixType::PlainObject, 1 + Is>::run(matrix, result),
              matched = true)
           : false)...};
  return matched;
}

template <typename MatrixType>
EIGEN_DEVICE_FUNC inline typename MatrixType::PlainObject inverse_cholesky_impl(const MatrixType& matrix,
                                                                                std::false_type) {
  typename MatrixType::PlainObject result(matrix.rows(), matrix.cols());
  if (!inverse_cholesky_dynamic_dispatch(matrix, result, std::make_index_sequence<6>{})) {
    compute_inverse_cholesky<MatrixType, typename MatrixType::PlainObject, Eigen::Dynamic>::run(matrix, result);
  }
  return result;
}

}  // end namespace detail

template <typename MatrixType>
EIGEN_DEVICE_FUNC inline typename MatrixType::PlainObject inverse_cholesky(const MatrixType& matrix) {
  eigen_assert(matrix.rows() == matrix.cols() && "You can't take the inverse of a non-square matrix!");

  return detail::inverse_cholesky_impl(matrix,
                                       std::integral_constant<bool, MatrixType::ColsAtCompileTime != Eigen::Dynamic>{});
}

}  // end namespace internal

}  // namespace Eigen

#endif  // EIGEN_CHOLESKY_INVERSE_IMPL_H
