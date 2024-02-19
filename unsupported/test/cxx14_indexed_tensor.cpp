// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015-2019 Godeffroy Valet <godeffroy.valet@m4x.org>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#include <Eigen/CXX11/Tensor>

using Eigen::Sizes;
using Eigen::TensorFixedSize;
using namespace Eigen::tensor_indices;

template <int DataLayout>
static void test_indexed_expression() {
  // fixed size
  {
    TensorFixedSize<float, Sizes<2, 3, 4, 1>, DataLayout> mat1;
    mat1.setRandom();

    TensorFixedSize<float, Sizes<3, 1, 4, 2>, DataLayout> result;
    result(i, j, k, l) = mat1.shuffle(std::array<int, 4>{{1, 3, 2, 0}})(i, j, k, l);

    for (int ii = 0; ii < result.dimension(0); ++ii) {
      for (int jj = 0; jj < result.dimension(1); ++jj) {
        for (int kk = 0; kk < result.dimension(2); ++kk) {
          for (int ll = 0; ll < result.dimension(3); ++ll) {
            VERIFY_IS_APPROX(result(ii, jj, kk, ll), mat1(ll, ii, kk, jj));
          }
        }
      }
    }
  }

  // dynamic size
  {
    Tensor<float, 4, DataLayout> mat1(2, 3, 4, 1);
    mat1.setRandom();

    Tensor<float, 4, DataLayout> result(3, 1, 4, 2);
    result(i, j, k, l) = mat1.shuffle(std::array<int, 4>{{1, 3, 2, 0}})(i, j, k, l);

    for (int ii = 0; ii < result.dimension(0); ++ii) {
      for (int jj = 0; jj < result.dimension(1); ++jj) {
        for (int kk = 0; kk < result.dimension(2); ++kk) {
          for (int ll = 0; ll < result.dimension(3); ++ll) {
            VERIFY_IS_APPROX(result(ii, jj, kk, ll), mat1(ll, ii, kk, jj));
          }
        }
      }
    }
  }
}

template <int DataLayout>
static void test_shuffling() {
  // fixed size
  {
    TensorFixedSize<float, Sizes<2, 3, 4, 1>, DataLayout> mat1;
    mat1.setRandom();

    TensorFixedSize<float, Sizes<3, 1, 4, 2>, DataLayout> result;
    result(i, j, k, l) = mat1(l, i, k, j);

    for (int ii = 0; ii < result.dimension(0); ++ii) {
      for (int jj = 0; jj < result.dimension(1); ++jj) {
        for (int kk = 0; kk < result.dimension(2); ++kk) {
          for (int ll = 0; ll < result.dimension(3); ++ll) {
            VERIFY_IS_APPROX(result(ii, jj, kk, ll), mat1(ll, ii, kk, jj));
          }
        }
      }
    }
  }

  // dynamic size
  {
    Tensor<float, 4, DataLayout> mat1(2, 3, 4, 1);
    mat1.setRandom();

    Tensor<float, 4, DataLayout> result(3, 1, 4, 2);
    result(i, j, k, l) = mat1(l, i, k, j);

    for (int ii = 0; ii < result.dimension(0); ++ii) {
      for (int jj = 0; jj < result.dimension(1); ++jj) {
        for (int kk = 0; kk < result.dimension(2); ++kk) {
          for (int ll = 0; ll < result.dimension(3); ++ll) {
            VERIFY_IS_APPROX(result(ii, jj, kk, ll), mat1(ll, ii, kk, jj));
          }
        }
      }
    }
  }
}

template <int DataLayout>
static void test_shuffling_map() {
  // fixed size
  {
    TensorFixedSize<float, Sizes<2, 3, 4, 1>, DataLayout> mat1;
    mat1.setRandom();

    TensorFixedSize<float, Sizes<3, 1, 4, 2>, DataLayout> result;
    TensorMap<decltype(result)> map(result.data(), result.dimensions());
    map(i, j, k, l) = mat1(l, i, k, j);

    for (int ii = 0; ii < result.dimension(0); ++ii) {
      for (int jj = 0; jj < result.dimension(1); ++jj) {
        for (int kk = 0; kk < result.dimension(2); ++kk) {
          for (int ll = 0; ll < result.dimension(3); ++ll) {
            VERIFY_IS_APPROX(result(ii, jj, kk, ll), mat1(ll, ii, kk, jj));
          }
        }
      }
    }
  }

  // dynamic size
  {
    Tensor<float, 4, DataLayout> mat1(2, 3, 4, 1);
    mat1.setRandom();

    Tensor<float, 4, DataLayout> result(3, 1, 4, 2);
    TensorMap<decltype(result)> map(result.data(), result.dimensions());
    map(i, j, k, l) = mat1(l, i, k, j);

    for (int ii = 0; ii < result.dimension(0); ++ii) {
      for (int jj = 0; jj < result.dimension(1); ++jj) {
        for (int kk = 0; kk < result.dimension(2); ++kk) {
          for (int ll = 0; ll < result.dimension(3); ++ll) {
            VERIFY_IS_APPROX(result(ii, jj, kk, ll), mat1(ll, ii, kk, jj));
          }
        }
      }
    }
  }
}

template <int DataLayout>
static void test_slicing() {
  // fixed size
  {
    TensorFixedSize<float, Sizes<2, 3, 4, 1>, DataLayout> mat1;
    mat1.setRandom();

    TensorFixedSize<float, Sizes<3, 4>, DataLayout> result;
    result(i, j) = mat1(1, i, j, 0);

    for (int ii = 0; ii < result.dimension(0); ++ii) {
      for (int jj = 0; jj < result.dimension(1); ++jj) {
        VERIFY_IS_APPROX(result(ii, jj), mat1(1, ii, jj, 0));
      }
    }
  }

  // dynamic size
  {
    Tensor<float, 4, DataLayout> mat1(2, 3, 4, 1);
    mat1.setRandom();

    Tensor<float, 2, DataLayout> result(3, 4);
    result(i, j) = mat1(1, i, j, 0);

    for (int ii = 0; ii < result.dimension(0); ++ii) {
      for (int jj = 0; jj < result.dimension(1); ++jj) {
        VERIFY_IS_APPROX(result(ii, jj), mat1(1, ii, jj, 0));
      }
    }
  }
}

template <int DataLayout>
static void test_slicing2() {
  // fixed size
  {
    TensorFixedSize<float, Sizes<3, 4, 5>, DataLayout> mat1;
    mat1.setRandom();

    TensorFixedSize<float, Sizes<5>, DataLayout> result;
    result(i) = mat1(0, 2, i);

    for (int ii = 0; ii < result.dimension(0); ++ii) {
      VERIFY_IS_APPROX(result(ii), mat1(0, 2, ii));
    }
  }

  // dynamic size
  {
    Tensor<float, 3, DataLayout> mat1(3, 4, 5);
    Tensor<float, 3, DataLayout> mat2(3, 4, 5);
    mat1.setRandom();
    mat2.setRandom();

    Tensor<float, 1, DataLayout> result(5);
    result(i) = mat1(0, 2, i);

    for (int ii = 0; ii < result.dimension(0); ++ii) {
      VERIFY_IS_APPROX(result(ii), mat1(0, 2, ii));
    }
  }
}

template <int DataLayout>
static void test_addition() {
  // fixed size
  {
    TensorFixedSize<float, Sizes<2, 3>, DataLayout> mat1;
    TensorFixedSize<float, Sizes<2, 3>, DataLayout> mat2;
    mat1.setRandom();
    mat2.setRandom();

    TensorFixedSize<float, Sizes<2, 3>, DataLayout> result;
    result(i, j) = mat1(i, j) + mat2(i, j);

    for (int ii = 0; ii < result.dimension(0); ++ii) {
      for (int jj = 0; jj < result.dimension(1); ++jj) {
        VERIFY_IS_APPROX(result(ii, jj), mat1(ii, jj) + mat2(ii, jj));
      }
    }
  }

  {
    TensorFixedSize<float, Sizes<2, 3>, DataLayout> mat1;
    TensorFixedSize<float, Sizes<2, 3>, DataLayout> mat2;
    mat1.setRandom();
    mat2.setRandom();

    TensorFixedSize<float, Sizes<2, 3>, DataLayout> result;
    result(i, j) = mat1(i, j);
    result(i, j) += mat2(i, j);

    for (int ii = 0; ii < result.dimension(0); ++ii) {
      for (int jj = 0; jj < result.dimension(1); ++jj) {
        VERIFY_IS_APPROX(result(ii, jj), mat1(ii, jj) + mat2(ii, jj));
      }
    }
  }

  {
    TensorFixedSize<float, Sizes<2, 2>, DataLayout> mat1;
    TensorFixedSize<float, Sizes<2, 2>, DataLayout> mat2;
    mat1.setRandom();
    mat2.setRandom();

    TensorFixedSize<float, Sizes<2, 2>, DataLayout> result;
    result(i, j) = mat1(i, j) + mat2(j, i);

    for (int ii = 0; ii < result.dimension(0); ++ii) {
      for (int jj = 0; jj < result.dimension(1); ++jj) {
        VERIFY_IS_APPROX(result(ii, jj), mat1(ii, jj) + mat2(jj, ii));
      }
    }
  }

  // dynamic size
  {
    Tensor<float, 2, DataLayout> mat1(2, 3);
    Tensor<float, 2, DataLayout> mat2(2, 3);
    mat1.setRandom();
    mat2.setRandom();

    Tensor<float, 2, DataLayout> result(2, 3);
    result(i, j) = mat1(i, j) + mat2(i, j);

    for (int ii = 0; ii < result.dimension(0); ++ii) {
      for (int jj = 0; jj < result.dimension(1); ++jj) {
        VERIFY_IS_APPROX(result(ii, jj), mat1(ii, jj) + mat2(ii, jj));
      }
    }
  }

  {
    Tensor<float, 2, DataLayout> mat1(2, 3);
    Tensor<float, 2, DataLayout> mat2(2, 3);
    mat1.setRandom();
    mat2.setRandom();

    Tensor<float, 2, DataLayout> result(2, 3);
    result(i, j) = mat1(i, j);
    result(i, j) += mat2(i, j);

    for (int ii = 0; ii < result.dimension(0); ++ii) {
      for (int jj = 0; jj < result.dimension(1); ++jj) {
        VERIFY_IS_APPROX(result(ii, jj), mat1(ii, jj) + mat2(ii, jj));
      }
    }
  }

  {
    Tensor<float, 2, DataLayout> mat1(2, 2);
    Tensor<float, 2, DataLayout> mat2(2, 2);
    mat1.setRandom();
    mat2.setRandom();

    Tensor<float, 2, DataLayout> result(2, 2);
    result(i, j) = mat1(i, j) + mat2(j, i);

    for (int ii = 0; ii < result.dimension(0); ++ii) {
      for (int jj = 0; jj < result.dimension(1); ++jj) {
        VERIFY_IS_APPROX(result(ii, jj), mat1(ii, jj) + mat2(jj, ii));
      }
    }
  }
}

template <int DataLayout>
static void test_substraction() {
  // fixed size
  {
    TensorFixedSize<float, Sizes<2, 3>, DataLayout> mat1;
    TensorFixedSize<float, Sizes<2, 3>, DataLayout> mat2;
    mat1.setRandom();
    mat2.setRandom();

    TensorFixedSize<float, Sizes<2, 3>, DataLayout> result;
    result(i, j) = mat1(i, j) - mat2(i, j);

    for (int ii = 0; ii < result.dimension(0); ++ii) {
      for (int jj = 0; jj < result.dimension(1); ++jj) {
        VERIFY_IS_APPROX(result(ii, jj), mat1(ii, jj) - mat2(ii, jj));
      }
    }
  }

  {
    TensorFixedSize<float, Sizes<2, 3>, DataLayout> mat1;
    TensorFixedSize<float, Sizes<2, 3>, DataLayout> mat2;
    mat1.setRandom();
    mat2.setRandom();

    TensorFixedSize<float, Sizes<2, 3>, DataLayout> result;
    result(i, j) = mat1(i, j);
    result(i, j) -= mat2(i, j);

    for (int ii = 0; ii < result.dimension(0); ++ii) {
      for (int jj = 0; jj < result.dimension(1); ++jj) {
        VERIFY_IS_APPROX(result(ii, jj), mat1(ii, jj) - mat2(ii, jj));
      }
    }
  }

  {
    TensorFixedSize<float, Sizes<2, 2>, DataLayout> mat1;
    TensorFixedSize<float, Sizes<2, 2>, DataLayout> mat2;
    mat1.setRandom();
    mat2.setRandom();

    TensorFixedSize<float, Sizes<2, 2>, DataLayout> result;
    result(i, j) = mat1(i, j) - mat2(j, i);

    for (int ii = 0; ii < result.dimension(0); ++ii) {
      for (int jj = 0; jj < result.dimension(1); ++jj) {
        VERIFY_IS_APPROX(result(ii, jj), mat1(ii, jj) - mat2(jj, ii));
      }
    }
  }

  // dynamic size
  {
    Tensor<float, 2, DataLayout> mat1(2, 3);
    Tensor<float, 2, DataLayout> mat2(2, 3);
    mat1.setRandom();
    mat2.setRandom();

    Tensor<float, 2, DataLayout> result(2, 3);
    result(i, j) = mat1(i, j) - mat2(i, j);

    for (int ii = 0; ii < result.dimension(0); ++ii) {
      for (int jj = 0; jj < result.dimension(1); ++jj) {
        VERIFY_IS_APPROX(result(ii, jj), mat1(ii, jj) - mat2(ii, jj));
      }
    }
  }

  {
    Tensor<float, 2, DataLayout> mat1(2, 3);
    Tensor<float, 2, DataLayout> mat2(2, 3);
    mat1.setRandom();
    mat2.setRandom();

    Tensor<float, 2, DataLayout> result(2, 3);
    result(i, j) = mat1(i, j);
    result(i, j) -= mat2(i, j);

    for (int ii = 0; ii < result.dimension(0); ++ii) {
      for (int jj = 0; jj < result.dimension(1); ++jj) {
        VERIFY_IS_APPROX(result(ii, jj), mat1(ii, jj) - mat2(ii, jj));
      }
    }
  }

  {
    Tensor<float, 2, DataLayout> mat1(2, 2);
    Tensor<float, 2, DataLayout> mat2(2, 2);
    mat1.setRandom();
    mat2.setRandom();

    Tensor<float, 2, DataLayout> result(2, 2);
    result(i, j) = mat1(i, j) - mat2(j, i);

    for (int ii = 0; ii < result.dimension(0); ++ii) {
      for (int jj = 0; jj < result.dimension(1); ++jj) {
        VERIFY_IS_APPROX(result(ii, jj), mat1(ii, jj) - mat2(jj, ii));
      }
    }
  }
}

template <int DataLayout>
static void test_tensor_product() {
  // fixed size
  {
    TensorFixedSize<float, Sizes<3>, DataLayout> mat1;
    TensorFixedSize<float, Sizes<3>, DataLayout> mat2;
    mat1.setRandom();
    mat2.setRandom();

    TensorFixedSize<float, Sizes<3, 3>, DataLayout> result;
    result(i, j) = mat1(i) * mat2(j);

    for (int ii = 0; ii < result.dimension(0); ++ii) {
      for (int jj = 0; jj < result.dimension(1); ++jj) {
        VERIFY_IS_APPROX(result(ii, jj), mat1(ii) * mat2(jj));
      }
    }
  }

  {
    TensorFixedSize<float, Sizes<2, 3>, DataLayout> mat1;
    TensorFixedSize<float, Sizes<4, 1>, DataLayout> mat2;
    mat1.setRandom();
    mat2.setRandom();

    TensorFixedSize<float, Sizes<2, 4, 1, 3>, DataLayout> result;
    result(i, j, k, l) = mat1(i, l) * mat2(j, k);

    for (int ii = 0; ii < result.dimension(0); ++ii) {
      for (int jj = 0; jj < result.dimension(1); ++jj) {
        for (int kk = 0; kk < result.dimension(2); ++kk) {
          for (int ll = 0; ll < result.dimension(3); ++ll) {
            VERIFY_IS_APPROX(result(ii, jj, kk, ll), mat1(ii, ll) * mat2(jj, kk));
          }
        }
      }
    }
  }

  // dynamic size
  {
    Tensor<float, 1, DataLayout> mat1(3);
    Tensor<float, 1, DataLayout> mat2(3);
    mat1.setRandom();
    mat2.setRandom();

    Tensor<float, 2, DataLayout> result(3, 3);
    result(i, j) = mat1(i) * mat2(j);

    for (int ii = 0; ii < result.dimension(0); ++ii) {
      for (int jj = 0; jj < result.dimension(1); ++jj) {
        VERIFY_IS_APPROX(result(ii, jj), mat1(ii) * mat2(jj));
      }
    }
  }

  {
    Tensor<float, 2, DataLayout> mat1(2, 3);
    Tensor<float, 2, DataLayout> mat2(4, 1);
    mat1.setRandom();
    mat2.setRandom();

    Tensor<float, 4, DataLayout> result(2, 4, 1, 3);
    result(i, j, k, l) = mat1(i, l) * mat2(j, k);

    for (int ii = 0; ii < result.dimension(0); ++ii) {
      for (int jj = 0; jj < result.dimension(1); ++jj) {
        for (int kk = 0; kk < result.dimension(2); ++kk) {
          for (int ll = 0; ll < result.dimension(3); ++ll) {
            VERIFY_IS_APPROX(result(ii, jj, kk, ll), mat1(ii, ll) * mat2(jj, kk));
          }
        }
      }
    }
  }
}

template <int DataLayout>
static void test_contraction() {
  // fixed size
  {
    TensorFixedSize<float, Sizes<4, 3, 5>, DataLayout> mat1;
    TensorFixedSize<float, Sizes<3>, DataLayout> mat2;
    mat1.setRandom();
    mat2.setRandom();

    TensorFixedSize<float, Sizes<4, 5>, DataLayout> result;
    result(i, k) = mat1(i, j, k) * mat2(j);

    for (int ii = 0; ii < result.dimension(0); ++ii) {
      for (int kk = 0; kk < result.dimension(1); ++kk) {
        float tmp = 0;
        for (int jj = 0; jj < mat1.dimension(1); ++jj) {
          tmp += mat1(ii, jj, kk) * mat2(jj);
        }
        VERIFY_IS_APPROX(result(ii, kk), tmp);
      }
    }
  }

  {
    TensorFixedSize<float, Sizes<2, 3>, DataLayout> mat1;
    TensorFixedSize<float, Sizes<3, 4>, DataLayout> mat2;
    mat1.setRandom();
    mat2.setRandom();

    TensorFixedSize<float, Sizes<2, 4>, DataLayout> result;
    result(i, k) = mat1(i, j) * mat2(j, k);

    for (int ii = 0; ii < result.dimension(0); ++ii) {
      for (int kk = 0; kk < result.dimension(1); ++kk) {
        float tmp = 0;
        for (int jj = 0; jj < mat1.dimension(1); ++jj) {
          tmp += mat1(ii, jj) * mat2(jj, kk);
        }
        VERIFY_IS_APPROX(result(ii, kk), tmp);
      }
    }
  }

  {
    TensorFixedSize<float, Sizes<4, 3>, DataLayout> mat1;
    TensorFixedSize<float, Sizes<3, 4>, DataLayout> mat2;
    mat1.setRandom();
    mat2.setRandom();

    float result;
    result = mat1(i, j) * mat2(j, i);

    float tmp = 0;
    for (int ii = 0; ii < mat1.dimension(0); ++ii) {
      for (int jj = 0; jj < mat1.dimension(1); ++jj) {
        tmp += mat1(ii, jj) * mat2(jj, ii);
      }
    }
    VERIFY_IS_APPROX(result, tmp);
  }

  // dynamic size
  {
    Tensor<float, 3, DataLayout> mat1(4, 3, 5);
    Tensor<float, 1, DataLayout> mat2(3);
    mat1.setRandom();
    mat2.setRandom();

    Tensor<float, 2, DataLayout> result(4, 5);
    result(i, k) = mat1(i, j, k) * mat2(j);

    for (int ii = 0; ii < result.dimension(0); ++ii) {
      for (int kk = 0; kk < result.dimension(1); ++kk) {
        float tmp = 0;
        for (int jj = 0; jj < mat1.dimension(1); ++jj) {
          tmp += mat1(ii, jj, kk) * mat2(jj);
        }
        VERIFY_IS_APPROX(result(ii, kk), tmp);
      }
    }
  }

  {
    Tensor<float, 2, DataLayout> mat1(2, 3);
    Tensor<float, 2, DataLayout> mat2(3, 4);
    mat1.setRandom();
    mat2.setRandom();

    Tensor<float, 2, DataLayout> result(2, 4);
    result(i, k) = mat1(i, j) * mat2(j, k);

    for (int ii = 0; ii < result.dimension(0); ++ii) {
      for (int kk = 0; kk < result.dimension(1); ++kk) {
        float tmp = 0;
        for (int jj = 0; jj < mat1.dimension(1); ++jj) {
          tmp += mat1(ii, jj) * mat2(jj, kk);
        }
        VERIFY_IS_APPROX(result(ii, kk), tmp);
      }
    }
  }

  {
    Tensor<float, 2, DataLayout> mat1(4, 3);
    Tensor<float, 2, DataLayout> mat2(3, 4);
    mat1.setRandom();
    mat2.setRandom();

    float result;
    result = mat1(i, j) * mat2(j, i);

    float tmp = 0;
    for (int ii = 0; ii < mat1.dimension(0); ++ii) {
      for (int jj = 0; jj < mat1.dimension(1); ++jj) {
        tmp += mat1(ii, jj) * mat2(jj, ii);
      }
    }
    VERIFY_IS_APPROX(result, tmp);
  }
}

template <int DataLayout>
static void test_combo() {
  // fixed size
  {
    TensorFixedSize<float, Sizes<2, 3>, DataLayout> mat1;
    TensorFixedSize<float, Sizes<4, 3>, DataLayout> mat2;
    TensorFixedSize<float, Sizes<4, 2>, DataLayout> mat3;
    TensorFixedSize<float, Sizes<2, 4>, DataLayout> mat4;
    mat1.setRandom();
    mat2.setRandom();
    mat3.setRandom();
    mat4.setRandom();

    TensorFixedSize<float, Sizes<2, 2>, DataLayout> result;
    result(i, l) = (mat1(i, j) * mat2(k, j) + mat4(i, k)) * mat3(k, l);

    for (int ii = 0; ii < result.dimension(0); ++ii) {
      for (int ll = 0; ll < result.dimension(1); ++ll) {
        float coeff = 0;
        for (int kk = 0; kk < mat2.dimension(0); ++kk) {
          float tmp = 0;
          for (int jj = 0; jj < mat1.dimension(1); ++jj) {
            tmp += mat1(ii, jj) * mat2(kk, jj);
          }
          coeff += (tmp + mat4(ii, kk)) * mat3(kk, ll);
        }
        VERIFY_IS_APPROX(result(ii, ll), coeff);
      }
    }
  }

  {
    TensorFixedSize<double, Sizes<2, 2>, DataLayout> mat1;
    TensorFixedSize<double, Sizes<2, 2>, DataLayout> mat2;
    mat1.setRandom();
    mat2.setRandom();

    TensorFixedSize<double, Sizes<2, 2, 2, 2>, DataLayout> result;
    result(i, j, k, l) = mat2(i, j) * mat1(k, l) + mat2(j, l) * mat1(i, k);

    for (int ii = 0; ii < result.dimension(0); ++ii) {
      for (int jj = 0; jj < result.dimension(1); ++jj) {
        for (int kk = 0; kk < result.dimension(2); ++kk) {
          for (int ll = 0; ll < result.dimension(3); ++ll) {
            VERIFY_IS_APPROX(result(ii, jj, kk, ll), mat2(ii, jj) * mat1(kk, ll) + mat2(jj, ll) * mat1(ii, kk));
          }
        }
      }
    }
  }

  // dynamic size
  {
    Tensor<float, 2, DataLayout> mat1(2, 3);
    Tensor<float, 2, DataLayout> mat2(4, 3);
    Tensor<float, 2, DataLayout> mat3(4, 2);
    Tensor<float, 2, DataLayout> mat4(2, 4);
    mat1.setRandom();
    mat2.setRandom();
    mat3.setRandom();
    mat4.setRandom();

    Tensor<float, 2, DataLayout> result(2, 2);
    result(i, l) = (mat1(i, j) * mat2(k, j) + mat4(i, k)) * mat3(k, l);

    for (int ii = 0; ii < result.dimension(0); ++ii) {
      for (int ll = 0; ll < result.dimension(1); ++ll) {
        float coeff = 0;
        for (int kk = 0; kk < mat2.dimension(0); ++kk) {
          float tmp = 0;
          for (int jj = 0; jj < mat1.dimension(1); ++jj) {
            tmp += mat1(ii, jj) * mat2(kk, jj);
          }
          coeff += (tmp + mat4(ii, kk)) * mat3(kk, ll);
        }
        VERIFY_IS_APPROX(result(ii, ll), coeff);
      }
    }
  }

  {
    Tensor<double, 2, DataLayout> mat1(2, 2);
    Tensor<double, 2, DataLayout> mat2(2, 2);
    mat1.setRandom();
    mat2.setRandom();

    Tensor<double, 4, DataLayout> result(2, 2, 2, 2);
    result(i, j, k, l) = mat2(i, j) * mat1(k, l) + mat2(j, l) * mat1(i, k);

    for (int ii = 0; ii < result.dimension(0); ++ii) {
      for (int jj = 0; jj < result.dimension(1); ++jj) {
        for (int kk = 0; kk < result.dimension(2); ++kk) {
          for (int ll = 0; ll < result.dimension(3); ++ll) {
            VERIFY_IS_APPROX(result(ii, jj, kk, ll), mat2(ii, jj) * mat1(kk, ll) + mat2(jj, ll) * mat1(ii, kk));
          }
        }
      }
    }
  }
}

template <int DataLayout>
static void test_multiply_scalar() {
  // fixed size
  {
    TensorFixedSize<float, Sizes<2, 3>, DataLayout> mat1;
    mat1.setRandom();

    TensorFixedSize<float, Sizes<2, 3>, DataLayout> result;

    result(i, j) = 3. * mat1(i, j);
    for (int ii = 0; ii < result.dimension(0); ++ii) {
      for (int jj = 0; jj < result.dimension(1); ++jj) {
        VERIFY_IS_APPROX(result(ii, jj), float(3.) * mat1(ii, jj));
      }
    }

    result(i, j) = mat1(i, j) * mat1(0, 1);
    for (int ii = 0; ii < result.dimension(0); ++ii) {
      for (int jj = 0; jj < result.dimension(1); ++jj) {
        VERIFY_IS_APPROX(result(ii, jj), mat1(ii, jj) * mat1(0, 1));
      }
    }
  }

  // dynamic size
  {
    Tensor<float, 2, DataLayout> mat1(2, 3);
    mat1.setRandom();

    Tensor<float, 2, DataLayout> result(2, 3);

    result(i, j) = 3. * mat1(i, j);
    for (int ii = 0; ii < result.dimension(0); ++ii) {
      for (int jj = 0; jj < result.dimension(1); ++jj) {
        VERIFY_IS_APPROX(result(ii, jj), float(3.) * mat1(ii, jj));
      }
    }

    result(i, j) = mat1(i, j) * mat1(0, 1);
    for (int ii = 0; ii < result.dimension(0); ++ii) {
      for (int jj = 0; jj < result.dimension(1); ++jj) {
        VERIFY_IS_APPROX(result(ii, jj), mat1(ii, jj) * mat1(0, 1));
      }
    }
  }
}

template <int DataLayout>
static void test_assign_scalar() {
  // fixed size
  {
    TensorFixedSize<float, Sizes<2, 3>, DataLayout> mat1;
    mat1.setRandom();

    TensorFixedSize<float, Sizes<2, 3>, DataLayout> result;
    result(i, j) = mat1(i, j);
    result(1, i) = 3.;

    for (int ii = 0; ii < mat1.dimension(0); ++ii) {
      for (int jj = 0; jj < mat1.dimension(1); ++jj) {
        if (ii == 1) {
          VERIFY_IS_APPROX(result(ii, jj), float(3.));
        } else {
          VERIFY_IS_APPROX(result(ii, jj), mat1(ii, jj));
        }
      }
    }

    float value = result(1, 0);
    VERIFY_IS_APPROX(value, float(3.));
  }

  // dynamic size
  {
    Tensor<float, 2, DataLayout> mat1(2, 3);
    mat1.setRandom();

    Tensor<float, 2, DataLayout> result(2, 3);
    result(i, j) = mat1(i, j);
    result(1, i) = 3.;

    for (int ii = 0; ii < mat1.dimension(0); ++ii) {
      for (int jj = 0; jj < mat1.dimension(1); ++jj) {
        if (ii == 1) {
          VERIFY_IS_APPROX(result(ii, jj), float(3.));
        } else {
          VERIFY_IS_APPROX(result(ii, jj), mat1(ii, jj));
        }
      }
    }

    float value = result(1, 0);
    VERIFY_IS_APPROX(value, float(3.));
  }
}

EIGEN_DECLARE_TEST(test_cxx14_indexed_tensor) {
  CALL_SUBTEST_1(test_indexed_expression<ColMajor>());
  CALL_SUBTEST_1(test_indexed_expression<RowMajor>());
  CALL_SUBTEST_2(test_shuffling<ColMajor>());
  CALL_SUBTEST_2(test_shuffling<RowMajor>());
  CALL_SUBTEST_3(test_shuffling_map<ColMajor>());
  CALL_SUBTEST_3(test_shuffling_map<RowMajor>());
  CALL_SUBTEST_4(test_slicing<ColMajor>());
  CALL_SUBTEST_4(test_slicing<RowMajor>());
  CALL_SUBTEST_5(test_slicing2<ColMajor>());
  CALL_SUBTEST_5(test_slicing2<RowMajor>());
  CALL_SUBTEST_6(test_addition<ColMajor>());
  CALL_SUBTEST_6(test_addition<RowMajor>());
  CALL_SUBTEST_7(test_substraction<ColMajor>());
  CALL_SUBTEST_7(test_substraction<RowMajor>());
  CALL_SUBTEST_8(test_tensor_product<ColMajor>());
  CALL_SUBTEST_8(test_tensor_product<RowMajor>());
  CALL_SUBTEST_9(test_contraction<ColMajor>());
  CALL_SUBTEST_9(test_contraction<RowMajor>());
  CALL_SUBTEST_10(test_combo<ColMajor>());
  CALL_SUBTEST_10(test_combo<RowMajor>());
  CALL_SUBTEST_11(test_multiply_scalar<ColMajor>());
  CALL_SUBTEST_11(test_multiply_scalar<RowMajor>());
  CALL_SUBTEST_12(test_assign_scalar<ColMajor>());
  CALL_SUBTEST_12(test_assign_scalar<RowMajor>());
}