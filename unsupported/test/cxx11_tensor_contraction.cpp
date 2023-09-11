// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#include <Eigen/CXX11/Tensor>
#include "AnnoyingScalar.h"

using Eigen::DefaultDevice;
using Eigen::Tensor;

typedef Tensor<float, 1>::DimensionPair DimPair;

template<int DataLayout>
static void test_evals()
{
  Tensor<float, 2, DataLayout> mat1(2, 3);
  Tensor<float, 2, DataLayout> mat2(2, 3);
  Tensor<float, 2, DataLayout> mat3(3, 2);

  mat1.setRandom();
  mat2.setRandom();
  mat3.setRandom();

  Tensor<float, 2, DataLayout> mat4(3,3);
  mat4.setZero();
  Eigen::array<DimPair, 1> dims3 = {{DimPair(0, 0)}};
  typedef TensorEvaluator<decltype(mat1.contract(mat2, dims3)), DefaultDevice> Evaluator;
  Evaluator eval(mat1.contract(mat2, dims3), DefaultDevice());
  eval.evalTo(mat4.data());
  EIGEN_STATIC_ASSERT(Evaluator::NumDims==2ul, YOU_MADE_A_PROGRAMMING_MISTAKE);
  VERIFY_IS_EQUAL(eval.dimensions()[0], 3);
  VERIFY_IS_EQUAL(eval.dimensions()[1], 3);

  VERIFY_IS_APPROX(mat4(0,0), mat1(0,0)*mat2(0,0) + mat1(1,0)*mat2(1,0));
  VERIFY_IS_APPROX(mat4(0,1), mat1(0,0)*mat2(0,1) + mat1(1,0)*mat2(1,1));
  VERIFY_IS_APPROX(mat4(0,2), mat1(0,0)*mat2(0,2) + mat1(1,0)*mat2(1,2));
  VERIFY_IS_APPROX(mat4(1,0), mat1(0,1)*mat2(0,0) + mat1(1,1)*mat2(1,0));
  VERIFY_IS_APPROX(mat4(1,1), mat1(0,1)*mat2(0,1) + mat1(1,1)*mat2(1,1));
  VERIFY_IS_APPROX(mat4(1,2), mat1(0,1)*mat2(0,2) + mat1(1,1)*mat2(1,2));
  VERIFY_IS_APPROX(mat4(2,0), mat1(0,2)*mat2(0,0) + mat1(1,2)*mat2(1,0));
  VERIFY_IS_APPROX(mat4(2,1), mat1(0,2)*mat2(0,1) + mat1(1,2)*mat2(1,1));
  VERIFY_IS_APPROX(mat4(2,2), mat1(0,2)*mat2(0,2) + mat1(1,2)*mat2(1,2));

  Tensor<float, 2, DataLayout> mat5(2,2);
  mat5.setZero();
  Eigen::array<DimPair, 1> dims4 = {{DimPair(1, 1)}};
  typedef TensorEvaluator<decltype(mat1.contract(mat2, dims4)), DefaultDevice> Evaluator2;
  Evaluator2 eval2(mat1.contract(mat2, dims4), DefaultDevice());
  eval2.evalTo(mat5.data());
  EIGEN_STATIC_ASSERT(Evaluator2::NumDims==2ul, YOU_MADE_A_PROGRAMMING_MISTAKE);
  VERIFY_IS_EQUAL(eval2.dimensions()[0], 2);
  VERIFY_IS_EQUAL(eval2.dimensions()[1], 2);

  VERIFY_IS_APPROX(mat5(0,0), mat1(0,0)*mat2(0,0) + mat1(0,1)*mat2(0,1) + mat1(0,2)*mat2(0,2));
  VERIFY_IS_APPROX(mat5(0,1), mat1(0,0)*mat2(1,0) + mat1(0,1)*mat2(1,1) + mat1(0,2)*mat2(1,2));
  VERIFY_IS_APPROX(mat5(1,0), mat1(1,0)*mat2(0,0) + mat1(1,1)*mat2(0,1) + mat1(1,2)*mat2(0,2));
  VERIFY_IS_APPROX(mat5(1,1), mat1(1,0)*mat2(1,0) + mat1(1,1)*mat2(1,1) + mat1(1,2)*mat2(1,2));

  Tensor<float, 2, DataLayout> mat6(2,2);
  mat6.setZero();
  Eigen::array<DimPair, 1> dims6 = {{DimPair(1, 0)}};
  typedef TensorEvaluator<decltype(mat1.contract(mat3, dims6)), DefaultDevice> Evaluator3;
  Evaluator3 eval3(mat1.contract(mat3, dims6), DefaultDevice());
  eval3.evalTo(mat6.data());
  EIGEN_STATIC_ASSERT(Evaluator3::NumDims==2ul, YOU_MADE_A_PROGRAMMING_MISTAKE);
  VERIFY_IS_EQUAL(eval3.dimensions()[0], 2);
  VERIFY_IS_EQUAL(eval3.dimensions()[1], 2);

  VERIFY_IS_APPROX(mat6(0,0), mat1(0,0)*mat3(0,0) + mat1(0,1)*mat3(1,0) + mat1(0,2)*mat3(2,0));
  VERIFY_IS_APPROX(mat6(0,1), mat1(0,0)*mat3(0,1) + mat1(0,1)*mat3(1,1) + mat1(0,2)*mat3(2,1));
  VERIFY_IS_APPROX(mat6(1,0), mat1(1,0)*mat3(0,0) + mat1(1,1)*mat3(1,0) + mat1(1,2)*mat3(2,0));
  VERIFY_IS_APPROX(mat6(1,1), mat1(1,0)*mat3(0,1) + mat1(1,1)*mat3(1,1) + mat1(1,2)*mat3(2,1));
}

template<int DataLayout>
static void test_scalar()
{
  Tensor<float, 1, DataLayout> vec1({6});
  Tensor<float, 1, DataLayout> vec2({6});

  vec1.setRandom();
  vec2.setRandom();

  Eigen::array<DimPair, 1> dims = {{DimPair(0, 0)}};
  Tensor<float, 0, DataLayout> scalar = vec1.contract(vec2, dims);

  float expected = 0.0f;
  for (int i = 0; i < 6; ++i) {
    expected += vec1(i) * vec2(i);
  }
  VERIFY_IS_APPROX(scalar(), expected);
}

template<int DataLayout>
static void test_multidims()
{
  Tensor<float, 3, DataLayout> mat1(2, 2, 2);
  Tensor<float, 4, DataLayout> mat2(2, 2, 2, 2);

  mat1.setRandom();
  mat2.setRandom();

  Tensor<float, 3, DataLayout> mat3(2, 2, 2);
  mat3.setZero();
  Eigen::array<DimPair, 2> dims = {{DimPair(1, 2), DimPair(2, 3)}};
  typedef TensorEvaluator<decltype(mat1.contract(mat2, dims)), DefaultDevice> Evaluator;
  Evaluator eval(mat1.contract(mat2, dims), DefaultDevice());
  eval.evalTo(mat3.data());
  EIGEN_STATIC_ASSERT(Evaluator::NumDims==3ul, YOU_MADE_A_PROGRAMMING_MISTAKE);
  VERIFY_IS_EQUAL(eval.dimensions()[0], 2);
  VERIFY_IS_EQUAL(eval.dimensions()[1], 2);
  VERIFY_IS_EQUAL(eval.dimensions()[2], 2);

  VERIFY_IS_APPROX(mat3(0,0,0), mat1(0,0,0)*mat2(0,0,0,0) + mat1(0,1,0)*mat2(0,0,1,0) +
                                mat1(0,0,1)*mat2(0,0,0,1) + mat1(0,1,1)*mat2(0,0,1,1));
  VERIFY_IS_APPROX(mat3(0,0,1), mat1(0,0,0)*mat2(0,1,0,0) + mat1(0,1,0)*mat2(0,1,1,0) +
                                mat1(0,0,1)*mat2(0,1,0,1) + mat1(0,1,1)*mat2(0,1,1,1));
  VERIFY_IS_APPROX(mat3(0,1,0), mat1(0,0,0)*mat2(1,0,0,0) + mat1(0,1,0)*mat2(1,0,1,0) +
                                mat1(0,0,1)*mat2(1,0,0,1) + mat1(0,1,1)*mat2(1,0,1,1));
  VERIFY_IS_APPROX(mat3(0,1,1), mat1(0,0,0)*mat2(1,1,0,0) + mat1(0,1,0)*mat2(1,1,1,0) +
                                mat1(0,0,1)*mat2(1,1,0,1) + mat1(0,1,1)*mat2(1,1,1,1));
  VERIFY_IS_APPROX(mat3(1,0,0), mat1(1,0,0)*mat2(0,0,0,0) + mat1(1,1,0)*mat2(0,0,1,0) +
                                mat1(1,0,1)*mat2(0,0,0,1) + mat1(1,1,1)*mat2(0,0,1,1));
  VERIFY_IS_APPROX(mat3(1,0,1), mat1(1,0,0)*mat2(0,1,0,0) + mat1(1,1,0)*mat2(0,1,1,0) +
                                mat1(1,0,1)*mat2(0,1,0,1) + mat1(1,1,1)*mat2(0,1,1,1));
  VERIFY_IS_APPROX(mat3(1,1,0), mat1(1,0,0)*mat2(1,0,0,0) + mat1(1,1,0)*mat2(1,0,1,0) +
                                mat1(1,0,1)*mat2(1,0,0,1) + mat1(1,1,1)*mat2(1,0,1,1));
  VERIFY_IS_APPROX(mat3(1,1,1), mat1(1,0,0)*mat2(1,1,0,0) + mat1(1,1,0)*mat2(1,1,1,0) +
                                mat1(1,0,1)*mat2(1,1,0,1) + mat1(1,1,1)*mat2(1,1,1,1));

  Tensor<float, 2, DataLayout> mat4(2, 2);
  Tensor<float, 3, DataLayout> mat5(2, 2, 2);

  mat4.setRandom();
  mat5.setRandom();

  Tensor<float, 1, DataLayout> mat6(2);
  mat6.setZero();
  Eigen::array<DimPair, 2> dims2({{DimPair(0, 1), DimPair(1, 0)}});
  typedef TensorEvaluator<decltype(mat4.contract(mat5, dims2)), DefaultDevice> Evaluator2;
  Evaluator2 eval2(mat4.contract(mat5, dims2), DefaultDevice());
  eval2.evalTo(mat6.data());
  EIGEN_STATIC_ASSERT(Evaluator2::NumDims==1ul, YOU_MADE_A_PROGRAMMING_MISTAKE);
  VERIFY_IS_EQUAL(eval2.dimensions()[0], 2);

  VERIFY_IS_APPROX(mat6(0), mat4(0,0)*mat5(0,0,0) + mat4(1,0)*mat5(0,1,0) +
                   mat4(0,1)*mat5(1,0,0) + mat4(1,1)*mat5(1,1,0));
  VERIFY_IS_APPROX(mat6(1), mat4(0,0)*mat5(0,0,1) + mat4(1,0)*mat5(0,1,1) +
                   mat4(0,1)*mat5(1,0,1) + mat4(1,1)*mat5(1,1,1));
}

template<int DataLayout>
static void test_holes() {
  Tensor<float, 4, DataLayout> t1(2, 5, 7, 3);
  Tensor<float, 5, DataLayout> t2(2, 7, 11, 13, 3);
  t1.setRandom();
  t2.setRandom();

  Eigen::array<DimPair, 2> dims = {{DimPair(0, 0), DimPair(3, 4)}};
  Tensor<float, 5, DataLayout> result = t1.contract(t2, dims);
  VERIFY_IS_EQUAL(result.dimension(0), 5);
  VERIFY_IS_EQUAL(result.dimension(1), 7);
  VERIFY_IS_EQUAL(result.dimension(2), 7);
  VERIFY_IS_EQUAL(result.dimension(3), 11);
  VERIFY_IS_EQUAL(result.dimension(4), 13);

  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      for (int k = 0; k < 5; ++k) {
        for (int l = 0; l < 5; ++l) {
          for (int m = 0; m < 5; ++m) {
            VERIFY_IS_APPROX(result(i, j, k, l, m),
                             t1(0, i, j, 0) * t2(0, k, l, m, 0) +
                             t1(1, i, j, 0) * t2(1, k, l, m, 0) +
                             t1(0, i, j, 1) * t2(0, k, l, m, 1) +
                             t1(1, i, j, 1) * t2(1, k, l, m, 1) +
                             t1(0, i, j, 2) * t2(0, k, l, m, 2) +
                             t1(1, i, j, 2) * t2(1, k, l, m, 2));
          }
        }
      }
    }
  }
}

template<int DataLayout>
static void test_full_redux()
{
  Tensor<float, 2, DataLayout> t1(2, 2);
  Tensor<float, 3, DataLayout> t2(2, 2, 2);
  t1.setRandom();
  t2.setRandom();

  Eigen::array<DimPair, 2> dims = {{DimPair(0, 0), DimPair(1, 1)}};
  Tensor<float, 1, DataLayout> result = t1.contract(t2, dims);
  VERIFY_IS_EQUAL(result.dimension(0), 2);
  VERIFY_IS_APPROX(result(0), t1(0, 0) * t2(0, 0, 0) +  t1(1, 0) * t2(1, 0, 0)
                            + t1(0, 1) * t2(0, 1, 0) +  t1(1, 1) * t2(1, 1, 0));
  VERIFY_IS_APPROX(result(1), t1(0, 0) * t2(0, 0, 1) +  t1(1, 0) * t2(1, 0, 1)
                            + t1(0, 1) * t2(0, 1, 1) +  t1(1, 1) * t2(1, 1, 1));

  dims[0] = DimPair(1, 0);
  dims[1] = DimPair(2, 1);
  result = t2.contract(t1, dims);
  VERIFY_IS_EQUAL(result.dimension(0), 2);
  VERIFY_IS_APPROX(result(0), t1(0, 0) * t2(0, 0, 0) +  t1(1, 0) * t2(0, 1, 0)
                            + t1(0, 1) * t2(0, 0, 1) +  t1(1, 1) * t2(0, 1, 1));
  VERIFY_IS_APPROX(result(1), t1(0, 0) * t2(1, 0, 0) +  t1(1, 0) * t2(1, 1, 0)
                            + t1(0, 1) * t2(1, 0, 1) +  t1(1, 1) * t2(1, 1, 1));
}

template<int DataLayout>
static void test_contraction_of_contraction()
{
  Tensor<float, 2, DataLayout> t1(2, 2);
  Tensor<float, 2, DataLayout> t2(2, 2);
  Tensor<float, 2, DataLayout> t3(2, 2);
  Tensor<float, 2, DataLayout> t4(2, 2);
  t1.setRandom();
  t2.setRandom();
  t3.setRandom();
  t4.setRandom();

  Eigen::array<DimPair, 1> dims = {{DimPair(1, 0)}};
  auto contract1 = t1.contract(t2, dims);
  auto diff = t3 - contract1;
  auto contract2 = t1.contract(t4, dims);
  Tensor<float, 2, DataLayout> result = contract2.contract(diff, dims);

  VERIFY_IS_EQUAL(result.dimension(0), 2);
  VERIFY_IS_EQUAL(result.dimension(1), 2);

  Eigen::Map<Eigen::Matrix<float, Dynamic, Dynamic, DataLayout>>
      m1(t1.data(), 2, 2), m2(t2.data(), 2, 2), m3(t3.data(), 2, 2),
      m4(t4.data(), 2, 2);
  Eigen::Matrix<float, Dynamic, Dynamic, DataLayout>
      expected = (m1 * m4) * (m3 - m1 * m2);

  VERIFY_IS_APPROX(result(0, 0), expected(0, 0));
  VERIFY_IS_APPROX(result(0, 1), expected(0, 1));
  VERIFY_IS_APPROX(result(1, 0), expected(1, 0));
  VERIFY_IS_APPROX(result(1, 1), expected(1, 1));
}

template<int DataLayout>
static void test_expr()
{
  Tensor<float, 2, DataLayout> mat1(2, 3);
  Tensor<float, 2, DataLayout> mat2(3, 2);
  mat1.setRandom();
  mat2.setRandom();

  Tensor<float, 2, DataLayout> mat3(2,2);

  Eigen::array<DimPair, 1> dims = {{DimPair(1, 0)}};
  mat3 = mat1.contract(mat2, dims);

  VERIFY_IS_APPROX(mat3(0,0), mat1(0,0)*mat2(0,0) + mat1(0,1)*mat2(1,0) + mat1(0,2)*mat2(2,0));
  VERIFY_IS_APPROX(mat3(0,1), mat1(0,0)*mat2(0,1) + mat1(0,1)*mat2(1,1) + mat1(0,2)*mat2(2,1));
  VERIFY_IS_APPROX(mat3(1,0), mat1(1,0)*mat2(0,0) + mat1(1,1)*mat2(1,0) + mat1(1,2)*mat2(2,0));
  VERIFY_IS_APPROX(mat3(1,1), mat1(1,0)*mat2(0,1) + mat1(1,1)*mat2(1,1) + mat1(1,2)*mat2(2,1));
}

template<int DataLayout>
static void test_out_of_order_contraction()
{
  Tensor<float, 3, DataLayout> mat1(2, 2, 2);
  Tensor<float, 3, DataLayout> mat2(2, 2, 2);

  mat1.setRandom();
  mat2.setRandom();

  Tensor<float, 2, DataLayout> mat3(2, 2);

  Eigen::array<DimPair, 2> dims = {{DimPair(2, 0), DimPair(0, 2)}};
  mat3 = mat1.contract(mat2, dims);

  VERIFY_IS_APPROX(mat3(0, 0),
                   mat1(0,0,0)*mat2(0,0,0) + mat1(1,0,0)*mat2(0,0,1) +
                   mat1(0,0,1)*mat2(1,0,0) + mat1(1,0,1)*mat2(1,0,1));
  VERIFY_IS_APPROX(mat3(1, 0),
                   mat1(0,1,0)*mat2(0,0,0) + mat1(1,1,0)*mat2(0,0,1) +
                   mat1(0,1,1)*mat2(1,0,0) + mat1(1,1,1)*mat2(1,0,1));
  VERIFY_IS_APPROX(mat3(0, 1),
                   mat1(0,0,0)*mat2(0,1,0) + mat1(1,0,0)*mat2(0,1,1) +
                   mat1(0,0,1)*mat2(1,1,0) + mat1(1,0,1)*mat2(1,1,1));
  VERIFY_IS_APPROX(mat3(1, 1),
                   mat1(0,1,0)*mat2(0,1,0) + mat1(1,1,0)*mat2(0,1,1) +
                   mat1(0,1,1)*mat2(1,1,0) + mat1(1,1,1)*mat2(1,1,1));

  Eigen::array<DimPair, 2> dims2 = {{DimPair(0, 2), DimPair(2, 0)}};
  mat3 = mat1.contract(mat2, dims2);

  VERIFY_IS_APPROX(mat3(0, 0),
                   mat1(0,0,0)*mat2(0,0,0) + mat1(1,0,0)*mat2(0,0,1) +
                   mat1(0,0,1)*mat2(1,0,0) + mat1(1,0,1)*mat2(1,0,1));
  VERIFY_IS_APPROX(mat3(1, 0),
                   mat1(0,1,0)*mat2(0,0,0) + mat1(1,1,0)*mat2(0,0,1) +
                   mat1(0,1,1)*mat2(1,0,0) + mat1(1,1,1)*mat2(1,0,1));
  VERIFY_IS_APPROX(mat3(0, 1),
                   mat1(0,0,0)*mat2(0,1,0) + mat1(1,0,0)*mat2(0,1,1) +
                   mat1(0,0,1)*mat2(1,1,0) + mat1(1,0,1)*mat2(1,1,1));
  VERIFY_IS_APPROX(mat3(1, 1),
                   mat1(0,1,0)*mat2(0,1,0) + mat1(1,1,0)*mat2(0,1,1) +
                   mat1(0,1,1)*mat2(1,1,0) + mat1(1,1,1)*mat2(1,1,1));

}

template<int DataLayout>
static void test_consistency()
{
  // this does something like testing (A*B)^T = (B^T * A^T)

  Tensor<float, 3, DataLayout> mat1(4, 3, 5);
  Tensor<float, 5, DataLayout> mat2(3, 2, 1, 5, 4);
  mat1.setRandom();
  mat2.setRandom();

  Tensor<float, 4, DataLayout> mat3(5, 2, 1, 5);
  Tensor<float, 4, DataLayout> mat4(2, 1, 5, 5);

  // contract on dimensions of size 4 and 3
  Eigen::array<DimPair, 2> dims1 = {{DimPair(0, 4), DimPair(1, 0)}};
  Eigen::array<DimPair, 2> dims2 = {{DimPair(4, 0), DimPair(0, 1)}};

  mat3 = mat1.contract(mat2, dims1);
  mat4 = mat2.contract(mat1, dims2);

  // check that these are equal except for ordering of dimensions
  if (DataLayout == ColMajor) {
    for (size_t i = 0; i < 5; i++) {
      for (size_t j = 0; j < 10; j++) {
        VERIFY_IS_APPROX(mat3.data()[i + 5 * j], mat4.data()[j + 10 * i]);
      }
    }
  } else {
    // Row major
    for (size_t i = 0; i < 5; i++) {
      for (size_t j = 0; j < 10; j++) {
        VERIFY_IS_APPROX(mat3.data()[10 * i + j], mat4.data()[i + 5 * j]);
      }
    }
  }
}

template<int DataLayout>
static void test_large_contraction()
{
  Tensor<float, 4, DataLayout> t_left(30, 50, 8, 31);
  Tensor<float, 5, DataLayout> t_right(8, 31, 7, 20, 10);
  Tensor<float, 5, DataLayout> t_result(30, 50, 7, 20, 10);

  t_left.setRandom();
  t_right.setRandom();

  // Add a little offset so that the results won't be close to zero.
  t_left += t_left.constant(1.0f);
  t_right += t_right.constant(1.0f);

  typedef Map<Eigen::Matrix<float, Dynamic, Dynamic, DataLayout>> MapXf;
  MapXf m_left(t_left.data(), 1500, 248);
  MapXf m_right(t_right.data(), 248, 1400);
  Eigen::Matrix<float, Dynamic, Dynamic, DataLayout> m_result(1500, 1400);

  // this contraction should be equivalent to a single matrix multiplication
  Eigen::array<DimPair, 2> dims = {{DimPair(2, 0), DimPair(3, 1)}};

  // compute results by separate methods
  t_result = t_left.contract(t_right, dims);
  m_result = m_left * m_right;

  for (int i = 0; i < t_result.dimensions().TotalSize(); i++) {
    VERIFY(&t_result.data()[i] != &m_result.data()[i]);
    VERIFY_IS_APPROX(t_result.data()[i], m_result.data()[i]);
  }
}

template<int DataLayout>
static void test_matrix_vector()
{
  Tensor<float, 2, DataLayout> t_left(30, 50);
  Tensor<float, 1, DataLayout> t_right(50);
  Tensor<float, 1, DataLayout> t_result(30);

  t_left.setRandom();
  t_right.setRandom();

  typedef Map<Eigen::Matrix<float, Dynamic, Dynamic, DataLayout>> MapXf;
  MapXf m_left(t_left.data(), 30, 50);
  MapXf m_right(t_right.data(), 50, 1);
  Eigen::Matrix<float, Dynamic, Dynamic, DataLayout> m_result(30, 1);

  // this contraction should be equivalent to a single matrix multiplication
  Eigen::array<DimPair, 1> dims{{DimPair(1, 0)}};

  // compute results by separate methods
  t_result = t_left.contract(t_right, dims);
  m_result = m_left * m_right;

  for (int i = 0; i < t_result.dimensions().TotalSize(); i++) {
    VERIFY(internal::isApprox(t_result(i), m_result(i, 0), 1));
  }
}


template<int DataLayout>
static void test_tensor_vector()
{
  Tensor<float, 3, DataLayout> t_left(7, 13, 17);
  Tensor<float, 2, DataLayout> t_right(1, 7);

  t_left.setRandom();
  t_right.setRandom();

  typedef typename Tensor<float, 1, DataLayout>::DimensionPair DimensionPair;
  Eigen::array<DimensionPair, 1> dim_pair01{{{0, 1}}};
  Tensor<float, 3, DataLayout> t_result = t_left.contract(t_right, dim_pair01);

  typedef Map<Eigen::Matrix<float, Dynamic, Dynamic, DataLayout>> MapXf;
  MapXf m_left(t_left.data(), 7, 13*17);
  MapXf m_right(t_right.data(), 1, 7);
  Eigen::Matrix<float, Dynamic, Dynamic, DataLayout> m_result = m_left.transpose() * m_right.transpose();

  for (int i = 0; i < t_result.dimensions().TotalSize(); i++) {
    VERIFY(internal::isApprox(t_result(i), m_result(i, 0), 1));
  }
}


template<int DataLayout>
static void test_small_blocking_factors()
{
  Tensor<float, 4, DataLayout> t_left(30, 5, 3, 31);
  Tensor<float, 5, DataLayout> t_right(3, 31, 7, 20, 1);
  t_left.setRandom();
  t_right.setRandom();

  // Add a little offset so that the results won't be close to zero.
  t_left += t_left.constant(1.0f);
  t_right += t_right.constant(1.0f);

  // Force the cache sizes, which results in smaller blocking factors.
  Eigen::setCpuCacheSizes(896, 1920, 2944);

  // this contraction should be equivalent to a single matrix multiplication
  Eigen::array<DimPair, 2> dims = {{DimPair(2, 0), DimPair(3, 1)}};
  Tensor<float, 5, DataLayout> t_result;
  t_result = t_left.contract(t_right, dims);

  // compute result using a simple eigen matrix product
  Map<Eigen::Matrix<float, Dynamic, Dynamic, DataLayout>> m_left(t_left.data(), 150, 93);
  Map<Eigen::Matrix<float, Dynamic, Dynamic, DataLayout>> m_right(t_right.data(), 93, 140);
  Eigen::Matrix<float, Dynamic, Dynamic, DataLayout> m_result = m_left * m_right;

  for (int i = 0; i < t_result.dimensions().TotalSize(); i++) {
    VERIFY_IS_APPROX(t_result.data()[i], m_result.data()[i]);
  }
}

template<int DataLayout>
static void test_tensor_product()
{
  Tensor<float, 2, DataLayout> mat1(2, 3);
  Tensor<float, 2, DataLayout> mat2(4, 1);
  mat1.setRandom();
  mat2.setRandom();

  Eigen::array<DimPair, 0> dims;
  Tensor<float, 4, DataLayout> result = mat1.contract(mat2, dims);

  VERIFY_IS_EQUAL(result.dimension(0), 2);
  VERIFY_IS_EQUAL(result.dimension(1), 3);
  VERIFY_IS_EQUAL(result.dimension(2), 4);
  VERIFY_IS_EQUAL(result.dimension(3), 1);
  for (int i = 0; i < result.dimension(0); ++i) {
    for (int j = 0; j < result.dimension(1); ++j) {
      for (int k = 0; k < result.dimension(2); ++k) {
        for (int l = 0; l < result.dimension(3); ++l) {
			VERIFY_IS_APPROX(result(i, j, k, l), mat1(i, j) * mat2(k, l) );
        }
      }
    }
  }
}


template<int DataLayout>
static void test_const_inputs()
{
  Tensor<float, 2, DataLayout> in1(2, 3);
  Tensor<float, 2, DataLayout> in2(3, 2);
  in1.setRandom();
  in2.setRandom();

  TensorMap<Tensor<const float, 2, DataLayout> > mat1(in1.data(), 2, 3);
  TensorMap<Tensor<const float, 2, DataLayout> > mat2(in2.data(), 3, 2);
  Tensor<float, 2, DataLayout> mat3(2,2);

  Eigen::array<DimPair, 1> dims = {{DimPair(1, 0)}};
  mat3 = mat1.contract(mat2, dims);

  VERIFY_IS_APPROX(mat3(0,0), mat1(0,0)*mat2(0,0) + mat1(0,1)*mat2(1,0) + mat1(0,2)*mat2(2,0));
  VERIFY_IS_APPROX(mat3(0,1), mat1(0,0)*mat2(0,1) + mat1(0,1)*mat2(1,1) + mat1(0,2)*mat2(2,1));
  VERIFY_IS_APPROX(mat3(1,0), mat1(1,0)*mat2(0,0) + mat1(1,1)*mat2(1,0) + mat1(1,2)*mat2(2,0));
  VERIFY_IS_APPROX(mat3(1,1), mat1(1,0)*mat2(0,1) + mat1(1,1)*mat2(1,1) + mat1(1,2)*mat2(2,1));
}

// Apply Sqrt to all output elements.
struct SqrtOutputKernel {
  template <typename Index, typename Scalar>
  EIGEN_ALWAYS_INLINE void operator()(
      const internal::blas_data_mapper<Scalar, Index, ColMajor>& output_mapper,
      const TensorContractionParams&, Index, Index, Index num_rows,
      Index num_cols) const {
    for (int i = 0; i < num_rows; ++i) {
      for (int j = 0; j < num_cols; ++j) {
        output_mapper(i, j) = std::sqrt(output_mapper(i, j));
      }
    }
  }
};

template <int DataLayout>
static void test_large_contraction_with_output_kernel() {
  Tensor<float, 4, DataLayout> t_left(30, 50, 8, 31);
  Tensor<float, 5, DataLayout> t_right(8, 31, 7, 20, 10);
  Tensor<float, 5, DataLayout> t_result(30, 50, 7, 20, 10);

  t_left.setRandom();
  t_right.setRandom();
  // Put trash in mat4 to verify contraction clears output memory.
  t_result.setRandom();

  // Add a little offset so that the results won't be close to zero.
  t_left += t_left.constant(1.0f);
  t_right += t_right.constant(1.0f);

  typedef Map<Eigen::Matrix<float, Dynamic, Dynamic, DataLayout>> MapXf;
  MapXf m_left(t_left.data(), 1500, 248);
  MapXf m_right(t_right.data(), 248, 1400);
  Eigen::Matrix<float, Dynamic, Dynamic, DataLayout> m_result(1500, 1400);

  // this contraction should be equivalent to a single matrix multiplication
  Eigen::array<DimPair, 2> dims({{DimPair(2, 0), DimPair(3, 1)}});

  // compute results by separate methods
  t_result = t_left.contract(t_right, dims, SqrtOutputKernel());

  m_result = m_left * m_right;

  for (std::ptrdiff_t i = 0; i < t_result.dimensions().TotalSize(); i++) {
    VERIFY(&t_result.data()[i] != &m_result.data()[i]);
    VERIFY_IS_APPROX(t_result.data()[i], std::sqrt(m_result.data()[i]));
  }
}

template<int DataLayout>
static void test_scalar_initialization()
{

#ifndef EIGEN_TEST_ANNOYING_SCALAR_DONT_THROW
  AnnoyingScalar::dont_throw = true;
#endif

  AnnoyingScalar::instances = 0;

  {

    Tensor<AnnoyingScalar, 2, DataLayout> A(2, 3);
    Tensor<AnnoyingScalar, 2, DataLayout> B(2, 3);
    Tensor<AnnoyingScalar, 2, DataLayout> result(A.dimension(1), B.dimension(1));

    // Tensor<AnnoyingScalar>.setRandom() causes overloaded ambiguous calls
    std::default_random_engine dre(time(0));
    std::uniform_real_distribution<float> distro(0, 1);

    for (Index i = 0; i < A.dimension(0); ++i) {
      for (Index j = 0; j < A.dimension(1); ++j) {
        A(i, j) = distro(dre);
      }
    }
    for (Index i = 0; i < B.dimension(0); ++i) {
      for (Index j = 0; j < B.dimension(1); ++j) {
        B(i, j) = distro(dre);
      }
    }
    result.setZero();

    typedef Tensor<AnnoyingScalar, 1>::DimensionPair Annoying_DimPair;
    Eigen::array<Annoying_DimPair, 1> dims = {{Annoying_DimPair(0, 0)}};
    typedef TensorEvaluator<decltype(A.contract(B, dims)), DefaultDevice> Evaluator;
    Evaluator eval(A.contract(B, dims), DefaultDevice());
    eval.evalTo(result.data());

    VERIFY_IS_APPROX(result(0,0), (A(0,0)*B(0,0) + A(1,0)*B(1,0)));
    VERIFY_IS_APPROX(result(0,1), (A(0,0)*B(0,1) + A(1,0)*B(1,1)));
    VERIFY_IS_APPROX(result(0,2), (A(0,0)*B(0,2) + A(1,0)*B(1,2)));
    VERIFY_IS_APPROX(result(1,0), (A(0,1)*B(0,0) + A(1,1)*B(1,0)));
    VERIFY_IS_APPROX(result(1,1), (A(0,1)*B(0,1) + A(1,1)*B(1,1)));
    VERIFY_IS_APPROX(result(1,2), (A(0,1)*B(0,2) + A(1,1)*B(1,2)));
    VERIFY_IS_APPROX(result(2,0), (A(0,2)*B(0,0) + A(1,2)*B(1,0)));
    VERIFY_IS_APPROX(result(2,1), (A(0,2)*B(0,1) + A(1,2)*B(1,1)));
    VERIFY_IS_APPROX(result(2,2), (A(0,2)*B(0,2) + A(1,2)*B(1,2)));

  }

  VERIFY(AnnoyingScalar::instances == 0 && "memory leak detected in contraction");

}

template<int DataLayout>
static void test_scalar_initialization_multidims()
{

#ifndef EIGEN_TEST_ANNOYING_SCALAR_DONT_THROW
  AnnoyingScalar::dont_throw = true;
#endif

  {
    Tensor<AnnoyingScalar, 3, DataLayout> A(2, 2, 2);
    Tensor<AnnoyingScalar, 4, DataLayout> B(2, 2, 2, 2);
    Tensor<AnnoyingScalar, 3, DataLayout> result(2, 2, 2);

    std::default_random_engine dre(time(0));
    std::uniform_real_distribution<float> distro(0, 1);

    for (Index i = 0; i < A.dimension(0); ++i) {
      for (Index j = 0; j < A.dimension(1); ++j) {
        for (Index k = 0; k < A.dimension(2); ++k) {
          A(i, j, k) = distro(dre);
        }
      }
    }
    for (Index i = 0; i < B.dimension(0); ++i) {
      for (Index j = 0; j < B.dimension(1); ++j) {
        for (Index k = 0; k < B.dimension(2); ++k) {
          for (Index l = 0; l < B.dimension(3); ++l) {
            B(i, j, k, l) = distro(dre);
          }
        }
      }
    }
    result.setZero();

    typedef Tensor<AnnoyingScalar, 1>::DimensionPair Annoying_DimPair;
    Eigen::array<Annoying_DimPair, 2> dims = {{Annoying_DimPair(1, 2), Annoying_DimPair(2, 3)}};
    typedef TensorEvaluator<decltype(A.contract(B, dims)), DefaultDevice> Evaluator;
    Evaluator eval(A.contract(B, dims), DefaultDevice());
    eval.evalTo(result.data());
    EIGEN_STATIC_ASSERT(Evaluator::NumDims==3ul, YOU_MADE_A_PROGRAMMING_MISTAKE);
    VERIFY_IS_EQUAL(eval.dimensions()[0], 2);
    VERIFY_IS_EQUAL(eval.dimensions()[1], 2);
    VERIFY_IS_EQUAL(eval.dimensions()[2], 2);

    VERIFY_IS_APPROX(result(0,0,0), A(0,0,0)*B(0,0,0,0) + A(0,1,0)*B(0,0,1,0) +
                                  A(0,0,1)*B(0,0,0,1) + A(0,1,1)*B(0,0,1,1));
    VERIFY_IS_APPROX(result(0,0,1), A(0,0,0)*B(0,1,0,0) + A(0,1,0)*B(0,1,1,0) +
                                  A(0,0,1)*B(0,1,0,1) + A(0,1,1)*B(0,1,1,1));
    VERIFY_IS_APPROX(result(0,1,0), A(0,0,0)*B(1,0,0,0) + A(0,1,0)*B(1,0,1,0) +
                                  A(0,0,1)*B(1,0,0,1) + A(0,1,1)*B(1,0,1,1));
    VERIFY_IS_APPROX(result(0,1,1), A(0,0,0)*B(1,1,0,0) + A(0,1,0)*B(1,1,1,0) +
                                  A(0,0,1)*B(1,1,0,1) + A(0,1,1)*B(1,1,1,1));
    VERIFY_IS_APPROX(result(1,0,0), A(1,0,0)*B(0,0,0,0) + A(1,1,0)*B(0,0,1,0) +
                                  A(1,0,1)*B(0,0,0,1) + A(1,1,1)*B(0,0,1,1));
    VERIFY_IS_APPROX(result(1,0,1), A(1,0,0)*B(0,1,0,0) + A(1,1,0)*B(0,1,1,0) +
                                  A(1,0,1)*B(0,1,0,1) + A(1,1,1)*B(0,1,1,1));
    VERIFY_IS_APPROX(result(1,1,0), A(1,0,0)*B(1,0,0,0) + A(1,1,0)*B(1,0,1,0) +
                                  A(1,0,1)*B(1,0,0,1) + A(1,1,1)*B(1,0,1,1));
    VERIFY_IS_APPROX(result(1,1,1), A(1,0,0)*B(1,1,0,0) + A(1,1,0)*B(1,1,1,0) +
                                  A(1,0,1)*B(1,1,0,1) + A(1,1,1)*B(1,1,1,1));
  }

  VERIFY(AnnoyingScalar::instances==0 && "memory leak detected in contraction");
}

template<int DataLayout>
static void test_scalar_initialization_in_large_contraction()
{

#ifndef EIGEN_TEST_ANNOYING_SCALAR_DONT_THROW
  AnnoyingScalar::dont_throw = true;
#endif

  AnnoyingScalar::instances = 0;

  {
    Tensor<AnnoyingScalar, 4, DataLayout> A(20, 45, 8, 31);
    Tensor<AnnoyingScalar, 5, DataLayout> B(8, 31, 7, 3, 5);
    Tensor<AnnoyingScalar, 5, DataLayout> result(20, 45, 7, 3, 5);

    result.setZero();

    // Tensor<AnnoyingScalar>.setRandom() causes overloaded ambiguous calls
    std::default_random_engine dre(time(0));
    std::uniform_real_distribution<float> distro(.0f, 10.f);

    for (Index i = 0; i < A.dimension(0); ++i) {
      for (Index j = 0; j < A.dimension(1); ++j) {
        for (Index k = 0; k < A.dimension(2); ++k) {
          for (Index l = 0; l < A.dimension(3); ++l) {
            A(i, j, k, l) = distro(dre);
          }
        }
      }
    }

    for (Index i = 0; i < B.dimension(0); ++i) {
      for (Index j = 0; j < B.dimension(1); ++j) {
        for (Index k = 0; k < B.dimension(2); ++k) {
          for (Index l = 0; l < B.dimension(3); ++l) {
            for (Index m = 0; m < B.dimension(4); ++m) {
              B(i, j, k, l, m) = distro(dre);
            }
          }
        }
      }
    }

    typedef Tensor<AnnoyingScalar, 1>::DimensionPair Annoying_DimPair;
    Eigen::array<DimPair, 2> dims({{Annoying_DimPair(2, 0), Annoying_DimPair(3, 1)}});
    typedef TensorEvaluator<decltype(A.contract(B, dims)), DefaultDevice> Evaluator;
    Evaluator eval(A.contract(B, dims), DefaultDevice());
    eval.evalTo(result.data());

  }

  VERIFY(AnnoyingScalar::instances == 0 && "memory leak detected in contraction on ThreadPoolDevice");

}

EIGEN_DECLARE_TEST(cxx11_tensor_contraction)
{
  CALL_SUBTEST_1(test_evals<ColMajor>());
  CALL_SUBTEST_1(test_evals<RowMajor>());
  CALL_SUBTEST_1(test_scalar<ColMajor>());
  CALL_SUBTEST_1(test_scalar<RowMajor>());
  CALL_SUBTEST_2(test_multidims<ColMajor>());
  CALL_SUBTEST_2(test_multidims<RowMajor>());
  CALL_SUBTEST_2(test_holes<ColMajor>());
  CALL_SUBTEST_2(test_holes<RowMajor>());
  CALL_SUBTEST_3(test_full_redux<ColMajor>());
  CALL_SUBTEST_3(test_full_redux<RowMajor>());
  CALL_SUBTEST_3(test_contraction_of_contraction<ColMajor>());
  CALL_SUBTEST_3(test_contraction_of_contraction<RowMajor>());
  CALL_SUBTEST_4(test_expr<ColMajor>());
  CALL_SUBTEST_4(test_expr<RowMajor>());
  CALL_SUBTEST_4(test_out_of_order_contraction<ColMajor>());
  CALL_SUBTEST_4(test_out_of_order_contraction<RowMajor>());
  CALL_SUBTEST_5(test_consistency<ColMajor>());
  CALL_SUBTEST_5(test_consistency<RowMajor>());
  CALL_SUBTEST_5(test_large_contraction<ColMajor>());
  CALL_SUBTEST_5(test_large_contraction<RowMajor>());
  CALL_SUBTEST_6(test_matrix_vector<ColMajor>());
  CALL_SUBTEST_6(test_matrix_vector<RowMajor>());
  CALL_SUBTEST_6(test_tensor_vector<ColMajor>());
  CALL_SUBTEST_6(test_tensor_vector<RowMajor>());
  CALL_SUBTEST_7(test_small_blocking_factors<ColMajor>());
  CALL_SUBTEST_7(test_small_blocking_factors<RowMajor>());
  CALL_SUBTEST_7(test_tensor_product<ColMajor>());
  CALL_SUBTEST_7(test_tensor_product<RowMajor>());
  CALL_SUBTEST_8(test_const_inputs<ColMajor>());
  CALL_SUBTEST_8(test_const_inputs<RowMajor>());
  CALL_SUBTEST_8(test_large_contraction_with_output_kernel<ColMajor>());
  CALL_SUBTEST_8(test_large_contraction_with_output_kernel<RowMajor>());

  // tests using AnnoyingScalar
  CALL_SUBTEST_9(test_scalar_initialization<ColMajor>());
  CALL_SUBTEST_9(test_scalar_initialization<RowMajor>());
  CALL_SUBTEST_10(test_scalar_initialization_multidims<ColMajor>());
  CALL_SUBTEST_10(test_scalar_initialization_multidims<RowMajor>());
  CALL_SUBTEST_11(test_scalar_initialization_in_large_contraction<ColMajor>());
  CALL_SUBTEST_11(test_scalar_initialization_in_large_contraction<RowMajor>());

  // Force CMake to split this test.
  // EIGEN_SUFFIXES;1;2;3;4;5;6;7;8;9;10;11

}
