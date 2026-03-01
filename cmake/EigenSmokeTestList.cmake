# List of tests that will be build and run during Eigen's smoke testing. If one
# of these tests doesn't exists or cannot be build with the current configuration
# it will just be skipped.
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0
#
# This is a curated subset (~60 tests) covering all major Eigen modules with
# representative scalar types (float, double, complex, half, bfloat16, int).
# The full test suite is run in non-MR CI pipelines.
set(ei_smoke_test_list
  # Core: basic operations and storage
  adjoint
  basicstuff
  block_basic
  constructor
  dense_storage
  diagonal
  diagonalmatrices
  dynalloc
  indexed_view
  linearstructure
  mapped_matrix
  nullary
  permutationmatrices
  redux_matrix
  ref
  resize
  selfadjoint
  swap
  triangular
  vectorwiseop

  # Core: array operations (representative subset of splits)
  array_cwise_operations
  array_cwise_real
  array_cwise_cast

  # Core: numeric types and special values
  half_float
  bfloat16_float
  integer_types
  numext
  special_numbers

  # Core: mixed types
  mixingtypes_vectorize

  # Products (representative subset)
  product_large_real
  product_small
  product_small_lazy_float
  product_trmm_float
  product_trsolve

  # Arch / vectorization
  packetmath_float
  packetmath_complex

  # Decompositions: Cholesky
  cholesky

  # Decompositions: LU
  lu
  determinant
  inverse

  # Decompositions: QR
  householder
  qr_colpivoting

  # Decompositions: SVD (one Jacobi, one BDC)
  jacobisvd_float
  bdcsvd_float_dynamic

  # Decompositions: Eigenvalues
  eigensolver_complex
  eigensolver_selfadjoint_dynamic

  # Schur
  schur_real

  # Geometry (representative subset)
  geo_transformations
  geo_homogeneous
  umeyama

  # Sparse (representative subset)
  sparse_basic
  sparse_product

  # Special functions
  special_functions

  # STL compatibility
  stl_iterators

  # Unsupported: Tensor (representative subset)
  tensor_simple
  tensor_contraction
  tensor_reduction
  tensor_morphing
  tensor_fft
  tensor_thread_pool_basic)
