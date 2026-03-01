# List of tests that will be build and run during Eigen's smoke testing. If one
# of these tests doesn't exists or cannot be build with the current configuration
# it will just be skipped.
#
# Heavy tests (>1.5 GB RSS under ASAN+UBSAN) are spread throughout the list
# so that with -j4 they are unlikely to be compiled concurrently, avoiding
# OOM on 16 GB CI runners.
set(ei_smoke_test_list
  # --- heavy: array_cwise_cast ~2.1 GB ---
  array_cwise_cast
  adjoint
  alignedvector3
  array_cwise_operations
  array_cwise_operations_int
  array_of_string
  array_replicate
  array_reverse
  # --- heavy: array_cwise_cast_dynamic ~2.1 GB ---
  array_cwise_cast_dynamic
  autodiff
  autodiff_scalar
  bandmatrix
  bdcsvd_assert
  bdcsvd_trivial
  bdcsvd_trivial_2d
  bdcsvd_compare
  # --- heavy: array_for_matrix ~2.6 GB ---
  array_for_matrix
  bdcsvd_float_fixed
  bdcsvd_float_dynamic
  bdcsvd_double_colmajor
  bdcsvd_double_semi
  bdcsvd_double_rowmajor
  bessel_functions
  bfloat16_float
  # --- heavy: mixingtypes_novectorize ~2 GB ---
  mixingtypes_novectorize
  blasutil
  block_basic
  block_extra
  BVH
  cholesky
  cholmod_support
  conservative_resize
  # --- heavy: mixingtypes_novectorize_dynamic ~2 GB ---
  mixingtypes_novectorize_dynamic
  constructor
  corners
  ctorleak
  dense_storage
  determinant
  diagonal
  diagonalmatrices
  # --- heavy: mixingtypes_vectorize ~2 GB ---
  mixingtypes_vectorize
  dynalloc
  eigensolver_complex
  eigensolver_selfadjoint_fixed
  eigensolver_selfadjoint_3x3_4x4
  eigensolver_selfadjoint_dynamic
  EulerAngles
  exceptions
  # --- heavy: mixingtypes_vectorize_dynamic ~2 GB ---
  mixingtypes_vectorize_dynamic
  fastmath
  first_aligned
  geo_alignedbox
  geo_eulerangles
  geo_homogeneous
  geo_hyperplane
  geo_orthomethods
  geo_parametrizedline
  # --- heavy: product_trmm_cplxfloat ~2.5 GB ---
  product_trmm_cplxfloat
  geo_transformations
  half_float
  hessenberg
  householder
  indexed_view
  inplace_decomposition
  integer_types
  # --- heavy: product_trmm_cplxdouble ~2.5 GB ---
  product_trmm_cplxdouble
  inverse
  is_same_dense
  jacobi
  jacobisvd_verify
  jacobisvd_trivial_2x2
  jacobisvd_misc
  jacobisvd_float
  # --- heavy: product_trmm_float ~2.5 GB ---
  product_trmm_float
  jacobisvd_double_fixed
  jacobisvd_double_rowmajor
  jacobisvd_double_dynamic
  jacobisvd_complex
  kronecker_product
  linearstructure
  lu
  # --- heavy: product_trmm_double ~2.5 GB ---
  product_trmm_double
  mapped_matrix
  mapstaticmethods
  mapstride
  matrix_square_root
  meta
  minres
  miscmatrices
  # --- heavy: product_extra ~2+ GB ---
  product_extra
  nestbyvalue
  nesting_ops
  nomalloc
  nullary
  num_dimensions
  NumericalDiff
  numext
  # --- heavy: product_small_gemm ~2+ GB ---
  product_small_gemm
  packetmath_float
  packetmath_integer
  packetmath_complex
  packetmath_special
  permutationmatrices
  polynomialsolver
  prec_inverse_4x4
  # --- heavy: product_symm ~2+ GB ---
  product_symm
  product_selfadjoint
  product_small
  product_syrk
  product_trmv
  product_trsolve
  qr
  qr_colpivoting
  # --- heavy: product_small_lazy_float ~1.7 GB ---
  product_small_lazy_float
  qr_cod
  qr_fullpivoting
  rand
  real_qz
  redux_matrix
  redux_vector
  ref
  # --- heavy: product_small_lazy_cplxfloat ~1.7 GB ---
  product_small_lazy_cplxfloat
  resize
  rvalue_types
  schur_complex
  schur_real
  selfadjoint
  sizeof
  sizeoverflow
  # --- heavy: product_small_lazy_double ~1.7 GB ---
  product_small_lazy_double
  smallvectors
  sparse_basic
  sparse_block
  sparse_extra
  sparse_permutations
  sparse_product
  sparse_ref
  # --- heavy: product_small_lazy_cplxdouble ~1.7 GB ---
  product_small_lazy_cplxdouble
  sparse_solvers
  sparse_vector
  special_functions
  special_numbers
  special_packetmath
  spqr_support
  stable_norm
  # --- heavy: array_cwise_real ~1.6 GB ---
  array_cwise_real
  stddeque
  stddeque_overload
  stdlist
  stdlist_overload
  stdvector
  stdvector_overload
  stl_iterators
  # --- heavy: array_cwise_complex ~1.5 GB ---
  array_cwise_complex
  array_cwise_math
  array_cwise_bitwise
  swap
  symbolic_index
  triangular
  type_alias
  umeyama
  unaryview
  unalignedcount
  vectorization_logic
  vectorwiseop
  visitor_matrix
  visitor_vector)
