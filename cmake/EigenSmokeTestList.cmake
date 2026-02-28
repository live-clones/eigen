# List of tests that will be build and run during Eigen's smoke testing. If one
# of these tests doesn't exists or cannot be build with the current configuration
# it will just be skipped.
#
# This is a minimal list (~59 tests, down from 166) chosen to cover all major
# Eigen modules and all key scalar types (float, double, complex<float>,
# complex<double>, half, bfloat16, integers) while minimizing compile time.
set(ei_smoke_test_list
  # --- Infrastructure / Low-level (4) ---
  meta                        # Type traits, metaprogramming utilities
  numext                      # Scalar math functions (abs, floor, isnan, etc.)
  dense_storage               # DenseStorage allocation, alignment, trivial constructibility
  nomalloc                    # No-heap-alloc check; includes Cholesky/LU/QR/SVD/Eigenvalues headers

  # --- Packet math (SIMD) — all scalar types (4) ---
  packetmath_float            # float, double
  packetmath_integer          # int8_t .. int64_t, uint8_t .. uint64_t
  packetmath_complex          # std::complex<float>, std::complex<double>
  packetmath_special          # Eigen::half, Eigen::bfloat16, bool

  # --- Core Operations (5) ---
  constructor                 # PlainObjectBase construction paths
  block_basic                 # Block operations (subsumes corners)
  indexed_view                # Modern slicing (seq, all, last; subsumes symbolic_index)
  mapped_matrix               # Map<> with alignments/strides (subsumes mapstride, mapstaticmethods)
  ref                         # Ref<> lightweight references

  # --- Array coefficient-wise (2) ---
  array_cwise_real            # float, double, half, bfloat16, complex<float>, complex<double>
  array_cwise_operations_int  # double, complex<float>, int, Index, uint32_t, uint64_t

  # --- Reductions / Nullary (2) ---
  redux_matrix                # float, double, int, int64_t, complex<float>, complex<double>
  nullary                     # LinSpaced, Zero, Ones, Identity, Random

  # --- Mixed scalar type products (1) ---
  mixingtypes_vectorize       # float*complex, double*complex with SIMD

  # --- Products — all scalar types (4) ---
  product_small               # float, bfloat16, half (fixed-size)
  product_small_gemm          # int, double (fixed-size GEMM)
  product_extra               # float, double, complex<float>, complex<double> (dynamic GEMM)
  product_trsolve             # float, double, complex<float>, complex<double> (triangular solves)

  # --- Cholesky (1) ---
  cholesky                    # LLT, LDLT (exercises selfadjoint/triangular products)

  # --- LU (3) ---
  lu_real                     # PartialPivLU, FullPivLU: float, double
  lu_complex                  # PartialPivLU, FullPivLU: complex<float>, complex<double>
  condition_estimator         # rcond() for LU + LLT + LDLT, all types

  # --- QR (1) ---
  qr_colpivoting              # ColPivHouseholderQR: float, double, complex<float>, complex<double>

  # --- SVD (3) ---
  bdcsvd_float_dynamic        # BdcSVD: float (dynamic)
  bdcsvd_double_colmajor      # BdcSVD: double (ColMajor)
  jacobisvd_complex           # JacobiSVD: complex<float>, complex<double>

  # --- Eigenvalues (3) ---
  eigensolver_selfadjoint_dynamic  # SelfAdjointEigenSolver: float, double, complex<double>
  schur_real                  # RealSchur: float, double
  real_qz                     # RealQZ: float, double

  # --- Geometry (3) ---
  geo_transformations         # Transform, Quaternion, AngleAxis: float, double
  geo_orthomethods            # Cross product, unitOrthogonal: float, double, complex<double>
  umeyama                     # Rigid-body alignment (uses Geometry + LU + SVD)

  # --- Sparse (6) ---
  sparse_basic                # Core sparse: float, double (ColMajor + RowMajor)
  sparse_basic_cplx           # Core sparse: complex<double>, long StorageIndex
  sparse_basic_extra          # Core sparse: short StorageIndex, big triplet regressions
  sparse_product              # Sparse products: float, double, complex<double>
  sparse_solvers              # Sparse triangular solvers: double, complex<double>
  sparseqr                    # SparseQR decomposition

  # --- STL Support (2) ---
  stdvector_overload          # Eigen types in std::vector with aligned allocator
  stl_iterators               # STL iterator API for dense/sparse

  # --- Type Support (2) ---
  half_float                  # Eigen::half (float16)
  bfloat16_float              # Eigen::bfloat16

  # --- Special / Safety (3) ---
  stable_norm                 # stableNorm, blueNorm with overflow/underflow
  fastmath                    # -ffast-math compilation mode
  sizeoverflow                # Integer overflow detection in size calculations

  # --- Unsupported Module (3) ---
  autodiff                    # AutoDiff (forward-mode AD)
  special_functions           # igamma, digamma, zeta, betainc (subsumes bessel_functions)
  kronecker_product           # Kronecker product (dense + sparse)

  # --- Tensor (unsupported) (6) ---
  tensor_simple               # Basic Tensor operations: construction, coefficient access, arithmetic
  tensor_contraction           # Tensor contractions (generalized matrix multiply)
  tensor_reduction             # Reductions: sum, mean, max, min, any, all
  tensor_morphing              # Reshape, slice, chip, stride, pad
  tensor_fft                   # Forward/inverse FFT on tensors
  tensor_thread_pool_basic     # ThreadPoolDevice: elementwise, chip, volume_patch, reductions

  # --- External Library Support (conditional) (1) ---
  cholmod_support             # CHOLMOD backend (skipped if not installed)
)
