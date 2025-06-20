# Changelog

## [Unreleased]

_Disclaimer:_ This release contains a lot of changes. Due to the large amount of changes, and the time between this release and the last, the release notes may be incomplete or could contain a small amount of wrong entries.  
If you notice that these notes are incomplete, please point it out in the issue tracker or forum so it can be fixed.

### Supported modules

#### Breaking changes

- [#485](https://gitlab.com/libeigen/eigen/-/merge_requests/485): Removes deprecated CMake package config variables, potentially breaking existing CMake configurations
- [#608](https://gitlab.com/libeigen/eigen/-/merge_requests/608): Removes CI jobs for C++03 compatibility, signaling transition to modern C++ standards
- [#649](https://gitlab.com/libeigen/eigen/-/merge_requests/649): Move Eigen::all, last, and lastp1 back to Eigen::placeholders namespace to reduce name collision risks
- [#725](https://gitlab.com/libeigen/eigen/-/merge_requests/725): Removed deprecated MappedSparseMatrix type from internal library code
- [#742](https://gitlab.com/libeigen/eigen/-/merge_requests/742): Updates minimum CMake version to 3.10, removes C++11 test disable option, and sets minimum GCC version to 5
- [#744](https://gitlab.com/libeigen/eigen/-/merge_requests/744): Updated compiler requirements by removing deprecated feature test macros and enforcing newer GCC and MSVC versions
- [#771](https://gitlab.com/libeigen/eigen/-/merge_requests/771): Renamed internal `size` function to `ssize` to prevent ADL conflicts and improve C++ standard compatibility
- [#808](https://gitlab.com/libeigen/eigen/-/merge_requests/808): Introduces explicit type casting requirements for `pmadd` function to improve type safety and compatibility with custom scalar types
- [#826](https://gitlab.com/libeigen/eigen/-/merge_requests/826): Significant updates to SVD module with new Options template parameter, introducing API breaking changes for improved flexibility
- [#840](https://gitlab.com/libeigen/eigen/-/merge_requests/840): Fixed CUDA feature flag handling to respect `EIGEN_NO_CUDA` compilation option
- [#857](https://gitlab.com/libeigen/eigen/-/merge_requests/857): Reintroduced `svd::compute(Matrix, options)` method to prevent breaking external projects
- [#862](https://gitlab.com/libeigen/eigen/-/merge_requests/862): Restores fixed sizes for U/V matrices in matrix decompositions for fixed-sized inputs
- [#911](https://gitlab.com/libeigen/eigen/-/merge_requests/911): Fixed critical assumption about RowMajorBit and RowMajor, potentially impacting matrix storage order logic
- [#932](https://gitlab.com/libeigen/eigen/-/merge_requests/932): Replaced `make_coherent` with `CoherentPadOp`, removing a function that modifies `const` inputs and introducing a more performant padding operator for derivative vector sizing
- [#946](https://gitlab.com/libeigen/eigen/-/merge_requests/946): Removed legacy macro EIGEN_EMPTY_STRUCT_CTOR, potentially impacting older GCC compatibility
- [#966](https://gitlab.com/libeigen/eigen/-/merge_requests/966): Simplified Accelerate LLT and LDLT solvers by removing explicit Symmetric flag requirement
- [#1015](https://gitlab.com/libeigen/eigen/-/merge_requests/1015): Disabled AVX512 GEMM kernels by default due to segmentation fault issues
- [#1240](https://gitlab.com/libeigen/eigen/-/merge_requests/1240): Changes default comparison overloads to return `bool` arrays and introduces `cwiseTypedLesser` for typed comparisons
- [#1254](https://gitlab.com/libeigen/eigen/-/merge_requests/1254): Backwards compatible implementation of DenseBase::select with swapped template argument order
- [#1280](https://gitlab.com/libeigen/eigen/-/merge_requests/1280): Disabled raw array indexed view access for 1D arrays to prevent potential bugs and improve library safety
- [#1301](https://gitlab.com/libeigen/eigen/-/merge_requests/1301): Introduces canonical range corrections for Euler angles with new default behavior, potentially breaking existing angle computations
- [#1497](https://gitlab.com/libeigen/eigen/-/merge_requests/1497): Removed non-standard `int` return types and unnecessary arguments from BLAS/LAPACK function interfaces to improve package compatibility
- [#1520](https://gitlab.com/libeigen/eigen/-/merge_requests/1520): Removes `using namespace Eigen` from blas/common.h to prevent symbol collisions
- [#1550](https://gitlab.com/libeigen/eigen/-/merge_requests/1550): Modify `rbegin`/`rend` handling for GPU, now explicitly marking unsupported with clearer compile-time errors
- [#1553](https://gitlab.com/libeigen/eigen/-/merge_requests/1553): Restored C++03 compatibility by modifying 2x2 matrix construction to support older C++ standards
- [#1696](https://gitlab.com/libeigen/eigen/-/merge_requests/1696): Makes fixed-size matrices and arrays `trivially_default_constructible`, requiring `EIGEN_NO_DEBUG` or `EIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT`
- [#1795](https://gitlab.com/libeigen/eigen/-/merge_requests/1795): Changes `Eigen::aligned_allocator` to no longer inherit from `std::allocator`, modifying allocator behavior and potentially breaking existing code
- [#1827](https://gitlab.com/libeigen/eigen/-/merge_requests/1827): Removes default assumption of `std::complex` for complex scalar types, allowing more flexibility with custom complex types

#### Major changes

- [#356](https://gitlab.com/libeigen/eigen/-/merge_requests/356): Introduced PocketFFT as a more performant and accurate replacement for KissFFT in Eigen's FFT module
- [#489](https://gitlab.com/libeigen/eigen/-/merge_requests/489): Added AVX512 and AVX2 support for Packet16i and Packet8i, enhancing vectorization capabilities for integer types
- [#515](https://gitlab.com/libeigen/eigen/-/merge_requests/515): Adds random matrix generation via SVD with two strategies for generating singular values
- [#610](https://gitlab.com/libeigen/eigen/-/merge_requests/610): Updates CMake configuration to centralize C++11 standard setting, simplifying build process
- [#673](https://gitlab.com/libeigen/eigen/-/merge_requests/673): Vectorized implementation of Visitor.h with up to 39% performance improvement using AVX2 instructions
- [#698](https://gitlab.com/libeigen/eigen/-/merge_requests/698): Optimizes `CommaInitializer` to reuse fixed dimensions more efficiently during matrix block initialization
- [#702](https://gitlab.com/libeigen/eigen/-/merge_requests/702): Added AVX vectorized implementation for float2half/half2float conversion functions with significant performance improvements
- [#732](https://gitlab.com/libeigen/eigen/-/merge_requests/732): Removes EIGEN_HAS_CXX11 macro, simplifying Eigen's codebase and focusing on C++11+ support
- [#736](https://gitlab.com/libeigen/eigen/-/merge_requests/736): SFINAE improvements for transpose methods in self-adjoint and triangular views
- [#764](https://gitlab.com/libeigen/eigen/-/merge_requests/764): Performance improvements for VSX and MMA GEMV operations on PowerPC, with up to 4X speedup
- [#796](https://gitlab.com/libeigen/eigen/-/merge_requests/796): Makes fixed-size Matrix and Array trivially copyable in C++20, improving memory management and compatibility
- [#817](https://gitlab.com/libeigen/eigen/-/merge_requests/817): Added support for int64 packets on x86 architectures, enabling more efficient vectorized operations
- [#820](https://gitlab.com/libeigen/eigen/-/merge_requests/820): Added reciprocal packet operation with optimized SSE, AVX, and AVX512 specializations for float, improving computational performance and accuracy
- [#824](https://gitlab.com/libeigen/eigen/-/merge_requests/824): Removed inline assembly for FMA (AVX) and added new packet operations pmsub, pnmadd, and pnmsub with performance improvements
- [#827](https://gitlab.com/libeigen/eigen/-/merge_requests/827): Optimized precipitation function implementation with IEEE compliance for 1/0 and 1/inf cases, improving performance and handling of special mathematical scenarios
- [#829](https://gitlab.com/libeigen/eigen/-/merge_requests/829): Replace Eigen type metaprogramming with standard C++ types and alias templates
- [#834](https://gitlab.com/libeigen/eigen/-/merge_requests/834): Introduces AVX512 optimized kernels for floating-point triangular solve operations, enhancing performance for smaller matrix sizes
- [#856](https://gitlab.com/libeigen/eigen/-/merge_requests/856): Adds support for Apple's Accelerate sparse matrix solvers with significant performance improvements for various factorization methods
- [#860](https://gitlab.com/libeigen/eigen/-/merge_requests/860): Adds AVX512 optimizations for matrix multiplication with significant performance improvements for single and double precision kernels
- [#868](https://gitlab.com/libeigen/eigen/-/merge_requests/868): Optimized SQRT/RSQRT implementations for modern x86 processors with improved performance and special value handling
- [#880](https://gitlab.com/libeigen/eigen/-/merge_requests/880): Fix critical SVD functionality bug for Microsoft Visual Studio (MSVC) compilation
- [#892](https://gitlab.com/libeigen/eigen/-/merge_requests/892): Added support for constant evaluation and improved alignment check assertions
- [#936](https://gitlab.com/libeigen/eigen/-/merge_requests/936): Performance improvements for GEMM on Power architecture with vector_pair loads and optimized matrix multiplication
- [#971](https://gitlab.com/libeigen/eigen/-/merge_requests/971): Introduces R-Bidiagonalization step to BDCSVD, optimizing SVD performance for tall and wide matrices using QR decomposition
- [#972](https://gitlab.com/libeigen/eigen/-/merge_requests/972): AVX512 optimizations for s/dgemm compute kernel, resolving previous architectural and build compatibility issues
- [#975](https://gitlab.com/libeigen/eigen/-/merge_requests/975): Introduced subMappers for Power GEMM packing, improving performance by approximately 10% through simplified address calculations
- [#978](https://gitlab.com/libeigen/eigen/-/merge_requests/978): Added efficient sparse subset of matrix inverse computation using Takahashi algorithm with improved numerical stability
- [#983](https://gitlab.com/libeigen/eigen/-/merge_requests/983): Extends SYCL backend's QueueInterface to accept existing SYCL queues for improved framework integration
- [#986](https://gitlab.com/libeigen/eigen/-/merge_requests/986): SYCL-2020 range handling updated to ensure at least one thread execution by replacing default ranges with ranges of size 1
- [#990](https://gitlab.com/libeigen/eigen/-/merge_requests/990): Adds product operations and static initializers for DiagonalMatrix, improving matrix algebra convenience
- [#992](https://gitlab.com/libeigen/eigen/-/merge_requests/992): Enhanced AVX512 TRSM kernels to respect EIGEN_NO_MALLOC memory allocation configuration
- [#996](https://gitlab.com/libeigen/eigen/-/merge_requests/996): Updates SYCL kernel naming to comply with SYCL-2020 specification, improving SYCL compatibility and integration
- [#1008](https://gitlab.com/libeigen/eigen/-/merge_requests/1008): Add Power10 (AltiVec) MMA instructions support for bfloat16 computations with enhanced performance
- [#1017](https://gitlab.com/libeigen/eigen/-/merge_requests/1017): Add support for AVX512-FP16 instruction set, introducing `Packet32h` and optimizing half-precision floating-point operations with up to 8-9x performance improvement
- [#1018](https://gitlab.com/libeigen/eigen/-/merge_requests/1018): Optimize gebp_kernel for arm64-neon with 3px8/2px8/1px8 configuration to improve matrix multiplication performance
- [#1024](https://gitlab.com/libeigen/eigen/-/merge_requests/1024): Partial Packet support for GEMM real-only operations on PowerPC, with compilation warning fixes and performance improvements
- [#1034](https://gitlab.com/libeigen/eigen/-/merge_requests/1034): Improved pow<double> performance with more efficient division algorithm, achieving 11-15% speedup
- [#1036](https://gitlab.com/libeigen/eigen/-/merge_requests/1036): Replaced malloc/free with conditional_aligned memory allocation in sparse classes to improve memory management and potential performance
- [#1038](https://gitlab.com/libeigen/eigen/-/merge_requests/1038): Vectorized implementations of `acos`, `asin`, and `atan` for float with significant performance improvements
- [#1073](https://gitlab.com/libeigen/eigen/-/merge_requests/1073): Adds AVX vectorized implementation for int32_t division with improved performance
- [#1076](https://gitlab.com/libeigen/eigen/-/merge_requests/1076): Adds vectorized integer division for int32 using AVX512, AVX, and SSE instructions with performance optimizations
- [#1082](https://gitlab.com/libeigen/eigen/-/merge_requests/1082): Adds vectorized implementation of atan2 with array syntax, providing significant performance improvements for mathematical computations
- [#1089](https://gitlab.com/libeigen/eigen/-/merge_requests/1089): Unconditionally enables CXX11 math features for all compilers supporting C++14 and later
- [#1103](https://gitlab.com/libeigen/eigen/-/merge_requests/1103): Added new utility for sorting inner vectors of sparse matrices and vectors with custom comparison function
- [#1126](https://gitlab.com/libeigen/eigen/-/merge_requests/1126): Added Intel DPCPP compiler support for SYCL backend with SYCL-2020 compatibility
- [#1147](https://gitlab.com/libeigen/eigen/-/merge_requests/1147): Comprehensive overhaul of SparseMatrix core functionality, improving performance, efficiency, and maintainability of sparse matrix operations
- [#1148](https://gitlab.com/libeigen/eigen/-/merge_requests/1148): Introduced runtime memory allocation guards and modified assertion behavior to improve debugging and error handling
- [#1152](https://gitlab.com/libeigen/eigen/-/merge_requests/1152): Adds template for QR permutation index type and improves ColPivHouseholderQR LAPACKE bindings
- [#1160](https://gitlab.com/libeigen/eigen/-/merge_requests/1160): Improved insert strategy for compressed sparse matrices with enhanced performance and capacity management
- [#1164](https://gitlab.com/libeigen/eigen/-/merge_requests/1164): Improved sparse matrix permutation performance by reducing memory allocations and optimizing data handling strategies
- [#1166](https://gitlab.com/libeigen/eigen/-/merge_requests/1166): Introduces custom ODR-safe assertion mechanism for improved C++20 module compatibility
- [#1168](https://gitlab.com/libeigen/eigen/-/merge_requests/1168): Adds thread-local storage for `is_malloc_allowed()` state to improve multi-threaded safety
- [#1170](https://gitlab.com/libeigen/eigen/-/merge_requests/1170): Significant performance improvements for sparse matrix insertion, reducing insertion times by orders of magnitude and optimizing memory management
- [#1196](https://gitlab.com/libeigen/eigen/-/merge_requests/1196): Introduced vectorized comparison optimizations with typed comparisons and new selection operation, improving performance for comparison operations
- [#1197](https://gitlab.com/libeigen/eigen/-/merge_requests/1197): Removed all LGPL licensed code and references to simplify licensing and improve compatibility with MPL2
- [#1203](https://gitlab.com/libeigen/eigen/-/merge_requests/1203): Introduces typed logical operators for full vectorization and generalized boolean evaluations across scalar types
- [#1210](https://gitlab.com/libeigen/eigen/-/merge_requests/1210): Optimizations for bfloat16 Matrix-Matrix Multiplication (MMA) with performance improvements up to 10%
- [#1211](https://gitlab.com/libeigen/eigen/-/merge_requests/1211): Add CArg function for vectorized complex argument calculations
- [#1233](https://gitlab.com/libeigen/eigen/-/merge_requests/1233): Vectorized `any()` and `all()` methods, improved performance for matrix operations and custom visitors
- [#1236](https://gitlab.com/libeigen/eigen/-/merge_requests/1236): Added partial linear access for bfloat16 GEMM MMA, improving performance by 30% with reduced memory loads
- [#1244](https://gitlab.com/libeigen/eigen/-/merge_requests/1244): Introduces mechanism to specify permutation index types for PartialPivLU and FullPivLU, improving compatibility with Lapacke ILP64 interfaces
- [#1260](https://gitlab.com/libeigen/eigen/-/merge_requests/1260): Upgrades NaN and Inf detection to use modern C++14 standard features for improved floating-point value handling
- [#1273](https://gitlab.com/libeigen/eigen/-/merge-requests/1273): Replaced internal pointer typedefs with standard `std::(u)intptr_t` types and removed ICC workaround
- [#1279](https://gitlab.com/libeigen/eigen/-/merge_requests/1279): Refactors indexed view expressions to enable non-const reference access with symbolic indices
- [#1281](https://gitlab.com/libeigen/eigen/-/merge_requests/1281): Introduces `insertFromTriplets` and `insertFromSortedTriplets` methods for efficient sparse matrix batch insertion and optimizes `setFromTriplets`
- [#1285](https://gitlab.com/libeigen/eigen/-/merge_requests/1285): Introduces Unified Shared Memory (USM) support for SYCL, simplifying device pointer management and improving expression construction efficiency
- [#1289](https://gitlab.com/libeigen/eigen/-/merge_requests/1289): Moves thread pool code from Tensor to Core module, enhancing multithreading infrastructure
- [#1293](https://gitlab.com/libeigen/eigen/-/merge_requests/1293): Enable new AVX512 GEMM kernel by default, improving performance for supported hardware
- [#1295](https://gitlab.com/libeigen/eigen/-/merge_requests/1295): Refactored IndexedView to simplify SFINAE usage, improve readability, and re-enable raw fixed-size array access
- [#1296](https://gitlab.com/libeigen/eigen/-/merge_requests/1296): Adds dynamic dispatch for BF16 GEMM on Power architecture with new VSX implementation, achieving up to 13.4X speedup and improved conversion performance
- [#1304](https://gitlab.com/libeigen/eigen/-/merge_requests/1304): Specialized vectorized casting evaluator for improved packet type conversion efficiency
- [#1307](https://gitlab.com/libeigen/eigen/-/merge_requests/1307): New VSX implementation of BF16 GEMV for Power architecture with up to 6.7X performance improvement
- [#1314](https://gitlab.com/libeigen/eigen/-/merge_requests/1314): Introduced `canonicalEulerAngles` method to replace deprecated `eulerAngles`, improving Euler angle calculation standardization and accuracy
- [#1329](https://gitlab.com/libeigen/eigen/-/merge_requests/1329): Added macros to customize ThreadPool synchronization primitives for enhanced performance and flexibility
- [#1330](https://gitlab.com/libeigen/eigen/-/merge_requests/1330): Added half precision type support for SYCL-2020 with efficient `Eigen::half` and `cl::sycl::half` conversions
- [#1336](https://gitlab.com/libeigen/eigen/-/merge_requests/1336): Introduces linear redux evaluators with efficient linear access methods for expressions, improving traversal and potential performance
- [#1347](https://gitlab.com/libeigen/eigen/-/merge_requests/1347): Adds compile-time and run-time assertions for `Ref<const>` construction to improve memory layout safety and error handling
- [#1375](https://gitlab.com/libeigen/eigen/-/merge_requests/1375): Add architecture definition files for Qualcomm Hexagon Vector Extension (HVX), introducing support for `EIGEN_VECTORIZE_HVX` and optimized vector operations
- [#1387](https://gitlab.com/libeigen/eigen/-/merge_requests/1387): Introduced a new method to improve handling of block expressions, offering a backwards compatible solution for converting blocks of block expressions
- [#1389](https://gitlab.com/libeigen/eigen/-/merge_requests/1389): New panel modes for GEMM MMA with real and complex number support, delivering performance improvements up to 2.84X for small matrices and 34-75% speed enhancements for large matrices
- [#1395](https://gitlab.com/libeigen/eigen/-/merge_requests/1395): Introduced ThreadPool in Eigen Core, enabling parallel vector and matrix computations with a new `CoreThreadPoolDevice`
- [#1408](https://gitlab.com/libeigen/eigen/-/merge_requests/1408): Generalized parallel GEMM implementation to support `Eigen::ThreadPool` in addition to OpenMP, enhancing library flexibility across different platforms
- [#1454](https://gitlab.com/libeigen/eigen/-/merge_requests/1454): Added half and quarter vector support for HVX architecture, enabling performance improvements for small matrix operations
- [#1511](https://gitlab.com/libeigen/eigen/-/merge_requests/1511): Added direct access methods and strides for IndexedView, enhancing matrix operation efficiency and usability
- [#1522](https://gitlab.com/libeigen/eigen/-/merge_requests/1522): Introduces SIMD vectorized sine and cosine functions for double precision using Veltkamp method and Padé approximant
- [#1544](https://gitlab.com/libeigen/eigen/-/merge_requests/1544): Added `Packet2l` for SSE to support vectorized `int64_t` operations
- [#1545](https://gitlab.com/libeigen/eigen/-/merge_requests/1545): Improved `CwiseUnaryView` with enhanced functionality for accessing and modifying complex array components
- [#1546](https://gitlab.com/libeigen/eigen/-/merge_requests/1546): Added optimized casting support between `double` and `int64_t` for SSE and AVX2 instruction sets
- [#1554](https://gitlab.com/libeigen/eigen/-/merge_requests/1554): Add `SimplicialNonHermitianLLT` and `SimplicialNonHermitianLDLT` solvers for complex symmetric matrices
- [#1556](https://gitlab.com/libeigen/eigen/-/merge_requests/1556): Reorganized CMake configuration to improve build efficiency and installation support, reducing configuration time and simplifying integration
- [#1565](https://gitlab.com/libeigen/eigen/-/merge_requests/1565): Enables symbols in compile-time expressions, enhancing `Eigen::indexing::last` usability and compile-time computation efficiency
- [#1572](https://gitlab.com/libeigen/eigen/-/merge_requests/1572): Implemented fully vectorized `double` to `int64_t` casting using AVX2, achieving 70% throughput improvement
- [#1578](https://gitlab.com/libeigen/eigen/-/merge_requests/1578): Updates to Geometry_SIMD.h to enhance SIMD performance and compatibility with modern architectures
- [#1593](https://gitlab.com/libeigen/eigen/-/merge_requests/1593): Introduced specialized vectorized evaluation for `(a < b).select(c, d)` ternary operations to improve performance
- [#1600](https://gitlab.com/libeigen/eigen/-/merge_requests/1600): Optimized transpose product calculations to reduce memory allocations and improve performance for matrix operations
- [#1636](https://gitlab.com/libeigen/eigen/-/merge_requests/1636): Enable `pointer_based_stl_iterator` to conform to `contiguous_iterator` concept in C++20, improving range and view compatibility
- [#1654](https://gitlab.com/libeigen/eigen/-/merge_requests/1654): Introduced `EIGEN_ALIGN_TO_AVOID_FALSE_SHARING` macro to reduce atomic false sharing in `RunQueue`, improving multithreaded performance
- [#1655](https://gitlab.com/libeigen/eigen/-/merge_requests/1655): Optimizes ThreadPool task submission with significant performance improvements, reducing execution time up to 49% in multi-threaded scenarios
- [#1662](https://gitlab.com/libeigen/eigen/-/merge_requests/1662): Speed up complex * complex matrix multiplication by dynamically adjusting block panel size for enhanced performance (8-33% improvement)
- [#1670](https://gitlab.com/libeigen/eigen/-/merge_requests/1670): Introduces new rational approximation for `tanh` with up to 50% performance gain and improved numerical accuracy
- [#1671](https://gitlab.com/libeigen/eigen/-/merge_requests/1671): Introduced a new inner product evaluator with direct reduction for improved dot product performance, supporting explicit unrolling for small vectors and enhancing SIMD operations
- [#1673](https://gitlab.com/libeigen/eigen/-/merge_requests/1673): Performance optimization for SVE intrinsics by replacing `_z` suffix with `_x` suffix to reduce instruction overhead
- [#1675](https://gitlab.com/libeigen/eigen/-/merge_requests/1675): Adds vectorized implementation of `tanh<double>` with significant performance speedups across different instruction set architectures
- [#1694](https://gitlab.com/libeigen/eigen/-/merge_requests/1694): Make fixed-size matrices and arrays trivially copy and move constructible, enabling better compiler optimizations
- [#1737](https://gitlab.com/libeigen/eigen/-/merge_requests/1737): Enhance fixed-size matrices to conform to `std::is_standard_layout`, improving type safety and memory handling
- [#1777](https://gitlab.com/libeigen/eigen/-/merge_requests/1777): Add support for LoongArch64 LSX architecture, expanding hardware compatibility
- [#1801](https://gitlab.com/libeigen/eigen/-/merge_requests/1801): Significantly improves Simplicial Cholesky `analyzePattern` performance using advanced sparse matrix algorithms, reducing computation time dramatically
- [#1813](https://gitlab.com/libeigen/eigen/-/merge_requests/1813): Increases maximum alignment to 256 bytes, enhancing support for `MaxSizeVector` and optimizing alignment for modern ARM architectures
- [#1820](https://gitlab.com/libeigen/eigen/-/merge_requests/1820): Improves fixed-size assignment handling by optimizing vectorized traversals and reducing compiler `Warray-bounds` warnings
- [#1838](https://gitlab.com/libeigen/eigen/-/merge_requests/1838): Simplified parallel task API for `ParallelFor` and `ParallelForAsync`, optimizing task function definition and completion handling

#### Other

##### Fixed

- [#611](https://gitlab.com/libeigen/eigen/-/merge_requests/611): Included `<unordered_map>` header to resolve header inclusion issue
- [#613](https://gitlab.com/libeigen/eigen/-/merge_requests/613): Fix `fix<N>` implementation for environments without variable templates support
- [#614](https://gitlab.com/libeigen/eigen/-/merge_requests/614): Fixed LAPACK test compilation issues with type mismatches in older Fortran code
- [#621](https://gitlab.com/libeigen/eigen/-/merge_requests/621): Fixed GCC 4.8 ARM compilation issues by improving register constraints and resolving warnings
- [#629](https://gitlab.com/libeigen/eigen/-/merge_requests/629): Fixed EIGEN_OPTIMIZATION_BARRIER compatibility for arm-clang compiler
- [#630](https://gitlab.com/libeigen/eigen/-/merge_requests/630): Fixed AVX2 integer packet issues and corrected AVX512 implementation details
- [#635](https://gitlab.com/libeigen/eigen/-/merge_requests/635): Fixed tridiagonalization selector issue by modifying `hCoeffs` vector handling to improve type compatibility
- [#638](https://gitlab.com/libeigen/eigen/-/merge_requests/638): Fixed missing packet types in pset1 function call, improving packet data handling robustness
- [#639](https://gitlab.com/libeigen/eigen/-/merge_requests/639): Fixed AVX2 PacketMath.h implementation with typo corrections and unaligned load resolution
- [#643](https://gitlab.com/libeigen/eigen/-/merge_requests/643): Minor fix for compilation error on HIP
- [#651](https://gitlab.com/libeigen/eigen/-/merge_requests/651): Remove `-fabi-version=6` flag from AVX512 builds to improve compatibility
- [#654](https://gitlab.com/libeigen/eigen/-/merge_requests/654): Silenced GCC string overflow warning in initializer_list_construction test
- [#656](https://gitlab.com/libeigen/eigen/-/merge_requests/656): Resolved strict aliasing bug causing product_small function failures in matrix multiplication
- [#657](https://gitlab.com/libeigen/eigen/-/merge_requests/657): Fixes implicit conversion warnings in tuple_test, improving type safety
- [#659](https://gitlab.com/libeigen/eigen/-/merge_requests/659): Fixed undefined behavior in BFloat16 float conversion by replacing `reinterpret_cast` with a safer alternative, improving reliability on PPC platforms
- [#664](https://gitlab.com/libeigen/eigen/-/merge_requests/664): Fixed MSVC compilation issues with complex compound assignment operators by disabling related tests
- [#665](https://gitlab.com/libeigen/eigen/-/merge_requests/665): Fix tuple compilation issues in Visual Studio 2017 by replacing tuple alias with TupleImpl
- [#666](https://gitlab.com/libeigen/eigen/-/merge_requests/666): Fixed MSVC+NVCC compilation issue with EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR macro
- [#680](https://gitlab.com/libeigen/eigen/-/merge_requests/680): Fixed PowerPC packing issue, correcting row and depth inversion in non-vectorized code with 10% performance improvement
- [#686](https://gitlab.com/libeigen/eigen/-/merge_requests/686): Reverted bit_cast implementation to use memcpy for CUDA to prevent undefined behavior
- [#689](https://gitlab.com/libeigen/eigen/-/merge_requests/689): Fixed broadcasting index-out-of-bounds error for vectorized 1-dimensional inputs, particularly for std::complex types
- [#691](https://gitlab.com/libeigen/eigen/-/merge_requests/691): Fixed Clang warnings by replacing bitwise operators with correct logical operators
- [#694](https://gitlab.com/libeigen/eigen/-/merge_requests/694): Fixed ZVector build issues for s390x cross-compilation, enabling packetmath tests under QEMU
- [#696](https://gitlab.com/libeigen/eigen/-/merge_requests/696): Fixed build compatibility issues with pload and ploadu functions on ARM and PPC architectures by removing const from visitor return type
- [#703](https://gitlab.com/libeigen/eigen/-/merge_requests/703): Fix NaN propagation in min/max functions with scalar inputs
- [#707](https://gitlab.com/libeigen/eigen/-/merge_requests/707): Fixed total deflation issue in BDCSVD for diagonal matrices
- [#709](https://gitlab.com/libeigen/eigen/-/merge_requests/709): Fixed BDCSVD total deflation logic to correctly handle diagonal matrices
- [#711](https://gitlab.com/libeigen/eigen/-/merge_requests/711): Bug fix for incorrect definition of EIGEN_HAS_FP16_C macro across different compilers
- [#713](https://gitlab.com/libeigen/eigen/-/merge_requests/713): Prevent integer overflow in EigenMetaKernel indexing for improved reliability, especially on Windows builds
- [#714](https://gitlab.com/libeigen/eigen/-/merge_requests/714): Fixed uninitialized matrix issue to prevent potential computation errors
- [#719](https://gitlab.com/libeigen/eigen/-/merge_requests/719): Fixed Sparse-Sparse product implementation for mixed StorageIndex types
- [#728](https://gitlab.com/libeigen/eigen/-/merge_requests/728): Fixed compilation errors for Windows build systems
- [#733](https://gitlab.com/libeigen/eigen/-/merge_requests/733): Fixed warnings about shadowing definitions to improve code clarity and maintainability
- [#741](https://gitlab.com/libeigen/eigen/-/merge_requests/741): Fixes HIP compilation failure in DenseBase by adding appropriate EIGEN_DEVICE_FUNC modifiers
- [#745](https://gitlab.com/libeigen/eigen/-/merge_requests/745): Fixed HIP compilation issues in selfAdjoint and triangular view classes
- [#746](https://gitlab.com/libeigen/eigen/-/merge_requests/746): Fixed handling of 0-sized matrices in LAPACKE-based Cholesky decomposition
- [#759](https://gitlab.com/libeigen/eigen/-/merge_requests/759): Fixed typo of `StableNorm` to `stableNorm` in IDRS.h file
- [#762](https://gitlab.com/libeigen/eigen/-/merge_requests/762): Fixed documentation code snippets to improve accuracy and readability
- [#765](https://gitlab.com/libeigen/eigen/-/merge_requests/765): Resolved Clang compiler ambiguity in index list overloads to improve code stability
- [#769](https://gitlab.com/libeigen/eigen/-/merge_requests/769): Fixed header inclusion issues in CholmodSupport to prevent direct access to internal files
- [#782](https://gitlab.com/libeigen/eigen/-/merge_requests/782): Fix a bug with the EIGEN_IMPLIES macro's side-effects introduced in a previous merge request
- [#785](https://gitlab.com/libeigen/eigen/-/merge_requests/785): Fixed Clang warnings related to alignment and floating-point precision
- [#789](https://gitlab.com/libeigen/eigen/-/merge_requests/789): Fixed inclusion of immintrin.h for F16C intrinsics when vectorization is disabled
- [#794](https://gitlab.com/libeigen/eigen/-/merge_requests/794): Fixed header guard conflicts between AltiVec and ZVector packages
- [#800](https://gitlab.com/libeigen/eigen/-/merge_requests/800): Fixes serialization API issues disrupting HIP GPU unit tests
- [#801](https://gitlab.com/libeigen/eigen/-/merge_requests/801): Fixes and cleanups for BFloat16 and Half numeric_limits, including AVX `psqrt` function workaround
- [#802](https://gitlab.com/libeigen/eigen/-/merge_requests/802): Fixed improper truncation of unsigned int to bool, improving type conversion reliability
- [#803](https://gitlab.com/libeigen/eigen/-/merge_requests/803): Fixed GCC 8.5 warning about missing base class initialization
- [#805](https://gitlab.com/libeigen/eigen/-/merge_requests/805): Fixed inconsistency in scalar and vectorized paths for array.exp() function
- [#806](https://gitlab.com/libeigen/eigen/-/merge_requests/806): Fix assertion messages in IterativeSolverBase to correctly reference its own class name
- [#809](https://gitlab.com/libeigen/eigen/-/merge_requests/809): Fixed broken assertions to improve runtime error checking and library reliability
- [#810](https://gitlab.com/libeigen/eigen/-/merge_requests/810): Fixed two corner cases in logistic sigmoid implementation for improved accuracy and robustness
- [#811](https://gitlab.com/libeigen/eigen/-/merge_requests/811): Fixed compilation issue with GCC < 10 and -std=c++2a standard
- [#812](https://gitlab.com/libeigen/eigen/-/merge_requests/812): Fix implicit conversion warning in vectorwise_reverse_inplace function by adding explicit casting
- [#815](https://gitlab.com/libeigen/eigen/-/merge_requests/815): Fixed implicit conversion warning in GEBP kernel's packing by changing variable types from `int` to `Index`
- [#818](https://gitlab.com/libeigen/eigen/-/merge_requests/818): Silenced specific MSVC compiler warnings in `construct_elements_of_array()` function
- [#822](https://gitlab.com/libeigen/eigen/-/merge_requests/822): Fixed potential overflow issue in random test by making casts explicit and adjusting variable types
- [#828](https://gitlab.com/libeigen/eigen/-/merge_requests/828): Fixed GEMV cache overflow issue for PowerPC architecture
- [#833](https://gitlab.com/libeigen/eigen/-/merge_requests/833): Fixes type discrepancy in 32-bit ARM platforms by replacing `int` with `int32_t` for proper bit pattern extraction
- [#835](https://gitlab.com/libeigen/eigen/-/merge_requests/835): Fixed ODR violations by removing unnamed namespaces and internal linkage from header files
- [#842](https://gitlab.com/libeigen/eigen/-/merge_requests/842): Fixed documentation typo in Complete Orthogonal Decomposition (COD) method reference
- [#843](https://gitlab.com/libeigen/eigen/-/merge_requests/843): Fixed naming collision with resolve.h by renaming local variables
- [#847](https://gitlab.com/libeigen/eigen/-/merge_requests/847): Cleaned up compiler warnings for PowerPC GEMM and GEMV implementations
- [#851](https://gitlab.com/libeigen/eigen/-/merge_requests/851): Fixed JacobiSVD_LAPACKE bindings to align with SVD module runtime options
- [#858](https://gitlab.com/libeigen/eigen/-/merge_requests/858): Fixed sqrt/rsqrt implementations for NEON with improved accuracy and special case handling
- [#859](https://gitlab.com/libeigen/eigen/-/merge_requests/859): Fixed MSVC+NVCC 9.2 pragma compatibility issue by replacing `_Pragma` with `__pragma`
- [#863](https://gitlab.com/libeigen/eigen/-/merge_requests/863): Modified test expression to avoid numerical differences during optimization
- [#865](https://gitlab.com/libeigen/eigen/-/merge_requests/865): Added assertion for edge case when requesting thin unitaries with incompatible matrix dimensions
- [#866](https://gitlab.com/libeigen/eigen/-/merge_requests/866): Fix crash bug in SPQRSupport by initializing pointers to nullptr to prevent invalid memory access
- [#870](https://gitlab.com/libeigen/eigen/-/merge_requests/870): Fixed test macro conflicts with STL headers in C++20 for GCC 9-11
- [#873](https://gitlab.com/libeigen/eigen/-/merge_requests/873): Disabled deprecated warnings in SVD tests to clean up build logs
- [#874](https://gitlab.com/libeigen/eigen/-/merge_requests/874): Fixed gcc-5 packetmath_12 bug with memory initialization in `packetmath_minus_zero_add()`
- [#875](https://gitlab.com/libeigen/eigen/-/merge_requests/875): Fixed compilation error in packetmath by introducing a wrapper struct for `psqrt` function
- [#876](https://gitlab.com/libeigen/eigen/-/merge_requests/876): Fixed AVX512 instruction handling and complex type computation issues for g++-11
- [#877](https://gitlab.com/libeigen/eigen/-/merge_requests/877): Disabled deprecated warnings for SVD tests on MSVC to improve build log clarity
- [#878](https://gitlab.com/libeigen/eigen/-/merge_requests/878): Fixed frexp packetmath tests for MSVC to handle non-finite input exponent behavior
- [#882](https://gitlab.com/libeigen/eigen/-/merge_requests/882): Fixed SVD compatibility issues for MSVC and CUDA by resolving Index type and function return warnings
- [#883](https://gitlab.com/libeigen/eigen/-/merge_requests/883): Adjusted matrix_power test tolerance for MSVC to reduce test failures
- [#885](https://gitlab.com/libeigen/eigen/-/merge_requests/885): Fixed enum conversion warnings in BooleanRedux component
- [#886](https://gitlab.com/libeigen/eigen/-/merge_requests/886): Fixed denormal test to skip when condition is false
- [#900](https://gitlab.com/libeigen/eigen/-/merge_requests/900): Fix swap test for size 1 matrix inputs to prevent assertion failures
- [#901](https://gitlab.com/libeigen/eigen/-/merge_requests/901): Fixed `construct_at` compilation issue on ROCm/HIP environments
- [#908](https://gitlab.com/libeigen/eigen/-/merge_requests/908): Corrected reference code for `ata_product` function in STL_interface.hh
- [#914](https://gitlab.com/libeigen/eigen/-/merge_requests/914): Disabled Schur non-convergence test to reduce flaky results and improve reliability
- [#915](https://gitlab.com/libeigen/eigen/-/merge_requests/915): Fixed missing pound directive to improve compilation and code robustness
- [#917](https://gitlab.com/libeigen/eigen/-/merge_requests/917): Resolved g++-10 docker compiler optimization issue in geo_orthomethods_4 test
- [#918](https://gitlab.com/libeigen/eigen/-/merge_requests/918): Added missing explicit reinterprets for `_mm512_shuffle_f32x4` to resolve g++ build errors
- [#919](https://gitlab.com/libeigen/eigen/-/merge_requests/919): Fixed a missing parenthesis in the tutorial documentation
- [#922](https://gitlab.com/libeigen/eigen/-/merge_requests/922): Work around MSVC compiler bug dropping `const` qualifier in method definitions
- [#923](https://gitlab.com/libeigen/eigen/-/merge_requests/923): Fixed AVX512 build compatibility issues with MSVC compiler
- [#924](https://gitlab.com/libeigen/eigen/-/merge_requests/924): Disabled f16c scalar conversions for MSVC to prevent compatibility issues
- [#925](https://gitlab.com/libeigen/eigen/-/merge_requests/925): Fixed ODR violation in trsm module by marking specific functions as inline
- [#926](https://gitlab.com/libeigen/eigen/-/merge_requests/926): Fixed compilation errors by correcting namespace usage in the codebase
- [#930](https://gitlab.com/libeigen/eigen/-/merge_requests/930): Fixed compilation issue in GCC 9 by adding missing typename and removing unused typedef
- [#934](https://gitlab.com/libeigen/eigen/-/merge_requests/934): Fixed order of arguments in BLAS SYRK implementation to resolve compilation errors
- [#937](https://gitlab.com/libeigen/eigen/-/merge_requests/937): Eliminates warnings related to unused trace statements, improving code cleanliness
- [#945](https://gitlab.com/libeigen/eigen/-/merge_requests/945): Restored correct max size expressions that were unintentionally modified in a previous merge request
- [#948](https://gitlab.com/libeigen/eigen/-/merge_requests/948): Fix compatibility issues between MSVC and CUDA for diagonal and transpose functionality
- [#949](https://gitlab.com/libeigen/eigen/-/merge_requests/949): Fixed ODR violations in lapacke_helpers module to improve library reliability
- [#953](https://gitlab.com/libeigen/eigen/-/merge_requests/953): Fixed ambiguous constructors for DiagonalMatrix to prevent compile-time errors with initializer lists
- [#958](https://gitlab.com/libeigen/eigen/-/merge_requests/958): Fixed compiler bugs for GCC 10 & 11 in Power GEMM inline assembly
- [#963](https://gitlab.com/libeigen/eigen/-/merge_requests/963): Fixed NaN propagation for scalar input by adding missing template parameter
- [#964](https://gitlab.com/libeigen/eigen/-/merge_requests/964): Fix compilation issue in HouseholderSequence.h related to InnerPanel template parameter
- [#974](https://gitlab.com/libeigen/eigen/-/merge_requests/974): Fixed BDCSVD crash caused by index out of bounds in matrix processing
- [#976](https://gitlab.com/libeigen/eigen/-/merge_requests/976): Fix LDLT decomposition with AutoDiffScalar when value is 0
- [#977](https://gitlab.com/libeigen/eigen/-/merge_requests/977): Fixed numerical stability issue in BDCSVD algorithm
- [#980](https://gitlab.com/libeigen/eigen/-/merge_requests/980): Fixed signed integer overflow in adjoint test to improve code safety
- [#987](https://gitlab.com/libeigen/eigen/-/merge_requests/987): Fixed integer shortening warnings in visitor tests
- [#988](https://gitlab.com/libeigen/eigen/-/merge_requests/988): Fixed MSVC build issues with AVX512 by temporarily disabling specific optimizations to reduce memory consumption and prevent compilation failures
- [#993](https://gitlab.com/libeigen/eigen/-/merge_requests/993): Corrected row vs column vector terminology typo in Matrix class tutorial documentation
- [#1003](https://gitlab.com/libeigen/eigen/-/merge_requests/1003): Eliminated undefined warnings for non-AVX512 compilation by adding appropriate macro guards
- [#1007](https://gitlab.com/libeigen/eigen/-/merge_requests/1007): Fixed One Definition Rule (ODR) violations by converting unnamed type declarations to named types
- [#1010](https://gitlab.com/libeigen/eigen/-/merge_requests/1010): Fixed inner iterator for sparse block to correctly handle outer index and improve sparse matrix operations
- [#1012](https://gitlab.com/libeigen/eigen/-/merge_requests/1012): Fixed vectorized Jacobi Rotation implementation by correcting logic for applying vectorized operations
- [#1014](https://gitlab.com/libeigen/eigen/-/merge_requests/1014): Fixed `aligned_realloc` to correctly check memory allocation constraints when pointer is null
- [#1019](https://gitlab.com/libeigen/eigen/-/merge_requests/1019): Prevent `<sstream>` inclusion when `EIGEN_NO_IO` is defined, improving embedded system compatibility
- [#1023](https://gitlab.com/libeigen/eigen/-/merge_requests/1023): Fixed flaky packetmath_1 test by adjusting inputs to prevent value cancellations
- [#1025](https://gitlab.com/libeigen/eigen/-/merge_requests/1025): Fixed Packet2d type implementation for non-VSX platforms to improve portability
- [#1027](https://gitlab.com/libeigen/eigen/-/merge_requests/1027): Fixed vectorized pow() function to handle edge cases with negative zero and negative infinity correctly
- [#1028](https://gitlab.com/libeigen/eigen/-/merge_requests/1028): Fixed build compatibility for non-VSX PowerPC architectures
- [#1030](https://gitlab.com/libeigen/eigen/-/merge_requests/1030): Resolves Half function definition conflict on aarch64 for GPU compilation
- [#1032](https://gitlab.com/libeigen/eigen/-/merge_requests/1032): Fixed invalid deprecation warnings in BDCSVD constructor handling
- [#1037](https://gitlab.com/libeigen/eigen/-/merge_requests/1037): Protected new pblend implementation with EIGEN_VECTORIZE_AVX2 to address build compatibility issues
- [#1039](https://gitlab.com/libeigen/eigen/-/merge_requests/1039): Fixed `psign` function for unsigned integer types, preventing incorrect behavior with bool types
- [#1042](https://gitlab.com/libeigen/eigen/-/merge_requests/1042): Fixed undefined behavior in array_cwise test related to signed integer overflow
- [#1044](https://gitlab.com/libeigen/eigen/-/merge_requests/1044): Fixed memory allocation issue by adding missing pointer in realloc call
- [#1045](https://gitlab.com/libeigen/eigen/-/merge_requests/1045): Fixed GeneralizedEigenSolver::info() method to improve initialization checks and error messaging
- [#1048](https://gitlab.com/libeigen/eigen/-/merge_requests/1048): Fixed test build errors in unary power operations with improved type handling for real and complex numbers
- [#1049](https://gitlab.com/libeigen/eigen/-/merge_requests/1049): Fixed typos in documentation table for slicing tutorial
- [#1051](https://gitlab.com/libeigen/eigen/-/merge_requests/1051): Fixed mixingtypes tests related to unary power operation
- [#1053](https://gitlab.com/libeigen/eigen/-/merge_requests/1053): Fixed MSVC compilation error in GeneralizedEigenSolver.h by adding a missing semi-colon
- [#1055](https://gitlab.com/libeigen/eigen/-/merge_requests/1055): Added safeguard in `aligned_realloc()` to prevent memory reallocation when `EIGEN_RUNTIME_NO_MALLOC` is defined
- [#1057](https://gitlab.com/libeigen/eigen/-/merge_requests/1057): Adjusted overflow threshold bounds in power function tests to prevent integer and floating-point overflow scenarios
- [#1060](https://gitlab.com/libeigen/eigen/-/merge_requests/1060): Fixed memory reallocation for non-trivial types to handle self-referencing pointers and improve stability
- [#1061](https://gitlab.com/libeigen/eigen/-/merge_requests/1061): Fixed bound for pow function to handle floating-point type limitations
- [#1063](https://gitlab.com/libeigen/eigen/-/merge_requests/1063): Fixed type safety and comparison issues in unary pow() function
- [#1065](https://gitlab.com/libeigen/eigen/-/merge_requests/1065): Fixes sparse matrix compilation issues on ROCm backend
- [#1069](https://gitlab.com/libeigen/eigen/-/merge_requests/1069): Removed faulty skew_symmetric_matrix3 test with uninitialized matrix comparison errors
- [#1070](https://gitlab.com/libeigen/eigen/-/merge_requests/1070): Fixed test for pow function handling of mixed integer types
- [#1077](https://gitlab.com/libeigen/eigen/-/merge_requests/1077): Fixed unused-result warning for ROCm gpuGetDevice function with better error reporting
- [#1085](https://gitlab.com/libeigen/eigen/-/merge_requests/1085): Fixed 4x4 matrix inverse computation when compiling with -Ofast optimization flag
- [#1094](https://gitlab.com/libeigen/eigen/-/merge_requests/1094): Fixed unused variable warnings in Eigen/Sparse module with clang 16.0.0git
- [#1096](https://gitlab.com/libeigen/eigen/-/merge_requests/1096): Fixed a bug in the `atan2` function related to `pselect` behavior with single-bit packets
- [#1104](https://gitlab.com/libeigen/eigen/-/merge_requests/1104): Fix NEON instruction fmla bug for half data type, preventing compiler errors and performance issues
- [#1105](https://gitlab.com/libeigen/eigen/-/merge_requests/1105): Fixed pragma check for disabling fastmath optimization
- [#1106](https://gitlab.com/libeigen/eigen/-/merge_requests/1106): Fixed handmade_aligned_malloc offset computation to prevent potential out-of-bounds memory writes and compiler warnings
- [#1107](https://gitlab.com/libeigen/eigen/-/merge_requests/1107): Disable patan for double precision on PowerPC to prevent build failures
- [#1112](https://gitlab.com/libeigen/eigen/-/merge_requests/1112): Corrected a typo in the CholmodSupport module
- [#1113](https://gitlab.com/libeigen/eigen/-/merge_requests/1113): Fixed duplicate execution code for Power 8 Altivec in pstore_partial function
- [#1115](https://gitlab.com/libeigen/eigen/-/merge_requests/1115): Fixed AVX2 psignbit implementation to resolve accuracy and reliability issues
- [#1116](https://gitlab.com/libeigen/eigen/-/merge_requests/1116): Corrected pnegate function to accurately handle floating-point zero by directly flipping the sign bit
- [#1118](https://gitlab.com/libeigen/eigen/-/merge_requests/1118): Fixed ambiguity in PowerPC vec_splats call for uint64_t type compatibility
- [#1120](https://gitlab.com/libeigen/eigen/-/merge_requests/1120): Fixed critical bugs in `handmade_aligned_realloc` to prevent memory management issues and potential undefined behavior
- [#1124](https://gitlab.com/libeigen/eigen/-/merge_requests/1124): Fixed sparseLU solver to handle destinations with non-unit stride
- [#1127](https://gitlab.com/libeigen/eigen/-/merge_requests/1127): Fixed serialization process for non-compressed matrices by correcting data buffer size calculation
- [#1130](https://gitlab.com/libeigen/eigen/-/merge_requests/1130): Fixed index type typo in sparse index sorting implementation
- [#1142](https://gitlab.com/libeigen/eigen/-/merge_requests/1142): Fixed incorrect NEON native fp16 multiplication kernel for ARM hardware
- [#1149](https://gitlab.com/libeigen/eigen/-/merge_requests/1149): Fixed `.gitignore` issue preventing `scripts/buildtests.in` from being added with `git add .`
- [#1150](https://gitlab.com/libeigen/eigen/-/merge_requests/1150): Fixes Altivec detection and VSX instruction handling for macOS PowerPC systems
- [#1151](https://gitlab.com/libeigen/eigen/-/merge_requests/1151): Fixed EIGEN_HAS_CXX17_OVERALIGN configuration for Intel C++ Compiler (icc)
- [#1153](https://gitlab.com/libeigen/eigen/-/merge_requests/1153): Fix macro guards for emulated FP16 operators on GPU, improving compatibility and reducing compilation errors
- [#1155](https://gitlab.com/libeigen/eigen/-/merge_requests/1155): Fixes overalign check preprocessor directive handling for improved compiler compatibility
- [#1156](https://gitlab.com/libeigen/eigen/-/merge_requests/1156): Fixed minor build and test issues including header paths, vectorization, GPU support, and removing unnecessary headers
- [#1161](https://gitlab.com/libeigen/eigen/-/merge_requests/1161): Fixes unused parameter warning on 32-bit ARM with Clang compiler
- [#1178](https://gitlab.com/libeigen/eigen/-/merge_requests/1178): Resolved compiler warnings related to sparse matrix operations
- [#1179](https://gitlab.com/libeigen/eigen/-/merge_requests/1179): Fixed consistency issue in reciprocal square root (rsqrt) vectorized implementation
- [#1180](https://gitlab.com/libeigen/eigen/-/merge_requests/1180): Fixed critical sparse matrix handling bugs for empty matrices to prevent segmentation faults
- [#1181](https://gitlab.com/libeigen/eigen/-/merge_requests/1181): Fixed bugs in GPU convolution operations by enabling GPU assertions
- [#1184](https://gitlab.com/libeigen/eigen/-/merge_requests/1184): Fixes pre-POWER8_VECTOR bugs in pcmp_lt and pnegate, and reactivates psqrt function
- [#1189](https://gitlab.com/libeigen/eigen/-/merge_requests/1189): Added EIGEN_DEVICE_FUNC qualifiers to SkewSymmetricDense to fix CUDA compatibility
- [#1202](https://gitlab.com/libeigen/eigen/-/merge_requests/1202): Fixed MSVC ARM build compatibility by resolving intrinsic function and vector type handling issues
- [#1212](https://gitlab.com/libeigen/eigen/-/merge_requests/1212): Disabled array BF16 to F32 conversions on Power architecture to improve performance and stability
- [#1213](https://gitlab.com/libeigen/eigen/-/merge_requests/1213): Resolved multiple compiler warnings to improve code quality and maintainability
- [#1215](https://gitlab.com/libeigen/eigen/-/merge_requests/1215): Fixed compiler warnings in test files to improve code quality and maintainability
- [#1216](https://gitlab.com/libeigen/eigen/-/merge_requests/1216): Fixed a typo in the NEON `make_packet2f` function to improve correctness
- [#1218](https://gitlab.com/libeigen/eigen/-/merge_requests/1218): Fix MSVC atan2 test to align with POSIX specification for underflow cases
- [#1220](https://gitlab.com/libeigen/eigen/-/merge_requests/1220): Fixed NEON packetmath compilation issues with GCC and resolved preinterpret stack overflow problem
- [#1221](https://gitlab.com/libeigen/eigen/-/merge_requests/1221): Guard complex sqrt function for compatibility with old MSVC compilers
- [#1222](https://gitlab.com/libeigen/eigen/-/merge_requests/1222): Fixed epsilon value for long double in double-doubles to improve algorithm convergence on PowerPC
- [#1228](https://gitlab.com/libeigen/eigen/-/merge_requests/1228): Fixed compiler compatibility for `vec_div` instructions on Power architecture
- [#1229](https://gitlab.com/libeigen/eigen/-/merge_requests/1229): Resolved MemorySanitizer (MSAN) failures in SVD tests by fixing uninitialized matrix entry issues
- [#1235](https://gitlab.com/libeigen/eigen/-/merge_requests/1235): Fixed ODR issues with Intel's AVX512 TRSM kernels by removing static qualifiers
- [#1239](https://gitlab.com/libeigen/eigen/-/merge_requests/1239): Fixed NEON integer shift operation test for signed shifts to prevent incorrect argument handling
- [#1245](https://gitlab.com/libeigen/eigen/-/merge_requests/1245): Fixed cwise test by resolving signed integer overflow issues using `.abs()` method
- [#1248](https://gitlab.com/libeigen/eigen/-/merge_requests/1248): Fixed typo in LinAlgSVD example code to enable compilation and correct least-squares solution
- [#1249](https://gitlab.com/libeigen/eigen/-/merge_requests/1249): Fixed MSVC test failures related to intrinsic operations by replacing set1 intrinsics with set intrinsics
- [#1252](https://gitlab.com/libeigen/eigen/-/merge_requests/1252): Implemented a workaround for a compiler bug in Tridiagonalization.h, improving stability across compiler environments
- [#1256](https://gitlab.com/libeigen/eigen/-/merge_requests/1256): Fixed bug in minmax_coeff_visitor when matrix contains only NaN values
- [#1257](https://gitlab.com/libeigen/eigen/-/merge_requests/1257): Fixed minmax visitor behavior for PropagateFast option to prevent out-of-bounds index issues with NaN matrices
- [#1263](https://gitlab.com/libeigen/eigen/-/merge_requests/1263): Fixed PowerPC and Clang compiler warnings to improve code stability
- [#1270](https://gitlab.com/libeigen/eigen/-/merge_requests/1270): Fixed ARM build compatibility issues including casting, MSVC packet conversion, and 32-bit ARM macro definitions
- [#1271](https://gitlab.com/libeigen/eigen/-/merge_requests/1271): Fixed issues with SparseMatrix::Map typedef and setFromTriplets method robustness
- [#1277](https://gitlab.com/libeigen/eigen/-/merge_requests/1277): Fix incorrect casting in AVX512DQ vectorization path
- [#1282](https://gitlab.com/libeigen/eigen/-/merge_requests/1282): ASAN fixes for AVX512 GEMM/TRSM kernels, addressing memory safety issues with buffer overrun prevention
- [#1283](https://gitlab.com/libeigen/eigen/-/merge_requests/1283): Corrected intrinsic function for accurate truncation during double-to-int casting
- [#1291](https://gitlab.com/libeigen/eigen/-/merge_requests/1291): Fixed `.gitignore` to prevent accidentally ignoring Eigen's Core directory on Windows
- [#1302](https://gitlab.com/libeigen/eigen/-/merge_requests/1302): Fixed typo in SSE packetmath implementation
- [#1308](https://gitlab.com/libeigen/eigen/-/merge_requests/1308): Fix `pow` function for `uint32_t` and disable problematic packet multiplication operation
- [#1311](https://gitlab.com/libeigen/eigen/-/merge_requests/1311): Fixed sparse iterator compatibility and warnings on macOS with Clang by modifying `StorageRef` and replacing deprecated `std::random_shuffle`
- [#1312](https://gitlab.com/libeigen/eigen/-/merge_requests/1312): Fixed boolean bitwise and warning in test code
- [#1318](https://gitlab.com/libeigen/eigen/-/merge_requests/1318): Add safeguard in JacobiSVD to handle non-finite inputs by setting `m_nonzeroSingularValues` to zero
- [#1319](https://gitlab.com/libeigen/eigen/-/merge_requests/1319): Fixed ColMajor BF16 GEMV implementation for RowMajor input vectors
- [#1321](https://gitlab.com/libeigen/eigen/-/merge_requests/1321): Cleaned up array_cwise test by suppressing MSVC warnings, resolving operator precedence, and removing redundant shift tests
- [#1322](https://gitlab.com/libeigen/eigen/-/merge_requests/1322): Fixed specialized `loadColData` implementation for BF16 GEMV, improving LLVM compatibility
- [#1323](https://gitlab.com/libeigen/eigen/-/merge_requests/1323): Fixed compiler warning related to modulo by zero in visitor pattern
- [#1327](https://gitlab.com/libeigen/eigen/-/merge_requests/1327): Fixed CUDA compilation issues by adjusting header inclusion order and resolving `EIGEN_AVOID_STL_ARRAY` related problems
- [#1333](https://gitlab.com/libeigen/eigen/-/merge_requests/1333): Fixed SVD initialization issues and compiler warnings in JacobiSVD and BDCSVD routines
- [#1339](https://gitlab.com/libeigen/eigen/-/merge_requests/1339): Fixes CUDA compilation issues with `EIGEN_HAS_ARM64_FP16_SCALAR_ARITHMETIC` by preventing miscompilation in host/device functions
- [#1343](https://gitlab.com/libeigen/eigen/-/merge_requests/1343): Fixed unary `pow()` error handling with improved edge case management and test robustness
- [#1344](https://gitlab.com/libeigen/eigen/-/merge_requests/1344): Prevent underflow in `prsqrt` function by adding numerical stability safeguards
- [#1349](https://gitlab.com/libeigen/eigen/-/merge_requests/1349): Fixed AVX `pstore` function for integer types to ensure correct aligned store intrinsics
- [#1350](https://gitlab.com/libeigen/eigen/-/merge_requests/1350): Fixed `safe_abs` in `int_pow` to improve compatibility with Clang compiler
- [#1351](https://gitlab.com/libeigen/eigen/-/merge_requests/1351): Fixed SVD test stability by removing deprecated test behavior
- [#1357](https://gitlab.com/libeigen/eigen/-/merge_requests/1357): Fixed `supportsMMA` to correctly handle `EIGEN_ALTIVEC_MMA_DYNAMIC_DISPATCH` compilation flag and compiler support
- [#1359](https://gitlab.com/libeigen/eigen/-/merge_requests/1359): Fixed AVX512 trsm kernel memory allocation issues in nomalloc environments
- [#1360](https://gitlab.com/libeigen/eigen/-/merge_requests/1360): Fixed return type of `ivcSize` in `IndexedViewMethods.h` to improve type safety and consistency
- [#1361](https://gitlab.com/libeigen/eigen/-/merge_requests/1361): Fixes Altivec compilation compatibility with C++20 and higher standards
- [#1362](https://gitlab.com/libeigen/eigen/-/merge_requests/1362): Fixed `_mm256_cvtps_ph` intrinsic argument to eliminate MSVC compilation warning
- [#1363](https://gitlab.com/libeigen/eigen/-/merge_requests/1363): Fixed `arg` function compatibility in CUDA environments, resolving compilation issues with MSVC and C++20
- [#1367](https://gitlab.com/libeigen/eigen/-/merge_requests/1367): Addresses GCC compiler warnings by fixing zero-sized block handling, assignment operators, and uninitialized variable issues
- [#1370](https://gitlab.com/libeigen/eigen/-/merge_requests/1370): Fix compiler warning for matrix-vector multiplication loop optimizations on x86-64 gcc 10+
- [#1371](https://gitlab.com/libeigen/eigen/-/merge_requests/1371): Fixed `-Wmaybe-uninitialized` warning in SVD implementation by improving dimension initialization and type safety
- [#1376](https://gitlab.com/libeigen/eigen/-/merge_requests/1376): Fixed nullptr dereference issue in triangular product for zero-sized matrices
- [#1377](https://gitlab.com/libeigen/eigen/-/merge_requests/1377): Fix undefined behavior in triangular solves for empty systems
- [#1379](https://gitlab.com/libeigen/eigen/-/merge_requests/1379): Prevents nullptr dereference in SVD implementation for small matrices
- [#1380](https://gitlab.com/libeigen/eigen/-/merge_requests/1380): Fixes undefined behavior related to scalar memory alignment with improved memory alignment checks
- [#1386](https://gitlab.com/libeigen/eigen/-/merge_requests/1386): Fixed ARM32 floating-point division issues, improving accuracy and reliability of float computations
- [#1388](https://gitlab.com/libeigen/eigen/-/merge_requests/1388): Fixed stage success check in Pardiso solver to only report success when `m_info == Eigen::Success`
- [#1392](https://gitlab.com/libeigen/eigen/-/merge_requests/1392): Fixed CUDA device function calls by adding `EIGEN_DEVICE_FUNC` attribute to static run methods
- [#1394](https://gitlab.com/libeigen/eigen/-/merge_requests/1394): Fixed extra semicolon in `XprHelper` causing compilation issues with `-Wextra-semi` flag
- [#1396](https://gitlab.com/libeigen/eigen/-/merge_requests/1396): Fixed sparse triangular view iterator by restoring `row()` and `col()` function implementations to prevent segmentation faults
- [#1399](https://gitlab.com/libeigen/eigen/-/merge_requests/1399): Disable denorm deprecation warnings in MSVC C++23 to reduce compiler noise
- [#1402](https://gitlab.com/libeigen/eigen/-/merge_requests/1402): Work around MSVC compiler issue with Block XprType by removing dependent typedef
- [#1407](https://gitlab.com/libeigen/eigen/-/merge_requests/1407): Fixed `Wshorten-64-to-32` warnings in `div_ceil` function to improve code robustness
- [#1409](https://gitlab.com/libeigen/eigen/-/merge_requests/1409): Resolved compiler warnings and critical bug fixes in `Memory.h`
- [#1411](https://gitlab.com/libeigen/eigen/-/merge_requests/1411): Fixed typo in `EIGEN_RUNTIME_NO_MALLOC` macro to resolve nomalloc test failure on AVX512
- [#1412](https://gitlab.com/libeigen/eigen/-/merge_requests/1412): Backports fix for disambiguation of overloads with empty index lists, resolving compilation errors
- [#1415](https://gitlab.com/libeigen/eigen/-/merge_requests/1415): Link pthread library for `product_threaded` test to resolve test execution issues
- [#1416](https://gitlab.com/libeigen/eigen/-/merge_requests/1416): Fixed `Wshorten-64-to-32` warning in gemm parallelizer to improve code quality
- [#1417](https://gitlab.com/libeigen/eigen/-/merge_requests/1417): Fixed `getNbThreads()` to correctly return 1 when threading is not parallelized
- [#1419](https://gitlab.com/libeigen/eigen/-/merge_requests/1419): Ensures `mc` is not smaller than `Traits::nr` to prevent potential calculation errors
- [#1422](https://gitlab.com/libeigen/eigen/-/merge_requests/1422): Fix 64-bit integer to float conversion precision on ARM architectures
- [#1425](https://gitlab.com/libeigen/eigen/-/merge_requests/1425): Fixes typecasting issue for arm32 architecture, restoring proper functionality
- [#1431](https://gitlab.com/libeigen/eigen/-/merge_requests/1431): Fixed `scalar_logistic_function` overflow handling for complex inputs by improving comparison mechanism
- [#1434](https://gitlab.com/libeigen/eigen/-/merge_requests/1434): Fixed CUDA syntax error introduced by clang-format
- [#1439](https://gitlab.com/libeigen/eigen/-/merge_requests/1439): Fixed `_BitScanReverse` function implementation for MSVC to correctly count leading zeros
- [#1444](https://gitlab.com/libeigen/eigen/-/merge_requests/1444): Fixed index type handling in `StorageIndex` to prevent overflow during resize operations in Eigen::SPQR module
- [#1447](https://gitlab.com/libeigen/eigen/-/merge_requests/1447): Addressed multiple AddressSanitizer (asan) errors including out-of-bounds, use-after-scope, and memory leak issues across various Eigen components
- [#1448](https://gitlab.com/libeigen/eigen/-/merge_requests/1448): Fixed MSAN failures by resolving uninitialized memory use in matrices
- [#1449](https://gitlab.com/libeigen/eigen/-/merge_requests/1449): Fixed GPU computation issue with Clang and AddressSanitizer by replacing function pointers with lambdas
- [#1451](https://gitlab.com/libeigen/eigen/-/merge_requests/1451): Fixed build error in SPQR module due to StorageIndex and Index type mismatch
- [#1456](https://gitlab.com/libeigen/eigen/-/merge_requests/1456): Improved memory safety by adding pointer checks before freeing to prevent potential undefined behavior
- [#1458](https://gitlab.com/libeigen/eigen/-/merge_requests/1458): Fixed `stableNorm` function to handle zero-sized input correctly
- [#1467](https://gitlab.com/libeigen/eigen/-/merge_requests/1467): Fixed compile-time error related to chip static assertions for dimension checks
- [#1468](https://gitlab.com/libeigen/eigen/-/merge_requests/1468): Addressed ARM32 floating-point computation issues by improving `Eigen::half` handling and numerical precision
- [#1476](https://gitlab.com/libeigen/eigen/-/merge_requests/1476): Fixed multiple One Definition Rule (ODR) violations across several Eigen library components
- [#1478](https://gitlab.com/libeigen/eigen/-/merge_requests/1478): Fixed comparison bug in detection of subnormal floating-point numbers
- [#1481](https://gitlab.com/libeigen/eigen/-/merge_requests/1481): Fixed CI compatibility issues for clang-6 during cross-compilation
- [#1482](https://gitlab.com/libeigen/eigen/-/merge_requests/1482): Fixed `preshear` transformation function implementation with corrected constructor and added verification test
- [#1485](https://gitlab.com/libeigen/eigen/-/merge_requests/1485): Fixed PPC architecture test failures related to random integer value ranges and signed integer overflow
- [#1486](https://gitlab.com/libeigen/eigen/-/merge_requests/1486): Fixed gcc-6 compiler bug in rand test by adding `noinline` attribute to preserve variable value
- [#1487](https://gitlab.com/libeigen/eigen/-/merge_requests/1487): Fixed skew symmetric matrix test by excluding problematic dimension cases to improve test reliability
- [#1488](https://gitlab.com/libeigen/eigen/-/merge_requests/1488): Fixed test compatibility issues with bfloat16 and half scalar types
- [#1489](https://gitlab.com/libeigen/eigen/-/merge_requests/1489): Fixed undefined behavior in `getRandomBits` when generating random values with zero bits
- [#1490](https://gitlab.com/libeigen/eigen/-/merge_requests/1490): Fixed undefined behavior in boolean packetmath test by correcting select mask loading
- [#1492](https://gitlab.com/libeigen/eigen/-/merge_requests/1492): Fixes C++20 compilation error related to arithmetic between different enumeration types
- [#1494](https://gitlab.com/libeigen/eigen/-/merge_requests/1494): Prevent segmentation fault in `CholmodBase::factorize()` when handling zero matrices
- [#1496](https://gitlab.com/libeigen/eigen/-/merge_requests/1496): Fixed division by zero undefined behavior in packet size logic
- [#1498](https://gitlab.com/libeigen/eigen/-/merge_requests/1498): Removes `r_cnjg` function to resolve conflicts with libf2c and inlines related complex conjugate functions
- [#1499](https://gitlab.com/libeigen/eigen/-/merge_requests/1499): Eliminated warning about writing bytes directly to non-trivial type using `void*` casting
- [#1500](https://gitlab.com/libeigen/eigen/-/merge_requests/1500): Fixes explicit scalar conversion issue in ternary expressions, resolving bug #2780
- [#1504](https://gitlab.com/libeigen/eigen/-/merge_requests/1504): Fixed undefined behavior in `pabsdiff` function on ARM to prevent compiler overflow issues
- [#1507](https://gitlab.com/libeigen/eigen/-/merge_requests/1507): Fixed deflation process in BDCSVD to improve numerical stability and correctness when handling large constant matrices
- [#1510](https://gitlab.com/libeigen/eigen/-/merge_requests/1510): Fixed potential infinite loop in real Schur decomposition and improved polynomial solver reliability
- [#1513](https://gitlab.com/libeigen/eigen/-/merge_requests/1513): Fixed pexp_complex_test to improve C++ standard compliance
- [#1514](https://gitlab.com/libeigen/eigen/-/merge_requests/1514): Fix exp complex test by using `int` instead of index type
- [#1517](https://gitlab.com/libeigen/eigen/-/merge_requests/1517): Fixed uninitialized memory usage in kronecker_product test by properly initializing matrices
- [#1518](https://gitlab.com/libeigen/eigen/-/merge_requests/1518): Fixed header guard inconsistencies in `GeneralMatrixMatrix.h` and `Parallelizer.h` to resolve build errors
- [#1521](https://gitlab.com/libeigen/eigen/-/merge_requests/1521): Fix crash in Incomplete Cholesky algorithm when input matrix has zeros on diagonal
- [#1524](https://gitlab.com/libeigen/eigen/-/merge_requests/1524): Fixed signed integer undefined behavior in random number generation functionality
- [#1526](https://gitlab.com/libeigen/eigen/-/merge_requests/1526): Fix build issue with MSVC GPU compilation by resolving `allocate()` function definition conflict
- [#1528](https://gitlab.com/libeigen/eigen/-/merge_requests/1528): Fixed QR colpivoting warnings by replacing `abs` with `numext::abs` to handle floating-point types correctly
- [#1529](https://gitlab.com/libeigen/eigen/-/merge_requests/1529): Fix triangular matrix-vector multiplication uninitialized warning by removing `const_cast` and simplifying implementation
- [#1531](https://gitlab.com/libeigen/eigen/-/merge_requests/1531): Add degenerate checks before calling BLAS routines to prevent crashes with zero-sized matrices/vectors
- [#1533](https://gitlab.com/libeigen/eigen/-/merge_requests/1533): Fixed edge-cases and test failures for complex `pexp` function
- [#1535](https://gitlab.com/libeigen/eigen/-/merge_requests/1535): Resolved deprecated anonymous enum-enum conversion warnings to improve code quality and compiler compatibility
- [#1536](https://gitlab.com/libeigen/eigen/-/merge_requests/1536): Fixed unaligned memory access in `trmv` function, resolving `nomalloc_3` test failure
- [#1537](https://gitlab.com/libeigen/eigen/-/merge_requests/1537): Fixed static_assert compatibility for C++14, improving compilation error clarity
- [#1538](https://gitlab.com/libeigen/eigen/-/merge_requests/1538): Fixes volume calculation for empty `AlignedBox` to return 0 instead of a negative value
- [#1540](https://gitlab.com/libeigen/eigen/-/merge_requests/1540): Fixed pexp test for 32-bit ARM architectures to handle subnormal number flushing
- [#1541](https://gitlab.com/libeigen/eigen/-/merge_requests/1541): Fixed `packetmath` plog test compatibility on Windows by updating comparison method using `numext::log`
- [#1549](https://gitlab.com/libeigen/eigen/-/merge_requests/1549): Fix const access in `CwiseUnaryView` with improved matrix mutability checks
- [#1551](https://gitlab.com/libeigen/eigen/-/merge_requests/1551): Resolved VS2015 compilation issue by adding explicit `static_cast` to handle `bool(...)` casting
- [#1552](https://gitlab.com/libeigen/eigen/-/merge_requests/1552): Fixed `CwiseUnaryView` compatibility with MSVC compiler by resolving default parameter declaration issues
- [#1559](https://gitlab.com/libeigen/eigen/-/merge_requests/1559): Fix SIMD intrinsics compatibility for 32-bit builds by introducing workarounds for `_mm_cvtsi128_si64` and `_mm_extract_epi64`
- [#1562](https://gitlab.com/libeigen/eigen/-/merge_requests/1562): Protect use of `alloca` to prevent breakages on 32-bit ARM systems
- [#1566](https://gitlab.com/libeigen/eigen/-/merge_requests/1566): Fixed `Packet2l` handling on Windows 32-bit platforms
- [#1567](https://gitlab.com/libeigen/eigen/-/merge_requests/1567): Fixes double to int64 conversion on 32-bit SSE architecture and adds Windows build smoketests
- [#1568](https://gitlab.com/libeigen/eigen/-/merge_requests/1568): Fix redefinition of `ScalarPrinter` for gcc compilation compatibility
- [#1570](https://gitlab.com/libeigen/eigen/-/merge_requests/1570): Fixed casting from `Packet2d` to `Packet2l` to use truncation instead of rounding
- [#1573](https://gitlab.com/libeigen/eigen/-/merge_requests/1573): Fixed compiler warnings related to unsigned type negation and type casting on MSVC
- [#1574](https://gitlab.com/libeigen/eigen/-/merge_requests/1574): Guarded AVX `Packet4l` definition to prevent potential compilation conflicts and improve stability
- [#1576](https://gitlab.com/libeigen/eigen/-/merge_requests/1576): Fixed preprocessor condition for fast float logistic implementation, restoring optimal performance by correcting `EIGEN_CPUCC` handling
- [#1577](https://gitlab.com/libeigen/eigen/-/merge_requests/1577): Fixed `preverse` function implementation for PowerPC architecture
- [#1585](https://gitlab.com/libeigen/eigen/-/merge_requests/1585): Fixed GCC bug handling `pfirst<Packet16i>` AVX512 intrinsic
- [#1588](https://gitlab.com/libeigen/eigen/-/merge_requests/1588): Fixed build compatibility for pblend, psin_double, and pcos_double when AVX is supported but AVX2 is not
- [#1591](https://gitlab.com/libeigen/eigen/-/merge_requests/1591): Fixes compilation problems with `PacketI` on PowerPC architecture
- [#1594](https://gitlab.com/libeigen/eigen/-/merge_requests/1594): Fix `tridiagonalization_inplace_selector::run()` method compatibility with CUDA by adding `EIGEN_DEVICE_FUNC` macro
- [#1598](https://gitlab.com/libeigen/eigen/-/merge_requests/1598): Fixed transposed matrix product memory allocation bug when using `noalias()`
- [#1601](https://gitlab.com/libeigen/eigen/-/merge_requests/1601): Fixed sine and cosine function implementation for PowerPC platforms
- [#1602](https://gitlab.com/libeigen/eigen/-/merge_requests/1602): Adjusted error bound for nonlinear tests with AVX to maintain accurate algorithm convergence
- [#1604](https://gitlab.com/libeigen/eigen/-/merge_requests/1604): Fixed AVX512 `preduce_mul` implementation for MSVC to correctly handle negative outputs
- [#1606](https://gitlab.com/libeigen/eigen/-/merge_requests/1606): Fixed undefined behavior in predux_mul test input generation to prevent signed integer overflow
- [#1607](https://gitlab.com/libeigen/eigen/-/merge_requests/1607): Fixed hard-coded magic bounds in nonlinear tests for improved cross-platform reliability
- [#1610](https://gitlab.com/libeigen/eigen/-/merge_requests/1610): Fixed generic nearest integer operations for GPU compatibility and performance
- [#1611](https://gitlab.com/libeigen/eigen/-/merge_requests/1611): Fixed CMake package include path configuration to ensure correct include directory setup
- [#1614](https://gitlab.com/libeigen/eigen/-/merge_requests/1614): Fix FFT functionality when destination does not have unit stride
- [#1616](https://gitlab.com/libeigen/eigen/-/merge_requests/1616): Fixed GCC 6 compilation error by removing namespace prefixes from struct specializations
- [#1622](https://gitlab.com/libeigen/eigen/-/merge_requests/1622): Fixed undefined behavior sanitizer (ubsan) failures in `array_for_matrix` with integer type handling
- [#1628](https://gitlab.com/libeigen/eigen/-/merge_requests/1628): Fixed threading tests by adjusting header inclusion order and resolving C++20 capture warnings
- [#1630](https://gitlab.com/libeigen/eigen/-/merge_requests/1630): Resolved warnings about repeated macro definitions to improve code reliability
- [#1631](https://gitlab.com/libeigen/eigen/-/merge_requests/1631): Fixed multiple GCC warnings related to enum comparisons, improving code clarity and maintainability
- [#1633](https://gitlab.com/libeigen/eigen/-/merge_requests/1633): Resolved compiler warnings introduced by previous warning fixes
- [#1635](https://gitlab.com/libeigen/eigen/-/merge_requests/1635): Fixed deprecated enum comparison warnings by improving type-safe comparisons
- [#1639](https://gitlab.com/libeigen/eigen/-/merge_requests/1639): Resolve AVX512FP16 build failure by implementing vectorized cast specializations for `packet16h` and `packet16f`
- [#1648](https://gitlab.com/libeigen/eigen/-/merge_requests/1648): Fix overflow warnings in `PacketMathFP16` by adding explicit `short` casts
- [#1649](https://gitlab.com/libeigen/eigen/-/merge_requests/1649): Fix compiler `-Wmaybe-uninitialized` warnings in BDCSVD by using placement new for object initialization
- [#1650](https://gitlab.com/libeigen/eigen/-/merge_requests/1650): Removes incorrect C++23 check for suppressing `has_denorm` deprecation warnings in MSVC
- [#1651](https://gitlab.com/libeigen/eigen/-/merge_requests/1651): Fixes compilation issues with `Eigen::half` to `_Float16` conversion in AVX512 code using `bit_cast` and user-defined literals
- [#1658](https://gitlab.com/libeigen/eigen/-/merge_requests/1658): Fixed pi definition in kissfft module to improve computational precision
- [#1679](https://gitlab.com/libeigen/eigen/-/merge_requests/1679): Addressed `Wmaybe-uninitialized` warnings in BDCSVD module, improving memory safety and code reliability
- [#1685](https://gitlab.com/libeigen/eigen/-/merge_requests/1685): Fixed out-of-range argument handling for `_mm_permute_pd` function to prevent undefined behavior
- [#1688](https://gitlab.com/libeigen/eigen/-/merge_requests/1688): Fixed bug in `atanh` function for input value -1, improving numerical stability and accuracy
- [#1690](https://gitlab.com/libeigen/eigen/-/merge_requests/1690): Fixed bug in `atanh` function implementation to improve accuracy and reliability
- [#1693](https://gitlab.com/libeigen/eigen/-/merge_requests/1693): Fix generic SSE2 ceil implementation for negative numbers near zero
- [#1697](https://gitlab.com/libeigen/eigen/-/merge_requests/1697): Removed unneeded `_mm_setzero_si128` function call, addressing issue #2858 and potentially improving code efficiency
- [#1699](https://gitlab.com/libeigen/eigen/-/merge_requests/1699): Fixed compiler warning in `EigenSolver::pseudoEigenvalueMatrix()` for matrix dimension handling
- [#1707](https://gitlab.com/libeigen/eigen/-/merge_requests/1707): Fix `erf(x)` computation to avoid NaN for large input values
- [#1708](https://gitlab.com/libeigen/eigen/-/merge_requests/1708): Fixed `atan` test for 32-bit ARM architecture by adjusting handling of flush-to-zero behavior
- [#1711](https://gitlab.com/libeigen/eigen/-/merge_requests/1711): Fixed compilation bug in `DenseBase::tail` for dynamic template arguments, improving function flexibility
- [#1718](https://gitlab.com/libeigen/eigen/-/merge_requests/1718): Fixed out-of-bounds access in triangular matrix multiplication code, improving safety and reliability
- [#1720](https://gitlab.com/libeigen/eigen/-/merge_requests/1720): Fixed NVCC build issues for CUDA 10+ by resolving warnings and assignment operator problems
- [#1722](https://gitlab.com/libeigen/eigen/-/merge_requests/1722): Fixed matrix parameter passing issues affecting internal data alignment in GCC arm environment
- [#1723](https://gitlab.com/libeigen/eigen/-/merge_requests/1723): Fixes compiler-specific issues with clang6 optimization, addressing problems in `small_product_5`, `cross3`, and SSE `pabs` functions
- [#1724](https://gitlab.com/libeigen/eigen/-/merge_requests/1724): Removes default FFT macros from CMake test declarations to eliminate macro redefinition warnings in FFTW tests
- [#1725](https://gitlab.com/libeigen/eigen/-/merge_requests/1725): Fixed clang6 and ARM architecture compatibility by modifying SSE instruction handling
- [#1726](https://gitlab.com/libeigen/eigen/-/merge_requests/1726): Fixed GPU builds by adding initializers for `constexpr` global variables to ensure CUDA compatibility
- [#1742](https://gitlab.com/libeigen/eigen/-/merge_requests/1742): Fix compilation issue in `Assign_MKL.h` by casting enum to `int` for comparison
- [#1760](https://gitlab.com/libeigen/eigen/-/merge_requests/1760): Fixed undefined behavior in `setZero` for null destination arrays and zero-sized blocks
- [#1761](https://gitlab.com/libeigen/eigen/-/merge_requests/1761): Fixed map fill logic to support more flexible stride configurations, including 0/0 stride and outer stride equal to underlying inner size
- [#1762](https://gitlab.com/libeigen/eigen/-/merge_requests/1762): Fixed IOFormat alignment computation for more consistent matrix output
- [#1764](https://gitlab.com/libeigen/eigen/-/merge_requests/1764): Fixed CI checkformat stage by updating base Ubuntu image to latest LTS version
- [#1769](https://gitlab.com/libeigen/eigen/-/merge_requests/1769): Fixes special packetmath `erfc` flushing for ARM32 architecture, improving subnormal number handling
- [#1785](https://gitlab.com/libeigen/eigen/-/merge_requests/1785): Fixes build issue by adding missing `#include <new>` header
- [#1790](https://gitlab.com/libeigen/eigen/-/merge_requests/1790): Fix uninitialized threshold read in `SparseQR::factorize()` method to improve code safety
- [#1792](https://gitlab.com/libeigen/eigen/-/merge_requests/1792): Resolves `std::fill_n` reference issue by removing `EIGEN_USING_STD` to prevent namespace conflicts
- [#1793](https://gitlab.com/libeigen/eigen/-/merge_requests/1793): Zero-initialize test arrays to prevent uninitialized memory reads and improve test reliability
- [#1799](https://gitlab.com/libeigen/eigen/-/merge_requests/1799): Fixed task retrieval logic in `NonBlockingThreadPool` to correctly enable task stealing between threads
- [#1802](https://gitlab.com/libeigen/eigen/-/merge_requests/1802): Fixed initialization order and removed unused variables in `NonBlockingThreadPool.h`
- [#1804](https://gitlab.com/libeigen/eigen/-/merge_requests/1804): Fixes potential data race on `spin_count_` in `NonBlockingThreadPool` by making it `const` and properly initializing it
- [#1806](https://gitlab.com/libeigen/eigen/-/merge_requests/1806): Fixed UTF-8 encoding errors in `SimplicialCholesky_impl.h` causing compilation issues in MSVC and Apple Clang
- [#1810](https://gitlab.com/libeigen/eigen/-/merge_requests/1810): Fixed midpoint calculation in `Eigen::ForkJoinScheduler` to prevent out-of-bounds errors and improve parallel computation reliability
- [#1814](https://gitlab.com/libeigen/eigen/-/merge_requests/1814): Added missing return statements in PowerPC architecture implementation to improve code reliability
- [#1816](https://gitlab.com/libeigen/eigen/-/merge_requests/1816): Fix Android NDK compatibility issue with `__cpp_lib_hardware_interference_size` macro
- [#1825](https://gitlab.com/libeigen/eigen/-/merge_requests/1825): Eliminate type-punning undefined behavior in `Eigen::half` by using safer bit-cast approach
- [#1831](https://gitlab.com/libeigen/eigen/-/merge_requests/1831): Fixed Power architecture builds for configurations without VSX and POWER8 support
- [#1833](https://gitlab.com/libeigen/eigen/-/merge_requests/1833): Fixed `Warray-bounds` warning in inner product implementation, preventing potential array access out-of-bounds errors
- [#1834](https://gitlab.com/libeigen/eigen/-/merge_requests/1834): Fixed uninitialized matrix elements in bicgstab test to improve test reliability
- [#1835](https://gitlab.com/libeigen/eigen/-/merge_requests/1835): Resolved bitwise operation compilation error when compiling with C++26
- [#1841](https://gitlab.com/libeigen/eigen/-/merge_requests/1841): Fixed documentation job configuration for nightly builds
- [#1842](https://gitlab.com/libeigen/eigen/-/merge_requests/1842): Resolved CMake BOOST warning by updating configuration to eliminate deprecated behavior
- [#1850](https://gitlab.com/libeigen/eigen/-/merge_requests/1850): Fixed x86 complex vectorized FMA implementation to improve computational accuracy and performance
- [#1851](https://gitlab.com/libeigen/eigen/-/merge_requests/1851): Fixed implementation of Givens rotation algorithm to improve accuracy and reliability

##### Improved

- [#544](https://gitlab.com/libeigen/eigen/-/merge_requests/544): Added GDB pretty printer support for Eigen::Block types to improve debugging experience
- [#572](https://gitlab.com/libeigen/eigen/-/merge_requests/572): Removed unnecessary `const` qualifiers from AutodiffScalar return types to improve code quality and readability
- [#605](https://gitlab.com/libeigen/eigen/-/merge_requests/605): Updated SparseExtra RandomSetter to use unordered_map for improved performance
- [#609](https://gitlab.com/libeigen/eigen/-/merge_requests/609): Optimize predux, predux_min, and predux_max operations for AArch64 architecture using specialized intrinsics
- [#615](https://gitlab.com/libeigen/eigen/-/merge_requests/615): Adds intrin header for Windows ARM to improve compatibility and intrinsic function support
- [#617](https://gitlab.com/libeigen/eigen/-/merge_requests/617): Extended matrixmarket reader/writer to support handling of dense matrices
- [#618](https://gitlab.com/libeigen/eigen/-/merge_requests/618): Added EIGEN_DEVICE_FUNC labels to improve CUDA 9 compatibility for gpu_basic tests
- [#631](https://gitlab.com/libeigen/eigen/-/merge_requests/631): Introduced error handling to prevent direct inclusion of internal Eigen headers
- [#632](https://gitlab.com/libeigen/eigen/-/merge_requests/632): Simplified CMake configuration by removing unused interface definitions
- [#633](https://gitlab.com/libeigen/eigen/-/merge_requests/633): Simplified CMake versioning for architecture-independent package configurations using `ARCH_INDEPENDENT` option
- [#634](https://gitlab.com/libeigen/eigen/-/merge_requests/634): Improved CMake package registry configuration for better dependency management
- [#641](https://gitlab.com/libeigen/eigen/-/merge_requests/641): Removed unnecessary std::tuple reference to simplify codebase
- [#647](https://gitlab.com/libeigen/eigen/-/merge_requests/647): Cleaned up EIGEN_STATIC_ASSERT to use standard C++11 static_assert, improving error messages and code organization
- [#648](https://gitlab.com/libeigen/eigen/-/merge_requests/648): Corrected typographical errors in copyright dates across project files
- [#652](https://gitlab.com/libeigen/eigen/-/merge_requests/652): Added a macro to pass arguments to ctest for running tests in parallel
- [#655](https://gitlab.com/libeigen/eigen/-/merge_requests/655): Improved CI test execution by running tests in parallel across all available CPU cores
- [#660](https://gitlab.com/libeigen/eigen/-/merge_requests/660): Corrected multiple typos in documentation and comments to improve code clarity and readability
- [#661](https://gitlab.com/libeigen/eigen/-/merge_requests/661): Corrected typographical errors in documentation and code comments
- [#662](https://gitlab.com/libeigen/eigen/-/merge_requests/662): Reorganized test main file for improved maintainability and code structure
- [#663](https://gitlab.com/libeigen/eigen/-/merge_requests/663): Reduced CUDA compilation warnings for versions 9.2 and 11.4
- [#668](https://gitlab.com/libeigen/eigen/-/merge_requests/668): Updated CMake Windows compiler and OS detection with more reliable and maintainable methods
- [#677](https://gitlab.com/libeigen/eigen/-/merge_requests/677): Optimized type punning in CUDA code by replacing memcpy with reinterpret_cast for improved GPU performance
- [#687](https://gitlab.com/libeigen/eigen/-/merge_requests/687): Adds nan-propagation options to elementwise min/max operations and reductions in matrix and array plugins
- [#692](https://gitlab.com/libeigen/eigen/-/merge_requests/692): Extend Eigen's Qt support to Qt6 by modifying compatibility functions in Transform.h
- [#693](https://gitlab.com/libeigen/eigen/-/merge_requests/693): Enhanced documentation for Stride class inner stride behavior in compile-time vectors
- [#697](https://gitlab.com/libeigen/eigen/-/merge_requests/697): Optimize CMake scripts to improve Eigen subproject integration and reduce default test build overhead
- [#700](https://gitlab.com/libeigen/eigen/-/merge_requests/700): Vectorized tanh and logistic functions for fp16 on Neon, improving computational performance
- [#701](https://gitlab.com/libeigen/eigen/-/merge_requests/701): Move alignment qualifier to improve consistency and resolve compiler warnings
- [#712](https://gitlab.com/libeigen/eigen/-/merge_requests/712): Improved documentation for Quaternion constructor from MatrixBase, clarifying element order and usage
- [#716](https://gitlab.com/libeigen/eigen/-/merge_requests/716): Converted diagnostic pragmas to standardized nv_diag format, improving code consistency and maintainability
- [#717](https://gitlab.com/libeigen/eigen/-/merge_requests/717): Moved pruning code from CompressedStorage to SparseVector.h to improve code organization
- [#718](https://gitlab.com/libeigen/eigen/-/merge_requests/718): Update SparseMatrix::Map and TransposedSparseMatrix to use consistent StorageIndex across implementations
- [#720](https://gitlab.com/libeigen/eigen/-/merge_requests/720): Fixed a documentation typo to improve clarity
- [#722](https://gitlab.com/libeigen/eigen/-/merge_requests/722): Optimized Umeyama algorithm computation by conditionally skipping unnecessary scaling calculations
- [#726](https://gitlab.com/libeigen/eigen/-/merge_requests/726): Added basic iterator support for Eigen::array to simplify array usage and transition from std::array
- [#727](https://gitlab.com/libeigen/eigen/-/merge_requests/727): Made numeric_limits members constexpr for improved compile-time evaluation
- [#734](https://gitlab.com/libeigen/eigen/-/merge_requests/734): Improved AVX2 optimization selection for non-multiple-of-8 data sizes
- [#735](https://gitlab.com/libeigen/eigen/-/merge_requests/735): Simplified C++11 feature checks by removing redundant macros and compiler version checks
- [#737](https://gitlab.com/libeigen/eigen/-/merge_requests/737): Refactored Lapacke LLT macro binding to improve code clarity and maintainability
- [#748](https://gitlab.com/libeigen/eigen/-/merge_requests/748): Improved Lapacke bindings for HouseholderQR and PartialPivLU by replacing macros with C++ code and extracting common binding logic
- [#753](https://gitlab.com/libeigen/eigen/-/merge_requests/753): Convert computational macros to type-safe constexpr functions for improved code quality
- [#756](https://gitlab.com/libeigen/eigen/-/merge_requests/756): Conditional inclusion of <atomic> header to improve compatibility with toolchains lacking atomic operations support
- [#757](https://gitlab.com/libeigen/eigen/-/merge_requests/757): Refactored IDRS code, replacing `norm()` with `StableNorm()` to improve code stability and numerical performance
- [#760](https://gitlab.com/libeigen/eigen/-/merge_requests/760): Removed `using namespace Eigen` from sample code to promote better coding practices
- [#761](https://gitlab.com/libeigen/eigen/-/merge_requests/761): Cleanup of obsolete compiler checks and flags, streamlining the codebase and reducing maintenance overhead
- [#763](https://gitlab.com/libeigen/eigen/-/merge_requests/763): Cleaned up CMake scripts by removing deprecated `COMPILE_FLAGS` and adopting modern `target_compile_options`
- [#767](https://gitlab.com/libeigen/eigen/-/merge_requests/767): Improved `exp()` function behavior for `-Inf` arguments in vectorized expressions with performance optimizations
- [#772](https://gitlab.com/libeigen/eigen/-/merge_requests/772): Cleanup of internal macros and sequence implementations to simplify codebase
- [#773](https://gitlab.com/libeigen/eigen/-/merge_requests/773): Optimized row-major sparse-dense matrix product implementation with two accumulation variables to improve computational efficiency
- [#774](https://gitlab.com/libeigen/eigen/-/merge_requests/774): Fixes for enabling HIP unit tests and updating CMake compatibility
- [#776](https://gitlab.com/libeigen/eigen/-/merge_requests/776): Improved CMake handling of `EIGEN_TEST_CUSTOM_CXX_FLAGS` by converting spaces to semicolons
- [#779](https://gitlab.com/libeigen/eigen/-/merge_requests/779): Optimize `exp<float>()` with reduced polynomial degree, expanded denormal range, and 4% speedup for AVX2
- [#780](https://gitlab.com/libeigen/eigen/-/merge_requests/780): Improved accuracy and performance of logistic sigmoid function implementation, reducing maximum relative error and extending computational range
- [#783](https://gitlab.com/libeigen/eigen/-/merge_requests/783): Simplified `logical_xor()` implementation for bool types, improving code clarity and efficiency
- [#786](https://gitlab.com/libeigen/eigen/-/merge_requests/786): Small cleanup of GDB pretty printer code, improving code readability and maintenance
- [#788](https://gitlab.com/libeigen/eigen/-/merge_requests/788): Small documentation and code quality improvements, including fixing warnings and documentation formatting
- [#790](https://gitlab.com/libeigen/eigen/-/merge_requests/790): Added missing internal namespace qualifiers to vectorization logic tests
- [#791](https://gitlab.com/libeigen/eigen/-/merge_requests/791): Added support for Cray, Fujitsu, and Intel ICX compilers with new preprocessor macros
- [#792](https://gitlab.com/libeigen/eigen/-/merge_requests/792): Enables manual specification of inner and outer strides for CWiseUnaryView, enhancing stride control and flexibility
- [#795](https://gitlab.com/libeigen/eigen/-/merge_requests/795): Refactored identifiers to reduce usage of reserved names in compliance with C++ standard guidelines
- [#797](https://gitlab.com/libeigen/eigen/-/merge_requests/797): Adds bounds checking to Eigen serializer to improve data integrity and prevent out-of-bounds access
- [#799](https://gitlab.com/libeigen/eigen/-/merge_requests/799): Performance improvement for logarithm function with 20% speedup for float and better denormal handling
- [#813](https://gitlab.com/libeigen/eigen/-/merge_requests/813): Corrected and clarified documentation for Least Squares Conjugate Gradient (LSCG) solver, improving mathematical descriptions and user understanding
- [#814](https://gitlab.com/libeigen/eigen/-/merge_requests/814): Updated comments to remove references to outdated macro and improve code clarity
- [#816](https://gitlab.com/libeigen/eigen/-/merge_requests/816): Port EIGEN_OPTIMIZATION_BARRIER to support soft float ARM architectures
- [#819](https://gitlab.com/libeigen/eigen/-/merge_requests/819): Enhance clang warning suppressions by checking for supported warnings before applying suppressions
- [#821](https://gitlab.com/libeigen/eigen/-/merge_requests/821): Prevent unnecessary heap allocation in diagonal product by setting NestByRefBit for more efficient memory management
- [#825](https://gitlab.com/libeigen/eigen/-/merge_requests/825): Introduced utility functions to reduce floating-point warnings and improve comparison precision
- [#830](https://gitlab.com/libeigen/eigen/-/merge_requests/830): Removed documentation referencing obsolete C++98/C++03 standards
- [#832](https://gitlab.com/libeigen/eigen/-/merge_requests/832): Improved AVX512 math function consistency and ICC compatibility for more reliable mathematical computations
- [#836](https://gitlab.com/libeigen/eigen/-/merge_requests/836): Refined compiler-specific `maxpd` workaround to target only GCC<6.3
- [#838](https://gitlab.com/libeigen/eigen/-/merge_requests/838): Corrected definition of EIGEN_HAS_AVX512_MATH in PacketMath to improve AVX512 math capabilities
- [#841](https://gitlab.com/libeigen/eigen/-/merge_requests/841): Consolidated and improved generic implementations of psqrt and prsqrt functions with correct handling of special cases
- [#844](https://gitlab.com/libeigen/eigen/-/merge_requests/844): Updated MPL2 license link to use HTTPS for improved security
- [#845](https://gitlab.com/libeigen/eigen/-/merge_requests/845): Improved numeric_limits implementation to ensure One Definition Rule (ODR) compliance and enhance static data member definitions
- [#846](https://gitlab.com/libeigen/eigen/-/merge_requests/846): Optimize performance by returning alphas() and betas() vectors as const references
- [#849](https://gitlab.com/libeigen/eigen/-/merge_requests/849): Improved documentation for MatrixXNt and MatrixNXt matrix patterns and fixed documentation compilation issues
- [#850](https://gitlab.com/libeigen/eigen/-/merge_requests/850): Added descriptive comments to Matrix typedefs to improve Doxygen documentation
- [#854](https://gitlab.com/libeigen/eigen/-/merge_requests/854): Added scaling function overload to handle vector rvalue references, improving diagonal matrix creation from temporary vectors
- [#861](https://gitlab.com/libeigen/eigen/-/merge_requests/861): Improved FixedInt constexpr support and resolved potential ODR violations
- [#864](https://gitlab.com/libeigen/eigen/-/merge_requests/864): Cleaned up unnecessary EIGEN_UNUSED decorations to improve code clarity and maintainability
- [#869](https://gitlab.com/libeigen/eigen/-/merge_requests/869): Improved SYCL support by simplifying CMake configuration and enhancing compatibility with C++ versions
- [#872](https://gitlab.com/libeigen/eigen/-/merge_requests/872): Improved sqrt/rsqrt handling of denormal numbers and performance optimizations for AVX512
- [#879](https://gitlab.com/libeigen/eigen/-/merge_requests/879): Improved efficiency of any/all reduction operations for row-major matrix layouts
- [#884](https://gitlab.com/libeigen/eigen/-/merge_requests/884): Simplified non-convergence checks in NonLinearOptimization tests to improve test reliability across different architectures
- [#887](https://gitlab.com/libeigen/eigen/-/merge_requests/887): Enhance vectorization logic tests for improved cross-platform compatibility and test reliability
- [#888](https://gitlab.com/libeigen/eigen/-/merge_requests/888): Optimized least_square_conjugate_gradient() performance using .noalias() to reduce temporary allocations
- [#889](https://gitlab.com/libeigen/eigen/-/merge_requests/889): Introduced `construct_at` and `destroy_at` wrappers, improving code clarity and modernizing memory management practices throughout Eigen
- [#890](https://gitlab.com/libeigen/eigen/-/merge_requests/890): Removed duplicate IsRowMajor declaration to reduce compilation warnings and improve code clarity
- [#891](https://gitlab.com/libeigen/eigen/-/merge_requests/891): Optimized SVD test memory consumption by splitting and reducing test matrix sizes
- [#893](https://gitlab.com/libeigen/eigen/-/merge_requests/893): Adds new CMake configuration options for more flexible build control of Eigen library components
- [#895](https://gitlab.com/libeigen/eigen/-/merge_requests/895): Added move constructors to SparseSolverBase and IterativeSolverBase for improved solver object management
- [#903](https://gitlab.com/libeigen/eigen/-/merge_requests/903): Replaces enum with constexpr for floating point bit size calculations, reducing type casts and improving code readability
- [#904](https://gitlab.com/libeigen/eigen/-/merge_requests/904): Converted static const class members to constexpr for improved compile-time efficiency
- [#909](https://gitlab.com/libeigen/eigen/-/merge_requests/909): Removed outdated GCC-4 warning workarounds, simplifying and improving code maintainability
- [#913](https://gitlab.com/libeigen/eigen/-/merge_requests/913): PowerPC MMA build configuration enhancement with dynamic dispatch option
- [#921](https://gitlab.com/libeigen/eigen/-/merge_requests/921): Optimized visitor traversal for RowMajor inputs, improving matrix operation performance
- [#927](https://gitlab.com/libeigen/eigen/-/merge_requests/927): Update warning suppression techniques for improved compiler compatibility
- [#931](https://gitlab.com/libeigen/eigen/-/merge_requests/931): Re-enabled Aarch64 CI pipelines to improve testing and validation for Aarch64 architecture
- [#939](https://gitlab.com/libeigen/eigen/-/merge_requests/939): Improved LAPACK module code organization by removing `.cpp` file inclusions
- [#940](https://gitlab.com/libeigen/eigen/-/merge_requests/940): Reintroduced std::remove* aliases to restore compatibility with third-party libraries
- [#941](https://gitlab.com/libeigen/eigen/-/merge_requests/941): Improve scalar test_isApprox handling of inf/nan values
- [#943](https://gitlab.com/libeigen/eigen/-/merge_requests/943): Enhanced `constexpr` helper functions in `XprHelper.h` to improve compile-time computations and code clarity
- [#944](https://gitlab.com/libeigen/eigen/-/merge_requests/944): Converted metaprogramming utility to constexpr function for improved compile-time evaluation and code simplification
- [#947](https://gitlab.com/libeigen/eigen/-/merge_requests/947): Added partial loading, storing, gathering, and scattering packet operations to improve memory access efficiency and performance
- [#951](https://gitlab.com/libeigen/eigen/-/merge_requests/951): Optimized Power GEMV predux operations for MMA, reducing instruction count and improving compatibility with GCC
- [#952](https://gitlab.com/libeigen/eigen/-/merge_requests/952): Introduced workarounds to allow all tests to pass with `EIGEN_TEST_NO_EXPLICIT_VECTORIZATION` setting
- [#959](https://gitlab.com/libeigen/eigen/-/merge_requests/959): Improved AVX512 implementation with header file renaming and hardware capability restrictions
- [#960](https://gitlab.com/libeigen/eigen/-/merge_requests/960): Removed AVX512VL dependency in trsm function, improving compatibility across different AVX configurations
- [#962](https://gitlab.com/libeigen/eigen/-/merge_requests/962): Optimized Householder sequence block handling to eliminate unnecessary heap allocations and improve performance
- [#967](https://gitlab.com/libeigen/eigen/-/merge_requests/967): Optimized GEMM MMA with vector_pairs loading and improved predux GEMV performance
- [#968](https://gitlab.com/libeigen/eigen/-/merge_requests/968): Made diagonal matrix `cols()` and `rows()` methods constexpr to improve compile-time evaluation
- [#969](https://gitlab.com/libeigen/eigen/-/merge_requests/969): Conditionally add `uninstall` target to prevent CMake installation conflicts
- [#984](https://gitlab.com/libeigen/eigen/-/merge_requests/984): Removes executable flag from files to improve project file permission management
- [#985](https://gitlab.com/libeigen/eigen/-/merge_requests/985): Improved logical shift operation implementations and fixed typo in SVE/PacketMath.h
- [#989](https://gitlab.com/libeigen/eigen/-/merge_requests/989): Resolves C++20 comparison operator ambiguity in template comparisons
- [#994](https://gitlab.com/libeigen/eigen/-/merge_requests/994): Marks `index_remap` as `EIGEN_DEVICE_FUNC` to enable GPU expression reshaping
- [#997](https://gitlab.com/libeigen/eigen/-/merge_requests/997): Enhances AVX512 TRSM kernels memory management by using `alloca` when `EIGEN_NO_MALLOC` is requested
- [#998](https://gitlab.com/libeigen/eigen/-/merge_requests/998): Improved tanh and erf vectorized implementation for EIGEN_FAST_MATH in VSX architecture
- [#999](https://gitlab.com/libeigen/eigen/-/merge_requests/999): Update Householder.h to use numext::sqrt for improved custom type support
- [#1000](https://gitlab.com/libeigen/eigen/-/merge_requests/1000): Performance optimization for GEMV on Power10 architecture using more load and store vector pairs
- [#1002](https://gitlab.com/libeigen/eigen/-/merge_requests/1002): Addressed clang-tidy warnings by reformatting function definitions in headers and improving code clarity
- [#1006](https://gitlab.com/libeigen/eigen/-/merge_requests/1006): Improved AutoDiff module header management by including necessary Core dependencies
- [#1009](https://gitlab.com/libeigen/eigen/-/merge_requests/1009): Corrected Doxygen group usage to improve documentation clarity and structure
- [#1011](https://gitlab.com/libeigen/eigen/-/merge_requests/1011): Optimized pblend AVX implementation, reducing execution time by 24.84%
- [#1013](https://gitlab.com/libeigen/eigen/-/merge_requests/1013): Added compiler flag to enable/disable AVX512 GEBP kernels, improving configuration flexibility
- [#1016](https://gitlab.com/libeigen/eigen/-/merge_requests/1016): Resolved Emscripten header inclusion issue with `immintrin.h`
- [#1020](https://gitlab.com/libeigen/eigen/-/merge_requests/1020): Modify ConjugateGradient to use numext::sqrt for improved type compatibility
- [#1021](https://gitlab.com/libeigen/eigen/-/merge_requests/1021): Updated AccelerateSupport documentation for improved clarity and accuracy
- [#1026](https://gitlab.com/libeigen/eigen/-/merge_requests/1026): Vectorized sign operator for real types to enhance computational performance across different CPU architectures
- [#1031](https://gitlab.com/libeigen/eigen/-/merge_requests/1031): Eliminated bool bitwise warnings by refactoring code to use logical operations instead of bitwise operations
- [#1035](https://gitlab.com/libeigen/eigen/-/merge_requests/1035): Removed redundant FP16C checks for AVX512 intrinsics, improving performance for float-to-half and half-to-float conversions
- [#1040](https://gitlab.com/libeigen/eigen/-/merge_requests/1040): Specialized `psign<Packet8i>` for AVX2 with up to 79.45% performance improvement and removed vectorization of `psign<bool>`
- [#1043](https://gitlab.com/libeigen/eigen/-/merge_requests/1043): Vectorized implementation of pow for integer base and exponent types, improving performance and numerical robustness
- [#1050](https://gitlab.com/libeigen/eigen/-/merge_requests/1050): Added index-out-of-bounds assertions in IndexedView to improve error detection and library safety
- [#1052](https://gitlab.com/libeigen/eigen/-/merge_requests/1052): Improved CMake configuration by disabling default benchmark builds and fixing test dependencies with sparse libraries
- [#1054](https://gitlab.com/libeigen/eigen/-/merge_requests/1054): Fixed documentation typo in TutorialSparse.dox
- [#1056](https://gitlab.com/libeigen/eigen/-/merge_requests/1056): Reduced compiler warnings in test code to improve build output and code quality
- [#1058](https://gitlab.com/libeigen/eigen/-/merge_requests/1058): Added missing comparison operators for GPU packets, resolving CUDA build issues and improving GPU computation support
- [#1064](https://gitlab.com/libeigen/eigen/-/merge_requests/1064): Improved constexpr compatibility for g++-6 and C++20, addressing build errors and compiler-specific constraints
- [#1066](https://gitlab.com/libeigen/eigen/-/merge_requests/1066): Improved `pow()` function to allow mixed types with safe type promotions
- [#1075](https://gitlab.com/libeigen/eigen/-/merge_requests/1075): Optimized sign function for complex numbers by conditionally using generic vectorization
- [#1078](https://gitlab.com/libeigen/eigen/-/merge_requests/1078): Added macro to configure `nr` trait in GEBP kernel for NEON architecture, potentially improving matrix computation performance
- [#1079](https://gitlab.com/libeigen/eigen/-/merge_requests/1079): Optimize GEBP kernel compilation time and memory usage with EIGEN_IF_CONSTEXPR
- [#1083](https://gitlab.com/libeigen/eigen/-/merge_requests/1083): Reduced memory footprint of GEBP kernel for non-ARM targets to mitigate MSVC heap memory issues
- [#1084](https://gitlab.com/libeigen/eigen/-/merge_requests/1084): Vectorized implementation of atan() for double precision, improving computational efficiency
- [#1086](https://gitlab.com/libeigen/eigen/-/merge_requests/1086): Conditional vectorization of `atan<double>` for Altivec with VSX support
- [#1087](https://gitlab.com/libeigen/eigen/-/merge_requests/1087): Simplified range reduction strategy for `atan<float>()` with 20-40% speedup on x86 architectures
- [#1088](https://gitlab.com/libeigen/eigen/-/merge_requests/1088): Replaced standard `assert` with `eigen_assert` for improved consistency and assertion control
- [#1091](https://gitlab.com/libeigen/eigen/-/merge_requests/1091): Added macros to AttributeMacros to improve clang-format compatibility and code formatting
- [#1093](https://gitlab.com/libeigen/eigen/-/merge_requests/1093): Improved handling of NaN inputs in atan2 function to enhance mathematical computation reliability
- [#1095](https://gitlab.com/libeigen/eigen/-/merge_requests/1095): Refactored special values tests for pow and added new test for atan2, improving mathematical function testing
- [#1099](https://gitlab.com/libeigen/eigen/-/merge_requests/1099): Clarified documentation requirement that indices must be sorted to improve library usability
- [#1100](https://gitlab.com/libeigen/eigen/-/merge_requests/1100): Enhanced resizing capabilities for dynamic empty matrices, improving matrix dimension handling and flexibility
- [#1101](https://gitlab.com/libeigen/eigen/-/merge_requests/1101): Improved memory management by using 1-byte offset for address alignment in handmade allocation functions
- [#1102](https://gitlab.com/libeigen/eigen/-/merge_requests/1102): Add assertion to validate outer index array size in SparseMapBase, improving input validation and preventing potential runtime errors
- [#1109](https://gitlab.com/libeigen/eigen/-/merge_requests/1109): Removed unnecessary assert in SparseMapBase to improve flexibility in sparse matrix initialization
- [#1110](https://gitlab.com/libeigen/eigen/-/merge_requests/1110): Removed unused parameter name to improve code readability
- [#1111](https://gitlab.com/libeigen/eigen/-/merge_requests/1111): Fixed Neon vectorization issues to improve ARM architecture performance and compatibility
- [#1114](https://gitlab.com/libeigen/eigen/-/merge_requests/1114): Enhanced BiCGSTAB parameter initialization to support custom types
- [#1117](https://gitlab.com/libeigen/eigen/-/merge_requests/1117): Small cleanup of IDRS.h, removing unused variable and improving comment formatting
- [#1122](https://gitlab.com/libeigen/eigen/-/merge_requests/1122): Reduced compiler warnings in test files by addressing narrowing conversions and improving code quality
- [#1128](https://gitlab.com/libeigen/eigen/-/merge_requests/1128): Enables direct access for NestByValue construct, improving performance and usability
- [#1129](https://gitlab.com/libeigen/eigen/-/merge_requests/1129): Added BDCSVD LAPACKE binding for more flexible and efficient SVD computations
- [#1131](https://gitlab.com/libeigen/eigen/-/merge_requests/1131): Increased L2 and L3 cache sizes for Power10 architecture to improve matrix operation performance by 1.33X
- [#1134](https://gitlab.com/libeigen/eigen/-/merge_requests/1134): Optimized `equalspace` packet operation to improve performance and computational efficiency
- [#1135](https://gitlab.com/libeigen/eigen/-/merge_requests/1135): Improved divide by zero error handling for better cross-platform compatibility
- [#1136](https://gitlab.com/libeigen/eigen/-/merge_requests/1136): Reviewed and cleaned up compiler version checks to improve maintainability and compatibility
- [#1137](https://gitlab.com/libeigen/eigen/-/merge_requests/1137): Improved bfloat16 support by replacing std::signbit with numext::signbit
- [#1138](https://gitlab.com/libeigen/eigen/-/merge_requests/1138): Improved test coverage for numext::signbit function
- [#1139](https://gitlab.com/libeigen/eigen/-/merge_requests/1139): Adds comparison, `+=`, and `-=` operators to `CompressedStorageIterator` to improve iterator functionality
- [#1140](https://gitlab.com/libeigen/eigen/-/merge_requests/1140): Improved SparseLU implementation by updating dense GEMM kernel and fixing initialization bug in SparseLUTransposeView
- [#1141](https://gitlab.com/libeigen/eigen/-/merge_requests/1141): Enables NEON absolute value operations for unsigned integer types, improving performance for `.cwiseAbs()` operations
- [#1144](https://gitlab.com/libeigen/eigen/-/merge_requests/1144): Improved C++ version detection macros and CMake tests to enhance compatibility and reduce CI failures
- [#1145](https://gitlab.com/libeigen/eigen/-/merge_requests/1145): Improved bfloat16 product test thresholds to enhance comparison reliability
- [#1146](https://gitlab.com/libeigen/eigen/-/merge_requests/1146): Enabled additional NEON instructions including complex psqrt and plset operations
- [#1154](https://gitlab.com/libeigen/eigen/-/merge_requests/1154): Significantly improved Power10 MMA bfloat16 GEMM performance with up to 61X speedup
- [#1158](https://gitlab.com/libeigen/eigen/-/merge_requests/1158): Clarified help message for spbenchsolver to improve matrix file naming instructions
- [#1165](https://gitlab.com/libeigen/eigen/-/merge_requests/1165): Added missing `EIGEN_DEVICE_FUNC` in assertions, improved code compatibility and clarity
- [#1167](https://gitlab.com/libeigen/eigen/-/merge_requests/1167): Improved `ColPivHouseholderQR` move assignment to enhance compiler compatibility
- [#1169](https://gitlab.com/libeigen/eigen/-/merge_requests/1169): Replaced deprecated CMake generator expression `$<CONFIGURATION>` with `$<CONFIG>` to improve build system compatibility
- [#1172](https://gitlab.com/libeigen/eigen/-/merge_requests/1172): Refactored SparseMatrix.h to improve code consistency and readability by directly referencing class members
- [#1174](https://gitlab.com/libeigen/eigen/-/merge_requests/1174): Performance optimization for bfloat16 matrix-matrix multiplication in non-standard matrix dimensions
- [#1175](https://gitlab.com/libeigen/eigen/-/merge_requests/1175): Improved `atan2` implementation with better corner case handling and performance optimization
- [#1176](https://gitlab.com/libeigen/eigen/-/merge_requests/1176): Optimized mathematical packet operations including `atan`, `atan2`, `acos`, and binary/unary power computations
- [#1186](https://gitlab.com/libeigen/eigen/-/merge_requests/1186): Update to ForwardDeclarations.h for improved header organization and maintainability
- [#1190](https://gitlab.com/libeigen/eigen/-/merge_requests/1190): Standardized zero comparisons using VERIFY_IS_EQUAL macro for improved code consistency and reliability
- [#1191](https://gitlab.com/libeigen/eigen/-/merge_requests/1191): Improved LAPACKE configuration with better complex type handling and LAPACK library compatibility
- [#1192](https://gitlab.com/libeigen/eigen/-/merge_requests/1192): Improved EIGEN_DEVICE_FUNC compatibility across CUDA 10/11/12 versions and cleaned up warnings
- [#1198](https://gitlab.com/libeigen/eigen/-/merge_requests/1198): Replaced eigen_asserts with eigen_internal_asserts in Power module to reduce unnecessary error checking in release builds
- [#1199](https://gitlab.com/libeigen/eigen/-/merge_requests/1199): Added IWYU export pragmas to top-level headers to improve tooling compatibility and code maintainability
- [#1206](https://gitlab.com/libeigen/eigen/-/merge_requests/1206): Enhances type handling for complex numbers in ColPivHouseholderQR_LAPACKE.h using LAPACKe specializations
- [#1207](https://gitlab.com/libeigen/eigen/-/merge_requests/1207): Optimized psign implementation for floating point types with reduced computational complexity
- [#1214](https://gitlab.com/libeigen/eigen/-/merge_requests/1214): Optimized BF16 to F32 array conversions on Power architectures by reducing vector instructions
- [#1219](https://gitlab.com/libeigen/eigen/-/merge_requests/1219): Optimized `pasin_float` function with bit manipulation for improved performance and fixed `psqrt_complex` error handling
- [#1223](https://gitlab.com/libeigen/eigen/-/merge_requests/1223): Vectorized implementation of atanh, added atan<half> definition, and new unit tests for mathematical functions
- [#1224](https://gitlab.com/libeigen/eigen/-/merge_requests/1224): Added Packet int divide support for Power10 architecture, improving computational performance
- [#1226](https://gitlab.com/libeigen/eigen/-/merge_requests/1226): Improved performance of pow() on Skylake by using pmsub instruction in twoprod
- [#1230](https://gitlab.com/libeigen/eigen/-/merge_requests/1230): Eliminates EIGEN_HAS_AVX512_MATH workaround, simplifying AVX512 packet math implementation
- [#1232](https://gitlab.com/libeigen/eigen/-/merge_requests/1232): Introduced guard mechanism to manage `long double` usage on GPU devices, improving compilation compatibility
- [#1234](https://gitlab.com/libeigen/eigen/-/merge_requests/1234): Streamlined BLAS/LAPACK routine declarations by removing unused headers and improving file organization
- [#1241](https://gitlab.com/libeigen/eigen/-/merge_requests/1241): Improved CMake configuration to prevent unintended modifications when Eigen is a sub-project
- [#1242](https://gitlab.com/libeigen/eigen/-/merge_requests/1242): Optimized memory allocation during tridiagonalization for eigenvector computation
- [#1250](https://gitlab.com/libeigen/eigen/-/merge_requests/1250): Replaced instances of 'Lesser' with 'Less' to improve terminology consistency
- [#1251](https://gitlab.com/libeigen/eigen/-/merge_requests/1251): Minor code style improvement by adding newline to end of file
- [#1253](https://gitlab.com/libeigen/eigen/-/merge_requests/1253): Simplified packetmath specializations using a macro, improving code readability and maintainability across backends
- [#1259](https://gitlab.com/libeigen/eigen/-/merge_requests/1259): Reinstated and expanded deadcode checks to improve code quality and maintainability
- [#1262](https://gitlab.com/libeigen/eigen/-/merge_requests/1262): Limits build and link jobs on PowerPC to reduce out-of-memory issues
- [#1267](https://gitlab.com/libeigen/eigen/-/merge_requests/1267): Corrected various typographical errors to improve code readability and documentation quality
- [#1268](https://gitlab.com/libeigen/eigen/-/merge_requests/1268): Improved CMake argument parsing to support semi-colon separated lists for better build system compatibility
- [#1272](https://gitlab.com/libeigen/eigen/-/merge_requests/1272): Optimized casting performance for x86_64 architectures, with significant speedups in bool and float casting operations
- [#1274](https://gitlab.com/libeigen/eigen/-/merge_requests/1274): Optimize float->bool cast performance for AVX2 with significant speed improvements
- [#1275](https://gitlab.com/libeigen/eigen/-/merge_requests/1275): Added vectorized integer casts for x86 and removed redundant unit tests, improving performance by up to 66.77%
- [#1276](https://gitlab.com/libeigen/eigen/-/merge_requests/1276): Optimized `generic_rsqrt_newton_step` function, improving accuracy and performance of square root calculations
- [#1284](https://gitlab.com/libeigen/eigen/-/merge_requests/1284): Clean up packet math implementation by removing unused traits, adding missing specializations, and setting blend properties
- [#1286](https://gitlab.com/libeigen/eigen/-/merge_requests/1286): Improves type safety for non-const symbolic indexed view expressions by adding explicit l-value qualification
- [#1288](https://gitlab.com/libeigen/eigen/-/merge_requests/1288): Updated documentation for Eigen 3.4.x to improve build process and clarity
- [#1294](https://gitlab.com/libeigen/eigen/-/merge_requests/1294): Improved accuracy of `erf()` function with refined rational approximation and enhanced clamping methods
- [#1303](https://gitlab.com/libeigen/eigen/-/merge_requests/1303): Improved `Erf()` function performance and accuracy, ensuring +/-1 return values at clamping points with computational speed enhancements
- [#1305](https://gitlab.com/libeigen/eigen/-/merge_requests/1305): Enhanced `StridedLinearBufferCopy` with half-`Packet` operations to improve computational efficiency
- [#1313](https://gitlab.com/libeigen/eigen/-/merge_requests/1313): Added pmul and abs2 operations for Packet4ul in AVX2 implementation
- [#1316](https://gitlab.com/libeigen/eigen/-/merge_requests/1316): Implemented `pcmp`, `pmin`, and `pmax` functions for `Packet4ui` type in SSE to improve vectorization compliance
- [#1317](https://gitlab.com/libeigen/eigen/-/merge_requests/1317): Optimized F32 to BF16 conversions with loop unrolling, achieving 1.8X faster performance for LLVM and vector pair improvements for GCC
- [#1320](https://gitlab.com/libeigen/eigen/-/merge_requests/1320): Improved memory management for FFTW/IMKL FFT backends using `std::shared_ptr`
- [#1324](https://gitlab.com/libeigen/eigen/-/merge_requests/1324): Update `ndtri` function to return NaN for out-of-range input values, improving consistency with SciPy and MATLAB
- [#1325](https://gitlab.com/libeigen/eigen/-/merge_requests/1325): Renamed `array_cwise` test to prevent naming conflicts and suppressed compiler warnings
- [#1328](https://gitlab.com/libeigen/eigen/-/merge_requests/1328): Implements specialized vectorization for `scalar_cast_op` evaluator, enhancing performance and safety in casting operations
- [#1334](https://gitlab.com/libeigen/eigen/-/merge_requests/1334): Improved unrolled assignment evaluator with more consistent linear access methods for small fixed-size arrays and matrices
- [#1337](https://gitlab.com/libeigen/eigen/-/merge_requests/1337): Clean-up of Redux.h and vectorization_logic test to improve code readability and test reliability
- [#1338](https://gitlab.com/libeigen/eigen/-/merge_requests/1338): Optimized error handling for `scalar_unary_pow_op` with improved performance and robustness for integer base and exponent scenarios
- [#1342](https://gitlab.com/libeigen/eigen/-/merge_requests/1342): Optimized Newton-Raphson step for reciprocal square root, reducing max relative error from 3 to 2 ulps in floating-point calculations
- [#1346](https://gitlab.com/libeigen/eigen/-/merge_requests/1346): Introduced move constructor for `Ref<const...>` to improve performance and reduce unnecessary copying
- [#1352](https://gitlab.com/libeigen/eigen/-/merge_requests/1352): Improved `rint`, `round`, `floor`, and `ceil` mathematical functions for enhanced precision and performance
- [#1353](https://gitlab.com/libeigen/eigen/-/merge_requests/1353): Removed deprecated function calls in SVD test suite to improve code maintainability
- [#1354](https://gitlab.com/libeigen/eigen/-/merge_requests/1354): Added optional offset parameter to `ploadu_partial` and `pstoreu_partial` to improve API consistency
- [#1356](https://gitlab.com/libeigen/eigen/-/merge_requests/1356): Fixed compilation warning by unconditionally defining `EIGEN_HAS_ARM64_FP16_VECTOR_ARITHMETIC` on ARM architectures
- [#1358](https://gitlab.com/libeigen/eigen/-/merge_requests/1358): Addressed multiple compiler warnings across various modules through strategic code refactoring and type handling improvements
- [#1364](https://gitlab.com/libeigen/eigen/-/merge_requests/1364): Optimized `check_rows_cols_for_overflow` with partial template specialization for more efficient matrix size checks
- [#1365](https://gitlab.com/libeigen/eigen/-/merge_requests/1365): Added missing x86 primary casts for float, int, and double type conversions across SIMD instruction sets
- [#1372](https://gitlab.com/libeigen/eigen/-/merge_requests/1372): Enhanced Power architecture support with partial packet resolution, CPU improvements, DataMapper updates, and `bfloat16` type compatibility
- [#1373](https://gitlab.com/libeigen/eigen/-/merge_requests/1373): Adds `max_digits10` function to `Eigen::NumTraits` for improved floating-point decimal digit representation
- [#1378](https://gitlab.com/libeigen/eigen/-/merge_requests/1378): Improved handling of reference forwarding by replacing `std::move()` with `std::forward()` to address clang-tidy warning
- [#1381](https://gitlab.com/libeigen/eigen/-/merge_requests/1381): Update boost MP test suite to reference new SVD test cases
- [#1384](https://gitlab.com/libeigen/eigen/-/merge_requests/1384): Added IWYU private pragmas to internal headers to enhance tooling capabilities and header management
- [#1385](https://gitlab.com/libeigen/eigen/-/merge_requests/1385): Renamed non-standard plugin headers to use `.inc` extension for improved header management
- [#1391](https://gitlab.com/libeigen/eigen/-/merge_requests/1391): Exported `ThreadPool` symbols from legacy header to silence Clang include-cleaner warnings
- [#1393](https://gitlab.com/libeigen/eigen/-/merge_requests/1393): Update ROCm configuration to use `ROCM_PATH` for improved compatibility with ROCm 6.0
- [#1397](https://gitlab.com/libeigen/eigen/-/merge_requests/1397): Consolidated and simplified multiple implementations of divup/div_up/div_ceil functions
- [#1398](https://gitlab.com/libeigen/eigen/-/merge_requests/1398): Eliminated use of `_res` identifier to resolve macro conflicts and improve code compilation
- [#1400](https://gitlab.com/libeigen/eigen/-/merge_requests/1400): Modifies `div_ceil` function to pass arguments by value, reducing potential ODR-usage errors
- [#1401](https://gitlab.com/libeigen/eigen/-/merge_requests/1401): Fixed a typo in code comments to improve documentation clarity
- [#1404](https://gitlab.com/libeigen/eigen/-/merge_requests/1404): Improve build system by avoiding documentation builds during cross-compilation or non-top-level builds
- [#1413](https://gitlab.com/libeigen/eigen/-/merge_requests/1413): Improved `traits<Ref>::match` to correctly handle strides for contiguous memory layouts, eliminating unnecessary copying and enhancing `Ref` class efficiency
- [#1421](https://gitlab.com/libeigen/eigen/-/merge_requests/1421): Gemv microoptimization improving loop performance and reducing compilation warnings
- [#1424](https://gitlab.com/libeigen/eigen/-/merge_requests/1424): Optimized matrix-vector operations in `GeneralMatrixVector.h` for improved performance when `PacketSize` is a power of two
- [#1428](https://gitlab.com/libeigen/eigen/-/merge_requests/1428): Introduced clang-format in CI to ensure consistent code formatting and improve code maintainability
- [#1429](https://gitlab.com/libeigen/eigen/-/merge_requests/1429): Applied clang-format to entire Eigen codebase for consistent code style and improved maintainability
- [#1430](https://gitlab.com/libeigen/eigen/-/merge_requests/1430): Introduced `.git-blame-ignore-revs` file to improve git blame functionality for contributors
- [#1432](https://gitlab.com/libeigen/eigen/-/merge_requests/1432): Comprehensive clang-format-17 update across Eigen library, improving code consistency and readability
- [#1433](https://gitlab.com/libeigen/eigen/-/merge_requests/1433): Improved formatting of `.git-blame-ignore-revs` file for better Git blame operations
- [#1437](https://gitlab.com/libeigen/eigen/-/merge_requests/1437): Improved random number generation functionality for scalar types, addressing entropy limitations and enhancing randomness across platforms
- [#1438](https://gitlab.com/libeigen/eigen/-/merge_requests/1438): Improved documentation for SparseLU module, clarifying function relationships and usage
- [#1443](https://gitlab.com/libeigen/eigen/-/merge_requests/1443): Updated continuous integration testing framework to enhance testing processes and reliability
- [#1446](https://gitlab.com/libeigen/eigen/-/merge_requests/1446): Removed C++11-specific code from count trailing/leading zeros implementations to improve portability
- [#1452](https://gitlab.com/libeigen/eigen/-/merge_requests/1452): Minor documentation improvements for basic slicing examples
- [#1459](https://gitlab.com/libeigen/eigen/-/merge_requests/1459): Added `constexpr` qualifiers to improve compile-time evaluation capabilities
- [#1461](https://gitlab.com/libeigen/eigen/-/merge_requests/1461): Eliminated unused warnings in failtest, improving code quality and developer experience
- [#1469](https://gitlab.com/libeigen/eigen/-/merge_requests/1469): Removed explicit member function specialization to improve compiler compatibility
- [#1470](https://gitlab.com/libeigen/eigen/-/merge_requests/1470): Improved codebase formatting for better readability and consistency
- [#1471](https://gitlab.com/libeigen/eigen/-/merge_requests/1471): Update LAPACK CPU time function naming conventions for improved consistency
- [#1473](https://gitlab.com/libeigen/eigen/-/merge_requests/1473): Improved documentation for LAPACK's `second` and `dsecnd` functions to enhance user understanding
- [#1483](https://gitlab.com/libeigen/eigen/-/merge_requests/1483): Integrated `stableNorm()` in ComplexEigenSolver to improve numerical stability
- [#1491](https://gitlab.com/libeigen/eigen/-/merge_requests/1491): Applied clang-format to improve code consistency in lapack and blas directories
- [#1495](https://gitlab.com/libeigen/eigen/-/merge_requests/1495): Optimized JacobiSVD implementation by removing unnecessary member variables `m_scaledMatrix` and `m_adjoint`, improving memory efficiency and compiler optimization
- [#1505](https://gitlab.com/libeigen/eigen/-/merge_requests/1505): Disable float16 packet casting for native AVX512 f16 support, enhancing stability of packet operations
- [#1506](https://gitlab.com/libeigen/eigen/-/merge_requests/1506): Replaced `Matrix::Options` with `traits<Matrix>::Options` to improve consistency across Eigen object types
- [#1515](https://gitlab.com/libeigen/eigen/-/merge_requests/1515): Enhanced random number generation for custom float types with improved accuracy and reduced rounding bias
- [#1519](https://gitlab.com/libeigen/eigen/-/merge_requests/1519): Updates `array_size` result type from enum to `constexpr` to improve type safety and reduce compiler warnings
- [#1523](https://gitlab.com/libeigen/eigen/-/merge_requests/1523): Optimized SparseQR module performance, reducing computation time from 256 to 200 seconds
- [#1525](https://gitlab.com/libeigen/eigen/-/merge_requests/1525): Speed up sparse x dense dot product with optimization techniques and inline methods, reducing SparseQR computation time
- [#1527](https://gitlab.com/libeigen/eigen/-/merge_requests/1527): Removed shadowed typedefs to improve code clarity and maintainability
- [#1530](https://gitlab.com/libeigen/eigen/-/merge_requests/1530): Eliminate FindCUDA CMake warning to improve build configuration process
- [#1532](https://gitlab.com/libeigen/eigen/-/merge_requests/1532): Improved error message clarity for C++14 requirement
- [#1539](https://gitlab.com/libeigen/eigen/-/merge_requests/1539): Improves static vector allocation alignment in TRMV module, ensuring consistent memory alignment for fixed-sized vectors
- [#1543](https://gitlab.com/libeigen/eigen/-/merge_requests/1543): Improved incomplete Cholesky decomposition with new `findOrInsertCoeff` method and enhanced verification of sparse matrix operations
- [#1547](https://gitlab.com/libeigen/eigen/-/merge_requests/1547): Improved const input handling and C++20 compatibility in unary views by preserving const-ness and updating type trait implementation
- [#1557](https://gitlab.com/libeigen/eigen/-/merge_requests/1557): Improved documentation consistency for the Jacobi module by adjusting documentation tag placement
- [#1560](https://gitlab.com/libeigen/eigen/-/merge_requests/1560): Added `cwiseSquare` function and improved tests for element-wise matrix operations
- [#1561](https://gitlab.com/libeigen/eigen/-/merge_requests/1561): Removed unnecessary `extern C` declarations in CholmodSupport module, simplifying code and maintaining library compatibility
- [#1563](https://gitlab.com/libeigen/eigen/-/merge_requests/1563): Introduced custom formatting for complex numbers improving Numpy and Native compatibility
- [#1564](https://gitlab.com/libeigen/eigen/-/merge_requests/1564): Vectorization and MSVC compatibility improvements for `cross3_product` function in Eigen's core operations
- [#1569](https://gitlab.com/libeigen/eigen/-/merge_requests/1569): Optimized move constructors and assignment operators for `SparseMatrix` to improve performance during object transfers
- [#1571](https://gitlab.com/libeigen/eigen/-/merge_requests/1571): Improved compatibility between `Eigen::array` and `std::array`, preparing for C++17 transition
- [#1580](https://gitlab.com/libeigen/eigen/-/merge_requests/1580): Added support for `Packet8l` in AVX512 architecture, optimizing performance for specific instruction set operations
- [#1581](https://gitlab.com/libeigen/eigen/-/merge_requests/1581): Add `constexpr` qualifiers to accessors in `DenseBase`, `Quaternions`, and `Translations` to improve compile-time computation capabilities
- [#1582](https://gitlab.com/libeigen/eigen/-/merge_requests/1582): Refactored indexed view template definitions to improve MSVC 14.16 compatibility and eliminate parameter redefinition warnings
- [#1583](https://gitlab.com/libeigen/eigen/-/merge_requests/1583): Optimized `pexp` function performance with speed improvements up to 6% across different SIMD architectures
- [#1584](https://gitlab.com/libeigen/eigen/-/merge_requests/1584): Implemented performance optimizations for Intel `pblend` functionality using more efficient integer operations and simplified mask creation
- [#1590](https://gitlab.com/libeigen/eigen/-/merge_requests/1590): Introduced optimizations for `pblend` functionality with improved bitmask generation and auto-vectorization techniques
- [#1592](https://gitlab.com/libeigen/eigen/-/merge_requests/1592): Fixed psincos implementation for PowerPC and 32-bit ARM, improving vectorized trigonometric computations
- [#1595](https://gitlab.com/libeigen/eigen/-/merge_requests/1595): Update CI scripts with Windows compatibility improvements, AVX tests, and local CI environment scripts
- [#1605](https://gitlab.com/libeigen/eigen/-/merge_requests/1605): Removed unnecessary semicolons to improve code readability and maintainability
- [#1609](https://gitlab.com/libeigen/eigen/-/merge_requests/1609): Improved test reliability for eigenvector orthonormality by adjusting error tolerance for scaled matrices
- [#1613](https://gitlab.com/libeigen/eigen/-/merge_requests/1613): Improved 128-bit integer operations for MSVC by replacing `__uint128_t` with MSVC-supported functions
- [#1615](https://gitlab.com/libeigen/eigen/-/merge_requests/1615): Updated `predux` for PowerPC `Packet4i` to align summation behavior with other architectures
- [#1618](https://gitlab.com/libeigen/eigen/-/merge_requests/1618): Fixed grammatical error in Matrix class documentation
- [#1619](https://gitlab.com/libeigen/eigen/-/merge_requests/1619): Suppress C++23 deprecation warnings for `std::has_denorm` and `std::has_denorm_loss`
- [#1621](https://gitlab.com/libeigen/eigen/-/merge_requests/1621): Adds validation checks for indices in `SparseMatrix::insert` to improve robustness and error handling
- [#1623](https://gitlab.com/libeigen/eigen/-/merge_requests/1623): Reformatted `EIGEN_STATIC_ASSERT()` macro as a statement macro to improve code consistency and maintainability
- [#1624](https://gitlab.com/libeigen/eigen/-/merge_requests/1624): Improved memory allocation and pointer arithmetic in `aligned_alloca` function to enhance performance and code quality
- [#1625](https://gitlab.com/libeigen/eigen/-/merge_requests/1625): Utilizes `__builtin_alloca_with_align` to optimize memory allocation efficiency and potentially improve performance
- [#1629](https://gitlab.com/libeigen/eigen/-/merge_requests/1629): Vectorized implementation of `isfinite` and `isinf` functions for improved performance
- [#1632](https://gitlab.com/libeigen/eigen/-/merge_requests/1632): Vectorized `allFinite()` function with approximately 2.7x performance speedup on AVX-compatible hardware
- [#1640](https://gitlab.com/libeigen/eigen/-/merge_requests/1640): Fixed markdown formatting in README.md for improved readability
- [#1641](https://gitlab.com/libeigen/eigen/-/merge_requests/1641): Introduced AVX512F-based casting optimization from `double` to `int64_t` for enhanced performance
- [#1656](https://gitlab.com/libeigen/eigen/-/merge_requests/1656): Corrected multiple typographical errors across the Eigen codebase using codespell
- [#1659](https://gitlab.com/libeigen/eigen/-/merge_requests/1659): Updated `.clang-format` configuration to improve JavaScript file formatting compatibility
- [#1660](https://gitlab.com/libeigen/eigen/-/merge_requests/1660): Updated `eigen_navtree_hacks.js` file to improve code readability, performance, and maintenance
- [#1661](https://gitlab.com/libeigen/eigen/-/merge_requests/1661): Improved `hlog` symbol lookup to allow local namespace definitions, enhancing function flexibility
- [#1663](https://gitlab.com/libeigen/eigen/-/merge_requests/1663): Optimized SSE/AVX complex multiplication kernels using `vfmaddsub` instructions for improved performance
- [#1665](https://gitlab.com/libeigen/eigen/-/merge_requests/1665): Cleanups to threaded product code and test cases for improved maintainability and readability
- [#1666](https://gitlab.com/libeigen/eigen/-/merge_requests/1666): Add `std::this_thread::yield()` to spinloops in threaded matrix multiplication to optimize CPU resource usage and instruction efficiency
- [#1667](https://gitlab.com/libeigen/eigen/-/merge_requests/1667): Optimized `StableNorm` performance for non-trivial sizes with improved consistency between aligned and unaligned inputs
- [#1668](https://gitlab.com/libeigen/eigen/-/merge_requests/1668): Added `<thread>` header to enable `std::this_thread::yield()` for improved thread management
- [#1669](https://gitlab.com/libeigen/eigen/-/merge_requests/1669): Introduced ARM NEON complex intrinsics `pmul` and `pmadd` for improved complex number computation performance on ARM architectures
- [#1672](https://gitlab.com/libeigen/eigen/-/merge_requests/1672): Vectorized implementation of `squaredNorm()` for complex types, improving performance of norm-related operations
- [#1676](https://gitlab.com/libeigen/eigen/-/merge_requests/1676): Improved documentation for `GeneralizedEigenSolver::eigenvectors()` method to ensure proper rendering and clarity
- [#1677](https://gitlab.com/libeigen/eigen/-/merge_requests/1677): Consolidated and optimized `patan()` implementations for float and double types, achieving significant performance speedups across various instruction set architectures
- [#1681](https://gitlab.com/libeigen/eigen/-/merge_requests/1681): Improved complex number trait handling by modifying `NumTraits<std::complex<Real_>>::IsSigned` and adding `pnmsub` tests
- [#1682](https://gitlab.com/libeigen/eigen/-/merge_requests/1682): Added support for nvc++ compiler with configuration macro and improved compilation compatibility
- [#1683](https://gitlab.com/libeigen/eigen/-/merge_requests/1683): Introduced SSE and AVX implementations for complex FMA operations, improving performance and computational accuracy
- [#1684](https://gitlab.com/libeigen/eigen/-/merge_requests/1684): Vectorized `atanh<double>` implementation with standard-compliant handling for |x| >= 1, delivering significant performance speedups across different instruction set architectures
- [#1689](https://gitlab.com/libeigen/eigen/-/merge_requests/1689): Fixed ARM SVE intrinsics bug and added `svsqrt_f32_x` sqrt support
- [#1691](https://gitlab.com/libeigen/eigen/-/merge_requests/1691): Updated `NonBlockingThreadPool.h` to use `eigen_plain_assert` for better C++26 compatibility
- [#1692](https://gitlab.com/libeigen/eigen/-/merge_requests/1692): Optimized dot product implementation with performance improvements for smaller vector sizes
- [#1700](https://gitlab.com/libeigen/eigen/-/merge_requests/1700): Improved test debugging capabilities by adding extra information to `float_pow_test_impl` and cleaning up `array_cwise` tests
- [#1701](https://gitlab.com/libeigen/eigen/-/merge_requests/1701): Add missing `EIGEN_DEVICE_FUNC` annotations to improve CUDA build compatibility
- [#1702](https://gitlab.com/libeigen/eigen/-/merge_requests/1702): Added `max_digits10` support for `mpreal` types in `NumTraits`
- [#1703](https://gitlab.com/libeigen/eigen/-/merge_requests/1703): Enhance inverse evaluator compatibility for CUDA device execution by marking as host+device function
- [#1706](https://gitlab.com/libeigen/eigen/-/merge_requests/1706): Improved speed and accuracy of `erf()` function with reduced maximum error and performance benchmarks
- [#1709](https://gitlab.com/libeigen/eigen/-/merge_requests/1709): Standardizes polynomial evaluation using `ppolevl` helper function across the codebase
- [#1710](https://gitlab.com/libeigen/eigen/-/merge_requests/1710): Introduced vectorized implementation of `erfc()` for float with significant performance improvements (45-72% speedup)
- [#1712](https://gitlab.com/libeigen/eigen/-/merge_requests/1712): Suppressed ARM out-of-bounds warnings for `reverseInPlace` function on fixed-size matrices
- [#1716](https://gitlab.com/libeigen/eigen/-/merge_requests/1716): Improved stack allocation assert handling to reduce performance overhead and enhance evaluator class usability
- [#1729](https://gitlab.com/libeigen/eigen/-/merge_requests/1729): Add nvc++ compiler support to Eigen v3.4, improving compilation compatibility
- [#1731](https://gitlab.com/libeigen/eigen/-/merge_requests/1731): Replace standard `__cplusplus` macro with library-specific `EIGEN_CPLUSPLUS` to improve MSVC compatibility
- [#1732](https://gitlab.com/libeigen/eigen/-/merge_requests/1732): Vectorized and improved `erfc(x)` function performance for double and float with up to 83% speedup and enhanced accuracy
- [#1734](https://gitlab.com/libeigen/eigen/-/merge_requests/1734): Enhances AVX implementation of `predux_any` function for improved vector reduction performance
- [#1736](https://gitlab.com/libeigen/eigen/-/merge_requests/1736): Added missing `EIGEN_DEVICE_FUNCTION` decorations to improve device compatibility
- [#1739](https://gitlab.com/libeigen/eigen/-/merge_requests/1739): Update overflow check implementation using C++ numeric limits for improved type safety and compatibility
- [#1741](https://gitlab.com/libeigen/eigen/-/merge_requests/1741): Improved lldb debugging support by ensuring non-inlined destructors for `MatrixBase` symbols
- [#1743](https://gitlab.com/libeigen/eigen/-/merge_requests/1743): Vectorized implementation of `erf(x)` for double with significant SIMD performance improvements across SSE 4.2, AVX2+FMA, and AVX512 architectures
- [#1747](https://gitlab.com/libeigen/eigen/-/merge_requests/1747): Optimized error function (erf) computation for large input values by eliminating redundant calculations
- [#1748](https://gitlab.com/libeigen/eigen/-/merge_requests/1748): Removed unnecessary `HasBlend` trait check, improving code readability and efficiency
- [#1749](https://gitlab.com/libeigen/eigen/-/merge_requests/1749): Disabled `fill_n` optimization for MSVC to improve performance of zero-initialization across compilers
- [#1750](https://gitlab.com/libeigen/eigen/-/merge_requests/1750): Optimized `exp(x)` function with performance improvements of 30-35%, enhancing computational efficiency for exponential calculations
- [#1752](https://gitlab.com/libeigen/eigen/-/merge_requests/1752): Improved `exp(x)` function performance with 3-4% speedup and prevented premature overflow
- [#1753](https://gitlab.com/libeigen/eigen/-/merge_requests/1753): Reinstates vectorized `erf<double>(x)` implementation for SSE and AVX architectures
- [#1754](https://gitlab.com/libeigen/eigen/-/merge_requests/1754): Simplified and optimized `pow()` function, achieving 5-6% performance speedup for float and double data types
- [#1755](https://gitlab.com/libeigen/eigen/-/merge_requests/1755): Optimized `setConstant` and `setZero` functions using `std::fill_n` and `memset` for improved performance across different matrix and array types
- [#1756](https://gitlab.com/libeigen/eigen/-/merge_requests/1756): Optimize `pow<float>(x,y)` with 25% speedup and improved accuracy for integer exponents
- [#1759](https://gitlab.com/libeigen/eigen/-/merge_requests/1759): Refactored special case handling in `pow(x,y)` and reintroduced repeated squaring for float and integer types
- [#1763](https://gitlab.com/libeigen/eigen/-/merge_requests/1763): Improved documentation for move constructor and move assignment methods
- [#1765](https://gitlab.com/libeigen/eigen/-/merge_requests/1765): Added CI deploy phase to tag successful nightly pipelines
- [#1766](https://gitlab.com/libeigen/eigen/-/merge_requests/1766): Updated ROCm docker image to improve CI reliability and functionality
- [#1773](https://gitlab.com/libeigen/eigen/-/merge_requests/1773): Improved CI pipeline to fetch commits using tags for better commit traceability and workflow consistency
- [#1774](https://gitlab.com/libeigen/eigen/-/merge_requests/1774): Introduced equality comparison operator for matrices with dissimilar sizes
- [#1775](https://gitlab.com/libeigen/eigen/-/merge_requests/1775): Simplifies nightly tag job by removing branch name from CI/CD pipeline
- [#1776](https://gitlab.com/libeigen/eigen/-/merge_requests/1776): Switched to Alpine image for more efficient nightly tag deployment
- [#1779](https://gitlab.com/libeigen/eigen/-/merge_requests/1779): Optimizes matrix construction and assignment using `fill_n` and `memset` for improved performance in matrix initialization
- [#1786](https://gitlab.com/libeigen/eigen/-/merge_requests/1786): Reinstates default threading behavior by using `omp_get_max_threads` when `setNbThreads` is not set
- [#1787](https://gitlab.com/libeigen/eigen/-/merge_requests/1787): Improved CUDA device compatibility by adding `EIGEN_DEVICE_FUNC` qualifiers and revising function implementations for better CUDA support
- [#1788](https://gitlab.com/libeigen/eigen/-/merge_requests/1788): Simplified CI configuration by removing unnecessary Ubuntu ToolChain PPA
- [#1791](https://gitlab.com/libeigen/eigen/-/merge_requests/1791): Adds ForkJoin-based ParallelFor algorithm to ThreadPool module, enhancing parallel computation performance
- [#1794](https://gitlab.com/libeigen/eigen/-/merge_requests/1794): Updated documentation to clarify cross product behavior for complex numbers
- [#1796](https://gitlab.com/libeigen/eigen/-/merge_requests/1796): Updated documentation to clarify block objects can have non-square dimensions
- [#1797](https://gitlab.com/libeigen/eigen/-/merge_requests/1797): Improved support for loongarch architecture in Eigen
- [#1800](https://gitlab.com/libeigen/eigen/-/merge_requests/1800): Documentation cleanup for `ForkJoin.h` with typo fixes and formatting improvements
- [#1803](https://gitlab.com/libeigen/eigen/-/merge_requests/1803): Fix threadpool compatibility issues for C++14 compilers, resolving initialization and warning problems
- [#1807](https://gitlab.com/libeigen/eigen/-/merge_requests/1807): Comprehensive documentation cleanup resolving Doxygen warnings and improving documentation clarity
- [#1808](https://gitlab.com/libeigen/eigen/-/merge_requests/1808): Minor documentation typo fixes in `ForkJoin.h`
- [#1809](https://gitlab.com/libeigen/eigen/-/merge_requests/1809): Improved tensor documentation by correcting class name references and streamlining documentation
- [#1811](https://gitlab.com/libeigen/eigen/-/merge_requests/1811): Improved cmake configuration for loongarch64 emulated tests, enhancing testing framework compatibility
- [#1815](https://gitlab.com/libeigen/eigen/-/merge_requests/1815): Updated check for `std::hardware_destructive_interference_size` to improve compatibility on Android platforms
- [#1817](https://gitlab.com/libeigen/eigen/-/merge_requests/1817): Introduced `EIGEN_CI_CTEST_ARGS` for custom test timeout control and standardized ctest-related argument naming
- [#1818](https://gitlab.com/libeigen/eigen/-/merge_requests/1818): Improved documentation generation with nightly Doxygen builds and enhanced error handling
- [#1821](https://gitlab.com/libeigen/eigen/-/merge_requests/1821): Improved BiCGSTAB numerical convergence by refining initialization and restart conditions
- [#1823](https://gitlab.com/libeigen/eigen/-/merge_requests/1823): Added graphviz dependency to improve documentation build and graph rendering
- [#1824](https://gitlab.com/libeigen/eigen/-/merge_requests/1824): Improved rcond estimate algorithm to return zero condition number for non-invertible matrices
- [#1826](https://gitlab.com/libeigen/eigen/-/merge_requests/1826): Adds missing MathJax/LaTeX configuration to improve mathematical formula rendering
- [#1829](https://gitlab.com/libeigen/eigen/-/merge_requests/1829): Refactored `AssignEvaluator.h` to modernize code, remove legacy enums, and improve maintainability
- [#1832](https://gitlab.com/libeigen/eigen/-/merge_requests/1832): Remove `fno-check-new` compiler flag for Clang to reduce build warnings
- [#1837](https://gitlab.com/libeigen/eigen/-/merge_requests/1837): Modify documentation build process to prevent automatic deletion of nightly docs on pipeline failures
- [#1843](https://gitlab.com/libeigen/eigen/-/merge_requests/1843): Improved STL feature detection for C++20 compatibility, preventing compilation issues across different compiler and library versions
- [#1846](https://gitlab.com/libeigen/eigen/-/merge_requests/1846): Refactored `AssignmentFunctors.h` to reduce code redundancy and improve consistency in assignment operations

##### Added

- [#121](https://gitlab.com/libeigen/eigen/-/merge_requests/121): Added a `make format` command to enforce consistent code styling across the project
- [#447](https://gitlab.com/libeigen/eigen/-/merge_requests/447): Introduces BiCGSTAB(L) algorithm for solving linear systems with potential improvements for non-symmetric systems
- [#482](https://gitlab.com/libeigen/eigen/-/merge_requests/482): Adds LLDB synthetic child provider for structured display of Eigen matrices and vectors during debugging
- [#646](https://gitlab.com/libeigen/eigen/-/merge_requests/646): Added new make targets `buildtests_gpu` and `check_gpu` to simplify GPU testing infrastructure
- [#688](https://gitlab.com/libeigen/eigen/-/merge_requests/688): Added nan-propagation options to matrix and array plugins for enhanced NaN value handling
- [#729](https://gitlab.com/libeigen/eigen/-/merge_requests/729): Implemented `reverse_iterator` for `Eigen::array<...>` to enhance iteration capabilities
- [#758](https://gitlab.com/libeigen/eigen/-/merge_requests/758): Added GPU unit tests for HIP using C++14, improving testing for GPU functionalities
- [#852](https://gitlab.com/libeigen/eigen/-/merge_requests/852): Adds convenience `constexpr std::size_t size() const` method to `Eigen::IndexList`
- [#965](https://gitlab.com/libeigen/eigen/-/merge_requests/965): Added three new fused multiply functions (pmsub, pnmadd, pnmsub) for PowerPC architecture
- [#981](https://gitlab.com/libeigen/eigen/-/merge_requests/981): Added MKL adapter and implementations for KFR and FFTS FFT libraries in Eigen's FFT module
- [#995](https://gitlab.com/libeigen/eigen/-/merge_requests/995): Added comprehensive documentation for the DiagonalBase class to improve library usability
- [#1004](https://gitlab.com/libeigen/eigen/-/merge_requests/1004): Adds `determinant()` method for various QR decomposition classes including HouseholderQR, ColPivHouseholderQR, FullPivHouseholderQR, and CompleteOrthogonalDecomposition
- [#1029](https://gitlab.com/libeigen/eigen/-/merge_requests/1029): Added fixed power unary operation for coefficientwise real-valued power operations on arrays
- [#1046](https://gitlab.com/libeigen/eigen/-/merge_requests/1046): Re-enabled pow function for complex number types, expanding mathematical computation capabilities
- [#1047](https://gitlab.com/libeigen/eigen/-/merge_requests/1047): Added skew symmetric matrix class for 3D vectors to enhance vector transformations
- [#1097](https://gitlab.com/libeigen/eigen/-/merge_requests/1097): Adds a new `signbit` function for efficient floating-point sign checking with AVX2 packet operation support
- [#1098](https://gitlab.com/libeigen/eigen/-/merge_requests/1098): Implemented cross product for 2D vectors, computing a scalar representing the signed area of the spanned parallelogram
- [#1121](https://gitlab.com/libeigen/eigen/-/merge_requests/1121): Adds serialization capabilities for sparse matrices and sparse vectors
- [#1133](https://gitlab.com/libeigen/eigen/-/merge_requests/1133): Introduces new `setEqualSpaced` function for creating equally spaced vectors with vectorized implementation
- [#1209](https://gitlab.com/libeigen/eigen/-/merge_requests/1209): Added functionality to directly print diagonal matrix expressions without requiring dense object assignment
- [#1297](https://gitlab.com/libeigen/eigen/-/merge_requests/1297): Added `Packet4ui`, `Packet8ui`, and `Packet4ul` packet types for SSE/AVX to support unsigned integer SIMD operations
- [#1299](https://gitlab.com/libeigen/eigen/-/merge_requests/1299): Added BF16 pcast functions and centralized type casting in `TypeCasting.h`
- [#1309](https://gitlab.com/libeigen/eigen/-/merge_requests/1309): Added `Abs2` method for `Packet4ul` data type to enhance vectorized operations
- [#1331](https://gitlab.com/libeigen/eigen/-/merge_requests/1331): Added new test to validate SYCL functionalities in Eigen core library
- [#1335](https://gitlab.com/libeigen/eigen/-/merge_requests/1335): Added new methods `removeOuterVectors()` and `insertEmptyOuterVectors()` for flexible sparse matrix manipulation
- [#1345](https://gitlab.com/libeigen/eigen/-/merge_requests/1345): Adds new Quaternion constructor that accepts a real scalar and 3D vector for more intuitive quaternion creation
- [#1403](https://gitlab.com/libeigen/eigen/-/merge_requests/1403): Adds component-wise cubic root (`cbrt`) functionality for arrays and matrices
- [#1414](https://gitlab.com/libeigen/eigen/-/merge_requests/1414): Implemented `plog_complex` function for vectorized complex logarithm calculations
- [#1436](https://gitlab.com/libeigen/eigen/-/merge_requests/1436): Added internal implementations for count trailing zeros (`ctz`) and count leading zeros (`clz`) functions
- [#1445](https://gitlab.com/libeigen/eigen/-/merge_requests/1445): Added factor getter functions for Cholmod LLT and LDLT solvers to access L, Lᵀ, and D factors
- [#1455](https://gitlab.com/libeigen/eigen/-/merge_requests/1455): Added test support for ROCm MI300 series architectures (gfx940, gfx941, gfx942)
- [#1462](https://gitlab.com/libeigen/eigen/-/merge_requests/1462): Adds ability to specify a custom temporary directory for file I/O outputs
- [#1493](https://gitlab.com/libeigen/eigen/-/merge_requests/1493): Added `trunc` operation for truncating floating-point numbers towards zero
- [#1501](https://gitlab.com/libeigen/eigen/-/merge_requests/1501): Implemented SIMD complex function `pexp_complex` for `float` to enhance performance of complex number operations
- [#1512](https://gitlab.com/libeigen/eigen/-/merge_requests/1512): Added `signDeterminant()` method to QR and related decompositions to determine determinant sign
- [#1612](https://gitlab.com/libeigen/eigen/-/merge_requests/1612): Added scalar bit shifting functions `logical_shift_left`, `logical_shift_right`, and `arithmetic_shift_right` for integer types
- [#1704](https://gitlab.com/libeigen/eigen/-/merge_requests/1704): Added free-function `swap` for dense and sparse matrices and blocks to improve C++ algorithm compatibility
- [#1714](https://gitlab.com/libeigen/eigen/-/merge_requests/1714): Added `std::nextafter` implementation for bfloat16 data type
- [#1715](https://gitlab.com/libeigen/eigen/-/merge_requests/1715): Adds `exp2(x)` function with improved numerical accuracy using TwoProd algorithm
- [#1719](https://gitlab.com/libeigen/eigen/-/merge_requests/1719): Added new tests for `sizeof()` with one dynamic dimension
- [#1733](https://gitlab.com/libeigen/eigen/-/merge_requests/1733): Added missing AVX `predux_any` functions to enhance vectorized reduction operations
- [#1758](https://gitlab.com/libeigen/eigen/-/merge_requests/1758): Added test case for `pcast` function with scalar types
- [#1778](https://gitlab.com/libeigen/eigen/-/merge_requests/1778): Added `install-doc` CMake target for documentation installation
- [#1805](https://gitlab.com/libeigen/eigen/-/merge_requests/1805): Added `matrixL()` and `matrixU()` functions to fetch L and U factors from IncompleteLUT sparse matrix decomposition
- [#1812](https://gitlab.com/libeigen/eigen/-/merge_requests/1812): Added automated Doxygen documentation build and deployment to GitLab Pages

##### Removed

- [#636](https://gitlab.com/libeigen/eigen/-/merge_requests/636): Removed stray references to deprecated DynamicSparseMatrix class
- [#740](https://gitlab.com/libeigen/eigen/-/merge_requests/740): Removed redundant `nonZeros()` method from `DenseBase` class, which simply called `size()`
- [#752](https://gitlab.com/libeigen/eigen/-/merge_requests/752): Deprecated unused macro EIGEN_GPU_TEST_C99_MATH to reduce code clutter
- [#768](https://gitlab.com/libeigen/eigen/-/merge_requests/768): Removed custom Find*.cmake scripts for BLAS, LAPACK, GLEW, and GSL, now using CMake's built-in modules
- [#793](https://gitlab.com/libeigen/eigen/-/merge_requests/793): Removed unused `EIGEN_HAS_STATIC_ARRAY_TEMPLATE` macro to clean up the codebase
- [#855](https://gitlab.com/libeigen/eigen/-/merge_requests/855): Removed unused macros related to `prsqrt` implementation, improving code clarity and maintainability
- [#897](https://gitlab.com/libeigen/eigen/-/merge_requests/897): Removed obsolete gcc 4.3 copy_bool workaround in testsuite
- [#1080](https://gitlab.com/libeigen/eigen/-/merge_requests/1080): Removed an unused typedef to improve code clarity and maintainability
- [#1092](https://gitlab.com/libeigen/eigen/-/merge_requests/1092): Removed references to M_PI_2 and M_PI_4 constants from Eigen codebase
- [#1200](https://gitlab.com/libeigen/eigen/-/merge_requests/1200): Remove custom implementations of `equal_to` and `not_equal_no` no longer needed in C++14
- [#1306](https://gitlab.com/libeigen/eigen/-/merge_requests/1306): Removed last remaining instances of unused `HasHalfPacket` enum
- [#1474](https://gitlab.com/libeigen/eigen/-/merge_requests/1474): Removes the Skyline module due to long-standing build issues and lack of tests
- [#1475](https://gitlab.com/libeigen/eigen/-/merge_requests/1475): Removed `MoreVectorization` feature, relocating `pasin` implementation to `GenericPacketMath` to reduce code complexity and potential ODR violations
- [#1477](https://gitlab.com/libeigen/eigen/-/merge_requests/1477): Removed obsolete relicense script to streamline codebase

##### Changes

### Unsupported

#### Breaking changes

- [#606](https://gitlab.com/libeigen/eigen/-/merge_requests/606): Removal of Sparse Dynamic Matrix from library API
- [#704](https://gitlab.com/libeigen/eigen/-/merge_requests/704): Removed problematic `take<n, numeric_list<T>>` implementation to resolve g++-11 compiler crash
- [#1423](https://gitlab.com/libeigen/eigen/-/merge_requests/1423): Adds static assertions to Tensor constructors to validate tensor dimension compatibility at compile-time

#### Major changes

- [#327](https://gitlab.com/libeigen/eigen/-/merge_requests/327): Reimplemented Tensor stream output with new predefined formats and improved IO functionality
- [#534](https://gitlab.com/libeigen/eigen/-/merge_requests/534): Introduces preliminary HIP bfloat16 GPU support for AMD GPUs
- [#577](https://gitlab.com/libeigen/eigen/-/merge_requests/577): Introduces IDR(s)STAB(l) method, a new iterative solver for sparse matrix problems combining features of IDR(s) and BiCGSTAB(l)
- [#612](https://gitlab.com/libeigen/eigen/-/merge_requests/612): Adds support for EIGEN_TENSOR_PLUGIN, EIGEN_TENSORBASE_PLUGIN, and EIGEN_READONLY_TENSORBASE_PLUGIN in tensor classes
- [#622](https://gitlab.com/libeigen/eigen/-/merge_requests/622): Renamed existing Tuple class to Pair and introduced a new Tuple class for improved device compatibility
- [#623](https://gitlab.com/libeigen/eigen/-/merge_requests/623): Introduces device-compatible Tuple implementation for GPU testing, addressing compatibility issues with std::tuple
- [#625](https://gitlab.com/libeigen/eigen/-/merge_requests/625): Introduced new GPU test utilities with flexible kernel execution functions for CPU and GPU environments
- [#676](https://gitlab.com/libeigen/eigen/-/merge_requests/676): Improved accuracy of full tensor reduction for half and bfloat16 types using tree summation algorithm
- [#681](https://gitlab.com/libeigen/eigen/-/merge_requests/681): Prevents integer overflows in EigenMetaKernel indexing for CUDA tensor operations
- [#1125](https://gitlab.com/libeigen/eigen/-/merge_requests/1125): Adds synchronize method to all device types, improving device operation consistency and flexibility
- [#1265](https://gitlab.com/libeigen/eigen/-/merge_requests/1265): Vectorize tensor.isnan() using typed predicates with performance optimizations for AVX512
- [#1287](https://gitlab.com/libeigen/eigen/-/merge_requests/1287): Fixed potential crash in tensor contraction with empty tensors by removing restrictive assert
- [#1627](https://gitlab.com/libeigen/eigen/-/merge_requests/1627): Added `.roll()` function for circular shifts in Tensor module, enabling NumPy/TensorFlow-like tensor rotation capabilities
- [#1828](https://gitlab.com/libeigen/eigen/-/merge_requests/1828): Enhances TensorRef implementation with improved type handling and immutability enforcement
- [#1848](https://gitlab.com/libeigen/eigen/-/merge_requests/1848): Cleaned up and improved TensorDeviceThreadPool implementation with method removals, enhanced C++20 compatibility, and simplified type erasure

#### Other

##### Fixed

- [#628](https://gitlab.com/libeigen/eigen/-/merge_requests/628): Renamed 'vec_all_nan' symbol in cxx11_tensor_expr test to resolve build conflicts with altivec.h on ppc64le platform
- [#653](https://gitlab.com/libeigen/eigen/-/merge_requests/653): Disabled specific HIP subtests that fail due to non-functional device side malloc/free
- [#671](https://gitlab.com/libeigen/eigen/-/merge_requests/671): Fixed GPU special function tests by correcting checks and updating verification methods
- [#679](https://gitlab.com/libeigen/eigen/-/merge_requests/679): Disabled Tree reduction for GPU to resolve memory errors and improve GPU operation stability
- [#695](https://gitlab.com/libeigen/eigen/-/merge_requests/695): Fix compilation compatibility issue with older Boost versions in boostmultiprec test
- [#705](https://gitlab.com/libeigen/eigen/-/merge_requests/705): Fixes TensorReduction test warnings and improves sum accuracy error bound calculation
- [#715](https://gitlab.com/libeigen/eigen/-/merge_requests/715): Fixed failing test for tensor reduction by improving error bound comparisons
- [#723](https://gitlab.com/libeigen/eigen/-/merge_requests/723): Fixed off-by-one error in tensor broadcasting affecting packet size handling
- [#730](https://gitlab.com/libeigen/eigen/-/merge_requests/730): Fixed stride computation for indexed views with non-Eigen index types to prevent potential signed integer overflow
- [#755](https://gitlab.com/libeigen/eigen/-/merge_requests/755): Fixed leftover else branch in unsupported code
- [#770](https://gitlab.com/libeigen/eigen/-/merge_requests/770): Fixed customIndices2Array function to correctly handle the first index in tensor module
- [#853](https://gitlab.com/libeigen/eigen/-/merge_requests/853): Resolved ODR failures in TensorRandom component to improve code stability and reliability
- [#894](https://gitlab.com/libeigen/eigen/-/merge_requests/894): Fixed tensor executor test and added support for tensor packets of size 1
- [#898](https://gitlab.com/libeigen/eigen/-/merge_requests/898): Fixed zeta function edge case for large inputs, preventing NaN and overflow issues
- [#902](https://gitlab.com/libeigen/eigen/-/merge_requests/902): Temporarily disabled aarch64 CI due to unavailable Windows on Arm machines
- [#916](https://gitlab.com/libeigen/eigen/-/merge_requests/916): Updated Altivec MMA dynamic dispatch flags to support binary values for improved TensorFlow compatibility
- [#929](https://gitlab.com/libeigen/eigen/-/merge_requests/929): Split general matrix-vector product interface for Power architectures to improve TensorFlow compatibility
- [#991](https://gitlab.com/libeigen/eigen/-/merge_requests/991): Resolved ambiguous comparison warnings in clang for C++20 by adjusting TensorBase comparison operators
- [#1001](https://gitlab.com/libeigen/eigen/-/merge_requests/1001): Fixed build compatibility for f16/bf16 Bessel function specializations on AVX512 for older compilers
- [#1033](https://gitlab.com/libeigen/eigen/-/merge_requests/1033): Fixed SYCL tests by correcting sigmoid function, binary logic operators, and resolving test failures in tensor math operations
- [#1123](https://gitlab.com/libeigen/eigen/-/merge_requests/1123): Fix reshaping strides handling for inputs with non-zero inner stride in Eigen's Tensor module
- [#1159](https://gitlab.com/libeigen/eigen/-/merge_requests/1159): Re-added missing header to restore GPU test functionality
- [#1185](https://gitlab.com/libeigen/eigen/-/merge_requests/1185): Improved special case handling in atan2 function to resolve test failure in TensorFlow with Clang
- [#1227](https://gitlab.com/libeigen/eigen/-/merge_requests/1227): Fixed null placeholder accessor issue in Reduction SYCL test to prevent segmentation faults
- [#1237](https://gitlab.com/libeigen/eigen/-/merge_requests/1237): Fixed GPU conv3d out-of-resources failure by adjusting 32-bit integer variable handling in kernel
- [#1243](https://gitlab.com/libeigen/eigen/-/merge_requests/1243): Fixed tensor comparison test in unsupported module
- [#1264](https://gitlab.com/libeigen/eigen/-/merge_requests/1264): Introduced EIGEN_NOT_A_MACRO macro to improve compatibility with TensorFlow build process
- [#1298](https://gitlab.com/libeigen/eigen/-/merge_requests/1298): Improved tensor select evaluator using typed ternary selection operator for better performance
- [#1355](https://gitlab.com/libeigen/eigen/-/merge_requests/1355): Disable FP16 arithmetic for arm32 to prevent compatibility issues with Clang compiler
- [#1369](https://gitlab.com/libeigen/eigen/-/merge_requests/1369): Fixed ARM build warnings by addressing type casting and variable shadowing issues in Eigen's Tensor module
- [#1382](https://gitlab.com/libeigen/eigen/-/merge_requests/1382): Fix tensor strided linear buffer copy to prevent negative index issues and improve integer arithmetic safety
- [#1383](https://gitlab.com/libeigen/eigen/-/merge_requests/1383): Introduces `EIGEN_TEMPORARY_UNALIGNED_SCALAR_UB` macro to handle unaligned scalar undefined behavior, primarily addressing TensorFlow Lite compatibility issues
- [#1410](https://gitlab.com/libeigen/eigen/-/merge_requests/1410): Fix integer overflow in `div_ceil` function preventing `cxx11_tensor_gpu_1` test from passing
- [#1435](https://gitlab.com/libeigen/eigen/-/merge_requests/1435): Protect kernel launch syntax from unintended clang-format modifications that cause syntax errors
- [#1453](https://gitlab.com/libeigen/eigen/-/merge_requests/1453): Fixed memory management issues in `TensorForcedEval` by using `shared_ptr` to prevent double-free and invalid memory access errors
- [#1516](https://gitlab.com/libeigen/eigen/-/merge_requests/1516): Fixed GPU build compatibility for `ptanh_float` function
- [#1542](https://gitlab.com/libeigen/eigen/-/merge_requests/1542): Split cxx11_tensor_gpu test to reduce Windows test timeouts and improve test suite reliability
- [#1575](https://gitlab.com/libeigen/eigen/-/merge_requests/1575): Fix `long double` random number generation fallback mechanism
- [#1596](https://gitlab.com/libeigen/eigen/-/merge_requests/1596): Resolved unused variable warnings in TensorIO component
- [#1597](https://gitlab.com/libeigen/eigen/-/merge_requests/1597): Fix enum comparison warnings in Autodiff module
- [#1599](https://gitlab.com/libeigen/eigen/-/merge_requests/1599): Fixed PPC runner cross-compilation attempt by preventing non-PPC target compilations
- [#1678](https://gitlab.com/libeigen/eigen/-/merge_requests/1678): Fixed `Wmaybe-uninitialized` warning in TensorVolumePatchOp by introducing `unreachable()` function
- [#1698](https://gitlab.com/libeigen/eigen/-/merge_requests/1698): Fixed implicit conversion issues in TensorChipping module
- [#1721](https://gitlab.com/libeigen/eigen/-/merge_requests/1721): Fixes compilation issue with `EIGEN_ALIGNED_ALLOCA` for nvc++ compiler by replacing unsupported `__builtin_alloca_with_align`
- [#1836](https://gitlab.com/libeigen/eigen/-/merge_requests/1836): Fixed compiler warning by adding explicit copy constructor to TensorRef class
- [#1840](https://gitlab.com/libeigen/eigen/-/merge_requests/1840): Fixed boolean scatter and random generation issues in tensor operations, improving reliability and test coverage
- [#1847](https://gitlab.com/libeigen/eigen/-/merge_requests/1847): Removed extra semicolon in `DeviceWrapper.h` to fix compilation warnings

##### Improved

- [#543](https://gitlab.com/libeigen/eigen/-/merge_requests/543): Improved PEP8 compliance and formatting in GDB pretty printer for better code readability
- [#616](https://gitlab.com/libeigen/eigen/-/merge_requests/616): Disabled CUDA Eigen::half host-side vectorization for compatibility with pre-CUDA 10.0 versions
- [#619](https://gitlab.com/libeigen/eigen/-/merge_requests/619): Improved documentation for unsupported sparse iterative solvers
- [#645](https://gitlab.com/libeigen/eigen/-/merge_requests/645): Introduced default constructor for eigen_packet_wrapper to simplify memory operations
- [#669](https://gitlab.com/libeigen/eigen/-/merge_requests/669): Optimized tensor_contract_gpu test by reducing contractions to improve test performance on Windows
- [#667](https://gitlab.com/libeigen/eigen/-/merge_requests/667): Significantly speeds up tensor reduction performance through loop strip mining and unrolling techniques
- [#678](https://gitlab.com/libeigen/eigen/-/merge_requests/678): Reorganized CUDA/Complex.h to GPU/Complex.h and removed deprecated TensorReductionCuda.h header
- [#724](https://gitlab.com/libeigen/eigen/-/merge_requests/724): Improved TensorIO compatibility with TensorMap containing const elements
- [#896](https://gitlab.com/libeigen/eigen/-/merge_requests/896): Removed ComputeCpp-specific code from SYCL Vptr, improving compatibility and performance
- [#942](https://gitlab.com/libeigen/eigen/-/merge_requests/942): Fixed navbar scroll behavior with table of contents by overriding Doxygen JavaScript
- [#982](https://gitlab.com/libeigen/eigen/-/merge_requests/982): Resolved ambiguities in Tensor comparison operators for C++20 compatibility
- [#1005](https://gitlab.com/libeigen/eigen/-/merge_requests/1005): Re-enabled unit tests for device side malloc in ROCm 5.2
- [#1119](https://gitlab.com/libeigen/eigen/-/merge_requests/1119): Added brackets around unsigned type names to improve code readability and consistency
- [#1341](https://gitlab.com/libeigen/eigen/-/merge_requests/1341): Replaced `CudaStreamDevice` with `GpuStreamDevice` in tensor GPU benchmarks for improved accuracy
- [#1406](https://gitlab.com/libeigen/eigen/-/merge_requests/1406): Replaced deprecated `divup` with `div_ceil` in TensorReduction to reduce warnings
- [#1441](https://gitlab.com/libeigen/eigen/-/merge_requests/1441): Improved clang-format CI configuration to operate in non-interactive mode and ensure proper installation
- [#1466](https://gitlab.com/libeigen/eigen/-/merge_requests/1466): Refined assertions for chipping operations in Tensor module, removing dimension checks and improving efficiency
- [#1479](https://gitlab.com/libeigen/eigen/-/merge_requests/1479): Corrected markdown formatting in Eigen::Tensor README.md for improved documentation readability
- [#1509](https://gitlab.com/libeigen/eigen/-/merge_requests/1509): Renamed `generic_fast_tanh_float` to `ptanh_float` for improved code clarity and maintainability
- [#1558](https://gitlab.com/libeigen/eigen/-/merge_requests/1558): Performance optimization for `Tensor::resize` by removing slow index checks and modernizing code
- [#1644](https://gitlab.com/libeigen/eigen/-/merge_requests/1644): Add async support for `chip` and `extract_volume_patches` operations in Eigen's Tensor module
- [#1645](https://gitlab.com/libeigen/eigen/-/merge_requests/1645): Explicitly capture `this` in lambda expressions in Tensor module to prevent compiler warnings and improve code clarity
- [#1653](https://gitlab.com/libeigen/eigen/-/merge_requests/1653): Corrected numerous typographical errors across Eigen's documentation and codebase to improve readability
- [#1680](https://gitlab.com/libeigen/eigen/-/merge_requests/1680): Enhances TensorChipping by detecting "effectively inner/outer" chipping with stride optimization
- [#1767](https://gitlab.com/libeigen/eigen/-/merge_requests/1767): Update ROCm Docker image to Ubuntu 22.04 for improved stability and reliability
- [#1768](https://gitlab.com/libeigen/eigen/-/merge_requests/1768): Update ROCm Docker image to Ubuntu 24.04 to address Ninja crashing issue
- [#1770](https://gitlab.com/libeigen/eigen/-/merge_requests/1770): Experimental Alpine Docker base image for CI to potentially improve build efficiency
- [#1771](https://gitlab.com/libeigen/eigen/-/merge_requests/1771): Updated deployment job to enhance efficiency and workflow reliability
- [#1772](https://gitlab.com/libeigen/eigen/-/merge_requests/1772): Update git clone strategy to improve branch setup and repository management
- [#1844](https://gitlab.com/libeigen/eigen/-/merge_requests/1844): Optimized division operations in `TensorVolumePatch.h` by reducing unnecessary divisions when packet size is 1
- [#1849](https://gitlab.com/libeigen/eigen/-/merge_requests/1849): Formatted `TensorDeviceThreadPool.h` and improved code using `if constexpr` for C++20

##### Added

- [#607](https://gitlab.com/libeigen/eigen/-/merge_requests/607): Added flowchart to help users select sparse iterative solvers in unsupported module
- [#624](https://gitlab.com/libeigen/eigen/-/merge_requests/624): Introduced `Serializer<T>` class for binary serialization, enhancing GPU testing data transfer capabilities
- [#798](https://gitlab.com/libeigen/eigen/-/merge_requests/798): Adds a Non-Negative Least Squares (NNLS) solver to Eigen's unsupported modules using an active-set algorithm
- [#973](https://gitlab.com/libeigen/eigen/-/merge_requests/973): Added `.arg()` method to Tensor class for retrieving indices of max/min values along specified dimensions

##### Removed

- [#637](https://gitlab.com/libeigen/eigen/-/merge_requests/637): Removes obsolete DynamicSparseMatrix references and typographical errors in unsupported directory

## [3.4.0]

Released on August 18, 2021

**Notice:** 3.4.x will be the last major release series of Eigen that will support c++03.

### Breaking changes

* Using float or double for indexing matrices, vectors and arrays will now fail to compile
* **Behavioral change:** `Transform::computeRotationScaling()` and `Transform::computeScalingRotation()` are now more continuous across degeneracies (see !349).

### New features

* Add c++11 **`initializer_list` constructors** to Matrix and Array [\[doc\]](http://eigen.tuxfamily.org/dox-devel/group__TutorialMatrixClass.html#title3)
* Add STL-compatible **iterators** for dense expressions [\[doc\]](http://eigen.tuxfamily.org/dox-devel/group__TutorialSTL.html).
* New versatile API for sub-matrices, **slices**, and **indexed views** [\[doc\]](http://eigen.tuxfamily.org/dox-devel/group__TutorialSlicingIndexing.html).
* Add C++11 **template aliases** for Matrix, Vector, and Array of common sizes, including generic `Vector<Type,Size>` and `RowVector<Type,Size>` aliases [\[doc\]](http://eigen.tuxfamily.org/dox-devel/group__matrixtypedefs.html).
* New support for `bfloat16`.

### New backends

* **Arm SVE:** fixed-length [Scalable Vector Extensions](https://developer.arm.com/Architectures/Scalable%20Vector%20Extensions) vectors for `uint32_t` and `float` are available.
* **MIPS MSA:**: [MIPS SIMD Architecture (MSA)](https://www.mips.com/products/architectures/ase/simd/) 
* **AMD ROCm/HIP:** generic GPU backend that unifies support for [NVIDIA/CUDA](https://developer.nvidia.com/cuda-toolkit) and [AMD/HIP](https://rocmdocs.amd.com/en/latest/).
* **Power 10 MMA:** initial support for [Power 10 matrix multiplication assist instructions](https://arxiv.org/pdf/2104.03142.pdf) for float32 and float64, real and complex.

### Improvements

* Eigen now uses the c++11 **alignas** keyword for static alignment. Users targeting C++17 only and recent compilers (e.g., GCC>=7, clang>=5, MSVC>=19.12) will thus be able to completely forget about all [issues](http://eigen.tuxfamily.org/dox-devel/group__TopicUnalignedArrayAssert.html) related to static alignment, including `EIGEN_MAKE_ALIGNED_OPERATOR_NEW`.
* Various performance improvements for products and Eigen's GEBP and GEMV kernels have been implemented:
  * By using half and quater-packets the performance of matrix multiplications of small to medium sized matrices has been improved
  * Eigen's GEMM now falls back to GEMV if it detects that a matrix is a run-time vector
  * The performance of matrix products using Arm Neon has been drastically improved (up to 20%)
  * Performance of many special cases of matrix products has been improved
* Large speed up from blocked algorithm for `transposeInPlace`.
* Speed up misc. operations by propagating compile-time sizes (col/row-wise reverse, PartialPivLU, and others)
* Faster specialized SIMD kernels for small fixed-size inverse, LU decomposition, and determinant.
* Improved or added vectorization of partial or slice reductions along the outer-dimension, for instance: `colmajor_mat.rowwise().mean()`.

### Elementwise math functions

* Many functions are now implemented and vectorized in generic (backend-agnostic) form.
* Many improvements to correctness, accuracy, and compatibility with c++ standard library.
  * Much improved implementation of `ldexp`.
  * Misc. fixes for corner cases, NaN/Inf inputs and singular points of many functions.
  * New implementation of the Payne-Hanek for argument reduction algorithm for `sin` and `cos` with huge arguments.
  * New faithfully rounded algorithm for `pow(x,y)`.
* Speedups from (new or improved) vectorized versions of `pow`, `log`, `sin`, `cos`, `arg`, `pow`, `log2`, complex `sqrt`, `erf`, `expm1`, `logp1`, `logistic`, `rint`, `gamma` and `bessel` functions, and more.
* Improved special function support (Bessel and gamma functions, `ndtri`, `erfc`, inverse hyperbolic functions and more)
* New elementwise functions for `absolute_difference`, `rint`.

### Dense matrix decompositions and solvers

* All dense linear solvers (i.e., Cholesky, *LU, *QR, CompleteOrthogonalDecomposition, *SVD) now inherit SolverBase and thus support `.transpose()`, `.adjoint()` and `.solve()` APIs.
* SVD implementations now have an `info()` method for checking convergence.
* Most decompositions now fail quickly when invalid inputs are detected.
* Optimized the product of a `HouseholderSequence` with the identity, as well as the evaluation of a `HouseholderSequence` to a dense matrix using faster blocked product.
* Fixed aliasing issues with in-place small matrix inversions.
* Fixed several edge-cases with empty or zero inputs.

### Sparse matrix support, decompositions and solvers

* Enabled assignment and addition with diagonal matrix expressions.
* Support added for SuiteSparse KLU routines via the `KLUSupport` module.  SuiteSparse must be installed to use this module.
* `SparseCholesky` now works with row-major matrices.
* Various bug fixes and performance improvements.

### Type support

* Improved support for `half`
  * Native support added for ARM `__fp16`, CUDA/HIP `__half`, and `F16C` conversion intrinsics.
  * Better vectorization support added across all backends.
* Improved bool support
  * Partial vectorization support added for boolean operations.
  * Significantly improved performance (x25) for logical operations with `Matrix` or `Tensor` of `bool`.
* Improved support for custom types
  * More custom types work out-of-the-box (see #2201).

### Backend-specific improvements

* **Arm NEON**
  * Now provides vectorization for `uint64_t`, `int64_t`, `uint32_t`, `int16_t`, `uint16_t`, `int16_t`, `int8_t`, and `uint8_t`
  * Emulates `bfloat16` support when using `Eigen::bfloat16`
  * Supports emulated and native `float16` when using `Eigen::half`
* **SSE/AVX/AVX512**
  * General performance improvements and bugfixes.
  * Enabled AVX512 instructions by default if available.
  * New `std::complex`, `half`, and `bfloat16` vectorization support added.
  * Many missing packet functions added.
* **Altivec/Power**
  * General performance improvement and bugfixes.
  * Enhanced vectorization of real and complex scalars.
  * Changes to the `gebp_kernel` specific to Altivec, using VSX implementation of the MMA instructions that gain speed improvements up to 4x for matrix-matrix products.
  * Dynamic dispatch for GCC greater than 10 enabling selection of MMA or VSX instructions based on `__builtin_cpu_supports`.
* **GPU (CUDA and HIP)**
  * Several optimized math functions added, better support for `std::complex`.
  * Added option to disable CUDA entirely by defining `EIGEN_NO_CUDA`.
  * Many more functions can now be used in device code (e.g. comparisons, small matrix inversion).
* **ZVector**
  * Vectorized `float` and `std::complex<float>` support added.
  * Added z14 support.
* **SYCL**
  * Redesigned SYCL implementation for use with the [https://eigen.tuxfamily.org/dox/unsupported/eigen_tensors.html Tensor] module, which can be enabled by defining `EIGEN_USE_SYCL`.
  * New generic memory model introduced used by `TensorDeviceSycl`.
  * Better integration with OpenCL devices.
  * Added many math function specializations.

### Miscellaneous API Changes

* New `setConstant(...)` methods for preserving one dimension of a matrix by passing in `NoChange`.
* Added `setUnit(Index i)` for vectors that sets the ''i'' th coefficient to one and all others to zero.
* Added `transpose()`, `adjoint()`, `conjugate()` methods to `SelfAdjointView`.
* Added `shiftLeft<N>()` and `shiftRight<N>()` coefficient-wise arithmetic shift functions to Arrays.
* Enabled adding and subtracting of diagonal expressions.
* Allow user-defined default cache sizes via defining `EIGEN_DEFAULT_L1_CACHE_SIZE`, ..., `EIGEN_DEFAULT_L3_CACHE_SIZE`.
* Added `EIGEN_ALIGNOF(X)` macro for determining alignment of a provided variable.
* Allow plugins for `VectorwiseOp` by defining a file `EIGEN_VECTORWISEOP_PLUGIN` (e.g. `-DEIGEN_VECTORWISEOP_PLUGIN=my_vectorwise_op_plugins.h`).
* Allow disabling of IO operations by defining `EIGEN_NO_IO`.

### Improvement to NaN propagation

* Improvements to NaN correctness for elementwise functions.
* New `NaNPropagation` template argument to control whether NaNs are propagated or suppressed in elementwise `min/max` and corresponding reductions on `Array`, `Matrix`, and `Tensor`.

### New low-latency non-blocking ThreadPool module
* Originally a part of the Tensor module, `Eigen::ThreadPool` is now separate and more portable, and forms the basis for multi-threading in TensorFlow, for example.

### Changes to Tensor module

* Support for c++03 was officially dropped in Tensor module, since most of the code was written in c++11 anyway. This will prevent building the code for CUDA with older version of `nvcc`.
* Performance optimizations of Tensor contraction
  * Speed up "outer-product-like" operations by parallelizing over the contraction dimension, using thread_local buffers and recursive work splitting.
  * Improved threading heuristics.
  * Support for fusing element-wise operations into contraction during evaluation.
* Performance optimizations of other Tensor operator
  * Speedups from improved vectorization, block evaluation, and multi-threading for most operators.
  * Significant speedup to broadcasting.
  * Reduction of index computation overhead, e.g. using fast divisors in TensorGenerator, squeezing dimensions in TensorPadding.
* Complete rewrite of the block (tiling) evaluation framework for tensor expressions lead to significant speedups and reduced number of memory allocations.
* Added new API for asynchronous evaluation of tensor expressions.
* Misc. minor behavior changes & fixes:
  * Fix const correctness for TensorMap.
  * Modify tensor argmin/argmax to always return first occurrence.
  * More numerically stable tree reduction.
  * Improve randomness of the tensor random generator.
  * Update the padding computation for `PADDING_SAME` to be consistent with TensorFlow.
  * Support static dimensions (aka IndexList) in resizing/reshape/broadcast.
  * Improved accuracy of Tensor FFT.

### Changes to sparse iterative solvers
* Added new IDRS iterative linear solver.

### Other relevant changes

* Eigen now provides an option to test with an external BLAS library

See the [announcement](https://www.eigen.tuxfamily.org/index.php?title=3.4) for more details.

## [3.3.9]

Released on December 4, 2020.

Changes since 3.3.8:

* Commit 4e5385c90: Introduce rendering Doxygen math formulas with MathJax and the option `EIGEN_DOC_USE_MATHJAX` to control this.
* #1746: Removed implementation of standard copy-constructor and standard copy-assign-operator from PermutationMatrix and Transpositions to allow malloc-less `std::move`.
* #2036: Make sure the find_standard_math_library_test_program compiles and doesn't optimize away functions we try to test for.
* #2046: Rename test/array.cpp to test/array_cwise.cpp to fix an issue with the C++ standard library header "array"
* #2040: Fix an issue in test/ctorleak that occured when disabling exceptions.
* #2011: Remove error counting in OpenMP parallel section in Eigen's GEMM parallelizing logic.
* #2012: Define coeff-wise binary array operators for base class to fix an issue when using Eigen with C++20
* Commit bfdd4a990: Fix an issue with Intel® MKL PARDISO support.

## [3.3.8]

Released on October 5, 2020.

Changes since 3.3.7:

* General bug fixes
  * #1995: Fix a failure in the GEBP kernel when using small L1 cache sizes, OpenMP and FMA.
  * #1990: Make CMake accept installation paths relative to `CMAKE_INSTALL_PREFIX`.
  * #1974: Fix issue when reserving an empty sparse matrix
  * #1823: Fix incorrect use of `std::abs`
  * #1788: Fix rule-of-three violations inside the stable modules. This fixes deprecated-copy warnings when compiling with GCC>=9 Also protect some additional Base-constructors from getting called by user code code (#1587)
  * #1796: Make matrix squareroot usable for Map and Ref types
  * #1281: Fix AutoDiffScalar's make_coherent for nested expression of constant ADs.
  * #1761: Fall back `is_integral` to `std::is_integral` in c++11 and fix `internal::is_integral<size_t/ptrdiff_t>` with MSVC 2013 and older.
  * #1741: Fix `self_adjoint*matrix`, `triangular*matrix`, and `triangular^1*matrix` with a destination having a non-trivial inner-stride
  * #1741: Fix SelfAdjointView::rankUpdate and product to triangular part for destination with non-trivial inner stride
  * #1741: Fix `C.noalias() = A*C;` with `C.innerStride()!=1`
  * #1695: Fix a numerical robustness issue in BDCSVD
  * #1692: Enable enum as sizes of Matrix and Array
  * #1689: Fix used-but-marked-unused warning
  * #1679: Avoid possible division by 0 in complex-schur
  * #1676: Fix C++17 template deduction in DenseBase
  * #1669: Fix PartialPivLU/inverse with zero-sized matrices.
  * #1557: Fix RealSchur and EigenSolver for matrices with only zeros on the diagonal.
* Performance related fixes
  * #1562: Optimize evaluation of small products of the form s*A*B by rewriting them as: s*(A.lazyProduct(B)) to save a costly temporary. Measured speedup from 2x to 5x...
  * Commit 165db26dc and 8ee2e10af: Fix performance issue with SimplicialLDLT for complexes coefficients
* Misc commits
  * Commit 5f1082d0b: Fix `QuaternionBase::cast` for quaternion map and wrapper.
  * Commit a153dbae9: Fix case issue with Lapack unit tests.
  * Commit 3d7e2a1f3: Fix possible conflict with an externally defined "real" type when using gcc-5.
  * Commit 1760432f6: Provide `numext::[u]int{32,64}_t`.
  * Commit 3d18879fc: Initialize isometric transforms like affine transforms.
  * Commit 160c0a340: Change typedefs from private to protected to fix MSVC compilation.
  * Commit 3cf273591: Fix compilation of FFTW unit test.
  * Commit 6abc9e537: Fix compilation of BLAS backend and frontend.
  * Commit 47e2f8a42: Fix real/imag namespace conflict.
  * Commit 71d0402e3: Avoid throwing in destructors.
  * Commit 0dd9643ad: Fix precision issue in `SelfAdjointEigenSolver.h`
  * Commit 6ed74ac97: Make `digits10()` return an integer.
  * Commit 841d844f9: Use pade for matrix exponential also for complex values.
  * Commit 4387298e8: Cast Index to RealScalar in SVDBase to fix an issue when RealScalar is not implicitly convertible to Index.
  * Commit fe8cd812b: Provide `EIGEN_HAS_C99_MATH` when using MSVC.
  * Commit 7c4208450: Various fixes in polynomial solver and its unit tests.
  * Commit e777674a8 and 4415d4e2d: Extend polynomial solver unit tests to complexes.
  * Commit 222ce4b49: Automatically switch between EigenSolver and ComplexEigenSolver, and fix a few Real versus Scalar issues.
  * Commit 7b93328ba: Enable construction of `Ref<VectorType>` from a runtime vector.
  * Commit c28ba89fe: Fix a problem of old gcc versions having problems with recursive #pragma GCC diagnostic push/pop.
  * Commit 210d510a9: Fix compilation with expression template scalar type.
  * Commit efd72cddc: Backport AVX512 implementation to 3.3.
  * Commit 5e484fa11: Fix StlDeque compilation issue with GCC 10.
  * Commit a796be81a: Avoid false-positive test results in non-linear optimization tests
  * Commit 9f202c6f1: Fix undefined behaviour caused by uncaught exceptions in OMP section of parallel GEBP kernel.
  * Commit 4707c3aa8: Fix a bug with half-precision floats on GPUs.
* Fixed warnings
  * Commit 14db78c53: Fix some maybe-uninitialized warnings in AmbiVector.h and test bdcsvd.
  * Commit f1b1f13d3: silent cmake warnings in Lapack CMakeLists.txt
  * Commit 8fb28db12: Rename variable which shadows class name in Polynomials module.
  * Commit f1c12d8ff: Workaround gcc's `alloc-size-larger-than=` warning in DenseStorage.h
  * Commit 6870a39fe: Hide some unused variable warnings in g++8.1 in Tensor contraction mapper.
  * Commit bb9981e24: Fix gcc 8.1 warning: "maybe use uninitialized" in std tests
  * Commit eea99eb4e: Fix always true warning with gcc 4.7in test numext.
  * Commit 65a6d4151: Fix nonnull-compare warning in test geo_alignedbox.
  * Commit 74a0c08d7: Disable ignoring attributes warning in vectorization logic test.
  * Commit 6c4d57dc9: Fix a gcc7 warning about bool * bool in abs2 default implementation.
  * Commit 89a86ed42: Fix a warning in SparseSelfAdjointView about a branch statement always evaluation to false.

## [3.3.8-rc1]

Released on September 14, 2020.

Changes since 3.3.7:

* General bug fixes
  * #1974: Fix issue when reserving an empty sparse matrix
  * #1823: Fix incorrect use of `std::abs`
  * #1788: Fix rule-of-three violations inside the stable modules. This fixes deprecated-copy warnings when compiling with GCC>=9 Also protect some additional Base-constructors from getting called by user code code (#1587)
  * #1796: Make matrix squareroot usable for Map and Ref types
  * #1281: Fix AutoDiffScalar's `make_coherent` for nested expression of constant ADs.
  * #1761: Fall back `is_integral` to `std::is_integral` in c++11 and fix `internal::is_integral<size_t/ptrdiff_t>` with MSVC 2013 and older.
  * #1741: Fix `self_adjoint*matrix`, `triangular*matrix`, and `triangular^1*matrix` with a destination having a non-trivial inner-stride
  * #1741: Fix SelfAdjointView::rankUpdate and product to triangular part for destination with non-trivial inner stride
  * #1741: Fix `C.noalias() = A*C;` with `C.innerStride()!=1`
  * #1695: Fix a numerical robustness issue in BDCSVD
  * #1692: Enable enum as sizes of Matrix and Array
  * #1689: Fix used-but-marked-unused warning
  * #1679: Avoid possible division by 0 in complex-schur
  * #1676: Fix C++17 template deduction in DenseBase
  * #1669: Fix PartialPivLU/inverse with zero-sized matrices.
  * #1557: Fix RealSchur and EigenSolver for matrices with only zeros on the diagonal.
* Performance related fixes
  * #1562: Optimize evaluation of small products of the form s*A*B by rewriting them as: s*(A.lazyProduct(B)) to save a costly temporary. Measured speedup from 2x to 5x...
  * Commit 165db26dc and 8ee2e10af: Fix performance issue with SimplicialLDLT for complexes coefficients
* Misc commits
  * Commit 5f1082d0b: Fix `QuaternionBase::cast` for quaternion map and wrapper.
  * Commit a153dbae9: Fix case issue with Lapack unit tests.
  * Commit 3d7e2a1f3: Fix possible conflict with an externally defined "real" type when using gcc-5.
  * Commit 1760432f6: Provide `numext::[u]int{32,64}_t`.
  * Commit 3d18879fc: Initialize isometric transforms like affine transforms.
  * Commit 160c0a340: Change typedefs from private to protected to fix MSVC compilation.
  * Commit 3cf273591: Fix compilation of FFTW unit test.
  * Commit 6abc9e537: Fix compilation of BLAS backend and frontend.
  * Commit 47e2f8a42: Fix real/imag namespace conflict.
  * Commit 71d0402e3: Avoid throwing in destructors.
  * Commit 0dd9643ad: Fix precision issue in SelfAdjointEigenSolver.h
  * Commit 6ed74ac97: Make digits10() return an integer.
  * Commit 841d844f9: Use pade for matrix exponential also for complex values.
  * Commit 4387298e8: Cast Index to RealScalar in SVDBase to fix an issue when RealScalar is not implicitly convertible to Index.
  * Commit fe8cd812b: Provide `EIGEN_HAS_C99_MATH` when using MSVC.
  * Commit 7c4208450: Various fixes in polynomial solver and its unit tests.
  * Commit e777674a8 and 4415d4e2d: Extend polynomial solver unit tests to complexes.
  * Commit 222ce4b49: Automatically switch between EigenSolver and ComplexEigenSolver, and fix a few Real versus Scalar issues.
  * Commit 5110d803e: Change license from LGPL to MPL2 with agreement from David Harmon. (grafted from 2df4f0024666a9085fe47f14e2290bd61676dbbd )
  * Commit 7b93328ba: Enable construction of `Ref<VectorType>` from a runtime vector.
  * Commit c28ba89fe: Fix a problem of old gcc versions having problems with recursive #pragma GCC diagnostic push/pop.
  * Commit 210d510a9: Fix compilation with expression template scalar type.
* Fixed warnings
  * Commit 14db78c53: Fix some maybe-uninitialized warnings in AmbiVector.h and test bdcsvd.
  * Commit f1b1f13d3: silent cmake warnings in Lapack CMakeLists.txt
  * Commit 8fb28db12: Rename variable which shadows class name in Polynomials module.
  * Commit f1c12d8ff: Workaround gcc's `alloc-size-larger-than=` warning in DenseStorage.h
  * Commit 6870a39fe: Hide some unused variable warnings in g++8.1 in Tensor contraction mapper.
  * Commit bb9981e24: Fix gcc 8.1 warning: "maybe use uninitialized" in std tests
  * Commit eea99eb4e: Fix always true warning with gcc 4.7in test `numext`.
  * Commit 65a6d4151: Fix nonnull-compare warning in test `geo_alignedbox`.
  * Commit 74a0c08d7: Disable ignoring attributes warning in vectorization logic test.
  * Commit 6c4d57dc9: Fix a gcc7 warning about bool * bool in abs2 default implementation.
  * Commit efd72cddc: Backport AVX512 implementation to 3.3.
  * Commit 5e484fa11: Fix StlDeque compilation issue with GCC 10.
  * Commit 89a86ed42: Fix a warning in SparseSelfAdjointView about a branch statement always evaluation to false.
  * Commit dd6de618: Fix a bug with half-precision floats on GPUs.

## [3.3.7]

Released on December 11, 2018.

Changes since 3.3.6:

* #1643: Fix compilation with GCC>=6 and compiler optimization turned off.

## [3.3.6]

Released on December 10, 2018.

Changes since 3.3.5:

* #1617: Fix triangular solve crashing for empty matrix.
* #785: Make dense Cholesky decomposition work for empty matrices.
* #1634: Remove double copy in move-ctor of non movable Matrix/Array.
* Changeset 588e1eb34eff: Workaround weird MSVC bug.
* #1637 Workaround performance regression in matrix products with gcc>=6 and clang>=6.0.
* Changeset bf0f100339c1: Fix some implicit 0 to Scalar conversions.
* #1605: Workaround ABI issue with vector types (aka `__m128`) versus scalar types (aka float).
* Changeset d1421c479baa: Fix for gcc<4.6 regarding usage of #pragma GCC diagnostic push/pop.
* Changeset c20b83b9d736: Fix conjugate-gradient for right-hand-sides with a very small magnitude.
* Changeset 281a877a3bf7: Fix product of empty arrays (returned 0 instead of 1).
* #1590: Fix collision with some system headers defining the macro FP32.
* #1584: Fix possible undefined behavior in random generation.
* Changeset d632d18db8ca: Fix fallback to BLAS for rankUpdate.
* Fixes for NVCC 9.
* Fix matrix-market IO.
* Various fixes in the doc.
* Various minor warning fixes/workarounds.

## [3.3.5]

Released on July 23, 2018.

Changes since 3.3.4:

* General bug fixes:
  * Fix GeneralizedEigenSolver when requesting for eigenvalues only (0d15855abb30)
  * #1560 fix product with a 1x1 diagonal matrix (90d7654f4a59)
  * #1543: fix linear indexing in generic block evaluation
  * Fix compilation of product with inverse transpositions (e.g., `mat * Transpositions().inverse()`) (14a13748d761)
  * #1509: fix `computeInverseWithCheck` for complexes (8be258ef0b6d)
  * #1521: avoid signalling `NaN` in hypot and make it std::complex<> friendly (a9c06b854991).
  * #1517: fix triangular product with unit diagonal and nested scaling factor: `(s*A).triangularView<UpperUnit>()*B` (a546d43bdd4f)
  * Fix compilation of stableNorm for some odd expressions as input (499e982b9281)
  * #1485: fix linking issue of non template functions (ae28c2aaeeda)
  * Fix overflow issues in BDCSVD (92060f82e1de)
  * #1468: add missing `std::` to `memcpy` (4565282592ae)
  * #1453: fix Map with non-default inner-stride but no outer-stride (af00212cf3a4)
  * Fix mixing types in sparse matrix products (7e5fcd0008bd)
  * #1544: Generate correct Q matrix in complex case (c0c410b508a1)
  * #1461: fix compilation of `Map<const Quaternion>::x()` (69652a06967d)

* Backends:
  * Fix MKL backend for symmetric eigenvalues on row-major matrices (4726d6a24f69)
  * #1527: fix support for MKL's VML (972424860545)
  * Fix incorrect ldvt in LAPACKE call from JacobiSVD (88c4604601b9)
  * Fix support for MKL's BLAS when using `MKL_DIRECT_CALL` (205731b87e19, b88c70c6ced7, 46e2367262e1)
  * Use MKL's lapacke.h header when using MKL (19bc9df6b726)

* Diagnostics:
  * #1516: add assertion for out-of-range diagonal index in `MatrixBase::diagonal(i)` (783d38b3c78c)
  * Add static assertion for fixed sizes `Ref<>` (e1203d5ceb8e)
  * Add static assertion on selfadjoint-view's UpLo parameter. (b84db94c677e, 0ffe8a819801)
  * #1479: fix failure detection in LDLT (67719139abc3)

* Compiler support:
  * #1555: compilation fix with XLC
  * Workaround MSVC 2013 ambiguous calls (1c7b59b0b5f4)
  * Adds missing `EIGEN_STRONG_INLINE` to help MSVC properly inlining small vector calculations (1ba3f10b91f2)
  * Several minor warning fixes: 3c87fc0f1042, ad6bcf0e8efc, "used uninitialized" (20efc44c5500), Wint-in-bool-context (131da2cbc695, b4f969795d1b)
  * #1428: make NEON vectorization compilable by MSVC. (* 3d1b3dbe5927, 4e1b7350182a)
  * Fix compilation and SSE support with PGI compiler (faabf000855d 90d33b09040f)
  * #1555: compilation fix with XLC (23eb37691f14)
  * #1520: workaround some `-Wfloat-equal` warnings by calling `std::equal_to` (7d9a9456ed7c)
  * Make the TensorStorage class compile with clang 3.9 (eff7001e1f0a)
  * Misc: some old compiler fixes (493691b29be1)
  * Fix MSVC warning C4290: C++ exception specification ignored except to indicate a function is not `__declspec(nothrow)` (524918622506)

* Architecture support:
  * Several AVX512 fixes for `log`, `sqrt`, `rsqrt`, non `AVX512ER` CPUs, `apply_rotation_in_the_plane` b64275e912ba cab3d626a59e 7ce234652ab9, d89b9a754371.
  * AltiVec fixes: 9450038e380d
  * NEON fixes: const-cast (e8a69835ccda), compilation of Jacobi rotations (c06cfd545b15,#1436).
  * Changeset d0658cc9d4a2: Define `pcast<>` for SSE types even when AVX is enabled. (otherwise float are silently reinterpreted as int instead of being converted)
  * #1494: makes `pmin`/`pmax` behave on Altivec/VSX as on x86 regarding NaNs (d0af83f82b19)

* Documentation:
  * Update manual pages regarding BDCSVD (#1538)
  * Add aliasing in common pitfaffs (2a5a8408fdc5)
  * Update `aligned_allocator` (21e03aef9f2b)
  * #1456: add perf recommendation for LLT and storage format (c8c154ebf130,  9aef1e23dbe0)
  * #1455: Cholesky module depends on Jacobi for rank-updates (2e6e26b851a8)
  * #1458: fix documentation of LLT and LDLT `info()` method (2a4cf4f473dd)
  * Warn about constness in `LLT::solveInPlace` (518f97b69bdf)
  * Fix lazyness of `operator*` with CUDA (c4dbb556bd36)
  * #336: improve doc for `PlainObjectBase::Map` (13dc446545fe)

* Other general improvements:
  * Enable linear indexing in generic block evaluation (31537598bf83, 5967bc3c2cdb, #1543).
  * Fix packet and alignment propagation logic of `Block<Xpr>` expressions. In particular, `(A+B).col(j)` now preserve vectorisation. (b323cc9c2c7f)
  * Several fixes regarding custom scalar type support: hypot (f8d6c791791d), boost-multiprec (acb8ef9b2478), literal casts (6bbd97f17534, 39f65d65894f),
  * LLT: avoid making a copy when decomposing in place (2f7e28920f4e), const the arg to `solveInPlace()` to allow passing `.transpose()`, `.block()`, etc. (c31c0090e998).
  * Add possibility to overwrite `EIGEN_STRONG_INLINE` (7094bbdf3f4d)
  * #1528: use `numeric_limits::min()` instead of `1/highest()` that might underflow (dd823c64ade7)
  * #1532: disable `stl::*_negate` in C++17 (they are deprecated) (88e9452099d5)
  * Add C++11 `max_digits10` for half (faf74dde8ed1)
  * Make sparse QR result sizes consistent with dense QR (4638bc4d0f96)

* Unsupported/unit-tests/cmake/unvisible internals/etc.
  * #1484: restore deleted line for 128 bits long doubles, and improve dispatching logic. (dffc0f957f19)
  * #1462: remove all occurences of the deprecated `__CUDACC_VER__` macro by introducing `EIGEN_CUDACC_VER` (a201b8438d36)
  * Changeset 2722aa8eb93f: Fix oversharding bug in parallelFor.
  * Changeset ea1db80eab46: commit 45e9c9996da790b55ed9c4b0dfeae49492ac5c46 (HEAD -> memory_fix)
  * Changeset 350957be012c: Fix int versus Index
  * Changeset 424038431015: fix linking issue
  * Changeset 3f938790b7e0: Fix short vs long
  * Changeset ba14974d054a: Fix cmake scripts with no fortran compiler
  * Changeset 2ac088501976: add cmake-option to enable/disable creation of tests
  * Changeset 56996c54158b: Use col method for column-major matrix
  * Changeset 762373ca9793: #1449: fix `redux_3` unit test
  * Changeset eda96fd2fa30: Fix uninitialized output argument.
  * Changeset 75a12dff8ca4: Handle min/max/inf/etc issue in `cuda_fp16.h` directly in `test/main.h`
  * Changeset 568614bf79b8: Add tests for sparseQR results (value and size) covering bugs 1522 and 1544
  * Changeset 12c9ece47d14: `SelfAdjointView<...,Mode>` causes a static assert since commit c73a77e47db8
  * Changeset 899fd2ef704f: weird compilation issue in `mapped_matrix.cpp`

## [3.3.4]

Released on June 15, 2017.

Changes since 3.3.3:

* General:
  * Improve speed of Jacobi rotation when mixing complex and real types.
  * #1405: enable StrictlyLower/StrictlyUpper triangularView as the destination of matrix*matrix products.
  * UmfPack support: enable changes in the control settings and add report functions.
  * #1423: fix LSCG's Jacobi preconditioner for row-major matrices.
  * #1424: fix compilation issue with abs and unsigned integers as scalar type.
  * #1410: fix lvalue propagation of Array/Matrix-Wrapper with a const nested expression.
  * #1403: fix several implicit scalar type conversion making SVD decompositions compatible with ceres::Jet.
  * Fix some real-to-scalar-to-real useless conversions in `ColPivHouseholderQR`.
* Regressions:
  * Fix `dense * sparse_selfadjoint_view` product.
  * #1417: make LinSpace compatible with std::complex.
  * #1400: fix `stableNorm` alignment issue with `EIGEN_DONT_ALIGN_STATICALLY`.
  * #1411: fix alignment issue in `Quaternion`.
  * Fix compilation of operations between nested Arrays.
  * #1435: fix aliasing issue in expressions like: `A = C - B*A`.
* Others:
  * Fix compilation with gcc 4.3 and ARM NEON.
  * Fix prefetches on ARM64 and ARM32.
  * Fix out-of-bounds check in COLAMD.
  * Few minor fixes regarding nvcc/CUDA support, including #1396.
  * Improve cmake scripts for Pastix and BLAS detection.
  * #1401: fix compilation of `cond ? x : -x` with `x` an `AutoDiffScalar`
  * Fix compilation of matrix log with Map as input.
  * Add specializations of `std::numeric_limits` for `Eigen::half` and and `AutoDiffScalar`
  * Fix compilation of streaming nested Array, i.e., `std::cout << Array<Array<...>>`

## [3.3.3]

Released on February 21, 2017.

Changes since 3.3.2:

* General:
  * Improve multi-threading heuristic for matrix products with a small number of columns.
  * #1395: fix compilation of JacobiSVD for vectors type.
  * Fix pruning in `(sparse*sparse).pruned()` when the result is nearly dense.
  * #1382: move using `std::size_t`/`ptrdiff_t` to Eigen's namespace.
  * Fix compilation and inlining when using clang-cl with visual studio.
  * #1392: fix `#include <Eigen/Sparse>` with mpl2-only.
* Regressions:
  * #1379: fix compilation in `sparse*diagonal*dense` with OpenMP.
  * #1373: add missing assertion on size mismatch with compound assignment operators (e.g., mat += mat.col(j))
  * #1375: fix cmake installation with cmake 2.8.
  * #1383: fix LinSpaced with integers for `LinPspaced(n,0,n-1)` with `n==0` or the `high<low` case.
  * #1381: fix `sparse.diagonal()` used as a rvalue.
  * #1384: fix evaluation of "sparse/scalar" that used the wrong evaluation path.
  * #478: fix regression in the eigen decomposition of zero matrices.
  * Fix a compilation issue with MSVC regarding the usage of `CUDACC_VER`
  * #1393: enable Matrix/Array explicit constructor from types with conversion operators.
  * #1394: fix compilation of `SelfAdjointEigenSolver<Matrix>(sparse*sparse)`.
* Others:
  * Fix ARM NEON wrapper for 16 byte systems.
  * #1391: include IO.h before DenseBase to enable its usage in DenseBase plugins.
  * #1389: fix std containers support with MSVC and AVX.
  * #1380: fix matrix exponential with `Map<>`.
  * #1369: fix type mismatch warning with OpenMP.
  * Fix usage of `size_t` instead of Index in sefl-adjoint `matrix * vector`
  * #1378: fix doc (`DiagonalIndex` vs `Diagonal`).

## [3.3.2]

Released on January 18, 2017.

Changes since 3.3.1:

* General:
  * Add `transpose`, `adjoint`, `conjugate` methods to `SelfAdjointView` (useful to write generic code)
  * Make sure that `HyperPlane::transform` maintains a unit normal vector in the Affine case.
  * Several documentation improvements, including: several doxygen workarounds, #1336, #1370, StorageIndex, selfadjointView, sparseView(), sparse triangular solve, AsciiQuickReference.txt, ...
* Regressions:
  * #1358: fix compilation of `sparse += sparse.selfadjointView()`.
  * #1359: fix compilation of `sparse /=scalar`, `sparse *=scalar`, and `col_major_sparse.row() *= scalar`.
  * #1361:  fix compilation of mat=perm.inverse()
  * Some fixes in sparse coeff-wise binary operations: add missing `.outer()` member to iterators, and properly report storage order.
  * Fix aliasing issue in code as `A.triangularView() = B*A.sefladjointView()*B.adjoint()`
* Performance:
  * Improve code generation for `mat*vec` on some compilers.
  * Optimize horizontal adds in SSE3 and AVX.
  * Speed up row-major TRSM (triangular solve with a matrix as right-hand-side) by reverting `vec/y` to `vec*(1/y)`. The rationale is:
    * div is extremely costly
    * this is consistent with the column-major case
    * this is consistent with all other BLAS implementations
  * Remove one temporary in `SparseLU::solve()`
* Others:
  * Fix BLAS backend for symmetric rank K updates.
  * #1360: fix `-0` vs `+0` issue with Altivec
  * #1363: fix mingw's ABI issue
  * #1367: fix compilation with gcc 4.1.
  * Fix ABI issue with AVX and old gcc versions.
  * Fix some warnings with ICC, Power8, etc.
  * Fix compilation with MSVC 2017

## [3.3.1]

Released on December 06, 2016.

Changes since 3.3.0:

* #426: add operators `&&` and `||` to dense and sparse matrices (only dense arrays were supported)
* #1319: add support for CMake's imported targets.
* #1343: fix compilation regression in `array = matrix_product` and `mat+=selfadjoint_view`
* Fix regression in assignment of sparse block to sparse block.
* Fix a memory leak in `Ref<SparseMatrix>` and `Ref<SparseVector>`.
* #1351: fix compilation of random with old compilers.
* Fix a performance regression in (mat*mat)*vec for which mat*mat was evaluated multiple times.
* Fix a regression in `SparseMatrix::ReverseInnerIterator`
* Fix performance issue of products for dynamic size matrices with fixed max size.
* implement `float`/`std::complex<float>` for ZVector
* Some fixes for expression-template scalar-types
* #1356: fix undefined behavior with nullptr.
* Workaround some compilation errors with MSVC and MSVC/clr
* #1348: document `EIGEN_MAX_ALIGN_BYTES` and `EIGEN_MAX_STATIC_ALIGN_BYTES`, and reflect in the doc that `EIGEN_DONT_ALIGN*` are deprecated.
* Bugs #1346,#1347: make Eigen's installation relocatable.
* Fix some harmless compilation warnings.

## [3.3]

Released on November 10, 2016

For a comprehensive list of change since the 3.2 series, see this [page](https://www.eigen.tuxfamily.org/index.php?title=3.3).


Main changes since 3.3-rc2:
* Fix regression in printing sparse expressions.
* Fix sparse solvers when using a SparseVector as the result and/or right-hand-side.

## [3.3-rc2]

Released on November 04, 2016

For a comprehensive list of change since the 3.2 series, see this [page](https://www.eigen.tuxfamily.org/index.php?title=3.3).

Main changes since 3.3-rc1:
* Core module
  * Add supports for AVX512 SIMD instruction set.
  * Bugs #698 and  #1004: Improve numerical robustness of LinSpaced methods for both real and integer scalar types ([details](http://eigen.tuxfamily.org/dox-devel/classEigen_1_1DenseBase.html#aaef589c1dbd7fad93f97bd3fa1b1e768)).
  * Fix a regression in `X = (X*X.transpose())/scalar` with `X` rectangular (`X` was resized before the evaluation).
  * #1311: Fix alignment logic in some cases of `(scalar*small).lazyProduct(small)`
  * #1317:  fix a performance regression from 3.2 with clang and some nested block expressions.
  * #1308: fix compilation of some small products involving nullary-expressions.
  * #1333: Fix a regression with `mat.array().sum()`
  * #1328: Fix a compilation issue with old compilers introduced in 3.3-rc1.
  * #1325: Fix compilation on NEON with clang
  * Properly handle negative inputs in vectorized sqrt.
  * Improve cost-model to determine the ideal number of threads in matrix-matrix products.
* Geometry module
  * #1304: Fix `Projective * scaling` and `Projective *= scaling`.
  * #1310: Workaround a compilation regression from 3.2 regarding triangular * homogeneous
  * #1312: Quaternion to AxisAngle conversion now ensures the angle will be in the range `[0,pi]`. This also increases accuracy when `q_w` is negative.
* Tensor module
  * Add support for OpenCL.
  * Improved random number generation.
* Other
  * #1330: SuiteSparse, explicitly handle the fact that Cholmod does not support single precision float numbers.
  * SuiteSparse, fix SPQR for rectangular matrices
  * Fix compilation of `qr.inverse()` for column and full pivoting variants

## [3.2.10]

Released on October 04, 2016

Changes since 3.2.9:

Main fixes and improvements:
* #1272: Core module, improve comma-initializer in handling empty matrices.
* #1276: Core module, remove all references to `std::binder*` when C++11 is enabled (those are deprecated).
* #1304: Geometry module, fix `Projective * scaling` and `Projective *= scaling`.
* #1300: Sparse module, compilation fix for some block expression and SPQR support.
* Sparse module, fix support for row (resp. column) of a column-major (resp. row-major) sparse matrix.
* LU module, fix 4x4 matrix inversion for non-linear destinations.
* Core module, a few fixes regarding custom complex types.
* #1275: backported improved random generator from 3.3
* Workaround MSVC 2013 compilation issue in Reverse
* Fix UmfPackLU constructor for expressions.
* #1273: fix shortcoming in eigen_assert macro
* #1249:  disable the use of `__builtin_prefetch` for compilers other than GCC, clang, and ICC.
* #1265: fix doc of QR decompositions

## [3.3-rc1]

Released on September 22, 2016

For a comprehensive list of change since the 3.2 series, see this [page](https://www.eigen.tuxfamily.org/index.php?title=3.3).

Main changes since 3.3-beta2:

* New features and improvements:
  * #645: implement eigenvector computation in GeneralizedEigenSolver
  * #1271: add a `SparseMatrix::coeffs()` method returning a linear view of the non-zeros (for compressed mode only).
  * #1286: Improve support for custom nullary functors: now the functor only has to expose one relevant operator among `f()`, `f(i)`, `f(i,j)`.
  * #1272: improve comma-initializer in handling empty matrices.
  * #1268: detect failure in LDLT and report them through info()
  * Add support for scalar factor in sparse-selfadjoint `*` dense products, and enable `+=`/`-=` assignment for such products.
  * Remove temporaries in product expressions matching `d?=a-b*c` by rewriting them as `d?=a; d?=b*c;`
  * Vectorization improvements for some small product cases.

* Doc:
  * #1265: fix outdated doc in QR facto
  * #828: improve documentation of sparse block methods, and sparse unary methods.
  * Improve documentation regarding nullary functors, and add an example demonstrating the use of nullary expression to perform fancy matrix manipulations.
  * Doc: explain how to use Accelerate as a LAPACK backend.

* Bug fixes and internal changes:
  * Numerous fixes regarding support for custom complex types.
  * #1273: fix shortcoming in `eigen_assert` macro
  * #1278: code formatting
  * #1270: by-pass hand written `pmadd` with recent clang versions.
  * #1282: fix implicit double to float conversion warning
  * #1167: simplify installation of header files using cmake's `install(DIRECTORY ...)` command
  * #1283: fix products involving an uncommon `vector.block(..)` expressions.
  * #1285: fix a minor regression in LU factorization.
  * JacobiSVD now consider any denormal number as zero.
  * Numerous fixes regarding support for CUDA/NVCC (including bugs #1266)
  * Fix an alignment issue in gemv, symv, and trmv for statically allocated temporaries.
  * Fix 4x4 matrix inversion for non-linear destinations.
  * Numerous improvements and fixes in half precision scalar type.
  * Fix vectorization logic for coeff-based product for some corner cases
  * Bugs #1260, #1261, #1264: several fixes in AutoDiffScalar.

## [3.3-beta2]

Released on July 26, 2016

For a comprehensive list of change since the 3.2 series, see this [page](https://www.eigen.tuxfamily.org/index.php?title=3.3).

Main changes since 3.3-beta1:

* Dense features:
  * #707: Add support for [inplace](http://eigen.tuxfamily.org/dox-devel/group__InplaceDecomposition.html) dense decompositions.
  * #977: normalize(d) left the input unchanged if its norm is 0 or too close to 0.
  * #977: add stableNormalize[d] methods: they are analogues to normalize[d] but with carefull handling of under/over-flow.
  * #279: Implement generic scalar*expr and expr*scalar operators. This is especially useful for custom scalar types, e.g., to enable `float*expr<multi_prec>` without conversion.
  * New unsupported/Eigen/SpecialFunctions module providing the following coefficient-wise math functions: erf, erfc, lgamma, digamma, polygamma, igamma, igammac, zeta, betainc.
  * Add fast reciprocal condition estimators in dense LU and Cholesky factorizations.
  * #1230: add support for `SelfadjointView::triangularView()` and `diagonal()`
  * #823: add `Quaternion::UnitRandom()` method.
  * Add exclusive or operator for bool arrays.
  * Relax dependency on MKL for `EIGEN_USE_BLAS` and `EIGEN_USE_LAPACKE`: any BLAS and LAPACK libraries can now be used as backend (see  [doc](http://eigen.tuxfamily.org/dox-devel/TopicUsingBlasLapack.html)).
  * Add static assertion to `x()`, `y()`, `z()`, `w()` accessors
  * #51: avoid dynamic memory allocation in fixed-size rank-updates, matrix products evaluated within a triangular part, and selfadjoint times matrix products.
  * #696: enable zero-sized block at compile-time by relaxing the respective assertion
  * #779: in `Map`, allows non aligned buffers for buffers smaller than the requested alignment.
  * Add a complete orthogonal decomposition class: [CompleteOrthogonalDecomposition](http://eigen.tuxfamily.org/dox-devel/classEigen_1_1CompleteOrthogonalDecomposition.html)
  * Improve robustness of JacoviSVD with complexes (underflow, noise amplification in complex to real conversion, compare off-diagonal entries to the current biggest diagonal entry instead of the global biggest, null inputs).
  * Change Eigen's ColPivHouseholderQR to use a numerically stable norm downdate formula (changeset 9da6c621d055)
  * #1214: consider denormals as zero in D&C SVD. This also workaround infinite binary search when compiling with ICC's unsafe optimizations.
  * Add log1p for arrays.
  * #1193: now `lpNorm<Infinity>` supports empty inputs.
  * #1151: remove useless critical section in matrix product
  * Add missing non-const reverse method in `VectorwiseOp` (e.g., this enables `A.rowwise().reverse() = ...`)
  * Update RealQZ to reduce 2x2 diagonal block of T corresponding to non reduced diagonal block of S to positive diagonal form.

* Sparse features:
  * #632: add support for "dense +/- sparse" operations. The current implementation is based on SparseView to make the dense subexpression compatible with the sparse one.
  * #1095: add Cholmod*::logDeterminant/determinant functions.
  * Add `SparseVector::conservativeResize()` method
  * #946: generalize `Cholmod::solve` to handle any rhs expressions.
  * #1150: make IncompleteCholesky more robust by iteratively increase the shift until the factorization succeed (with at most 10 attempts)
  * #557: make InnerIterator of sparse storage types more versatile by adding default-ctor, copy-ctor/assignment.
  * #694: document that `SparseQR::matrixR` is not sorted.
  * Block expressions now expose all the information defining the block.
  * Fix GMRES returned error.
  * #1119: add support for SuperLU 5

* Performance improvements:
  *  #256: enable vectorization with unaligned loads/stores. This concerns all architectures and all sizes. This new behavior can be disabled by defining `EIGEN_UNALIGNED_VECTORIZE=0`
  * Add support for s390x(zEC13) ZVECTOR instruction set.
  * Optimize mixing of real with complex matrices by avoiding a conversion from real to complex when the real types do not match exactly. (see bccae23d7018)
  * Speedup square roots in performance critical methods such as norm, normalize(d).
  * #1154: use dynamic scheduling for spmv products.
  * #667,  #1181: improve perf with MSVC and ICC through `FORCE_INLINE`
  * Improve heuristics for switching between coeff-based and general matrix product implementation at compile-time.
  * Add vectorization of tanh for float (SSE/AVX)
  * Improve cost estimates of numerous functors.
  * Numerous improvements regarding half-packet vectorization: coeff-based products (e.g.,  `Matrix4f*Vector4f` is now vectorized again when using AVX), reductions, linear vs inner traversals.
  * Fix performance regression: with AVX, unaligned stores were emitted instead of aligned ones for fixed size assignment.
  * #1201: optimize `affine*vector` products.
  * #1191: prevent Clang/ARM from rewriting VMLA into VMUL+VADD.
  * Small speed-up in `Quaternion::slerp`.
  * #1201:  improve code generation of affine*vec with MSVC

* Doc:
  * Add [documentation and exemple](http://eigen.tuxfamily.org/dox-devel/group__MatrixfreeSolverExample.html) for matrix-free solving.
  * A new documentation [page](http://eigen.tuxfamily.org/dox-devel/group__CoeffwiseMathFunctions.html) summarizing coefficient-wise math functions.
  * #1144: clarify the doc about aliasing in case of resizing and matrix product.
  * A new documentation [page](http://eigen.tuxfamily.org/dox-devel/group__DenseDecompositionBenchmark.html) summarizing the true performance of Eigen's dense decomposition algorithms.

* Misc improvements:
  * Allow one generic scalar argument for all binary operators/functions.
  * Add a `EIGEN_MAX_CPP_VER` option to limit the C++ version to be used, as well as [fine grained options](http://eigen.tuxfamily.org/dox-devel/TopicPreprocessorDirectives.html#title1) to control individual language features.
  * A new [ScalarBinaryOpTraits](http://eigen.tuxfamily.org/dox-devel/structEigen_1_1ScalarBinaryOpTraits.html) class allowing to control how different scalar types are mixed.
  * `NumTraits` now exposes a `digits10` function making `internal::significant_decimals_impl` deprecated.
  * Countless improvements and fixes in Tensors module.
  * #1156: fix several function declarations whose arguments were passed by value instead of being passed by reference
  * #1164: fix `std::list` and `std::deque` specializations such that our aligned allocator is automatically activated only when the user did not specified an allocator (or specified the default std::allocator).
  * #795: mention allocate_shared as a candidate for aligned_allocator.
  * #1170: skip calls to memcpy/memmove for empty inputs.
  * #1203: by-pass large stack-allocation in stableNorm if `EIGEN_STACK_ALLOCATION_LIMIT` is too small
  * Improve constness of blas level-2/3 interface.
  * Implement stricter argument checking for SYRK and SY2K
  * Countless improvements in the documentations.
  * Internal: Remove `posix_memalign`, `_mm_malloc`, and `_aligned_malloc` special paths.
  * Internal: Remove custom unaligned loads for SSE
  * Internal: introduce `[U]IntPtr` types to be used for casting pointers to integers.
  * Internal: `NumTraits` now exposes `infinity()`
  * Internal: `EvalBeforeNestingBit` is now deprecated.
  * #1213: workaround gcc linking issue with anonymous enums.
  * #1242: fix comma initializer with empty matrices.
  * #725: make move ctor/assignment noexcept
  * Add minimal support for `Array<string>`
  * Improve support for custom scalar types bases on expression template (e.g., `boost::multiprecision::number<>` type). All dense decompositions are successfully tested.

* Most visible fixes:
  * #1144: fix regression in `x=y+A*x`  (aliasing issue)
  * #1140: fix usage of `_mm256_set_m128` and `_mm256_setr_m128` in AVX support
  * #1141: fix some missing initializations in CholmodSupport
  * #1143: workaround gcc bug #10200
  * #1145, #1147,  #1148,  #1149: numerous fixes in PastixSupport
  * #1153: don't rely on `__GXX_EXPERIMENTAL_CXX0X__` to detect C++11 support.
  * #1152: fix data race in static initialization of blas routines.
  * fix some buffer overflow in product block size computation.
  * #96,  #1006: fix by value argument in result_of
  * #178: clean several `const_cast`.
  * Fix compilation in `ceil()` function.
  * #698: fix linspaced for integer types.
  * #1161: fix division by zero for huge scalar types in cache block size computation.
  * #774: fix a numerical issue in Umeyama algorithm that produced unwanted reflections.
  * #901: fix triangular-view with unit diagonal of sparse rectangular matrices.
  * #1166: fix shortcoming in gemv when the destination is not a vector at compile-time.
  * #1172: make `SparseMatrix::valuePtr` and `innerIndexPtr` properly return null for empty matrices
  * #537:  fix a compilation issue in Quaternion with Apples's compiler
  * #1186: fix usage of `vreinterpretq_u64_f64` (NEON)
  * #1190: fix usage of `__ARM_FEATURE_FMA` on Clang/ARM
  * #1189:  fix pow/atan2 compilation for `AutoDiffScalar`
  * Fix detection of same input-output when applied permutations, or on solve operations.
  * Workaround a division by zero in triangular solve when outerstride==0
  * Fix compilation of s`parse.cast<>().transpose()`.
  * Fix double-conversion warnings throughout the code.
  *  #1207: fix logical-op warnings
  *  #1222,  #1223: fix compilation in `AutoDiffScalar`.
  *  #1229: fix usage of `Derived::Options` in MatrixFunctions.
  *  #1224: fix regression in `(dense*dense).sparseView()`.
  *  #1231: fix compilation regression regarding complex_array/=real_array.
  *  #1221: disable gcc 6 warning: ignoring attributes on template argument.
  * Workaround clang/llvm bug 27908
  *  #1236: fix possible integer overflow in sparse matrix product.
  *  #1238: fix `SparseMatrix::sum()` overload for un-compressed mode
  * #1240: remove any assumption on NEON vector types
  * Improves support for MKL's PARDISO solver.
  * Fix support for Visual 2010.
  * Fix support for gcc 4.1.
  * Fix support for ICC 2016
  * Various Altivec/VSX fixes: exp, support for clang 3.9,
  *  #1258: fix compilation of `Map<SparseMatrix>::coeffRef`
  *  #1249: fix compilation with compilers that do not support `__builtin_prefetch` .
  *  #1250: fix `pow()` for `AutoDiffScalar` with custom nested scalar type.

## [3.2.9]

Released on July 18, 2016

Changes since 3.2.8:

* Main fixes and improvements:
  * Improve numerical robustness of JacobiSVD (backported from 3.3)
  * #1017: prevents underflows in `makeHouseholder`
  * Fix numerical accuracy issue in the extraction of complex eigenvalue pairs in real generalized eigenvalue problems.
  * Fix support for `vector.homogeneous().asDiagonal()`
  * #1238: fix `SparseMatrix::sum()` overload for un-compressed mode
  * #1213: workaround gcc linking issue with anonymous enums.
  * #1236: fix possible integer overflow in sparse-sparse product
  * Improve detection of identical matrices when applying a permutation (e.g., `mat = perm * mat`)
  * Fix usage of nesting type in blas_traits. In practice, this fixes compilation of expressions such as `A*(A*A)^T`
  * CMake: fixes support of Ninja generator
  * Add a StorageIndex typedef to sparse matrices and expressions to ease porting code to 3.3 (see http://eigen.tuxfamily.org/index.php?title=3.3#Index_typedef)
  * #1200: make `aligned_allocator` c++11 compatible (backported from 3.3)
  * #1182: improve generality of `abs2` (backported from 3.3)
  * #537: fix compilation of Quaternion with Apples's compiler
  * #1176: allow products between compatible scalar types
  * #1172: make `valuePtr` and `innerIndexPtr` properly return null for empty sparse matrices.
  * #1170: skip calls to `memcpy`/`memmove` for empty inputs.

* Others:
  * #1242: fix comma initializer with empty matrices.
  * Improves support for MKL's PARDISO solver.
  * Fix a compilation issue with Pastix solver.
  * Add some missing explicit scalar conversions
  * Fix a compilation issue with matrix exponential (unsupported MatrixFunctions module).
  * #734: fix a storage order issue in unsupported Spline module
  * #1222: fix a compilation issue in AutoDiffScalar
  * #1221: shutdown some GCC6's warnings.
  * #1175: fix index type conversion warnings in sparse to dense conversion.

## [3.2.8]

Released on February 16, 2016

Changes since 3.2.7:

* Main fixes and improvements:
  * Make `FullPivLU::solve` use `rank()` instead of `nonzeroPivots()`.
  * Add `EIGEN_MAPBASE_PLUGIN`
  * #1166: fix issue in matrix-vector products when the destination is not a vector at compile-time.
  * #1100: Improve cmake/pkg-config support.
  * #1113: fix name conflict with C99's "I".
  * Add missing delete operator overloads in `EIGEN_MAKE_ALIGNED_OPERATOR_NEW`
  * Fix `(A*B).maxCoeff(i)` and similar.
  * Workaround an ICE with VC2015 Update1 x64.
  * #1156: fix several function declarations whose arguments were passed by value instead of being passed by reference
  * #1164: fix `std::list` and `std::deque` specializations such that our aligned allocator is automatically activatived only when the user did not specified an allocator (or specified the default `std::allocator`).

* Others:
  * Fix BLAS backend (aka MKL) for empty matrix products.
  * #1134: fix JacobiSVD pre-allocation.
  * #1111: fix infinite recursion in `sparse_column_major.row(i).nonZeros()` (it now produces a compilation error)
  * #1106: workaround a compilation issue in Sparse module for msvc-icc combo
  * #1153: remove the usage of `__GXX_EXPERIMENTAL_CXX0X__` to detect C++11 support
  * #1143: work-around gcc bug in COLAMD
  * Improve support for matrix products with empty factors.
  * Fix and clarify documentation of Transform wrt `operator*(MatrixBase)`
  * Add a matrix-free conjugate gradient example.
  * Fix cost computation in CwiseUnaryView (internal)
  * Remove custom unaligned loads for SSE.
  * Some warning fixes.
  * Several other documentation clarifications.

## [3.3-beta1]

Released on December 16, 2015

For a comprehensive list of change since the 3.2 series, see this [page](https://www.eigen.tuxfamily.org/index.php?title=3.3).

Main changes since 3.3-alpha1:

* Dense features:
  * Add `LU::transpose().solve()` and `LU::adjoint().solve()` API.
  * Add `Array::rsqrt()` method as a more efficient shorcut for `sqrt().inverse()`.
  * Add `Array::sign()` method for real and complexes.
  * Add `lgamma`, `erf`, and `erfc` functions for arrays.
  * Add support for row/col-wise `lpNorm()`.
  * Add missing `Rotation2D::operator=(Matrix2x2)`.
  * Add support for `permutation * homogenous`.
  * Improve numerical accuracy in LLT and triangular solve by using true scalar divisions (instead of x * (1/y)).
  * Add `EIGEN_MAPBASE_PLUGIN` and `EIGEN_QUATERNION_PLUGIN`.
  *  #1074: forbid the creation of PlainObjectBase objects.

* Sparse features:
  * Add IncompleteCholesky preconditioner.
  * Improve support for [matrix-free iterative solvers](http://eigen.tuxfamily.org/dox/group__MatrixfreeSolverExample.html)
  * Extend `setFromTriplets` API to allow passing a functor object controlling how to collapse duplicated entries.
  * #918: add access to UmfPack return code and parameters.
  * Add support for `dense.cwiseProduct(sparse)`, thus enabling `(dense*sparse).diagonal()` expressions.
  * Add support to directly evaluate the product of two sparse matrices within a dense matrix.
  * #1064: add support for `Ref<SparseVector>`.
  * Add supports for real mul/div `sparse<complex>` operations.
  * #1086: replace deprecated `UF_long` by `SuiteSparse_long`.
  * Make `Ref<SparseMatrix>` more versatile.

* Performance improvements:
  * #1115: enable static alignment and thus small size vectorization on ARM.
  * Add temporary-free evaluation of `D.nolias() *= C + A*B`.
  * Add vectorization of round, ceil and floor for SSE4.1/AVX.
  * Optimize assignment into a `Block<SparseMatrix>` by using Ref and avoiding useless updates in non-compressed mode. This make row-by-row filling of a row-major sparse matrix very efficient.
  * Improve internal cost model leading to faster code in some cases (see changeset 1bcb41187a45).
  * #1090: improve redux evaluation logic.
  * Enable unaligned vectorization of small fixed size matrix products.

* Misc improvements:
  * Improve support for `isfinite`/`isnan`/`isinf` in fast-math mode.
  * Make the IterativeLinearSolvers module compatible with MPL2-only mode by defaulting to COLAMDOrdering and NaturalOrdering for ILUT and ILLT respectively.
  * Avoid any OpenMP calls if multi-threading is explicitly disabled at runtime.
  * Make abs2 compatible with custom complex types.
  * #1109: use noexcept instead of throw for C++11 compilers.
  * #1100: Improve cmake/pkg-config support.
  * Countless improvements and fixes in Tensors module.

* Most visible fixes:
  * #1105: fix default preallocation when moving from compressed to uncompressed mode in SparseMatrix.
  * Fix UmfPackLU constructor for expressions.
  * Fix degenerate cases in syrk and trsm BLAS API.
  * Fix matrix to quaternion (and angleaxis) conversion for matrix expression.
  * Fix compilation of sparse-triangular to dense assignment.
  * Fix several minor performance issues in the nesting of matrix products.
  * #1092: fix iterative solver ctors for expressions as input.
  * #1099: fix missing include for CUDA.
  * #1102: fix multiple definition linking issue.
  * #1088: fix setIdenity for non-compressed sparse-matrix.
  * Fix `SparseMatrix::insert`/`coeffRef` for non-empty compressed matrix.
  * #1113: fix name conflict with C99's "I".
  * #1075: fix `AlignedBox::sample` for runtime dimension.
  * #1103: fix NEON vectorization of `complex<double>` multiplication.
  * #1134: fix JacobiSVD pre-allocation.
  * Fix ICE with VC2015 Update1.
  * Improve cmake install scripts.

## [3.2.7]

Released on November 5, 2015

Changes since 3.2.6:

* Main fixes and improvements:
  * Add support for `dense.cwiseProduct(sparse)`.
  * Fix a regression regarding `(dense*sparse).diagonal()`.
  * Make the `IterativeLinearSolvers` module compatible with MPL2-only mode by defaulting to `COLAMDOrdering` and `NaturalOrdering` for ILUT and ILLT respectively.
  * #266: backport support for c++11 move semantic
  * `operator/=(Scalar)` now performs a true division (instead of `mat*(1/s)`)
  * Improve numerical accuracy in LLT and triangular solve by using true scalar divisions (instead of `mat * (1/s)`)
  * #1092: fix iterative solver constructors for expressions as input
  * #1088: fix `setIdenity` for non-compressed sparse-matrix
  * #1086: add support for recent SuiteSparse versions

* Others:
  * Add overloads for real-scalar times `SparseMatrix<complex>` operations. This avoids real to complex conversions, and also fixes a compilation issue with MSVC.
  * Use explicit Scalar types for AngleAxis initialization
  * Fix several shortcomings in cost computation (avoid multiple re-evaluation in some very rare cases).
  * #1090: fix a shortcoming in redux logic for which slice-vectorization plus unrolling might happen.
  * Fix compilation issue with MSVC by backporting `DenseStorage::operator=` from devel branch.
  * #1063: fix nesting of unsupported/AutoDiffScalar to prevent dead references when computing second-order derivatives
  * #1100: remove explicit `CMAKE_INSTALL_PREFIX` prefix to conform to cmake install's `DESTINATION` parameter.
  * unsupported/ArpackSupport is now properly installed by make install.
  * #1080: warning fixes

## [3.2.6]

Released on October 1, 2015

Changes since 3.2.5:

* fix some compilation issues with MSVC 2013, including bugs #1000 and #1057
* SparseLU: fixes to support `EIGEN_DEFAULT_TO_ROW_MAJOR` (#1053), and for empty (#1026) and some structurally rank deficient matrices (#792)
* #1075: fix `AlignedBox::sample()` for Dynamic dimension
* fix regression in AMD ordering when a column has only one off-diagonal non-zero (used in sparse Cholesky)
* fix Jacobi preconditioner with zero diagonal entries
* fix Quaternion identity initialization for non-implicitly convertible types
* #1059: fix `predux_max<Packet4i>` for NEON
* #1039: fix some issues when redefining `EIGEN_DEFAULT_DENSE_INDEX_TYPE`
* #1062: fix SelfAdjointEigenSolver for RowMajor matrices
* MKL: fix support for the 11.2 version, and fix a naming conflict (#1067)
  * #1033: explicit type conversion from 0 to RealScalar

## [3.3-alpha1]

Released on September 4, 2015

See the [announcement](https://www.eigen.tuxfamily.org/index.php?title=3.3).

## [3.2.5]

Released on June 16, 2015

Changes since 3.2.4:

* Changes with main impact:
  * Improve robustness of SimplicialLDLT to semidefinite problems by correctly handling structural zeros in AMD reordering
  * Re-enable supernodes in SparseLU (fix a performance regression in SparseLU)
  * Use zero guess in `ConjugateGradients::solve`
  * Add `PermutationMatrix::determinant` method
  * Fix `SparseLU::signDeterminant()` method, and add a SparseLU::determinant() method
  * Allows Lower|Upper as a template argument of CG and MINRES: in this case the full matrix will be considered
  * #872: remove usage of std::bind* functions (deprecated in c++11)

* Numerical robustness improvements:
  * #1014: improve numerical robustness of the 3x3 direct eigenvalue solver
  * #1013: fix 2x2 direct eigenvalue solver for identical eigenvalues
  * #824: improve accuracy of `Quaternion::angularDistance`
  * #941: fix an accuracy issue in ColPivHouseholderQR by continuing the decomposition on a small pivot
  * #933: improve numerical robustness in RealSchur
  * Fix default threshold value in SPQR

* Other changes:
  * Fix usage of `EIGEN_NO_AUTOMATIC_RESIZING`
  * Improved support for custom scalar types in SparseLU
  * Improve cygwin compatibility
  * #650: fix an issue with sparse-dense product and rowmajor matrices
  * #704: fix MKL support (HouseholderQR)
  * #705: fix handling of Lapack potrf return code (LLT)
  * #714: fix matrix product with OpenMP support
  * #949: add static assertions for incompatible scalar types in many of the dense decompositions
  * #957,  #1000: workaround MSVC/ICC compilation issues when using sparse blocks
  * #969: fix ambiguous calls to Ref
  * #972, #986: add support for coefficient-based product with 0 depth
  * #980: fix taking a row (resp. column) of a column-major (resp. row-major) sparse matrix
  * #983: fix an alignement issue in Quaternion
  * #985: fix RealQZ when either matrix had zero rows or columns
  * #987: fix alignement guess in diagonal product
  * #993: fix a pitfall with matrix.inverse()
  * #996, #1016: fix scalar conversions
  * #1003: fix handling of pointers non aligned on scalar boundary in slice-vectorization
  * #1010: fix member initialization in IncompleteLUT
  * #1012: enable alloca on Mac OS or if alloca is defined as macro
  * Doc and build system: #733, #914, #952,  #961, #999

## [3.2.4]

Released on January 21, 2015

Changes since 3.2.3:

* Fix compilation regression in Rotation2D
* #920: fix compilation issue with MSVC 2015.
* #921: fix utilization of bitwise operation on enums in `first_aligned`.
* Fix compilation with NEON on some platforms.

## [3.2.3]

Released on December 16, 2014

Changes since 3.2.2:

* Core:
  * Enable `Mx0 * 0xN` matrix products.
  * #859: fix returned values for vectorized versions of `exp(NaN)`, `log(NaN)`, `sqrt(NaN)` and `sqrt(-1)`.
  * #879: tri1 = mat * tri2 was compiling and running incorrectly if tri2 was not numerically triangular. Workaround the issue by evaluating mat*tri2 into a temporary.
  * #854: fix numerical issue in SelfAdjointEigenSolver::computeDirect for 3x3 matrices.
  * #884: make sure there no call to malloc for zero-sized matrices or for a Ref<> without temporaries.
  * #890: fix aliasing detection when applying a permutation.
  * #898: MSVC optimization by adding inline hint to const_cast_ptr.
  * #853: remove enable_if<> in Ref<> ctor.

* Dense solvers:
  * #894: fix the sign returned by LDLT for multiple calls to `compute()`.
  * Fix JacobiSVD wrt underflow and overflow.
  * #791: fix infinite loop in JacobiSVD in the presence of NaN.

* Sparse:
  * Fix out-of-bounds memory write when the product of two sparse matrices is completely dense and performed using pruning.
  * UmfPack support: fix redundant evaluation/copies when calling `compute()`, add support for generic expressions as input, and fix extraction of the L and U factors (#911).
  * Improve `SparseMatrix::block` for const matrices (the generic path was used).
  * Fix memory pre-allocation when permuting inner vectors of a sparse matrix.
  * Fix `SparseQR::rank` for a completely empty matrix.
  * Fix `SparseQR` for row-major inputs.
  * Fix `SparseLU::absDeterminant` and add respective unit test.
  * BiCGSTAB: make sure that good initial guesses are not destroyed by a bad preconditioner.

* Geometry:
  * Fix `Hyperplane::Through(a,b,c)` when points are aligned or identical.
  * Fix linking issues in OpenGLSupport.

* OS, build system and doc:
  * Various compilation fixes including: #821,  #822, #857, #871, #873.
  * Fix many compilation warnings produced by recent compilers including: #909.
  * #861: enable posix_memalign with PGI.
  * Fix BiCGSTAB doc example.

## [3.2.2]

Released on August 4, 2014

Changes since 3.2.1:

* Core:
  * Relax Ref such that `Ref<MatrixXf>` accepts a `RowVectorXf` which can be seen as a degenerate `MatrixXf(1,N)`
  * Fix performance regression for the vectorization of sub columns/rows of matrices.
  * `EIGEN_STACK_ALLOCATION_LIMIT`: Raise its default value to 128KB, make use of it to assert on  maximal fixed size object, and allows it to be 0 to mean "no limit".
  * #839: Fix 1x1 triangular matrix-vector product.
  * #755: `CommaInitializer` produced wrong assertions in absence of Return-Value-Optimization.

* Dense solvers:
  * Add a `rank()` method with threshold control to JacobiSVD, and make solve uses it to return the minimal norm solution for rank-deficient problems.
  * Various numerical fixes in JacobiSVD, including:#843, and the move from Lapack to Matlab strategy for the default threshold.
  * Various numerical fixes in LDLT, including the case of semi-definite complex matrices.
  * Fix `ColPivHouseholderQR::rank()`.
  * #222: Make temporary matrix column-major independently of `EIGEN_DEFAULT_TO_ROW_MAJOR` in BlockHouseholder.

* Sparse:
  * #838: Fix `dense * sparse` and `sparse * dense` outer products and detect outer products from either the lhs or rhs.
  * Make the ordering method of SimplicialL[D]LT configurable.
  * Fix regression in the restart mechanism of BiCGSTAB.
  * #836: extend SparseQR to support more columns than rows.
  * #808: Use double instead of float for the increasing size ratio in `CompressedStorage::resize`, fix implicit conversions from int/longint to float/double, and fix `set_from_triplets` temporary matrix type.
  * #647: Use `smart_copy` instead of bitwise memcpy in CompressedStorage.
  * GMRES: Initialize essential Householder vector with correct dimension.

* Geometry:
  * #807: Missing scalar type cast in `umeyama()`
  * #806: Missing scalar type cast in `Quaternion::setFromTwoVectors()`
  * #759: Removed hard-coded double-math from `Quaternion::angularDistance`.

* OS, build system and doc:
  * Fix compilation with Windows CE.
  * Fix some ICEs with VC11.
  * Check IMKL version for compatibility with Eigen
  * #754: Only inserted (`!defined(_WIN32_WCE)`) analog to alloc and free implementation.
  * #803: Avoid `char*` to `int*` conversion.
  * #819: Include path of details.h file.
  * #738: Use the "current" version of cmake project directories to ease the inclusion of Eigen within other projects.
  * #815: Fix doc of FullPivLU wrt permutation matrices.
  * #632: doc: Note that `dm2 = sm1 + dm1` is not possible
  * Extend AsciiQuickReference (real, imag, conjugate, rot90)

## [3.2.1]

Released on February 26, 2014

Changes since 3.2.0:

* Eigen2 support is now deprecated and will be removed in version 3.3.
* Core:
  * Bug fix for Ref object containing a temporary matrix.
  * #654: Allow construction of row vector from 1D array.
  * #679: Support `cwiseMin()` and `cwiseMax()` on maps.
  * Support `conservativeResize()` on vectors.
  * Improve performance of vectorwise and replicate expressions.
  * #642: Add vectorization of sqrt for doubles, and make sqrt really safe if `EIGEN_FAST_MATH` is disabled.
  * #616: Try harder to align columns when printing matrices and arrays.
  * #579: Add optional run-time parameter to fixed-size block methods.
  * Implement `.all()` and `.any()` for zero-sized objects
  * #708: Add placement new and delete for arrays.
  * #503: Better C++11 support.
* Dense linear algebra:
  * #689: Speed up some matrix-vector products by using aligned loads if possible.
  * Make solve in `FullPivHouseholderQR` return least-square solution if there is no exact solution.
  * #678: Fix `fullPivHouseholderQR` for rectangular matrices.
  * Fix a 0/0 issue in JacobiSVD.
  * #736: Wrong result in `LDLT::isPositiveDefinite()` for semi-definite matrices.
  * #740: Fix overflow issue in `stableNorm()`.
  * Make pivoting HouseholderQR compatible with custom scalar types.
* Geometry:
  * Fix compilation of Transform * UniformScaling
* Sparse matrices:
  * Fix elimination tree and SparseQR for fat rectangular matrices.
  * #635: add `isCompressed` to `MappedSparseMatrix` for compatibility.
  * #664: Support iterators without `operator<` in `setFromTriplets()`.
  * Fixes in SparseLU: infinite loop, aliasing issue when solving, overflow in memory allocation, use exceptions only if enabled (#672).
  * Fixes in SparseQR: reduce explicit zero, assigning result to map, assert catching non-conforming sizes, memory leak.
  * #681: Uninitialized value in CholmodSupport which may lead to incorrect results.
  * Fix some issues when using a non-standard index type (#665 and more)
  * Update constrained CG (unsupported module) to Eigen3.
* OS and build system:
  * MacOS put OpenGL header files somewhere else from where we expected it.
  * Do not assume that `alloca()` is 16-byte aligned on Windows.
  * Compilation fixes when using ICC with Visual Studio.
  * Fix Fortran compiler detection in CMake files.
* Fix some of our tests (bugs #744 and #748 and more).
* Fix a few compiler warnings (bug #317 and more).
* Documentation fixes (bugs #609, #638 and #739 and more).

## [3.1.4]

Released on August 02, 2013

Changes since 3.1.3:

* #620: Fix robustness and performance issues in JacobiSVD::solve.
* #613: Fix accuracy of SSE sqrt for very small numbers.
* #608: Fix sign computation in LDLT.
* Fix write access to CwiseUnaryView expressions.
* Fix compilation of `transposeInPlace()` for Array expressions.
* Fix non const `data()` member in Array and Matrix wrappers.
* Fix a few warnings and compilation issues with recent compiler versions.
* Documentation fixes.

## [3.0.7]

Released on August 02, 2013

Changes since 3.0.6:

* Fix traits of `Map<Quaternion>`.
* Fix a few warnings (#507) and documentation (#531).

## [3.2.0]

Released on July 24, 2013.

Major new features and optimizations since 3.1:

* Dense world
  * New [`Ref<>`](http://eigen.tuxfamily.org/dox-devel/classEigen_1_1Ref.html) class allowing to write non templated function taking various kind of Eigen dense objects without copies.
  * New [RealQZ](http://eigen.tuxfamily.org/dox-devel/classEigen_1_1RealQZ.html) factorization and [GeneralizedEigenSolver](http://eigen.tuxfamily.org/dox-devel/classEigen_1_1GeneralizedEigenSolver.html).
  * Add vector-wise normalized and normalize functions, and hasNaN/allFinite members.
  * Add mixed static/dynamic-size `.block<.,.>()` functions.
  * Optimize outer products for non rank-1 update operations.
  * Optimize diagonal products (enable vectorization in more cases).
  * Improve robustness and performance in `JacobiSVD::solve()`.
* Sparse world
  * New [SparseLU](http://eigen.tuxfamily.org/dox-devel/group__SparseLU__Module.html) module: built-in sparse LU with supernodes and numerical row pivoting (port of SuperLU making the SuperLUSupport module obsolete).
  * New [SparseQR](http://eigen.tuxfamily.org/dox-devel/group__SparseQR__Module.html) module: rank-revealing sparse QR factorization with numerical column pivoting.
  * New [COLAMD](http://eigen.tuxfamily.org/dox-devel/classEigen_1_1COLAMDOrdering.html) ordering and unified [ordering API](http://eigen.tuxfamily.org/dox-devel/group__OrderingMethods__Module.html).
  * Add support for generic blocks of sparse matrices (read-only).
  * Add conservative resize feature on sparse matrices.
  * Add uniform support for solving sparse systems with sparse right hand sides.
  * Add support for sparse matrix time sparse self-adjoint view products.
  * Improve BiCGSTAB robustness with restart.
* Support to external libraries
  * New [MetisSupport](http://eigen.tuxfamily.org/dox-devel/group__MetisSupport__Module.html) module: wrapper to the famous graph partitioning library.
  * New [SPQRSupport](http://eigen.tuxfamily.org/dox-devel/group__SPQRSupport__Module.html) module: wrapper to suitesparse's supernodal QR solver.

Eigen 3.2 represents about 600 commits since Eigen 3.1.

## [3.2-rc2]

Released on July 19, 2013.

Changes since 3.2-rc1:

* Rename `DenseBase::isFinite()` to `allFinite()` to avoid a future naming collision.
* Fix an ICE with ICC 11.1.

## [3.2-rc1]

Released on July 17, 2013.

Main changes since 3.2-beta1:
* New features:
  * #562: Add vector-wise normalized and normalize functions.
  * #564: Add `hasNaN` and `isFinite` members.
  * #579: Add support for mixed static/dynamic-size `.block()`.
  * #588: Add support for determinant in SparseLU.
  * Add support in SparseLU to solve with L and U factors independently.
  * Allow multiplication-like binary operators to be applied on type combinations supported by `scalar_product_traits`.
  * #596: Add conversion from `SparseQR::matrixQ()` to a `SparseMatrix`.
  * #553: Add support for sparse matrix time sparse self-adjoint view products.

* Accuracy and performance:
  * Improve BiCGSTAB robustness: fix a divide by zero and allow to restart with a new initial residual reference.
  * #71: Enable vectorization of diagonal products in more cases.
  * #620: Fix robustness and performance issues in JacobiSVD::solve.
  * #609: Improve accuracy and consistency of the eulerAngles functions.
  * #613: Fix accuracy of SSE sqrt for very small numbers.
  * Enable SSE with ICC even when it mimics a gcc version lower than 4.2.
  * Add SSE4 min/max for integers.
  * #590 & #591: Minor improvements in NEON vectorization.

* Bug fixes:
  * Fix `HouseholderSequence::conjugate()` and `::adjoint()`.
  * Fix SparseLU for dense matrices and matrices in non compressed mode.
  * Fix `SparseMatrix::conservativeResize()` when one dimension is null.
  * Fix `transposeInpPlace` for arrays.
  * Fix `handmade_aligned_realloc`.
  * #554: Fix detection of the presence of `posix_memalign` with mingw.
  * #556: Workaround mingw bug with `-O3` or `-fipa-cp-clone` options.
  * #608: Fix sign computation in LDLT.
  * #567: Fix iterative solvers to immediately return when the initial guess is the true solution and for trivial solution.
  * #607: Fix support for implicit transposition from dense to sparse vectors.
  * #611: Fix support for products of the form `diagonal_matrix * sparse_matrix * diagonal_matrix`.

* Others:
  * #583: Add compile-time assertion to check DenseIndex is signed.
  * #63: Add lapack unit tests. They are automatically downloaded and configured if `EIGEN_ENABLE_LAPACK_TESTS` is ON.
  * #563: Assignment to `Block<SparseMatrix>` is now allowed on non-compressed matrices.
  * #626: Add assertion on input ranges for coeff* and insert members for sparse objects.
  * #314: Move special math functions from internal to numext namespace.
  * Fix many warnings and compilation issues with recent compiler versions.
  * Many other fixes including #230, #482, #542, #561, #564, #565, #566, #578, #581, #595, #597, #598, #599, #605, #606, #615.

## [3.1.3]

Released on April 16, 2013

Changes since 3.1.2:

* #526 - Fix linear vectorized transversal in linspace.
* #551 - Fix compilation issue when using `EIGEN_DEFAULT_DENSE_INDEX_TYPE`.
* #533 - Fix some missing const qualifiers in Transpose
* Fix a compilation with CGAL::Gmpq by adding explicit internal:: namespace when calling `abs()`.
* Fix computation of outer-stride when calling `.real()` or `.imag()`.
* Fix `handmade_aligned_realloc` (affected `conservativeResize()`).
* Fix sparse vector assignment from a sparse matrix.
* Fix `log(0)` with SSE.
* Fix bug in aligned_free with windows CE.
* Fix traits of `Map<Quaternion>`.
* Fix a few warnings (#507, #535, #581).
* Enable SSE with ICC even when it mimics a gcc version lower than 4.2
* Workaround [gcc-4.7 bug #53900](http://gcc.gnu.org/bugzilla/show_bug.cgi?id=53900) (too aggressive optimization in our alignment check)

## [3.2-beta1]

Released on March 07, 2013

Main changes since 3.1:

* Dense modules
  * A new [`Ref<>`](http://eigen.tuxfamily.org/dox-devel/classEigen_1_1Ref.html) class allowing to write non templated function taking various kind of Eigen dense objects without copies.
  * New [RealQZ](http://eigen.tuxfamily.org/dox-devel/classEigen_1_1RealQZ.html) factorization and [GeneralizedEigenSolver](http://eigen.tuxfamily.org/dox-devel/classEigen_1_1GeneralizedEigenSolver.html)
  * Optimized outer products for non rank-1 update operations.

* Sparse modules
  * New [SparseLU](http://eigen.tuxfamily.org/dox-devel/group__SparseLU__Module.html) module: built-in sparse LU with supernodes and numerical row pivoting (port of SuperLU making the SuperLUSupport module obsolete).
  * New [SparseQR](http://eigen.tuxfamily.org/dox-devel/group__SparseQR__Module.html) module: rank-revealing sparse QR factorization with numerical column pivoting.
  * OrderingMethods: extended with [COLAMD](http://eigen.tuxfamily.org/dox-devel/classEigen_1_1COLAMDOrdering.html) ordering and a unified [ordering](http://eigen.tuxfamily.org/dox-devel/group__OrderingMethods__Module.html) API.
  * Support for generic blocks of sparse matrices.
  * Add conservative resize feature on sparse matrices.
  * Add uniform support for solving sparse systems with sparse right hand sides.

* Support to external libraries
  * New [MetisSupport](http://eigen.tuxfamily.org/dox-devel/group__MetisSupport__Module.html) module: wrapper to the famous graph partitioning library.
  * New [SPQRSupport](http://eigen.tuxfamily.org/dox-devel/group__SPQRSupport__Module.html) module: wrapper to suitesparse's supernodal QR solver.

* Misc
  * Improved presentation and clarity of Doxygen generated documentation (modules are now organized into chapters, treeview panel and search engine for quick navagitation).
  * New compilation token `EIGEN_INITIALIZE_MATRICES_BY_NAN` to help debugging.
  * All bug fixes of the 3.1 branch, plus a couple of other fixes (including 211, 479, 496, 508, 552)

## [3.1.2]

Released on Nov 05, 2012

Changes since 3.1.1:

* #524 - Pardiso's parameter array does not have to be aligned!
* #521 - Disable `__cpuidex` on architectures different that x86 or x86-64 with MSVC.
* #519 - `AlignedBox::dim()` was wrong for dynamic dimensions.
* #515 - Fix missing explicit scalar conversion.
* #511 - Fix pretty printers on windows.
* #509 - Fix warnings with gcc 4.7
* #501 - Remove aggressive mat/scalar optimization (was replaced by `mat*(1/scalar)` for non integer types).
* #479 - Use EISPACK's strategy re max number of iters in Schur decomposition.
* Add support for scalar multiple of diagonal matrices.
* Forward `resize()` function from Array/Matrix wrappers to the nested expression such that `mat.array().resize(a,b)` is now allowed.
* Windows CE: fix the lack of the `aligned_malloc` function on this platform.
* Fix comma initializer when inserting empty matrices.
* Fix `dense=sparse*diagonal` products.
* Fix compilation with `m.array().min(scalar)` and `m.array().max(scalar)`.
* Fix out-of-range memory access in GEMV (the memory was not used for the computation, only to assemble unaligned packets from aligned packet loads).
* Fix various regressions with MKL support.
* Fix aliasing issue in sparse matrix assignment.
* Remove stupid assert in blue norm.
* Workaround a weird compilation error with MSVC.

## [3.1.1]

Released on July 22, 2012

Changes since 3.1.0:
* [relicense to MPL2](https://www.eigen.tuxfamily.org/index.php?title=Main_Page#License)
* add a `EIGEN_MPL2_ONLY` build option to generate compiler errors when including non-MPL2 modules
* remove dynamic allocation for triangular matrix-matrix products of fixed size objects
* Fix possible underflow issues in SelfAdjointEigenSolver
* Fix issues with fixed-size Diagonal (sub/super diagonal size computation was wrong)
* #487 - Geometry module: `isometry * scaling` compilation error
* #486 - MKL support: fixed multiple-references linker errors with various decompositions
* #480 - work around compilation error on Android NDK due to isfinite being defined as a macro
* #485 - IterativeLinearSolvers: conflict between a typedef and template type parameter
* #479 - Eigenvalues/Schur: Adjust max iterations count to matrix size
* Fixed Geometry module compilation under MSVC
* Fixed Sparse module compilation under MSVC 2005

## [3.0.6]

Released on July 9, 2012

Changes since 3.0.5:
* #447 - fix infinite recursion in `ProductBase::coeff()`
* #478 - fix RealSchur on a zero matrix
* #477 - fix warnings with gcc 4.7
* #475 - `.exp()` now returns `+inf` when overflow occurs (SSE)
* #466 - fix a possible race condition in OpenMP environment (for non OpenMP thread model it is recommended to upgrade to 3.1)
* #362 - fix missing specialization for affine-compact `*` projective
* #451 - fix a clang warning
* Fix compilation of `somedensematrix.llt().matrixL().transpose()`
* Fix miss-use of the cost-model in Replicate
* Fix use of int versus Index types for `Block::m_outerStride`
* Fix ambiguous calls to some std functions
* Fix geometry tutorial on scalings
* Fix RVCT 3.1 compiler errors
* Fix implicit scalar conversion in Transform
* Fix typo in NumericalDiff (unsupported module)
* Fix LevenbergMarquart for non double scalar type (unsupported module)

## [3.1.0]

Released on June 24, 2012.

Major changes between Eigen 3.0 and Eigen 3.1:
* New features
  * **New set of officially supported Sparse Modules**
  ** This includes sparse matrix storage, assembly, and many built-in (Cholesky, CG, BiCGSTAB, ILU), and third-party (PaStiX, Cholmod, UmfPack, SuperLU, Pardiso) solvers
  ** See this [page](http://eigen.tuxfamily.org/dox-devel/TutorialSparse.html) for an overview of the features
  * **Optional support for Intel MKL**
  ** This includes the BLAS, LAPACK, VML, and Pardiso components
  ** See this [page](http://eigen.tuxfamily.org/dox-devel/TopicUsingIntelMKL.html) for the details
  * Core
  ** New vector-wise operators: `*`, `/`, `*=`, `/=`
  ** New coefficient-wise operators: `&&`, `||`,  `min(Scalar)`, `max(Scalar)`, `pow`, `operator/(Scalar,ArrayBase)`
  * Decompositions
  ** Add incremental rank-updates in LLTand LDLT
  ** New `SelfAdjointEigenSolver::computeDirect()` function for fast eigen-decomposition through closed-form formulas (only for 2x2 and 3x3 real matrices)
* Optimizations
  * Memory optimizations in JacobiSVD and triangular solves.
  * Optimization of reductions via partial unrolling (e.g., dot, sum, norm, etc.)
  * Improved performance of small matrix-matrix products and some Transform<> operations

Eigen 3.1 represents about 600 commits since Eigen 3.0.

## [3.1.0-rc2]

Released on June 21, 2012.

Changes since 3.1.0-rc1:
* Fix a couple of compilation warnings
* Improved documentation, in particular regarding the Geometry and Sparse tutorials, and sparse solver modules
* Fix double preconditioner allocation in `JacobiSVD`
* #466: `RealSchur` failed on a zero matrix
* Update Adolc and MPReal support modules

## [3.1.0-rc1]

Released on June 14, 2012

Main changes since 3.1.0-beta1:
* #466: fix a possible race condition issue. from now, multithreaded applications that call Eigen from multiple thread must initialize Eigen by calling `initParallel()`.
* For consistency, `SimplicialLLT` and `SimplicialLDLT` now factorizes `P A P^-1` (instead of `P^-1 A P`).
* #475: now the vectorized `exp` operator returns +inf when overflow occurs
* Fix the use of MKL with MSVC by disabling MKL's pow functions.
* Avoid dynamic allocation for fixed size triangular solving
* Fix a compilation issue with ICC 11.1
* Fix ambiguous calls in the math functors
* Fix BTL interface.

## [3.1.0-beta1]

Released on June 7, 2012

Main changes since 3.1.0-alpha2:
* **API changes**
  * `SimplicialLLt` and `SimplicialLDLt` are now renamed `SimplicialLLT` and `SimplicialLDLT` for consistency with the other modules.
  * The Pardiso support module is now spelled "PardisoSupport"
* Dense modules:
  * Add `operator/(Scalar,ArrayBase)` and coefficient-wise pow operator.
  * Fix automatic evaluation of expressions nested by Replicate (performance improvement)
  * #447 - fix infinite recursion in `ProductBase::coeff()`
  * #455 - add support for c++11 in `aligned_allocator`
  * `LinSpace`: add a missing variant, and fix the size=1 case
* Sparse modules:
  * Add an **IncompleteLU** preconditioner with dual thresholding.
  * Add an interface to the parallel **Pastix** solver
  * Improve applicability of permutations (add `SparseMatrixBase::twistedBy`, handle non symmetric permutations)
  * `CholmodDecomposition` now has explicit variants: `CholmodSupernodalLLT`, `CholmodSimplicialLLT`, `CholmodSimplicialLDLT`
  * Add analysePattern/factorize methods to iterative solvers
  * Preserve explicit zero in a sparse assignment
  * Speedup `sparse * dense` products
  * Fix a couple of issues with Pardiso support
* Geometry module:
  * Improve performance of some `Transform<>` operations by better preserving the alignment status.
  * #415 - wrong return type in `Rotation2D::operator*=`
  * #439 - add `Quaternion::FromTwoVectors()` static constructor
  * #362 - missing specialization for affine-compact `*` projective
* Others:
  * add support for RVCT 3.1 compiler
  * New tutorial page on Map
  * and many other bug fixes such as: #417, #419, #450

## [3.0.5]

Released February 10, 2012

Changes since 3.0.4:
* #417 - fix nesting of `Map` expressions
* #415 - fix return value of `Rotation2D::operator*=`
* #410 - fix a possible out of range access in `EigenSolver`
* #406 - fix infinite loop/deadlock when using OpenMP and Eigen
* Changeset 4462 - fix broken asserts revealed by Clang
* Changeset 4457 - fix description of `rankUpdate()` in quick reference guide
* Changeset 4455 - fix out-of-range int constant in 4x4 inverse
* #398 - fix in slerp: the returned quaternion was not always normalized
* Changeset 4432 - fix asserts in eigenvalue decompositions
* Changeset 4416 - fix MSVC integer overflow warning

## [3.1.0-alpha2]

Released February 6, 2012

Main changes since 3.0.1-alpha1:
* New optional support for Intel MKL and other BLAS including: ([details](http://eigen.tuxfamily.org/dox-devel/TopicUsingIntelMKL.html))
  * BLAS (arbitrary BLAS)
  * Intel LAPACKE
  * Intel VML (coefficient-wise math operations)
  * Intel PARDISO (sparse direct solver)
* Dense modules:
  * improved performance of small matrix-matrix products
  * Feature #319 - add a rankUpdate function to LDLt and LLT for updates/downdates
  * Feature #400 - new coefficient wise min/max functions taking one scalar argument
* Sparse modules:
  * new fast sparse matrix assembly interface from a random list of triplets (see `SparseMatrix::setFromTriplets()`)
  * new shifting feature in SimplicialCholesky (see `SimplicialCholeskyBase::setShift()`)
  * add checks for positive definiteness in SimplicialCholesky
  * improved heuristic to predict the nnz of a `sparse*sparse` product
  * add support for uncompressed SparseMatrix in CholmodSupport
* Geometry module:
  * Feature #297 - add `ParametrizedLine::intersectionPoint()` and `intersectionParam()` functions
* Others:
  * fix many warnings and compilation issues with ICC 12 and -strict-ansi
  * fix some ICE with MSVC10
  * add the possibility to disable calls to cpuid (`-DEIGEN_NO_CPUID`) and other asm directives
  * and many other bug fixes such as: #406, #410, #398, #396, #394, #354, #352, #301,


## [3.1.0-alpha1]

Released December 6, 2011

Main changes since 3.0:
* Officially supported set of sparse modules. See this [page](http://eigen.tuxfamily.org/dox-devel/TutorialSparse.html) for an overview of the features. Main changes:
  * new `SparseCore` module equivalent to the old `Sparse` module, the `Sparse` module is now a super module including all sparse-related modules
    * the `SparseMatrix` class is now more versatile and supports an uncompressed mode for fast element insertion
    * the `SparseMatrix` class now offer a unique and simplified API to insert elements
    * `DynamicSparseMatrix` has been deprecated (moved into `unsupported/SparseExtra`)
    * new conservative `sparse * sparse` matrix product which is also used by default
  * new `SparseCholesky` module featuring the SimplicialLLT and SimplicialLDLT built-in solvers
  * new `IterativeLinearSolvers` module featuring a conjugate gradient and stabilized bi-conjugate gradient iterative solvers with a basic Jacobi preconditioner
* New `SelfAdjointEigenSolver::computeDirect()` function for fast eigen-decomposition through closed-form formulas (only for 2x2 and 3x3 real matrices)
* New `LLT::rankUpdate()` function supporting both updates and down-dates
* Optimization of reduction via partial unrolling (e.g., dot, sum, norm, etc.)
* New coefficient-wise operators: `&&` and `||`
* Feature #157 - New vector-wise operations for arrays: `*`, `/`, `*=`, and `/=`.
* Feature #206 - Pre-allocation of intermediate buffers in JacobiSVD
* Feature #370 - New typedefs for AlignedBox
* All the fixes and improvements of the 3.0 branch up to the 3.0.4 release (see below)



## [3.0.4]

Released December 6, 2011

Changes since 3.0.3:

* #363 - check for integer overflow in size computations
* #369 - Quaternion alignment is broken (and more alignment fixes)
* #354 - Converge better in SelfAdjointEigenSolver, and allow better handling of non-convergent cases
* #347 - Fix compilation on ARM NEON with LLVM 3.0 and iOS SDK 5.0
* #372 - Put unsupported modules documentation at the right place
* #383 - Fix C++11 compilation problem due to some constructs mis-interpreted as c++11 user-defined literals
* #373 - Compilation error with clang 2.9 when exceptions are disabled
* Fix compilation issue with `QuaternionBase::cast`


## [2.0.17]

Released December 6, 2011

Changes since 2.0.16:

* Fix a compilation bug in `aligned_allocator`: the allocate method should take a void pointer
* Fix a typo in ParametrizedLine documentation


## [3.0.3]

Released October 6, 2011

Changes since 3.0.2:

* Fix compilation errors when Eigen2 support is enabled.
* Fix bug in evaluating expressions of the form `matrix1 * matrix2 * scalar1 * scalar2`.
* Fix solve using LDLT for singular matrices if solution exists.
* Fix infinite loop when computing SVD of some matrices with very small numbers.
* Allow user to specify pkgconfig destination.
* Several improvements to the documentation.


## [3.0.2]

Released August 26, 2011

Changes since 3.0.1:

* `Windows.h`: protect min/max calls from macros having the same name (no need to `#undef` min/max anymore).
* MinGW: fix compilation issues and pretty gdb printer.
* Standard compliance: fix aligned_allocator and remove uses of long long.
* MPReal: updates for the new version.
* Other fixes:
  * fix aligned_stack_memory_handler for null pointers.
  * fix std::vector support with gcc 4.6.
  * fix linking issue with OpenGL support.
  * fix SelfAdjointEigenSolver for 1x1 matrices.
  * fix a couple of warnings with new compilers.
  * fix a few documentation issues.


## [3.0.1]

Released May 30, 2011

Changes since 3.0.0:

* Fix many bugs regarding ARM and NEON (Now all tests succeed on ARM/NEON).
* Fix compilation on gcc 4.6
* Improved support for custom scalar types:
  * Fix memory leak issue for scalar types throwing exceptions.
  * Fix implicit scalar type conversion.
  * Math functions can be defined in the scalar type's namespace.
* Fix bug in trapezoidal matrix time matrix product.
* Fix asin.
* Fix compilation with MSVC 2005 (SSE was wrongly enabled).
* Fix bug in `EigenSolver`: normalize the eigen vectors.
* Fix Qt support in Transform.
* Improved documentation.

## [2.0.16]

Released May 28, 2011

Changes since 2.0.15:

* Fix bug in 3x3 tridiagonlisation (and consequently in 3x3 selfadjoint eigen decomposition).
* Fix compilation for new gcc 4.6.
* Fix performance regression since 2.0.12: in some matrix-vector product, complex matrix expressions were not pre-evaluated.
* Fix documentation of Least-Square.
* New feature: support for `part<SelfAdjoint>`.
* Fix bug in SparseLU::setOrderingMethod.

## [3.0.0]

Released March 19, 2011, at the [meeting](https://www.eigen.tuxfamily.org/index.php?title=Paris_2011_Meeting).

See the [Eigen 3.0 release notes](https://www.eigen.tuxfamily.org/index.php?title=3.0).

Only change since 3.0-rc1:
* Fixed compilation of the unsupported 'openglsupport' test.

## [3.0-rc1]

Released March 14, 2011.

Main changes since 3.0-beta4:

* Core: added new `EIGEN_RUNTIME_NO_MALLOC` option and new `set_is_malloc_allowed()` option to finely control where dynamic memory allocation is allowed. Useful for unit-testing of functions that must not cause dynamic memory allocations.
* Core: SSE performance fixes (follow-up from #203).
* Core: Fixed crashes when using `EIGEN_DONT_ALIGN` or `EIGEN_DONT_ALIGN_STATICALLY` (#213 and friends).
* Core: `EIGEN_DONT_ALIGN` and `EIGEN_DONT_ALIGN_STATICALLY` are now covered by unit tests.
* Geometry: Fixed transform * matrix products (#207).
* Geometry: compilation fix for mixing CompactAffine with Homogeneous objects
* Geometry: compilation fix for 1D transform
* SVD: fix non-computing constructors (correctly forward `computationOptions`) (#206)
* Sparse: fix resizing when the destination sparse matrix is row major (#37)
* more Eigen2Support improvements
* more unit test fixes/improvements
* more documentation improvements
* more compiler warnings fixes
* fixed GDB pretty-printer for dynamic-size matrices (#210)

## [3.0-beta4]

Released February 28, 2011.

Main changes since 3.0-beta3:

* Non-vectorization bug fixes:
  * fix #89: work around an extremely evil compiler bug on old GCC (<= 4.3) with the standard `assert()` macro
  * fix Umfpack back-end in the complex case
* Vectorization bug fixes:
  * fix a segfault in "slice vectorization" when the destination might not be aligned on a scalar (`complex<double>`)
  * fix #195: fast SSE unaligned loads fail on GCC/i386 and on Clang
  * fix #186: worked around a GCC 4.3 i386 backend issue with SSE
  * fix #203: SSE: a workaround used in pset1() resulted in poor assembly
  * worked around a GCC 4.2.4 internal compiler error with vectorization of complex numbers
  * lots of AltiVec compilation fixes
  * NEON compilation fixes
* API additions and error messages improvements
  * Transform: prevent bad user code from compiling
  * fix #190: directly pass Transform Options to Matrix, allowing to use RowMajor. Fix issues in Transform with non-default Options.
  * factorize implementation of standard real unary math functions, and add acos, asin
* Build/tests system
  * Lots of unit test improvements
  * fix installation of unsupported modules
  * fixed many compiler warnings, especially on the Intel compiler and on LLVM/Clang
  * CTest/CMake improvements
  * added option to build in 32bit mode
* BLAS/LAPACK implementation improvements
  * The Blas library and tests are now automatically built as part of the tests.
  * expanded LAPACK interface (including syev)
  * now Sparse solver backends use our own BLAS/LAPACK implementation
  * fix #189 (cblat1 test failure)
* Documentation
  * improved conservativeResize methods documentation
  * documented sorting of eigenvalues
  * misc documentation improvements
  * improve documentation of plugins

## [3.0-beta3]

Released February 12, 2011.

The biggest news is that the API is now **100% stable**.

Main changes since 3.0-beta2:

* The "too many to list them all" category:
  * lots of bug fixes
  * lots of performance fixes
  * lots of compiler support fixes
  * lots of warning fixes
  * lots of unit tests improvements and fixes
  * lots of documentation improvements
  * lots of build system fixes
* API changes:
  * replaced `ei_` prefix by `internal::` namespace. For example, `ei_cos(x)` becomes `internal::cos(x)`.
  * renamed `PlanarRotation` -> `JacobiRotation`
  * renamed `DenseStorageBase` -> `PlainObjectBase`
  * HouseholderSequence API cleanup
  * refactored internal metaprogramming helpers to follow closely the standard library
  * made UpperBidiagonalization internal
  * made BandMatrix/TridiagonalMatrix internal
  * Core: also see below, "const correctness".
  * Sparse: `EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET` must be defined to use Eigen/Sparse
  * Core: `random<interger type>()` now spans over range of width `RAND_MAX`
* New API:
  * Core: added Map static methods taking strides
  * SVD: added `jacobiSvd()` method
  * Sparse: many misc improvements and new features. Improved support for Cholmod, Amd, SuperLU and other back-ends.
  * Core: allow mixed real-complex dot products
  * Geometry: allow mixed real-complex cross products
  * Geometry: allow to pass Options parameters to Transform, Quaternion and other templates, to control memory alignment
  * QR: add threshold API to FullPivHouseholderQR
  * Core: added tan function
* Const correctness:
  * Eigen now properly enforces const-correctness everywhere, for example with Map objects. This will break compilation of code that relied on former behavior.
  * A new kind of test suite was added to check that, 'failtest'.
* BLAS/LAPACK:
  * Complete BLAS library built on top of Eigen. Imported BLAS test suite, which allowed to fix many issues.
  * Partial LAPACK implementation. Passing part of the LAPACK test suite, which also allowed to fix some issues.
* Eigen 2 Support:
  * tons of improvements in `EIGEN2_SUPPORT`
  * new incremental migration path: see http://eigen.tuxfamily.org/dox-devel/Eigen2SupportModes.html
  * imported a copy of the Eigen 2 test suite, made sure that Eigen 3 passes it. That also allowed to fix several issues.


## [3.0-beta2]

Released October 15, 2010.

Main changes since 3.0-beta1:

* Add support for the vectorization of `std::complex<>` with SSE, AltiVec and NEON.
* Add support for mixed `real * complex` matrix products with vectorization.
* Finalize the JacobiSVD class with: compile time options, thin/full decompositions, and least-square solving.
* Several improvement of the Transform class. In particular, there is no default mode anymore.
* New methods: `middleRows()`, `middleCols()`, `TriangularMatrix::conjugate()`
* New unsupported modules: OpenGL, MPFR C++
* Many improvements in the support of empty objects.
* Many improvements of the vectorization logic.
* Add the possibility to extend QuaternionBase.
* Vectorize Quaternion multiplication with double.
* Significant improvements of the documentation.
* Improved compile time errors.
* Enforce static allocation of temporary buffers in gemm (when possible).
* Fix aligned_delete for null pointers and non trivial dtors.
* Fix eigen decomposition of 3x3 float matrices.
* Fix 4x4 matrix inversions (vectorization).
* Many fixes in QR: solving with `m>n`, use of rank, etc.
* Fixes for MSVC for windows mobile and CLang.
* Remove the Taucs backend (obsolete).
* Remove the old SVD class (was causing too much troubles, a new decompozition based on bidiagonalisation/householder should come back soon, `JacobiSVD` can be used meanwhile).

## [2.0.15]

Released July 16, 2010

Changes since 2.0.14:

* Fix bug: certain cases of matrix-vector product (depending on storage order) were blocked by an assertion failure.
* Fix LU and QR solve when rank==0, fix LLT when the matrix is purely 0.
* Fix a couple of bugs with QR solving especially with rows>cols.
* Fix bug with custom scalar types that have non-trivial destructor.
* Fix for ICC in SSE code.
* Fix some C++ issues found by Clang (patch by Nick Lewycky).

## [3.0-beta1]

Released July 5, 2010

See the [announcement](https://www.eigen.tuxfamily.org/index.php?title=3.0).

## [2.0.14]

Released June 22, 2010

Changes since 2.0.13:

* Fix #141: crash in SSE (alignment problem) when using dynamic-size matrices with a max-size fixed at compile time that is not a multiple of 16 bytes. For example, `Matrix<double,Dynamic,Dynamic,AutoAlign,5,5>`.
* Fix #142: LU of fixed-size matrices was causing dynamic memory allocation (patch by Stuart Glaser).
* Fix #127: remove useless static keywords (also fixes warnings with clang++).

## [2.0.13]

Released June 10, 2010

Changes since 2.0.12:

* Fix #132: crash in certain matrix-vector products. Unit test added.
* Fix #125: colwise `norm()` and `squaredNorm()` on complex types do not return real types
* Fully support the QCC/QNX compiler (thanks to Piotr Trojanek). The support in 2.0.12 was incomplete. The whole test suite is now successful.
* As part of the QCC support work, a lot of standards compliance work: put `std::` in front of a lot of things such as `size_t`, check whether the math library needs to be linked to explicitly.
* Fix precision issues in LDLT. The `isPositiveDefinite()` method is now always returning true, but it was conceptually broken anyway, since a non-pivoting LDLT decomposition can't know that.
* Compilation fix in `ldlt()` on expressions.
* Actually install the Eigen/Eigen and Eigen/Dense public headers!
* Fix readcost for complex types.
* Fix compilation of the BTL benchmarks.
* Some dox updates.

## [2.0.12]

Released February 12, 2010

Changes since 2.0.11:

* `EIGEN_DEFAULT_TO_ROW_MAJOR` is fully supported and tested.
* Several important fixes for row-major matrices.
* Fix support of several algorithms for mixed fixed-dynamic size matrices where the fixed dimension is greater than the dynamic dimension. For example: `Matrix<float,3,Dynamic>(3,2)`
* fix `EIGEN_DONT_ALIGN`: now it _really_ disables vectorization (was giving a `#error` unless you also used `EIGEN_DONT_VECTORIZE`).
* Fix #92: Support QNX's QCC compiler (patch by Piotr Trojanek)
* Fix #90, missing type cast in LU, allow to use LU with MPFR (patch by 'Wolf').
* Fix ICC compiler support: work around a bug present at least in ICC 11.1.
* Compilation fixes for `computeInverse()` on expressions.
* Fix a gap in a unit-test (thanks to Jitse Niesen)
* Backport improvements to benchmarking code.
* Documentation fixes

## [2.0.11]

Released January 10, 2010

Changes since 2.0.10:

* Complete rewrite of the 4x4 matrix inversion: we now use the usual cofactors approach, so no numerical stability problems anymore (bug #70)
* Still 4x4 matrix inverse: SSE path for the float case, borrowing code by Intel, giving very high performance.
* Fix crash happening on 32-bit x86 Linux with SSE, when double's were created at non-8-byte-aligned locations (bug #79).
* Fix bug in Part making it crash in certain products (bug #80).
* Precision improvements in Quaternion SLERP (bug #71).
* Fix sparse triangular solver for lower/row-major matrices (bug #74).
* Fix MSVC 2010 compatibility.
* Some documentation improvements.

## [2.0.10]

Released November 25, 2009

Changes since 2.0.9:

* Rewrite 4x4 matrix inverse to improve precision, and add a new unit test to guarantee that precision. It's less fast, but it's still faster than the cofactors method.
* Fix bug #62: crash in SSE code with MSVC 2008 (Thanks to Hauke Heibel).
* Fix bug #65: `MatrixBase::nonZeros()` was recursing infinitely
* Fix PowerPC platform detection on Mac OSX.
* Prevent the construction of bogus MatrixBase objects and generate good compilation errors for that. Done by making the default constructor protected, and adding some private constructors.
* Add option to initialize all matrices by zero: just #define `EIGEN_INITIALIZE_MATRICES_BY_ZERO`
* Improve Map documentation
* Install the pkg-config file to share/pkgconfig, instead of lib/pkgconfig (thanks to Thomas Capricelli)
* fix warnings
* fix compilation with MSVC 2010
* adjust to repository name change

## [2.0.9]

Released October 24, 2009

Changes since 2.0.8:

* Really fix installation and the pkg-config file.
* Install the `NewStdVector` header that was introduced in 2.0.6.

## [2.0.8]

Released October 23, 2009

Changes since 2.0.7:

* fix installation error introduced in 2.0.7: it was choking on the pkg-config file eigen2.pc not being found. The fix had been proposed long ago by Ingmar Vanhassel for the development branch, and when recently the pkg-config support was back-ported to the 2.0 branch, nobody thought of backporting this fix too, and apparently nobody tested "make install" !
* SVD: add default constructor. Users were relying on the compiler to generate one, and apparenty 2.0.7 triggered a little MSVC 2008 subtlety in this respect. Also added an assert.

## [2.0.7]

Released October 22, 2009

Changes since 2.0.6:

* fix bug #61: crash when using Qt `QVector` on Windows 32-bit. By Hauke Heibel.
* fix bug #10: the `reallocateSparse` function was half coded
* fix bug in `SparseMatrix::resize()` not correctly initializing by zero
* fix another bug in `SparseMatrix::resize()` when `outerSize==0`. By Hauke Heibel.
* fully support GCC 3.3. It was working in 2.0.2, there was a compilation error in 2.0.6, now for the first time in 2.0.7 it's 100% tested (the test suite passes without any errors, warnings, or failed tests).
* SVD: add missing assert (help catch mistakes)
* fixed warnings in unit-tests (Hauke Heibel)
* finish syncing `Memory.h` with the devel branch. This is cleaner and fixes a warning. The choice of system aligned malloc function may be affected by this change.
* add pkg-config support by Rhys Ulerich.
* documentation fix and doc-generation-script updates by Thomas Capricelli

## [2.0.6]

Released September 23, 2009

Changes since 2.0.5:

* fix bug: visitors didn't work on row-vectors.
* fix bug #50: compilation errors with `swap()`.
* fix bug #42: Add `Transform::Identity()` as mentioned in the tutorial.
* allow to disable all alignment code by defining `EIGEN_DONT_ALIGN` (backport from devel branch).
* backport the devel branch's `StdVector` header as `NewStdVector`. You may also #define `EIGEN_USE_NEW_STDVECTOR` to make `StdVector` use it automatically. However, by default it isn't used by `StdVector`, to preserve compatibility.
* Vectorized quaternion product (for float) by Rohit Garg (backport from devel branch).
* allow to override `EIGEN_RESTRICT` and add `EIGEN_DONT_USE_RESTRICT_KEYWORD`
* fix a warning in `ei_aligned_malloc`; fixed by backporting the body from the devel branch; may result in a different choice of system aligned malloc function.
* update the documentation.

## [2.0.5]

Released August 22, 2009

Changes since 2.0.4:

* fix bug: in rare situations involving mixed storage orders, a matrix product could be evaluated as its own transpose
* fix bug: `LU::solve()` crashed when called on the LU decomposition of a zero matrix
* fix bug: `EIGEN_STACK_ALLOCATION_LIMIT` was too high, resulting in stack overflow for a user. Now it is set as in the devel branch.
* fix compilation bug: our `StdVector` header didn't work with GCC 4.1. (Bug #41)
* fix compilation bug: missing return statement in `Rotation2D::operator*=` (Bug #36)
* in StdVector, a more useful `#error` message about the #including order
* add `EIGEN_TRANSFORM_PLUGIN` allowing to customize the Transform class
* fix a warning with MSVC
* fix a bug in our cmake code when building unit-tests (thanks to Marcus Hanwell)
* work around a bug in cmake that made it fail to build unit-tests when fortran wasn't installed
* in our cmake code, remove the part about retrieving the mercurial info and appending it to the version number in the dox
* dox: remove the old example list
* fix the option to build a binary library, although it's not very useful and will be removed
* add basic .hgignore file and script to build the docs (thanks to Thomas Capricelli)

## [2.0.4]

Released August 1, 2009

Changes since 2.0.3:
* Several fixes in the overloaded new and delete operators. Thanks to Hauke Heibel.
* compilation fix: add the missing `ei_atan2` function. Thanks to Manuel Yguel.
* Use `ei_atan2` instead of using `std::atan2` directly.
* several compilation fixes in the Qt interoperability code: methods `toQTransform()` and `toQMatrix()`. Thanks to Anthony Truchet.
* compilation fix and simplification in Matrix assignment
* compilation fixes in `a *= b` and  `a = a*b` when a has to be resized.
* remove a "stupid" version of `ei_pow`. for integers for gcc >= 4.3
* bug fix in `Quaternion::setFromTwoVectors()`
* several ctest improvements: use our own dashboard, use a separate project for the 2.0 branch.
* documentation: improvement on the pages on unaligned arrays (the online copies have been updated immediately).

## [2.0.3]

Released June 21, 2009

Changes since 2.0.2:
* precision and reliability fixes in various algorithms, especially LLT, QR, Tridiagonalization, and also a precision improvement in LU.
* fix LLT and LDLT solve() on uninitialized result (was causing an assertion).
* add Eigen/Eigen and Eigen/Dense headers for convenience
* document a newly found cause for the "unaligned array" assertion
* backport documentation improvements on transpose() and adjoint()
* updates in the Sparse module (was needed to support KDE 4.3)

## [2.0.2]

Released May 22, 2009

Changes since 2.0.1:
* Fix `linearRegression()` compilation, actually it is reimplemented using the better fitHyperplane() which does total least-squares.
* Add missing `setZero()` etc... variants taking size parameters and resizing. These were mentioned in the tutorial but weren't implemented.
* Fix `posix_memalign` platform check. This fixes portability issues. Thanks to Ross Smith.
* Fix detection of SSE2 on the Windows 64-bit platform.
* Fix compatibility with the old GCC 3.3: it is now fully supported again.
* Fix warnings with recent GCC (4.4.0 and 4.3.3).

## [2.0.1]

Released April 14, 2009

Changes since 2.0.0:
* disable alignment altogether on exotic platforms on which we don't vectorize anyway. This allows e.g. to use Eigen on ARM platforms.
* new StdVector header with a new workaround for the problems with std::vector.
* workarounds for MSVC internal compiler errors
* MSVC 9 compilation fix (patch by Hauke Heibel)
* fixes for various bugs in Maps/Blocks that could give wrong results
* fix bug in 4x4 matrix inverse that could give wrong results
* compilation fix in SliceVectorization
* fix wrong static assertion (patch by Markus Moll)
* add missing operators in `aligned_allocator` (thanks to Hauke Heibel)

## [2.0.0]

Released February 2, 2009