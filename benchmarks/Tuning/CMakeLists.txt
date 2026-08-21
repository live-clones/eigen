# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

eigen_add_benchmark(bench_blocking_sizes bench_blocking_sizes.cpp)
eigen_add_benchmark(bench_aocl bench_aocl.cpp)

# bench_blas_gemm.cpp includes <cblas.h>, which CMake's FindBLAS never reports a
# path for; vendors.cmake resolves it per vendor. Without it this is the only
# target that cannot be built, so it is skipped rather than failing the project --
# the comparison benchmarks call the Fortran BLAS directly and need no header.
if(BLAS_FOUND AND EIGEN_BENCH_BLAS_INCLUDE_DIRS)
  eigen_add_benchmark(bench_blas_gemm bench_blas_gemm.cpp
    LIBRARIES ${BLAS_LIBRARIES}
    DEFINITIONS HAVE_BLAS)
elseif(BLAS_FOUND)
  message(STATUS "bench_blas_gemm: skipped; no cblas.h was found for the BLAS in use")
endif()

add_executable(print_blocking print_blocking.cpp)
target_include_directories(print_blocking PRIVATE ${EIGEN_SOURCE_DIR})
target_compile_options(print_blocking PRIVATE -O3 -DNDEBUG)
