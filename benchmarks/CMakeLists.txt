# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

cmake_minimum_required(VERSION 3.10)
project(EigenBenchmarks CXX)

# Eigen is a header-only library; find it relative to this directory.
set(EIGEN_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")

# Eigen's own find modules (FindAOCL, FindBLASEXT, ...) are otherwise unreachable
# from this standalone project.
list(APPEND CMAKE_MODULE_PATH "${EIGEN_SOURCE_DIR}/cmake")

# C++17 applies to the benchmark sources only. Eigen's C++14 contract
# (AGENTS.md rule 3) constrains the headers under Eigen/, not their consumers,
# and Core/bench_gemm.cpp already uses `if constexpr`, which a compiler
# defaulting to gnu++14 rejects. Only a default: a plain set() would shadow a
# -DCMAKE_CXX_STANDARD= given on the command line.
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_CXX_STANDARD LESS 17)
  message(FATAL_ERROR
    "CMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}: these benchmarks need C++17 or later "
    "(Core/bench_gemm.cpp uses `if constexpr`). Eigen's own C++14 contract covers the "
    "headers under Eigen/, not this consumer project.")
endif()
if(NOT DEFINED CMAKE_CXX_STANDARD_REQUIRED)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

find_package(benchmark REQUIRED)
find_package(Threads REQUIRED)

# Reference BLAS selection for the comparison benchmarks (-DEIGEN_BENCH_REFERENCE=<vendor>).
# It must run before any unqualified find_package(BLAS): FindBLAS creates the
# BLAS::BLAS imported target once and never revises it, so a generic search first
# would pin that target to a vendor the table did not choose. On success it also
# sets BLAS_FOUND and BLAS_LIBRARIES, which is what Tuning/CMakeLists.txt guards on.
include(${CMAKE_CURRENT_SOURCE_DIR}/comparison/vendors.cmake)
eigen_bench_select_blas_vendor()
if(NOT BLAS_FOUND)
  find_package(BLAS QUIET)
endif()

# Helper: add a Google Benchmark target.
#   eigen_add_benchmark(name source [LIBRARIES lib1 lib2 ...] [DEFINITIONS def1 def2 ...])
function(eigen_add_benchmark name source)
  cmake_parse_arguments(BENCH "" "" "LIBRARIES;DEFINITIONS" ${ARGN})
  if(NOT IS_ABSOLUTE "${source}")
    set(source "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
  endif()
  add_executable(${name} ${source})
  target_include_directories(${name} PRIVATE ${EIGEN_SOURCE_DIR})
  target_link_libraries(${name} PRIVATE benchmark::benchmark benchmark::benchmark_main)
  if(BENCH_LIBRARIES)
    target_link_libraries(${name} PRIVATE ${BENCH_LIBRARIES})
  endif()
  target_compile_options(${name} PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/O2>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O3>
  )
  target_compile_definitions(${name} PRIVATE NDEBUG)
  if(BENCH_DEFINITIONS)
    target_compile_definitions(${name} PRIVATE ${BENCH_DEFINITIONS})
  endif()
endfunction()

enable_testing()

add_subdirectory(Core)
add_subdirectory(Cholesky)
add_subdirectory(LU)
add_subdirectory(QR)
add_subdirectory(SVD)
add_subdirectory(Eigenvalues)
add_subdirectory(Geometry)
add_subdirectory(Sparse)
add_subdirectory(FFT)
add_subdirectory(Householder)
add_subdirectory(Solvers)
add_subdirectory(Tuning)
add_subdirectory(BLAS)
add_subdirectory(ThreadPool)
add_subdirectory(comparison)
