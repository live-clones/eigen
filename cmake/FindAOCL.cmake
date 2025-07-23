# FindAOCL.cmake
#
# This module locates the AMD AOCL MathLib, BLAS, and LAPACK libraries.
# It searches for the core AOCL library (amdlibm, alm, aocl_core) and then
# for the BLAS and LAPACK libraries provided by AOCL.
#
# Usage:
#   Set the AOCL_ROOT environment variable to your AOCL installation root (e.g.,
#   export AOCL_ROOT=<path-to-aocl-install-dir>)
#   Then include this module in your CMakeLists.txt:
#     find_package(AOCL)
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
# Developed by: Sharad Saurabh Bhaskar, (shbhsaka@amd.com) Advanced Micro Devices, Inc.
#
# This implementation is inspired by the structure and approach of FindMKL.cmake
# from the Eigen library (https://github.com/PX4/eigen/blob/master/bench/btl/cmake/FindMKL.cmake),
# adapted for use with AMD AOCL libraries. No direct code is copied; the design
# reflects common CMake conventions tailored to AOCL requirements.
#
# If no license is specified, this file is considered proprietary to AMD.
# FindAOCL.cmake
#
# This module locates the AMD AOCL MathLib, BLAS, and LAPACK libraries.
# It searches for the core AOCL library (amdlibm, alm) and then
# for the BLAS and LAPACK libraries provided by AOCL.
#

if(NOT DEFINED AOCL_ROOT)
  if(DEFINED ENV{AOCL_ROOT})
    set(AOCL_ROOT $ENV{AOCL_ROOT})
    message(STATUS "AOCL_ROOT set from environment: ${AOCL_ROOT}")
  else()
    message(WARNING "AOCL_ROOT is not set. AOCL support will be disabled.")
    set(AOCL_LIBRARIES "")
  endif()
endif()

if(AOCL_LIBRARIES)
  set(AOCL_FIND_QUIETLY TRUE)
endif()

# Determine default include directories
set(AOCL_INCLUDE_DIRS "")
if(AOCL_ROOT AND EXISTS "${AOCL_ROOT}/include")
  list(APPEND AOCL_INCLUDE_DIRS "${AOCL_ROOT}/include")
endif()
if(EXISTS "/opt/amd/aocl/include")
  list(APPEND AOCL_INCLUDE_DIRS "/opt/amd/aocl/include")
endif()

if(CMAKE_MINOR_VERSION GREATER 4)
  if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")
    # Search for the core AOCL math library.
    find_library(AOCL_CORE_LIB
      NAMES amdlibm alm almfast
      PATHS
        ${AOCL_ROOT}/lib
        /opt/amd/aocl/lib64
        ${LIB_INSTALL_DIR}
    )
    if(AOCL_CORE_LIB)
      message(STATUS "Found AOCL core library: ${AOCL_CORE_LIB}")
    else()
      message(WARNING "AOCL core library not found in ${AOCL_ROOT}/lib or default locations.")
    endif()
    # Now search for AOCL BLAS library.
    find_library(AOCL_BLAS_LIB
      NAMES blis-mt blis
      PATHS
        ${AOCL_ROOT}/lib
        /opt/amd/aocl/lib64
        ${LIB_INSTALL_DIR}
    )
    if(AOCL_BLAS_LIB)
      message(STATUS "Found AOCL BLAS library: ${AOCL_BLAS_LIB}")
    else()
      message(WARNING "AOCL BLAS library not found in ${AOCL_ROOT}/lib or default locations.")
    endif()

    # Now search for AOCL LAPACK library.
    find_library(AOCL_LAPACK_LIB
      NAMES flame
      PATHS
        ${AOCL_ROOT}/lib
        /opt/amd/aocl/lib64
        ${LIB_INSTALL_DIR}
    )
    if(AOCL_LAPACK_LIB)
      message(STATUS "Found AOCL LAPACK library: ${AOCL_LAPACK_LIB}")
    else()
      message(WARNING "AOCL LAPACK library not found in ${AOCL_ROOT}/lib or default locations.")
    endif()

  else()
    # For 32-bit systems, similar search paths.
    find_library(AOCL_CORE_LIB
      NAMES amdlibm alm almfast
      PATHS
        ${AOCL_ROOT}/lib
        /opt/amd/aocl/lib32
        ${LIB_INSTALL_DIR}
    )
    if(AOCL_CORE_LIB)
      message(STATUS "Found AOCL core library: ${AOCL_CORE_LIB}")
    else()
      message(WARNING "AOCL core library not found in ${AOCL_ROOT}/lib or default locations.")
    endif()

    find_library(AOCL_BLAS_LIB
      NAMES blis-mt blis
      PATHS
        ${AOCL_ROOT}/lib
        /opt/amd/aocl/lib32
        ${LIB_INSTALL_DIR}
    )
    if(AOCL_BLAS_LIB)
      message(STATUS "Found AOCL BLAS library: ${AOCL_BLAS_LIB}")
    else()
      message(WARNING "AOCL BLAS library not found in ${AOCL_ROOT}/lib or default locations.")
    endif()

    find_library(AOCL_LAPACK_LIB
      NAMES flame
      PATHS
        ${AOCL_ROOT}/lib
        /opt/amd/aocl/lib32
        ${LIB_INSTALL_DIR}
    )
    if(AOCL_LAPACK_LIB)
      message(STATUS "Found AOCL LAPACK library: ${AOCL_LAPACK_LIB}")
    else()
      message(WARNING "AOCL LAPACK library not found in ${AOCL_ROOT}/lib or default locations.")
    endif()
  endif()
endif()

# Combine the found libraries into one variable.
if(AOCL_CORE_LIB)
  set(AOCL_LIBRARIES ${AOCL_CORE_LIB})
endif()
if(AOCL_BLAS_LIB)
  list(APPEND AOCL_LIBRARIES ${AOCL_BLAS_LIB})
endif()
if(AOCL_LAPACK_LIB)
  list(APPEND AOCL_LIBRARIES ${AOCL_LAPACK_LIB})
endif()
if(AOCL_LIBRARIES)
  # Link against the standard math and pthread libraries as well as librt
  list(APPEND AOCL_LIBRARIES m pthread rt)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(AOCL DEFAULT_MSG AOCL_LIBRARIES AOCL_INCLUDE_DIRS)
mark_as_advanced(AOCL_LIBRARIES AOCL_INCLUDE_DIRS)
