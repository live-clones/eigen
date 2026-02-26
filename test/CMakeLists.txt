# check if we have a Fortran compiler
include(CheckLanguage)
check_language(Fortran)
if(CMAKE_Fortran_COMPILER)
  enable_language(Fortran)
  set(EIGEN_Fortran_COMPILER_WORKS ON)
else()
  set(EIGEN_Fortran_COMPILER_WORKS OFF)
  # search for a default Lapack library to complete Eigen's one
  find_package(LAPACK QUIET)
endif()

# TODO do the same for EXTERNAL_LAPACK
option(EIGEN_TEST_EXTERNAL_BLAS "Use external BLAS library for testsuite" OFF)
if(EIGEN_TEST_EXTERNAL_BLAS)
  find_package(BLAS REQUIRED)
  message(STATUS "BLAS_COMPILER_FLAGS: ${BLAS_COMPILER_FLAGS}")
  add_definitions("-DEIGEN_USE_BLAS") # is adding  ${BLAS_COMPILER_FLAGS} necessary?
  list(APPEND EXTERNAL_LIBS "${BLAS_LIBRARIES}")
endif()

# configure blas/lapack (use Eigen's ones)
set(EIGEN_BLAS_LIBRARIES eigen_blas)
set(EIGEN_LAPACK_LIBRARIES eigen_lapack)

set(EIGEN_TEST_MATRIX_DIR "" CACHE STRING "Enable testing of realword sparse matrices contained in the specified path")
if(EIGEN_TEST_MATRIX_DIR)
  if(NOT WIN32)
    message(STATUS "Test realworld sparse matrices: ${EIGEN_TEST_MATRIX_DIR}")
    add_definitions( -DTEST_REAL_CASES="${EIGEN_TEST_MATRIX_DIR}" )
  else()
    message(STATUS "REAL CASES CAN NOT BE CURRENTLY TESTED ON WIN32")
  endif()
endif()

set(SPARSE_LIBS " ")

find_package(CHOLMOD)
if(CHOLMOD_FOUND AND EIGEN_BUILD_BLAS AND EIGEN_BUILD_LAPACK)
  add_definitions("-DEIGEN_CHOLMOD_SUPPORT")
  include_directories(${CHOLMOD_INCLUDES})
  set(SPARSE_LIBS ${SPARSE_LIBS} ${CHOLMOD_LIBRARIES} ${EIGEN_BLAS_LIBRARIES} ${EIGEN_LAPACK_LIBRARIES})
  set(CHOLMOD_ALL_LIBS  ${CHOLMOD_LIBRARIES} ${EIGEN_BLAS_LIBRARIES} ${EIGEN_LAPACK_LIBRARIES})
  ei_add_property(EIGEN_TESTED_BACKENDS "CHOLMOD, ")
else()
  ei_add_property(EIGEN_MISSING_BACKENDS "CHOLMOD, ")
endif()

find_package(UMFPACK)
if(UMFPACK_FOUND AND EIGEN_BUILD_BLAS)
  add_definitions("-DEIGEN_UMFPACK_SUPPORT")
  include_directories(${UMFPACK_INCLUDES})
  set(SPARSE_LIBS ${SPARSE_LIBS} ${UMFPACK_LIBRARIES} ${EIGEN_BLAS_LIBRARIES})
  set(UMFPACK_ALL_LIBS ${UMFPACK_LIBRARIES} ${EIGEN_BLAS_LIBRARIES})
  ei_add_property(EIGEN_TESTED_BACKENDS "UMFPACK, ")
else()
  ei_add_property(EIGEN_MISSING_BACKENDS "UMFPACK, ")
endif()

find_package(KLU)
if(KLU_FOUND AND EIGEN_BUILD_BLAS)
  add_definitions("-DEIGEN_KLU_SUPPORT")
  include_directories(${KLU_INCLUDES})
  set(SPARSE_LIBS ${SPARSE_LIBS} ${KLU_LIBRARIES} ${EIGEN_BLAS_LIBRARIES})
  set(KLU_ALL_LIBS ${KLU_LIBRARIES} ${EIGEN_BLAS_LIBRARIES})
  ei_add_property(EIGEN_TESTED_BACKENDS "KLU, ")
else()
  ei_add_property(EIGEN_MISSING_BACKENDS "KLU, ")
endif()

find_package(SuperLU 4.0)
if(SuperLU_FOUND AND EIGEN_BUILD_BLAS)
  add_definitions("-DEIGEN_SUPERLU_SUPPORT")
  include_directories(${SUPERLU_INCLUDES})
  set(SPARSE_LIBS ${SPARSE_LIBS} ${SUPERLU_LIBRARIES} ${EIGEN_BLAS_LIBRARIES})
  set(SUPERLU_ALL_LIBS ${SUPERLU_LIBRARIES} ${EIGEN_BLAS_LIBRARIES})
  ei_add_property(EIGEN_TESTED_BACKENDS  "SuperLU, ")
else()
  ei_add_property(EIGEN_MISSING_BACKENDS  "SuperLU, ")
endif()


find_package(PASTIX QUIET COMPONENTS METIS SEQ)
# check that the PASTIX found is a version without MPI
find_path(PASTIX_pastix_nompi.h_INCLUDE_DIRS
  NAMES pastix_nompi.h
  HINTS ${PASTIX_INCLUDE_DIRS}
)
if (NOT PASTIX_pastix_nompi.h_INCLUDE_DIRS)
  message(STATUS "A version of Pastix has been found but pastix_nompi.h does not exist in the include directory."
                 " Because Eigen tests require a version without MPI, we disable the Pastix backend.")
endif()
if(PASTIX_FOUND AND PASTIX_pastix_nompi.h_INCLUDE_DIRS)
  add_definitions("-DEIGEN_PASTIX_SUPPORT")
  include_directories(${PASTIX_INCLUDE_DIRS_DEP})
  if(SCOTCH_FOUND)
    include_directories(${SCOTCH_INCLUDE_DIRS})
    set(PASTIX_LIBRARIES ${PASTIX_LIBRARIES} ${SCOTCH_LIBRARIES})
  elseif(METIS_FOUND)
    include_directories(${METIS_INCLUDE_DIRS})
    set(PASTIX_LIBRARIES ${PASTIX_LIBRARIES} ${METIS_LIBRARIES})
  else()
    ei_add_property(EIGEN_MISSING_BACKENDS  "PaStiX, ")
  endif()
  set(SPARSE_LIBS ${SPARSE_LIBS} ${PASTIX_LIBRARIES_DEP} ${ORDERING_LIBRARIES})
  set(PASTIX_ALL_LIBS ${PASTIX_LIBRARIES_DEP})
  ei_add_property(EIGEN_TESTED_BACKENDS  "PaStiX, ")
else()
  ei_add_property(EIGEN_MISSING_BACKENDS  "PaStiX, ")
endif()

if(METIS_FOUND)
  add_definitions("-DEIGEN_METIS_SUPPORT")
  include_directories(${METIS_INCLUDE_DIRS})
  ei_add_property(EIGEN_TESTED_BACKENDS "METIS, ")
else()
  ei_add_property(EIGEN_MISSING_BACKENDS "METIS, ")
endif()

find_package(SPQR)
if(SPQR_FOUND AND CHOLMOD_FOUND AND EIGEN_BUILD_BLAS AND EIGEN_BUILD_LAPACK AND (EIGEN_Fortran_COMPILER_WORKS OR LAPACK_FOUND) )
  add_definitions("-DEIGEN_SPQR_SUPPORT")
  include_directories(${SPQR_INCLUDES})
  set(SPQR_ALL_LIBS ${SPQR_LIBRARIES} ${CHOLMOD_LIBRARIES} ${EIGEN_LAPACK_LIBRARIES} ${EIGEN_BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
  set(SPARSE_LIBS ${SPARSE_LIBS} ${SPQR_ALL_LIBS})
  ei_add_property(EIGEN_TESTED_BACKENDS "SPQR, ")
else()
  ei_add_property(EIGEN_MISSING_BACKENDS "SPQR, ")
endif()

find_package(Accelerate)
if(Accelerate_FOUND)
  add_definitions("-DEIGEN_ACCELERATE_SUPPORT")
  include_directories(${Accelerate_INCLUDES})
  set(SPARSE_LIBS ${SPARSE_LIBS} ${Accelerate_LIBRARIES})
  set(Accelerate_ALL_LIBS ${Accelerate_LIBRARIES})
  ei_add_property(EIGEN_TESTED_BACKENDS "Accelerate, ")
else()
  ei_add_property(EIGEN_MISSING_BACKENDS "Accelerate, ")
endif()

option(EIGEN_TEST_NOQT "Disable Qt support in unit tests" OFF)
if(NOT EIGEN_TEST_NOQT)
  find_package(Qt4)
  if(QT4_FOUND)
    include(${QT_USE_FILE})
    ei_add_property(EIGEN_TESTED_BACKENDS  "Qt4 support, ")
  else()
    ei_add_property(EIGEN_MISSING_BACKENDS  "Qt4 support, ")
  endif()
endif()

if(TEST_LIB)
  add_definitions("-DEIGEN_EXTERN_INSTANTIATIONS=1")
endif()

check_cxx_compiler_flag("-ffast-math" COMPILER_SUPPORT_FASTMATH)
if(COMPILER_SUPPORT_FASTMATH)
  set(EIGEN_FASTMATH_FLAGS "-ffast-math")
else()
  check_cxx_compiler_flag("/fp:fast" COMPILER_SUPPORT_FPFAST)
  if(COMPILER_SUPPORT_FPFAST)
    set(EIGEN_FASTMATH_FLAGS "/fp:fast")
  endif()
endif()
# The fastmath test intentionally uses NaN/infinity under -ffast-math.
# Suppress the clang warning about this being technically undefined.
check_cxx_compiler_flag("-Wno-nan-infinity-disabled" COMPILER_SUPPORT_WNO_NAN_INF)
if(COMPILER_SUPPORT_WNO_NAN_INF)
  set(EIGEN_FASTMATH_FLAGS "${EIGEN_FASTMATH_FLAGS} -Wno-nan-infinity-disabled")
endif()

# boost MP unit test
find_package(Boost 1.53.0 CONFIG)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
  ei_add_property(EIGEN_TESTED_BACKENDS "Boost.Multiprecision, ")
else()
  ei_add_property(EIGEN_MISSING_BACKENDS "Boost.Multiprecision, ")
endif()

# CUDA unit tests
option(EIGEN_TEST_CUDA "Enable CUDA support in unit tests" OFF)
option(EIGEN_TEST_CUDA_CLANG "Use clang instead of nvcc to compile the CUDA tests" OFF)
option(EIGEN_TEST_CUDA_NVC "Use nvc++ (NVHPC) instead of nvcc to compile the CUDA tests" OFF)

if(EIGEN_TEST_CUDA_CLANG AND NOT CMAKE_CXX_COMPILER MATCHES "clang")
  message(WARNING "EIGEN_TEST_CUDA_CLANG is set, but CMAKE_CXX_COMPILER does not appear to be clang.")
endif()

if(EIGEN_TEST_CUDA_NVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "NVHPC")
  message(WARNING "EIGEN_TEST_CUDA_NVC is set, but CMAKE_CXX_COMPILER does not appear to be nvc++.")
endif()

find_package(CUDA 9.0)

# HIP unit tests
option(EIGEN_TEST_HIP "Add HIP support." OFF)

# ---------------------------------------------------------------
# Module subdirectories
# ---------------------------------------------------------------

# Load the smoke test list BEFORE add_subdirectory() calls so that
# ei_add_test_internal can label tests in subdirectory scopes.
include("EigenSmokeTestList")
set_property(GLOBAL PROPERTY EIGEN_SMOKE_TESTS_LIST ${ei_smoke_test_list})

set_property(GLOBAL PROPERTY EIGEN_CURRENT_SUBPROJECT "Official")
add_custom_target(BuildOfficial)

add_subdirectory(Core)
add_subdirectory(Cholesky)
add_subdirectory(LU)
add_subdirectory(QR)
add_subdirectory(SVD)
add_subdirectory(Eigenvalues)
add_subdirectory(Geometry)
add_subdirectory(SparseCore)
add_subdirectory(SparseCholesky)
add_subdirectory(SparseLU)
add_subdirectory(SparseQR)
add_subdirectory(IterativeLinearSolvers)
add_subdirectory(StlSupport)
add_subdirectory(ThreadPool)
add_subdirectory(Support)
add_subdirectory(GPU)

# SYCL tests (remain in test/ root)
if(EIGEN_TEST_SYCL)
  set(EIGEN_SYCL ON)
  include(SyclConfigureTesting)

  ei_add_test(sycl_basic)
  set(EIGEN_SYCL OFF)
endif()

cmake_dependent_option(EIGEN_TEST_BUILD_DOCUMENTATION "Test building the doxygen documentation" OFF "EIGEN_BUILD_DOC" OFF)
if(EIGEN_TEST_BUILD_DOCUMENTATION)
  add_dependencies(buildtests doc)
endif()

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
string(TOLOWER "${CMAKE_CXX_COMPILER}" cmake_cxx_compiler_tolower)
if(cmake_cxx_compiler_tolower MATCHES "qcc")
  set(CXX_IS_QCC "ON")
endif()

ei_add_property(EIGEN_TESTING_SUMMARY "CXX:               ${CMAKE_CXX_COMPILER}\n")
if(CMAKE_COMPILER_IS_GNUCXX AND NOT CXX_IS_QCC)
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} --version COMMAND head -n 1 OUTPUT_VARIABLE EIGEN_CXX_VERSION_STRING OUTPUT_STRIP_TRAILING_WHITESPACE)
  ei_add_property(EIGEN_TESTING_SUMMARY "CXX_VERSION:       ${EIGEN_CXX_VERSION_STRING}\n")
endif()
ei_add_property(EIGEN_TESTING_SUMMARY "CXX_FLAGS:         ${CMAKE_CXX_FLAGS}\n")
if (EIGEN_TEST_CUSTOM_CXX_FLAGS)
  ei_add_property(EIGEN_TESTING_SUMMARY "Custom CXX flags:  ${EIGEN_TEST_CUSTOM_CXX_FLAGS}\n")
endif()
ei_add_property(EIGEN_TESTING_SUMMARY "Sparse lib flags:  ${SPARSE_LIBS}\n")

option(EIGEN_TEST_EIGEN2 "Run whole Eigen2 test suite against EIGEN2_SUPPORT" OFF)
mark_as_advanced(EIGEN_TEST_EIGEN2)
if(EIGEN_TEST_EIGEN2)
  message(WARNING "The Eigen2 test suite has been removed")
endif()

# NOTE: ei_add_smoke_tests is called from the top-level CMakeLists.txt after
# all test subdirectories (including unsupported/) have been processed, so that
# unsupported smoke tests are included in the buildsmoketests target.
