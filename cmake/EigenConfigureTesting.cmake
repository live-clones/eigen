include(EigenTesting)
include(CheckCXXSourceCompiles)

# configure the "site" and "buildname" 
ei_set_sitename()

# retrieve and store the build string
ei_set_build_string()

add_custom_target(buildtests)
add_custom_target(check COMMAND "ctest" ${EIGEN_CTEST_ARGS})
add_dependencies(check buildtests)

# Convenience target for only building GPU tests.
add_custom_target(buildtests_gpu)
add_custom_target(check_gpu COMMAND "ctest" "--output-on-failure" 
                                            "--no-compress-output"
                                            "--build-no-clean"
                                            "-T" "test"
                                            "-L" "gpu")
add_dependencies(check_gpu buildtests_gpu)

# check whether /bin/bash exists (disabled as not used anymore)
# find_file(EIGEN_BIN_BASH_EXISTS "/bin/bash" PATHS "/" NO_DEFAULT_PATH)

# This call activates testing and generates the DartConfiguration.tcl
include(CTest)

set(EIGEN_TEST_BUILD_FLAGS "" CACHE STRING "Options passed to the build command of unit tests")
set(EIGEN_DASHBOARD_BUILD_TARGET "buildtests" CACHE STRING "Target to be built in dashboard mode, default is buildtests")
set(EIGEN_CTEST_ERROR_EXCEPTION "" CACHE STRING "Regular expression for build error messages to be filtered out")

# Overwrite default DartConfiguration.tcl such that ctest can build our unit tests.
# Recall that our unit tests are not in the "all" target, so we have to explicitly ask ctest to build our custom 'buildtests' target.
# At this stage, we can also add custom flags to the build tool through the user defined EIGEN_TEST_BUILD_FLAGS variable.
file(READ  "${CMAKE_CURRENT_BINARY_DIR}/DartConfiguration.tcl" EIGEN_DART_CONFIG_FILE)
# try to grab the default flags
string(REGEX MATCH "MakeCommand:.*-- (.*)\nDefaultCTestConfigurationType" EIGEN_DUMMY ${EIGEN_DART_CONFIG_FILE})
if(NOT CMAKE_MATCH_1)
string(REGEX MATCH "MakeCommand:.*[^c]make (.*)\nDefaultCTestConfigurationType" EIGEN_DUMMY ${EIGEN_DART_CONFIG_FILE})
endif()
string(REGEX REPLACE "MakeCommand:.*DefaultCTestConfigurationType" "MakeCommand: ${CMAKE_COMMAND} --build . --target ${EIGEN_DASHBOARD_BUILD_TARGET} --config \"\${CTEST_CONFIGURATION_TYPE}\" -- ${CMAKE_MATCH_1} ${EIGEN_TEST_BUILD_FLAGS}\nDefaultCTestConfigurationType"
       EIGEN_DART_CONFIG_FILE2 ${EIGEN_DART_CONFIG_FILE})
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/DartConfiguration.tcl" ${EIGEN_DART_CONFIG_FILE2})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CTestCustom.cmake.in ${CMAKE_BINARY_DIR}/CTestCustom.cmake)

# some documentation of this function would be nice
ei_init_testing()

# configure Eigen related testing options
option(EIGEN_NO_ASSERTION_CHECKING "Disable checking of assertions using exceptions" OFF)
option(EIGEN_DEBUG_ASSERTS "Enable advanced debugging of assertions" OFF)


# Add a target that gathers the common flags and dependencies for compiling tests
add_library(EigenTestDeps INTERFACE)
if(EIGEN_NO_ASSERTION_CHECKING)
  target_compile_definitions(EigenTestDeps INTERFACE EIGEN_NO_ASSERTION_CHECKING=1)
else()
  if(EIGEN_DEBUG_ASSERTS)
    target_compile_definitions(EigenTestDeps INTERFACE EIGEN_DEBUG_ASSERTS=1)
  endif()
endif()
target_compile_definitions(EigenTestDeps INTERFACE EIGEN_TEST_MAX_SIZE=${EIGEN_TEST_MAX_SIZE})

if(MSVC)
  target_compile_options(EigenTestDeps INTERFACE "/bigobj")
endif()

if(EIGEN_STANDARD_LIBRARIES_TO_LINK_TO)
  target_link_libraries(EigenTestDeps INTERFACE ${EIGEN_STANDARD_LIBRARIES_TO_LINK_TO})
endif()

set(EIGEN_TEST_CUSTOM_CXX_FLAGS     "" CACHE STRING "Additional compiler flags when compiling unit tests.")
# convert space separated argument into CMake lists for downstream consumption
separate_arguments(EIGEN_TEST_CUSTOM_CXX_FLAGS NATIVE_COMMAND ${EIGEN_TEST_CUSTOM_CXX_FLAGS})
if(EIGEN_TEST_CUSTOM_CXX_FLAGS)
  target_compile_options(EigenTestDeps INTERFACE "${EIGEN_TEST_CUSTOM_CXX_FLAGS}")
endif()

set(EIGEN_TEST_CUSTOM_LINKER_FLAGS  "" CACHE STRING "Additional linker flags when linking unit tests.")
if(EIGEN_TEST_CUSTOM_LINKER_FLAGS)
  target_link_libraries(EigenTestDeps INTERFACE ${EIGEN_TEST_CUSTOM_LINKER_FLAGS})
endif()


if(CMAKE_COMPILER_IS_GNUCXX)
  option(EIGEN_COVERAGE_TESTING "Enable/disable gcov" OFF)
  if(EIGEN_COVERAGE_TESTING)
    set(COVERAGE_FLAGS "-fprofile-arcs -ftest-coverage")
    set(CTEST_CUSTOM_COVERAGE_EXCLUDE "/test/")
    target_compile_options(EigenTestDeps ${COVERAGE_FLAGS})
  endif()
  
elseif(MSVC)
  target_compile_definitions(EigenTestDeps _CRT_SECURE_NO_WARNINGS _SCL_SECURE_NO_WARNINGS)
endif()

option(EIGEN_TEST_EXTERNAL_BLAS "Use external BLAS library for testsuite" OFF)
if(EIGEN_TEST_EXTERNAL_BLAS)
  find_package(BLAS REQUIRED)
  message(STATUS "BLAS_COMPILER_FLAGS: ${BLAS_COMPILER_FLAGS}")
  # TODO recent versions of CMake provide an imported target for BLAS. Use that if available.
  target_link_libraries(EigenTestDeps INTERFACE ${BLAS_LIBRARIES})
  target_compile_options(EigenTestDeps INTERFACE ${BLAS_COMPILER_FLAGS})
  target_compile_definitions(EigenTestDeps INTERFACE -DEIGEN_USE_BLAS)
endif()

option(EIGEN_TEST_EXTERNAL_LAPACKE "Use external LAPACKE library for testsuite" OFF)
if(EIGEN_TEST_EXTERNAL_LAPACKE)
  # I'm not sure on how many platforms that works. However, having lapacke testing
  # on some platforms should be better than having no lapacke testing at all.
  find_library(LAPACKE_LIBRARY lapacke)
  target_link_libraries(EigenTestDeps INTERFACE ${LAPACKE_LIBRARY})
endif()

target_link_libraries(EigenTestDeps INTERFACE Eigen3::Eigen EigenWarnings)
target_include_directories(EigenTestDeps INTERFACE ${Eigen3_SOURCE_DIR}/test)

if(NOT MSVC)
  # We assume that other compilers are partly compatible with GNUCC
  option(EIGEN_TEST_SSE2 "Enable/Disable SSE2 in tests/examples" OFF)
  if(EIGEN_TEST_SSE2)
    target_compile_options(EigenTestDeps INTERFACE -msse2)
    message(STATUS "Enabling SSE2 in tests/examples")
  endif()

  option(EIGEN_TEST_SSE3 "Enable/Disable SSE3 in tests/examples" OFF)
  if(EIGEN_TEST_SSE3)
    target_compile_options(EigenTestDeps INTERFACE -msse3)
    message(STATUS "Enabling SSE3 in tests/examples")
  endif()

  option(EIGEN_TEST_SSSE3 "Enable/Disable SSSE3 in tests/examples" OFF)
  if(EIGEN_TEST_SSSE3)
    target_compile_options(EigenTestDeps INTERFACE -mssse3)
    message(STATUS "Enabling SSSE3 in tests/examples")
  endif()

  option(EIGEN_TEST_SSE4_1 "Enable/Disable SSE4.1 in tests/examples" OFF)
  if(EIGEN_TEST_SSE4_1)
    target_compile_options(EigenTestDeps INTERFACE -msse4.1)
    message(STATUS "Enabling SSE4.1 in tests/examples")
  endif()

  option(EIGEN_TEST_SSE4_2 "Enable/Disable SSE4.2 in tests/examples" OFF)
  if(EIGEN_TEST_SSE4_2)
    target_compile_options(EigenTestDeps INTERFACE -msse4.2)
    message(STATUS "Enabling SSE4.2 in tests/examples")
  endif()

  option(EIGEN_TEST_AVX "Enable/Disable AVX in tests/examples" OFF)
  if(EIGEN_TEST_AVX)
    target_compile_options(EigenTestDeps INTERFACE -mavx)
    message(STATUS "Enabling AVX in tests/examples")
  endif()

  option(EIGEN_TEST_FMA "Enable/Disable FMA in tests/examples" OFF)
  if(EIGEN_TEST_FMA AND NOT EIGEN_TEST_NEON)
    # The combination of NEON and FMA is handled separately below
    target_compile_options(EigenTestDeps INTERFACE -mfma)
    message(STATUS "Enabling FMA in tests/examples")
  endif()

  option(EIGEN_TEST_AVX2 "Enable/Disable AVX2 in tests/examples" OFF)
  if(EIGEN_TEST_AVX2)
    target_compile_options(EigenTestDeps INTERFACE -mavx2 -mfma)
    message(STATUS "Enabling AVX2 in tests/examples")
  endif()

  option(EIGEN_TEST_AVX512 "Enable/Disable AVX512 in tests/examples" OFF)
  if(EIGEN_TEST_AVX512)
    target_compile_options(EigenTestDeps INTERFACE -mavx512f -mfma)
    message(STATUS "Enabling AVX512 in tests/examples")
  endif()

  option(EIGEN_TEST_AVX512DQ "Enable/Disable AVX512DQ in tests/examples" OFF)
  if(EIGEN_TEST_AVX512DQ)
    target_compile_options(EigenTestDeps INTERFACE -mavx512dq -mfma)
    message(STATUS "Enabling AVX512DQ in tests/examples")
  endif()

  option(EIGEN_TEST_F16C "Enable/Disable F16C in tests/examples" OFF)
  if(EIGEN_TEST_F16C)
    target_compile_options(EigenTestDeps INTERFACE -mf16c)
    message(STATUS "Enabling F16C in tests/examples")
  endif()

  option(EIGEN_TEST_ALTIVEC "Enable/Disable AltiVec in tests/examples" OFF)
  if(EIGEN_TEST_ALTIVEC)
    target_compile_options(EigenTestDeps INTERFACE -maltivec -mabi=altivec)
    message(STATUS "Enabling AltiVec in tests/examples")
  endif()

  option(EIGEN_TEST_VSX "Enable/Disable VSX in tests/examples" OFF)
  if(EIGEN_TEST_VSX)
    target_compile_options(EigenTestDeps INTERFACE -m64 -mvsx)
    message(STATUS "Enabling VSX in tests/examples")
  endif()

  option(EIGEN_TEST_MSA "Enable/Disable MSA in tests/examples" OFF)
  if(EIGEN_TEST_MSA)
    target_compile_options(EigenTestDeps INTERFACE -mmsa)
    message(STATUS "Enabling MSA in tests/examples")
  endif()

  option(EIGEN_TEST_NEON "Enable/Disable Neon in tests/examples" OFF)
  if(EIGEN_TEST_NEON)
    if(EIGEN_TEST_FMA)
      target_compile_options(EigenTestDeps INTERFACE -mfpu=neon-vfpv4)
    else()
      target_compile_options(EigenTestDeps INTERFACE -mfpu=neon)
    endif()
    target_compile_options(EigenTestDeps INTERFACE -mfloat-abi=hard)
    message(STATUS "Enabling NEON in tests/examples")
  endif()

  option(EIGEN_TEST_NEON64 "Enable/Disable Neon in tests/examples" OFF)
  if(EIGEN_TEST_NEON64)
    # TODO this is just leaving flags as before????
    message(STATUS "Enabling NEON in tests/examples")
  endif()

  option(EIGEN_TEST_Z13 "Enable/Disable S390X(zEC13) ZVECTOR in tests/examples" OFF)
  if(EIGEN_TEST_Z13)
    target_compile_options(EigenTestDeps INTERFACE -march=z13 -mzvector)
    message(STATUS "Enabling S390X(zEC13) ZVECTOR in tests/examples")
  endif()

  option(EIGEN_TEST_Z14 "Enable/Disable S390X(zEC14) ZVECTOR in tests/examples" OFF)
  if(EIGEN_TEST_Z14)
    target_compile_options(EigenTestDeps INTERFACE -march=z14 -mzvector)
    message(STATUS "Enabling S390X(zEC13) ZVECTOR in tests/examples")
  endif()
else()
  # TODO what is this doing, and should it be doing that?
  # TODO In modern CMAKE, CMAKE_CXX_FLAGS should be mostly left along
  # replace all /Wx by /W4
  string(REGEX REPLACE "/W[0-9]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

  option(EIGEN_TEST_SSE2 "Enable/Disable SSE2 in tests/examples" OFF)
  if(EIGEN_TEST_SSE2)
    if(NOT CMAKE_CL_64)
      # arch is not supported on 64 bit systems, SSE is enabled automatically.
      target_compile_options(EigenTestDeps INTERFACE /arch:SSE2)
    endif()
    message(STATUS "Enabling SSE2 in tests/examples")
  endif()

  option(EIGEN_TEST_AVX "Enable/Disable AVX in tests/examples" OFF)
  if(EIGEN_TEST_AVX)
    target_compile_options(EigenTestDeps INTERFACE /arch:AVX)
    message(STATUS "Enabling AVX in tests/examples")
  endif()

  option(EIGEN_TEST_FMA "Enable/Disable FMA/AVX2 in tests/examples" OFF)
  if(EIGEN_TEST_FMA AND NOT EIGEN_TEST_NEON)
    # TODO what happens with EIGEN_TEST_FMA and EIGEN_TEST_NEON?
    target_compile_options(EigenTestDeps INTERFACE /arch:AVX2)
    message(STATUS "Enabling FMA/AVX2 in tests/examples")
  endif()
endif()

find_package(OpenMP)
if(COMPILER_SUPPORT_OPENMP)
  option(EIGEN_TEST_OPENMP "Enable/Disable OpenMP in tests/examples" OFF)
  if(EIGEN_TEST_OPENMP)
    target_link_libraries(EigenTestDeps INTERFACE OpenMP::OpenMP_CXX)
    message(STATUS "Enabling OpenMP in tests/examples")
  endif()
endif()


option(EIGEN_TEST_NO_EXPLICIT_VECTORIZATION "Disable explicit vectorization in tests/examples" OFF)
option(EIGEN_TEST_X87 "Force using X87 instructions. Implies no vectorization." OFF)
option(EIGEN_TEST_32BIT "Force generating 32bit code." OFF)
option(EIGEN_TEST_NO_EXPLICIT_ALIGNMENT "Disable explicit alignment (hence vectorization) in tests/examples" OFF)

if(EIGEN_TEST_X87)
  set(EIGEN_TEST_NO_EXPLICIT_VECTORIZATION ON)
  if(CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(EigenTestDeps INTERFACE -mfpmath=387)
    message(STATUS "Forcing use of x87 instructions in tests/examples")
  else()
    message(STATUS "EIGEN_TEST_X87 ignored on your compiler")
  endif()
endif()

if(EIGEN_TEST_32BIT)
  if(CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(EigenTestDeps INTERFACE -m32)
    message(STATUS "Forcing generation of 32-bit code in tests/examples")
  else()
    message(STATUS "EIGEN_TEST_32BIT ignored on your compiler")
  endif()
endif()

if(EIGEN_TEST_NO_EXPLICIT_VECTORIZATION)
  target_compile_definitions(EigenTestDeps INTERFACE EIGEN_DONT_VECTORIZE=1)
  message(STATUS "Disabling vectorization in tests/examples")
endif()

if(EIGEN_TEST_NO_EXPLICIT_ALIGNMENT)
  target_compile_definitions(EigenTestDeps INTERFACE EIGEN_DONT_ALIGN=1)
  message(STATUS "Disabling alignment in tests/examples")
endif()

option(EIGEN_TEST_NO_EXCEPTIONS "Disables C++ exceptions" OFF)
if(EIGEN_TEST_NO_EXCEPTIONS)
  # TODO is this compatible with msvc?
  target_compile_options(EigenTestDeps INTERFACE -fno-exceptions)
  message(STATUS "Disabling exceptions in tests/examples")
endif()

# add SYCL
option(EIGEN_TEST_SYCL "Add Sycl support." OFF)
option(EIGEN_SYCL_TRISYCL "Use the triSYCL Sycl implementation (ComputeCPP by default)." OFF)
if(EIGEN_TEST_SYCL)
  # TODO why do we modify that path here?
  set (CMAKE_MODULE_PATH "${CMAKE_ROOT}/Modules" "cmake/Modules/" "${CMAKE_MODULE_PATH}")
  find_package(Threads REQUIRED)
  if(EIGEN_SYCL_TRISYCL)
    message(STATUS "Using triSYCL")
    include(FindTriSYCL)
  else()
    message(STATUS "Using ComputeCPP SYCL")
    include(FindComputeCpp)
    set(COMPUTECPP_DRIVER_DEFAULT_VALUE OFF)
    if (NOT MSVC)
      set(COMPUTECPP_DRIVER_DEFAULT_VALUE ON)
    endif()
    option(COMPUTECPP_USE_COMPILER_DRIVER
            "Use ComputeCpp driver instead of a 2 steps compilation"
            ${COMPUTECPP_DRIVER_DEFAULT_VALUE}
            )
  endif(EIGEN_SYCL_TRISYCL)
  option(EIGEN_DONT_VECTORIZE_SYCL "Don't use vectorisation in the SYCL tests." OFF)
  if(EIGEN_DONT_VECTORIZE_SYCL)
    message(STATUS "Disabling SYCL vectorization in tests/examples")
    # When disabling SYCL vectorization, also disable Eigen default vectorization
    target_compile_definitions(EigenTestDeps INTERFACE EIGEN_DONT_VECTORIZE=1 EIGEN_DONT_VECTORIZE_SYCL=1)
  endif()
endif()

# CUDA configuration
if(EIGEN_TEST_CUDA)
  set(CMAKE_CUDA_ARCHITECTURES ${EIGEN_CUDA_COMPUTE_ARCH})
  if(EIGEN_TEST_CUDA_CLANG)
    # TODO
  else()
    # add cuda specific flags for all cuda language files
    target_compile_options(EigenTestDeps INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-Xcudafe;--display_error_number>)
  endif()
endif()