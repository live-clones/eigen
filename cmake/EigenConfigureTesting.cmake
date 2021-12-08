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

if(EIGEN_TEST_CUSTOM_CXX_FLAGS)
  target_compile_options(EigenTestDeps INTERFACE "${EIGEN_TEST_CUSTOM_CXX_FLAGS}")
endif()

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
  # these are currently shadowed by Eigens of FindBLAS script
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

