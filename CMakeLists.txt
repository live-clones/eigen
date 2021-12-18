# cmake_minimum_require must be the first command of the file
cmake_minimum_required(VERSION 3.10.0)

# NOTE Remove setting the policy once the minimum required CMake version is
# increased to at least 3.15. Retain enabling the export to package registry.
if (POLICY CMP0090)
  # The export command does not populate package registry by default
  cmake_policy (SET CMP0090 NEW)

  # Unless otherwise specified, always export to package registry to ensure
  # backwards compatibility.
  if (NOT DEFINED CMAKE_EXPORT_PACKAGE_REGISTRY)
    set (CMAKE_EXPORT_PACKAGE_REGISTRY ON)
  endif (NOT DEFINED CMAKE_EXPORT_PACKAGE_REGISTRY)
endif (POLICY CMP0090)

if (POLICY CMP0079)
  # target_link_libraries looks up linked libraries in calling scope instead of original target scope
  cmake_policy (SET CMP0079 NEW)
endif()

project(Eigen3)

# Remove this block after bumping CMake to v3.21.0
# PROJECT_IS_TOP_LEVEL is defined then by default
if(CMAKE_VERSION VERSION_LESS 3.21.0)
  if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(PROJECT_IS_TOP_LEVEL TRUE)
  else()
    set(PROJECT_IS_TOP_LEVEL FALSE)
  endif()
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE BOOL "Require C++ standard")
set(CMAKE_CXX_EXTENSIONS OFF CACHE BOOL "Allow C++ extensions")

# guard against in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt. ")
endif()


# Alias Eigen_*_DIR to Eigen3_*_DIR:
# TODO do we need this?
set(Eigen_SOURCE_DIR ${Eigen3_SOURCE_DIR})
set(Eigen_BINARY_DIR ${Eigen3_BINARY_DIR})

# guard against bad build-type strings
# TODO do we need this?
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

#############################################################################
# Setting up CMake utilities                                                #
#############################################################################

include(CheckCXXCompilerFlag)
include(CheckLanguage)
include(GNUInstallDirs)
include(CMakeDependentOption)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(EigenUtils)


#############################################################################
# retrieve version information                                              #
#############################################################################

# automatically parse the version number
extract_eigen_version("${PROJECT_SOURCE_DIR}/Eigen/src/Core/util/Macros.h")
extract_git_version("${CMAKE_SOURCE_DIR}")

set(EIGEN_VERSION_NUMBER ${EIGEN_WORLD_VERSION}.${EIGEN_MAJOR_VERSION}.${EIGEN_MINOR_VERSION})
# if we have a git revision, show it as part of the version number
if(EIGEN_GIT_REVNUM)
  set(EIGEN_VERSION "${EIGEN_VERSION_NUMBER} (git rev ${EIGEN_GIT_REVNUM})")
else()
  set(EIGEN_VERSION "${EIGEN_VERSION_NUMBER}")
endif()

# Determine if we should build shared libraries on this platform.
get_cmake_property(EIGEN_BUILD_SHARED_LIBS TARGET_SUPPORTS_SHARED_LIBS)

option(EIGEN_BUILD_BTL "Build benchmark suite" OFF)

# Disable pkgconfig only for native Windows builds
if(NOT WIN32 OR NOT CMAKE_HOST_SYSTEM_NAME MATCHES Windows)
  option(EIGEN_BUILD_PKGCONFIG "Build pkg-config .pc file for Eigen" ON)
endif()

set(CMAKE_INCLUDE_CURRENT_DIR OFF)

option(EIGEN_DEFAULT_TO_ROW_MAJOR "Use row-major as default matrix storage order" OFF)
if(EIGEN_DEFAULT_TO_ROW_MAJOR)
  add_definitions("-DEIGEN_DEFAULT_TO_ROW_MAJOR")
endif()

include(cmake/EigenWarnings.cmake)

# Backward compatibility support for EIGEN_INCLUDE_INSTALL_DIR
# TODO is this still needed?
if(EIGEN_INCLUDE_INSTALL_DIR)
  message(WARNING "EIGEN_INCLUDE_INSTALL_DIR is deprecated. Use INCLUDE_INSTALL_DIR instead.")
endif()

if(EIGEN_INCLUDE_INSTALL_DIR AND NOT INCLUDE_INSTALL_DIR)
  set(INCLUDE_INSTALL_DIR ${EIGEN_INCLUDE_INSTALL_DIR}
      CACHE PATH "The directory relative to CMAKE_INSTALL_PREFIX where Eigen header files are installed")
else()
  set(INCLUDE_INSTALL_DIR
      "${CMAKE_INSTALL_INCLUDEDIR}/eigen3"
      CACHE PATH "The directory relative to CMAKE_INSTALL_PREFIX where Eigen header files are installed"
      )
endif()
set(CMAKEPACKAGE_INSTALL_DIR
    "${CMAKE_INSTALL_DATADIR}/eigen3/cmake"
    CACHE PATH "The directory relative to CMAKE_INSTALL_PREFIX where Eigen3Config.cmake is installed"
    )
set(PKGCONFIG_INSTALL_DIR
    "${CMAKE_INSTALL_DATADIR}/pkgconfig"
    CACHE PATH "The directory relative to CMAKE_INSTALL_PREFIX where eigen3.pc is installed"
    )

foreach(var INCLUDE_INSTALL_DIR CMAKEPACKAGE_INSTALL_DIR PKGCONFIG_INSTALL_DIR)
  # If an absolute path is specified, make it relative to "{CMAKE_INSTALL_PREFIX}".
  if(IS_ABSOLUTE "${${var}}")
    file(RELATIVE_PATH "${var}" "${CMAKE_INSTALL_PREFIX}" "${${var}}")
  endif()
endforeach()

install(FILES
  signature_of_eigen3_matrix_library
  DESTINATION ${INCLUDE_INSTALL_DIR} COMPONENT Devel
  )

if(EIGEN_BUILD_PKGCONFIG)
    configure_file(eigen3.pc.in eigen3.pc @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/eigen3.pc
        DESTINATION ${PKGCONFIG_INSTALL_DIR}
        )
endif()

install(DIRECTORY Eigen DESTINATION ${INCLUDE_INSTALL_DIR} COMPONENT Devel)

cmake_dependent_option(BUILD_TESTING "Enable creation of tests." ON "PROJECT_IS_TOP_LEVEL" OFF)
option(EIGEN_BUILD_TESTING "Enable creation of Eigen tests." ${BUILD_TESTING})
if(EIGEN_BUILD_TESTING)
  option(EIGEN_TEST_CUDA "Enable CUDA support in unit tests" OFF)
  if(EIGEN_TEST_CUDA)
    # TODO this was necessary on my system, because otherwise errors about unsupported float128 popped up
    # a related problem: https://stackoverflow.com/questions/43316533/float128-is-not-supported-on-this-target
    if(CMAKE_CUDA_COMPILER MATCHES "clang")
      # TODO should we do this in the CMakeLists.txt, or should this be the users responsibility to submit the correct
      # CUDAFLAGS for their preferred cuda compiler?
      # NOTE: This needs to happen before the check_language call, because otherwise there will be clang errors and
      # cmake will think cuda is not usable.
      set(CMAKE_CUDA_FLAGS "-stdlib=libc++")
    endif()
      # enable language should be in the top-level file, https://cmake.org/cmake/help/latest/command/enable_language.html
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
      enable_language(CUDA)
    else()
      message(WARNING "EIGEN_TEST_CUDA was set, but no CUDA compiler was found. Disabling CUDA tests")
      set(EIGEN_TEST_CUDA OFF)
    endif()
  endif()
  include(EigenConfigureTesting)

  # TODO what was the point of EXCLUDE_FROM_ALL here?
  add_subdirectory(test)
  add_subdirectory(failtest)
endif()

option(EIGEN_BUILD_DOC "Enable creation of Eigen documentation" ON)
if(EIGEN_BUILD_DOC)
  include(EigenDocs)
  add_subdirectory(doc EXCLUDE_FROM_ALL)
endif()

include(CMakeDetermineFortranCompiler)
option(EIGEN_BUILD_BLAS "Toggles the building of the Eigen Blas library" ${CMAKE_Fortran_COMPILER})
option(EIGEN_BUILD_LAPACK "Toggles the building of the included Eigen LAPACK library" ${CMAKE_Fortran_COMPILER})
add_subdirectory(blas)
add_subdirectory(lapack)
add_subdirectory(unsupported)

add_subdirectory(demos EXCLUDE_FROM_ALL)

# must be after test and unsupported, for configuring buildtests.in
add_subdirectory(scripts EXCLUDE_FROM_ALL)

# TODO: consider also replacing EIGEN_BUILD_BTL by a custom target "make btl"?
if(EIGEN_BUILD_BTL)
  add_subdirectory(bench/btl EXCLUDE_FROM_ALL)
endif()

find_package(CLANG_FORMAT 9 EXACT)
if(CLANG_FORMAT_FOUND)
set(FORMAT_SOURCES)
list(APPEND FORMAT_SUBDIRS blas bench demos "doc" Eigen include lapack scripts share unsupported test failtest)
foreach(DIR ${FORMAT_SUBDIRS})
    set(ABS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${DIR})
    file(GLOB_RECURSE ${DIR}_SOURCES ${ABS_DIR}/*.cc ${ABS_DIR}/*.h ${ABS_DIR}/*.cpp ${ABS_DIR}/*.hpp ${ABS_DIR}/*.c)
    list(APPEND FORMAT_SOURCES ${${DIR}_SOURCES})
  endforeach()
    file(GLOB FORMAT_SOURCES_WITHOUTENDING LIST_DIRECTORIES false ${CMAKE_CURRENT_SOURCE_DIR}/Eigen/* ${CMAKE_CURRENT_SOURCE_DIR}/Eigen/CXX11/* ${CMAKE_CURRENT_SOURCE_DIR}/unsupported/Eigen/* ${CMAKE_CURRENT_SOURCE_DIR}/unsupported/Eigen/CXX11/*)
    list(FILTER FORMAT_SOURCES_WITHOUTENDING EXCLUDE REGEX ".*.txt$")
    list (APPEND FORMAT_SOURCES ${FORMAT_SOURCES_WITHOUTENDING})
    add_custom_target(format
    COMMAND ${CLANG_FORMAT_EXECUTABLE} -i -style=file ${FORMAT_SOURCES}
    DEPENDS ${FORMAT_SOURCES})
endif()

if(NOT WIN32)
  add_subdirectory(bench/spbench EXCLUDE_FROM_ALL)
endif()

configure_file(scripts/cdashtesting.cmake.in cdashtesting.cmake @ONLY)

if(EIGEN_BUILD_TESTING)
  ei_testing_print_summary()
endif()

message(STATUS "")
message(STATUS "Configured Eigen ${EIGEN_VERSION_NUMBER}")
message(STATUS "")

if(PROJECT_IS_TOP_LEVEL)
  string(TOLOWER "${CMAKE_GENERATOR}" cmake_generator_tolower)
  if(cmake_generator_tolower MATCHES "makefile")
    message(STATUS "Available targets (use: make TARGET):")
  else()
    message(STATUS "Available targets (use: cmake --build . --target TARGET):")
  endif()
  message(STATUS "---------+--------------------------------------------------------------")
  message(STATUS "Target   |   Description")
  message(STATUS "---------+--------------------------------------------------------------")
  message(STATUS "install  | Install Eigen. Headers will be installed to:")
  message(STATUS "         |     <CMAKE_INSTALL_PREFIX>/<INCLUDE_INSTALL_DIR>")
  message(STATUS "         |   Using the following values:")
  message(STATUS "         |     CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
  message(STATUS "         |     INCLUDE_INSTALL_DIR:  ${INCLUDE_INSTALL_DIR}")
  message(STATUS "         |   Change the install location of Eigen headers using:")
  message(STATUS "         |     cmake . -DCMAKE_INSTALL_PREFIX=yourprefix")
  message(STATUS "         |   Or:")
  message(STATUS "         |     cmake . -DINCLUDE_INSTALL_DIR=yourdir")
  message(STATUS "doc      | Generate the API documentation, requires Doxygen & LaTeX")
  if(EIGEN_BUILD_TESTING)
    message(STATUS "check    | Build and run the unit-tests. Read this page:")
    message(STATUS "         |   http://eigen.tuxfamily.org/index.php?title=Tests")
  endif()
  if(CLANG_FORMAT_FOUND)
    message(STATUS "format   | Formats the source code according to .clang-format file")
  endif()
  message(STATUS "blas     | Build BLAS library (not the same thing as Eigen)")
  message(STATUS "uninstall| Remove files installed by the install target")
  message(STATUS "---------+--------------------------------------------------------------")
  message(STATUS "")
endif()

# TODO why do we need both sets of variables?
set ( EIGEN_VERSION_STRING ${EIGEN_VERSION_NUMBER} )
set ( EIGEN_VERSION_MAJOR  ${EIGEN_WORLD_VERSION} )
set ( EIGEN_VERSION_MINOR  ${EIGEN_MAJOR_VERSION} )
set ( EIGEN_VERSION_PATCH  ${EIGEN_MINOR_VERSION} )

include (CMakePackageConfigHelpers)

# Imported target support
add_library (eigen INTERFACE)
target_compile_features(eigen INTERFACE cxx_std_14)
add_library (Eigen3::Eigen ALIAS eigen)
target_include_directories (eigen INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${INCLUDE_INSTALL_DIR}>
)


#############################################################################
# find how to link to the standard libraries                                #
#############################################################################
find_package(StandardMathLibrary)
if(NOT STANDARD_MATH_LIBRARY_FOUND)
  message(FATAL_ERROR
          "Can't link to the standard math library. Please report to the Eigen developers, telling them about your platform.")
else()
  # Don't add the math library for the installed version, there it might be different
  target_link_libraries(eigen INTERFACE $<BUILD_INTERFACE:${STANDARD_MATH_LIBRARY}>)
endif()

# Export as title case Eigen
set_target_properties (eigen PROPERTIES EXPORT_NAME Eigen)

install (TARGETS eigen EXPORT Eigen3Targets)

option(EIGEN_BUILD_CMAKE_PACKAGE "Enables the creation of EigenConfig.cmake and related files" ON)
if(EIGEN_BUILD_CMAKE_PACKAGE)
configure_package_config_file (
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Eigen3Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/Eigen3Config.cmake
  INSTALL_DESTINATION ${CMAKEPACKAGE_INSTALL_DIR}
  NO_SET_AND_CHECK_MACRO # Eigen does not provide legacy style defines
  NO_CHECK_REQUIRED_COMPONENTS_MACRO # Eigen does not provide components
)

# NOTE Remove the first code path once the minimum required CMake version is
# bumped to 3.14 or above.
if (CMAKE_VERSION VERSION_LESS 3.14)
  # Remove CMAKE_SIZEOF_VOID_P from Eigen3ConfigVersion.cmake since Eigen does
  # not depend on architecture specific settings or libraries. More
  # specifically, an Eigen3Config.cmake generated from a 64 bit target can be
  # used for 32 bit targets as well (and vice versa).
  set (_Eigen3_CMAKE_SIZEOF_VOID_P ${CMAKE_SIZEOF_VOID_P})
  unset (CMAKE_SIZEOF_VOID_P)
  write_basic_package_version_file (Eigen3ConfigVersion.cmake
                                    VERSION ${EIGEN_VERSION_NUMBER}
                                    COMPATIBILITY SameMajorVersion)
  set (CMAKE_SIZEOF_VOID_P ${_Eigen3_CMAKE_SIZEOF_VOID_P})
else (CMAKE_VERSION VERSION_LESS 3.14)
  write_basic_package_version_file (Eigen3ConfigVersion.cmake
                                    VERSION ${EIGEN_VERSION_NUMBER}
                                    COMPATIBILITY SameMajorVersion
                                    ARCH_INDEPENDENT)
endif (CMAKE_VERSION VERSION_LESS 3.14)

# The Eigen target will be located in the Eigen3 namespace. Other CMake
# targets can refer to it using Eigen3::Eigen.
export (TARGETS eigen NAMESPACE Eigen3:: FILE Eigen3Targets.cmake)
# Export Eigen3 package to CMake registry such that it can be easily found by
# CMake even if it has not been installed to a standard directory.
export (PACKAGE Eigen3)

install (EXPORT Eigen3Targets NAMESPACE Eigen3:: DESTINATION ${CMAKEPACKAGE_INSTALL_DIR})

install (FILES ${CMAKE_CURRENT_BINARY_DIR}/Eigen3Config.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/Eigen3ConfigVersion.cmake
         DESTINATION ${CMAKEPACKAGE_INSTALL_DIR})

# Add uninstall target
add_custom_target ( uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EigenUninstall.cmake)
endif()

if (EIGEN_SPLIT_TESTSUITE)
  ei_split_testsuite("${EIGEN_SPLIT_TESTSUITE}")
endif()
