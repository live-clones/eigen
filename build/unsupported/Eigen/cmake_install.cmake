# Install script for directory: /home/sbronder/open_source/eigen/unsupported/Eigen

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Devel" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE FILE FILES
    "/home/sbronder/open_source/eigen/unsupported/Eigen/AdolcForward"
    "/home/sbronder/open_source/eigen/unsupported/Eigen/AlignedVector3"
    "/home/sbronder/open_source/eigen/unsupported/Eigen/ArpackSupport"
    "/home/sbronder/open_source/eigen/unsupported/Eigen/AutoDiff"
    "/home/sbronder/open_source/eigen/unsupported/Eigen/BVH"
    "/home/sbronder/open_source/eigen/unsupported/Eigen/EulerAngles"
    "/home/sbronder/open_source/eigen/unsupported/Eigen/FFT"
    "/home/sbronder/open_source/eigen/unsupported/Eigen/IterativeSolvers"
    "/home/sbronder/open_source/eigen/unsupported/Eigen/KroneckerProduct"
    "/home/sbronder/open_source/eigen/unsupported/Eigen/LevenbergMarquardt"
    "/home/sbronder/open_source/eigen/unsupported/Eigen/MatrixFunctions"
    "/home/sbronder/open_source/eigen/unsupported/Eigen/MPRealSupport"
    "/home/sbronder/open_source/eigen/unsupported/Eigen/NNLS"
    "/home/sbronder/open_source/eigen/unsupported/Eigen/NonLinearOptimization"
    "/home/sbronder/open_source/eigen/unsupported/Eigen/NumericalDiff"
    "/home/sbronder/open_source/eigen/unsupported/Eigen/OpenGLSupport"
    "/home/sbronder/open_source/eigen/unsupported/Eigen/Polynomials"
    "/home/sbronder/open_source/eigen/unsupported/Eigen/SparseExtra"
    "/home/sbronder/open_source/eigen/unsupported/Eigen/SpecialFunctions"
    "/home/sbronder/open_source/eigen/unsupported/Eigen/Splines"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Devel" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE DIRECTORY FILES "/home/sbronder/open_source/eigen/unsupported/Eigen/src" FILES_MATCHING REGEX "/[^/]*\\.h$")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/sbronder/open_source/eigen/build/unsupported/Eigen/CXX11/cmake_install.cmake")

endif()

