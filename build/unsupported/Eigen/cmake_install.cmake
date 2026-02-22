# Install script for directory: /home/rmlarsen/eigen/rmlarsen/eigen-new/unsupported/Eigen

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
  set(CMAKE_INSTALL_SO_NO_EXE "1")
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
    "/home/rmlarsen/eigen/rmlarsen/eigen-new/unsupported/Eigen/AdolcForward"
    "/home/rmlarsen/eigen/rmlarsen/eigen-new/unsupported/Eigen/AlignedVector3"
    "/home/rmlarsen/eigen/rmlarsen/eigen-new/unsupported/Eigen/ArpackSupport"
    "/home/rmlarsen/eigen/rmlarsen/eigen-new/unsupported/Eigen/AutoDiff"
    "/home/rmlarsen/eigen/rmlarsen/eigen-new/unsupported/Eigen/BVH"
    "/home/rmlarsen/eigen/rmlarsen/eigen-new/unsupported/Eigen/EulerAngles"
    "/home/rmlarsen/eigen/rmlarsen/eigen-new/unsupported/Eigen/FFT"
    "/home/rmlarsen/eigen/rmlarsen/eigen-new/unsupported/Eigen/IterativeSolvers"
    "/home/rmlarsen/eigen/rmlarsen/eigen-new/unsupported/Eigen/KroneckerProduct"
    "/home/rmlarsen/eigen/rmlarsen/eigen-new/unsupported/Eigen/LevenbergMarquardt"
    "/home/rmlarsen/eigen/rmlarsen/eigen-new/unsupported/Eigen/MatrixFunctions"
    "/home/rmlarsen/eigen/rmlarsen/eigen-new/unsupported/Eigen/MPRealSupport"
    "/home/rmlarsen/eigen/rmlarsen/eigen-new/unsupported/Eigen/NNLS"
    "/home/rmlarsen/eigen/rmlarsen/eigen-new/unsupported/Eigen/NonLinearOptimization"
    "/home/rmlarsen/eigen/rmlarsen/eigen-new/unsupported/Eigen/NumericalDiff"
    "/home/rmlarsen/eigen/rmlarsen/eigen-new/unsupported/Eigen/OpenGLSupport"
    "/home/rmlarsen/eigen/rmlarsen/eigen-new/unsupported/Eigen/Polynomials"
    "/home/rmlarsen/eigen/rmlarsen/eigen-new/unsupported/Eigen/SparseExtra"
    "/home/rmlarsen/eigen/rmlarsen/eigen-new/unsupported/Eigen/SpecialFunctions"
    "/home/rmlarsen/eigen/rmlarsen/eigen-new/unsupported/Eigen/Splines"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Devel" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE DIRECTORY FILES "/home/rmlarsen/eigen/rmlarsen/eigen-new/unsupported/Eigen/src" FILES_MATCHING REGEX "/[^/]*\\.h$")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/rmlarsen/eigen/rmlarsen/eigen-new/build/unsupported/Eigen/CXX11/cmake_install.cmake")
endif()

