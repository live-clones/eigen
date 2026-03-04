# Install script for directory: /home/rmlarsen/eigen/rmlarsen3/eigen-new/benchmarks

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

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/rmlarsen/eigen/rmlarsen3/eigen-new/build-bench-clang-old/Core/cmake_install.cmake")
  include("/home/rmlarsen/eigen/rmlarsen3/eigen-new/build-bench-clang-old/Cholesky/cmake_install.cmake")
  include("/home/rmlarsen/eigen/rmlarsen3/eigen-new/build-bench-clang-old/LU/cmake_install.cmake")
  include("/home/rmlarsen/eigen/rmlarsen3/eigen-new/build-bench-clang-old/QR/cmake_install.cmake")
  include("/home/rmlarsen/eigen/rmlarsen3/eigen-new/build-bench-clang-old/SVD/cmake_install.cmake")
  include("/home/rmlarsen/eigen/rmlarsen3/eigen-new/build-bench-clang-old/Eigenvalues/cmake_install.cmake")
  include("/home/rmlarsen/eigen/rmlarsen3/eigen-new/build-bench-clang-old/Geometry/cmake_install.cmake")
  include("/home/rmlarsen/eigen/rmlarsen3/eigen-new/build-bench-clang-old/Sparse/cmake_install.cmake")
  include("/home/rmlarsen/eigen/rmlarsen3/eigen-new/build-bench-clang-old/FFT/cmake_install.cmake")
  include("/home/rmlarsen/eigen/rmlarsen3/eigen-new/build-bench-clang-old/Householder/cmake_install.cmake")
  include("/home/rmlarsen/eigen/rmlarsen3/eigen-new/build-bench-clang-old/Solvers/cmake_install.cmake")
  include("/home/rmlarsen/eigen/rmlarsen3/eigen-new/build-bench-clang-old/Tuning/cmake_install.cmake")
  include("/home/rmlarsen/eigen/rmlarsen3/eigen-new/build-bench-clang-old/BLAS/cmake_install.cmake")
  include("/home/rmlarsen/eigen/rmlarsen3/eigen-new/build-bench-clang-old/ThreadPool/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/rmlarsen/eigen/rmlarsen3/eigen-new/build-bench-clang-old/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
