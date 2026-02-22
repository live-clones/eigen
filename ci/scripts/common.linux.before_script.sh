#!/bin/bash

set -x

echo "Running ${CI_JOB_NAME}"

# Get architecture and display CI configuration.
export ARCH=`uname -m`
export NPROC=`nproc`
echo "arch=$ARCH, target=${EIGEN_CI_TARGET_ARCH}"
echo "Processors: ${NPROC}"
echo "CI Variables:"
export | grep EIGEN

# Set noninteractive, otherwise tzdata may be installed and prompt for a
# geographical region.
export DEBIAN_FRONTEND=noninteractive
apt-get update -y > /dev/null
apt-get install -y --no-install-recommends ninja-build cmake git ca-certificates > /dev/null

# Ensure cmake >= 3.14 (required for FetchContent_MakeAvailable / gtest).
# On older distros (e.g. Ubuntu 18.04), the system cmake may be too old.
CMAKE_MAJOR=$(cmake --version | head -1 | cut -d' ' -f3 | cut -d. -f1)
CMAKE_MINOR=$(cmake --version | head -1 | cut -d' ' -f3 | cut -d. -f2)
if [ "$CMAKE_MAJOR" -eq 3 ] && [ "$CMAKE_MINOR" -lt 14 ]; then
  apt-get install -y --no-install-recommends python3-pip > /dev/null 2>&1
  pip3 install --quiet cmake
  hash -r
fi

# Install required dependencies and set up compilers.
# These are required even for testing to ensure that dynamic runtime libraries
# are available.
if [[ "$ARCH" == "${EIGEN_CI_TARGET_ARCH}" || "${EIGEN_CI_TARGET_ARCH}" == "any" ]]; then
  apt-get install -y --no-install-recommends ${EIGEN_CI_INSTALL} > /dev/null;
  export EIGEN_CI_CXX_IMPLICIT_INCLUDE_DIRECTORIES="";
  export EIGEN_CI_CXX_COMPILER_TARGET="";
else
  apt-get install -y --no-install-recommends ${EIGEN_CI_CROSS_INSTALL} > /dev/null;
  export EIGEN_CI_C_COMPILER=${EIGEN_CI_CROSS_C_COMPILER};
  export EIGEN_CI_CXX_COMPILER=${EIGEN_CI_CROSS_CXX_COMPILER};
  export EIGEN_CI_CXX_COMPILER_TARGET=${EIGEN_CI_CROSS_TARGET_TRIPLE};
  # Tell the compiler where to find headers and libraries if using clang.
  # NOTE: this breaks GCC since it messes with include path order
  #       (https://gcc.gnu.org/bugzilla/show_bug.cgi?id=70129)
  if [[ "${EIGEN_CI_CROSS_CXX_COMPILER}" == *"clang"* ]]; then
    export CPLUS_INCLUDE_PATH="/usr/${EIGEN_CI_CROSS_TARGET_TRIPLE}/include";
    export LIBRARY_PATH="/usr/${EIGEN_CI_CROSS_TARGET_TRIPLE}/lib64";
  fi
fi

echo "Compilers: ${EIGEN_CI_C_COMPILER} ${EIGEN_CI_CXX_COMPILER}"

if [ -n "$EIGEN_CI_BEFORE_SCRIPT" ]; then eval "$EIGEN_CI_BEFORE_SCRIPT"; fi

set +x
