#!/bin/bash

# Initialize default variables used by the CI.
export EIGEN_CI_ADDITIONAL_ARGS=""
export EIGEN_CI_BEFORE_SCRIPT=""
export EIGEN_CI_BUILD_TARGET="buildtests"
export EIGEN_CI_BUILDDIR=".build"
export EIGEN_CI_C_COMPILER="clang"
export EIGEN_CI_CXX_COMPILER="clang++"
export EIGEN_CI_TEST_CUSTOM_CXX_FLAGS=""
export EIGEN_CI_CTEST_LABEL="Official"
export EIGEN_CI_CTEST_REGEX=""
export EIGEN_CI_CTEST_ARGS=""

# Hexagon-specific variables
export EIGEN_CI_HEXAGON_ARCH="v68"
export EIGEN_CI_HEXAGON_TOOLCHAIN_VERSION="20.1.4"
export EIGEN_CI_HEXAGON_HVX_LENGTH="128B"
export EIGEN_CI_HEXAGON_EMULATION="false"
export EIGEN_CI_TEST_TIMEOUT="300"

