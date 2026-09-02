#!/bin/bash
# Build Eigen benchmarks for a given ISA target.
#
# Expected environment variables:
#   EIGEN_CI_BUILDDIR         - build directory (default: .bench-build)
#   EIGEN_CI_CXX_COMPILER     - C++ compiler
#   EIGEN_CI_C_COMPILER        - C compiler
#   EIGEN_BENCH_ISA_FLAGS     - ISA-specific compiler flags (e.g. "-mavx2 -mfma")
#   EIGEN_BENCH_CMAKE_ARGS    - extra configure options, e.g. the GPU subtree's
#                               -DEIGEN_BENCH_CUDA=ON -DEIGEN_BENCH_CPU=OFF
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

set -ex

rootdir=$(pwd)
builddir=${EIGEN_CI_BUILDDIR:-.bench-build}
mkdir -p "${builddir}"
cd "${builddir}"

# The ISA jobs name versioned compilers (e.g. g++-10), which the CI image ships
# without unversioned c++/cc aliases, so those have to be passed to CMake. The
# GPU job leaves the compiler variables empty and builds with the image's own
# toolchain instead; passing -DCMAKE_CXX_COMPILER= would fail configuration with
# "No CMAKE_CXX_COMPILER could be found", so the arguments are only added when
# the variables are set.
compiler_args=()
if [ -n "${EIGEN_CI_CXX_COMPILER}" ]; then
  compiler_args+=("-DCMAKE_CXX_COMPILER=${EIGEN_CI_CXX_COMPILER}")
fi
if [ -n "${EIGEN_CI_C_COMPILER}" ]; then
  compiler_args+=("-DCMAKE_C_COMPILER=${EIGEN_CI_C_COMPILER}")
fi

# Install Google Benchmark from source if not already present.
# The common before_script already installs cmake/ninja; we only need
# git and ca-certificates for the clone.
if ! pkg-config --exists benchmark 2>/dev/null; then
  apt-get update -qq
  apt-get install -y --no-install-recommends git ca-certificates
  git clone --depth 1 --branch v1.9.1 https://github.com/google/benchmark.git /tmp/gbench
  cmake -G Ninja -S /tmp/gbench -B /tmp/gbench-build \
    "${compiler_args[@]}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBENCHMARK_ENABLE_TESTING=OFF \
    -DCMAKE_INSTALL_PREFIX=/usr/local
  cmake --build /tmp/gbench-build --target install
  rm -rf /tmp/gbench /tmp/gbench-build
fi

# Configure benchmarks.  ISA flags are passed via CMAKE_CXX_FLAGS so they
# apply globally to all benchmark targets.
cmake -G Ninja \
  "${compiler_args[@]}" \
  -DCMAKE_CXX_FLAGS="${EIGEN_BENCH_ISA_FLAGS}" \
  -DCMAKE_BUILD_TYPE=Release \
  ${EIGEN_BENCH_CMAKE_ARGS} \
  "${rootdir}/benchmarks"

# Build all benchmark targets.  The nightly/weekly scope filtering happens
# at run time, not build time.
cmake --build . -- -k0 || cmake --build . -- -k0 -j1

cd "${rootdir}"
