#!/bin/bash
# Cross-compiles the Windows unit tests on a Linux runner using MSVC via Wine.
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

set -x

rootdir=$(pwd)

# Puts the MSVC wrappers (cl, link, ...) on PATH.
. "${rootdir}/ci/scripts/setup.msvc-wine.sh"

mkdir -p ${EIGEN_CI_BUILDDIR}
cd ${EIGEN_CI_BUILDDIR}

# Split EIGEN_CI_ADDITIONAL_ARGS on spaces, as the linux build script does.
CC=cl CXX=cl cmake -G Ninja                                                 \
  -DCMAKE_SYSTEM_NAME=Windows                                               \
  -DCMAKE_BUILD_TYPE=MinSizeRel                                             \
  -DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded                            \
  -DEIGEN_TEST_CUSTOM_CXX_FLAGS="${EIGEN_CI_TEST_CUSTOM_CXX_FLAGS}"         \
  ${EIGEN_CI_ADDITIONAL_ARGS} "${rootdir}"

# CTest bakes the absolute build directory into every add_test() in
# CTestTestfile.cmake.  Record it so the Windows test job can rewrite those
# paths to wherever the artifacts are unpacked; see
# ci/scripts/test.windows-cross.before_script.ps1.
pwd > .eigen_ci_builddir

# Record the toolset these binaries were built with, so the Windows test job can
# report it next to the runtime it actually has.
printf '%s\n' "${EIGEN_CI_MSVC_TOOLSET}" > .eigen_ci_msvc_toolset

target=""
if [[ ${EIGEN_CI_BUILD_TARGET} ]]; then
  target="--target ${EIGEN_CI_BUILD_TARGET}"
fi

cmake --build . ${target} -- -k0
success=$?

cd ${rootdir}

exit ${success}
