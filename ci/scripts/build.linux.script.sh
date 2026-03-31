#!/bin/bash

set -x

# Create and enter build directory.
rootdir=`pwd`
mkdir -p ${EIGEN_CI_BUILDDIR}
cd ${EIGEN_CI_BUILDDIR}

# Configure build.
cmake -G Ninja                                                   \
  -DCMAKE_CXX_COMPILER=${EIGEN_CI_CXX_COMPILER}                  \
  -DCMAKE_C_COMPILER=${EIGEN_CI_C_COMPILER}                      \
  -DCMAKE_CXX_COMPILER_TARGET=${EIGEN_CI_CXX_COMPILER_TARGET}    \
  ${EIGEN_CI_ADDITIONAL_ARGS} ${rootdir}

target=""
if [[ ${EIGEN_CI_BUILD_TARGET} ]]; then
  target="--target ${EIGEN_CI_BUILD_TARGET}"
fi

# Builds (particularly gcc) sometimes get killed, potentially when running
# out of resources.  In that case, keep trying to build the remaining
# targets (k0), then try to build again with a single thread (j1) to minimize
# resource use.
# EIGEN_CI_BUILD_JOBS can be set to limit parallelism for memory-hungry
# compilers (e.g. NVHPC).
jobs=""
if [[ -n "${EIGEN_CI_BUILD_JOBS}" ]]; then
  jobs="-j${EIGEN_CI_BUILD_JOBS}"
fi

# For phony meta-targets (e.g. buildtests), shuffle the dependency list so
# that memory-hungry compilations (like bdcsvd with nvc++) are spread out
# instead of all running at once.  Falls back to the normal build if the
# target is not a phony or if ninja is not available.
shuffled=false
if [[ -n "${EIGEN_CI_BUILD_TARGET}" ]] && command -v ninja >/dev/null 2>&1; then
  deps=$(ninja -t query "${EIGEN_CI_BUILD_TARGET}" 2>/dev/null \
         | awk '/^  input:/{found=1; next} /^  outputs:/{found=0} found && /^    /{print $1}')
  if [[ -n "$deps" ]]; then
    shuffled_deps=$(echo "$deps" | shuf)
    ninja -k0 ${jobs} ${shuffled_deps} || ninja -k0 -j1 ${shuffled_deps}
    shuffled=true
  fi
fi

if [[ "$shuffled" != "true" ]]; then
  cmake --build . ${target} -- -k0 ${jobs} || cmake --build . ${target} -- -k0 -j1
fi

# Return to root directory.
cd ${rootdir}

set +x
